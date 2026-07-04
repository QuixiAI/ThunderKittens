// DeepSeek MLA decode (paged, MQA latent cache), CUDA/SM86 port of ThunderMittens
// kernels/mla core: mla_q_norm_rope (GPT-J interleaved RoPE, optional RMSNorm),
// mla_kv_insert_fp8 (V4 packed cache: 448 NoPE -> e4m3 with per-64 UE8M0 scales,
// 64 RoPE bf16; 576B data + 8B scale rows), mla_decode (bf16 absorb path:
// QK=LATENT+ROPE dot, LATENT-only accumulate), mla_decode_fp8 (dequant-on-read).
// Partitioned + sparse variants and the bf16 insert are the follow-up; the
// D=512 v2 reducer already exists in paged_attn_v2.cu.
//
// Build:
//   /usr/local/cuda/bin/nvcc mla.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -I../quant -o mla.out
#include "quant_formats.cuh"   // tmq::e4m3_decode
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

using bf16 = __nv_bfloat16;

namespace tms {

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// float -> e4m3 RNE encode (same as quant_rt's, local copy to keep this file standalone)
__device__ __forceinline__ uint8_t e4m3_encode(float x) {
    const unsigned sign = (x < 0.0f) ? 0x80u : 0x00u;
    float a = fabsf(x);
    if (!(a < 448.0f)) return uint8_t(sign | 0x7Eu);
    if (a < 0.0009765625f) return uint8_t(sign | unsigned(int(rintf(a * 512.0f))));
    int e; frexpf(a, &e);
    const int E = e - 1;
    if (E < -6) {
        int mant = int(rintf(a * 512.0f));
        if (mant >= 8) return uint8_t(sign | (1u << 3));
        return uint8_t(sign | unsigned(mant));
    }
    const float two_m = a / exp2f(float(E));
    int mant = int(rintf((two_m - 1.0f) * 8.0f));
    int exp = E + 7;
    if (mant >= 8) { mant = 0; exp += 1; }
    if (exp > 15 || (exp == 15 && mant > 6)) return uint8_t(sign | 0x7Eu);
    return uint8_t(sign | (unsigned(exp) << 3) | unsigned(mant));
}

// ---- fused Q path: optional RMSNorm over full head dim + GPT-J interleaved RoPE on the
// last rope_dim dims. One warp per (token, head); lane owns D/32 CONTIGUOUS elements so
// every interleaved pair lives in one lane. norm_mode: 0 none, 1 rms, 2 rms*weight. ----
template <int D>
__global__ void mla_q_norm_rope(const bf16* q, const bf16* cosb, const bf16* sinb,
                                const int* positions, bf16* out, int num_heads,
                                int nope_dim, int rope_dim, int norm_mode, float eps,
                                const bf16* norm_weight) {
    static_assert(D % 64 == 0, "head_dim must be divisible by 64");
    constexpr int PER_LANE = D / 32;
    const int row = blockIdx.x;                  // (token, head)
    const int token = row / num_heads;
    const int lane = threadIdx.x;
    const int pos = positions[token];
    const int rope_half = rope_dim / 2;
    const int64_t base = int64_t(row) * D + lane * PER_LANE;

    float rms = 1.0f;
    if (norm_mode != 0) {
        float ss = 0.0f;
        #pragma unroll
        for (int k = 0; k < PER_LANE; k++) { const float v = float(q[base + k]); ss += v * v; }
        ss = warp_sum(ss);
        rms = rsqrtf(ss / float(D) + eps);
    }
    const int64_t wbase = int64_t(lane) * PER_LANE;
    const int64_t csbase = int64_t(pos) * rope_half;
    #pragma unroll
    for (int k = 0; k < PER_LANE; k += 2) {
        const int g0 = lane * PER_LANE + k;      // even global index (pair start)
        float v0 = float(q[base + k]) * rms;
        float v1 = float(q[base + k + 1]) * rms;
        if (norm_mode == 2) {
            v0 *= float(norm_weight[wbase + k]);
            v1 *= float(norm_weight[wbase + k + 1]);
        }
        if (g0 >= nope_dim) {                    // interleaved rotate within the pair
            const int p = (g0 - nope_dim) / 2;
            const float c = float(cosb[csbase + p]), s = float(sinb[csbase + p]);
            const float r0 = v0 * c - v1 * s, r1 = v0 * s + v1 * c;
            v0 = r0; v1 = r1;
        }
        out[base + k] = bf16(v0);
        out[base + k + 1] = bf16(v1);
    }
}

// ---- V4 packed insert: kv (T, 512) -> data_cache (nb, bs, 576) + scale_cache (nb, bs, 8).
// NoPE [0,448): e4m3 with per-64-elem UE8M0 power-of-2 scale; RoPE [448,512): GPT-J
// interleaved rotate, stored bf16. Lane owns 16 contiguous elems; 4 lanes = one 64-block. ----
__global__ void mla_kv_insert_fp8(const bf16* kv, const bf16* cosb, const bf16* sinb,
                                  const int* positions, const int64_t* slot_mapping,
                                  uint8_t* data_cache, uint8_t* scale_cache, int block_size) {
    constexpr int LAT = 512, NOPE = 448, PER_LANE = 16, NOPE_LANES = NOPE / PER_LANE;  // 28
    const int token = blockIdx.x, lane = threadIdx.x;
    const int64_t slot = slot_mapping[token];
    if (slot < 0) return;
    const int64_t dslot = (slot / block_size) * block_size + slot % block_size;
    const int64_t dst_data = dslot * 576, dst_scale = dslot * 8;
    const int pos = positions[token];
    const int64_t kbase = int64_t(token) * LAT + lane * PER_LANE;

    float v[PER_LANE];
    #pragma unroll
    for (int k = 0; k < PER_LANE; k++) v[k] = float(kv[kbase + k]);

    float amax = 0.0f;
    #pragma unroll
    for (int k = 0; k < PER_LANE; k++) amax = fmaxf(amax, fabsf(v[k]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float exponent = ceilf(log2f(fmaxf(amax, 1e-4f) / 448.0f));
    const float inv_scale = exp2f(-exponent);

    if (lane < NOPE_LANES) {
        #pragma unroll
        for (int k = 0; k < PER_LANE; k++)
            data_cache[dst_data + lane * PER_LANE + k] = e4m3_encode(v[k] * inv_scale);
        if ((lane & 3) == 0) {
            const int e = min(max(int(exponent) + 127, 0), 255);
            scale_cache[dst_scale + lane / 4] = uint8_t(e);
        }
    } else {
        const int rl = (lane - NOPE_LANES) * PER_LANE;   // rope-local start: 0,16,32,48
        bf16* rope_out = reinterpret_cast<bf16*>(data_cache + dst_data + NOPE);
        #pragma unroll
        for (int j = 0; j < PER_LANE; j += 2) {
            const int p = (rl + j) / 2;
            const float c = float(cosb[int64_t(pos) * 32 + p]);
            const float s = float(sinb[int64_t(pos) * 32 + p]);
            rope_out[rl + j]     = bf16(v[j] * c - v[j + 1] * s);
            rope_out[rl + j + 1] = bf16(v[j] * s + v[j + 1] * c);
        }
    }
    if (lane == 0) scale_cache[dst_scale + 7] = 0;
}

// ---- bf16 absorb-path decode: score over QK=LATENT+ROPE, accumulate over LATENT only ----
template <int LATENT, int ROPE>
__global__ void mla_decode(const bf16* q, const bf16* kv_cache, const int* block_table,
                           const int* context_lens, bf16* out,
                           int block_size, int bt_stride, float scale, int num_heads) {
    constexpr int QK = LATENT + ROPE, VQK = QK / 32, VAV = LATENT / 32;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
    const int context_len = context_lens[batch];
    const int64_t q_base = (int64_t(batch) * num_heads + head) * QK;

    float qv[VQK], acc[VAV];
    #pragma unroll
    for (int i = 0; i < VQK; i++) qv[i] = float(q[q_base + lane + 32 * i]);
    #pragma unroll
    for (int i = 0; i < VAV; i++) acc[i] = 0.0f;
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = 0; t < context_len; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        const int64_t base = (int64_t(block) * block_size + slot) * QK;   // MQA: no head axis
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VQK; i++) partial += qv[i] * float(kv_cache[base + lane + 32 * i]);
        const float score = warp_sum(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VAV; i++)
            acc[i] = acc[i] * alpha + beta * float(kv_cache[base + lane + 32 * i]);
        l = l * alpha + beta;
        m = nm;
    }
    const int64_t out_base = (int64_t(batch) * num_heads + head) * LATENT;
    #pragma unroll
    for (int i = 0; i < VAV; i++)
        out[out_base + lane + 32 * i] = (l == 0.0f) ? bf16(0.0f) : bf16(acc[i] / l);
}

// ---- V4 fp8 decode: dequant-on-read over the packed 576/8 cache, score+value over 512 ----
__global__ void mla_decode_fp8(const bf16* q, const uint8_t* data_cache, const uint8_t* scale_cache,
                               const int* block_table, const int* context_lens, bf16* out,
                               int block_size, int bt_stride, float scale, int num_heads) {
    constexpr int LATENT = 512, NOPE = 448, VPL = LATENT / 32;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
    const int context_len = context_lens[batch];
    const int64_t q_base = (int64_t(batch) * num_heads + head) * LATENT;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = 0; t < context_len; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        const int64_t dslot = int64_t(block) * block_size + slot;
        const int64_t dbase = dslot * 576, sbase = dslot * 8;
        const bf16* rope = reinterpret_cast<const bf16*>(data_cache + dbase + NOPE);

        float lat[VPL], partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) {
            const int d = lane + 32 * i;
            if (d < NOPE) {
                const int e = int(scale_cache[sbase + d / 64]);
                lat[i] = tmq::e4m3_decode(data_cache[dbase + d]) * exp2f(float(e - 127));
            } else {
                lat[i] = float(rope[d - NOPE]);
            }
            partial += qv[i] * lat[i];
        }
        const float score = warp_sum(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++) acc[i] = acc[i] * alpha + beta * lat[i];
        l = l * alpha + beta;
        m = nm;
    }
    const int64_t out_base = (int64_t(batch) * num_heads + head) * LATENT;
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        out[out_base + lane + 32 * i] = (l == 0.0f) ? bf16(0.0f) : bf16(acc[i] / l);
}

}  // namespace tms

// ============================== test harness ==============================
using namespace tms;
static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }
static double b2d(bf16 x) { return double(__bfloat162float(x)); }

int main() {
    srand(41);
    int rc = 0;

    // ---- 1. mla_q_norm_rope: D 192 (128+64) and 512 (448+64), norm modes 0/1/2 ----
    for (int D : {192, 512}) {
        const int T_ = 12, H = 4, nope = D - 64, rope_dim = 64, rope_half = 32, PMAX = 64;
        const int M = T_ * H;
        std::vector<bf16> q(size_t(M) * D), W(D), cb(size_t(PMAX) * rope_half), sb(cb.size());
        std::vector<int> pos(T_);
        for (auto& x : q) x = bf16(frand());
        for (auto& x : W) x = bf16(frand() * 0.5f + 1.0f);
        for (int p = 0; p < PMAX; p++) for (int i = 0; i < rope_half; i++) {
            const double th = p / pow(10000.0, 2.0 * i / rope_dim);
            cb[size_t(p) * rope_half + i] = bf16(float(cos(th)));
            sb[size_t(p) * rope_half + i] = bf16(float(sin(th)));
        }
        for (int t = 0; t < T_; t++) pos[t] = (t * 5) % PMAX;
        bf16 *dq, *dW, *dc, *ds, *dout; int* dpos;
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dW, D * 2);
        cudaMalloc(&dc, cb.size() * 2); cudaMalloc(&ds, sb.size() * 2);
        cudaMalloc(&dout, q.size() * 2); cudaMalloc(&dpos, T_ * 4);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dW, W.data(), D * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dc, cb.data(), cb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, sb.data(), sb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dpos, pos.data(), T_ * 4, cudaMemcpyHostToDevice);
        for (int mode = 0; mode < 3; mode++) {
            if (D == 192) mla_q_norm_rope<192><<<M, 32>>>(dq, dc, ds, dpos, dout, H, nope, rope_dim, mode, 1e-6f, dW);
            else          mla_q_norm_rope<512><<<M, 32>>>(dq, dc, ds, dpos, dout, H, nope, rope_dim, mode, 1e-6f, dW);
            cudaDeviceSynchronize();
            std::vector<bf16> got(q.size());
            cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
            double maxd = 0;
            for (int r = 0; r < M; r++) {
                const int t = r / H;
                std::vector<double> v(D);
                double ss = 0;
                for (int d = 0; d < D; d++) { v[d] = b2d(q[size_t(r) * D + d]); ss += v[d] * v[d]; }
                if (mode) {
                    const double rms = 1.0 / sqrt(ss / D + 1e-6);
                    for (int d = 0; d < D; d++) v[d] *= rms * (mode == 2 ? b2d(W[d]) : 1.0);
                }
                for (int g = nope; g < D; g += 2) {
                    const int p = (g - nope) / 2;
                    const double c = b2d(cb[size_t(pos[t]) * rope_half + p]);
                    const double s = b2d(sb[size_t(pos[t]) * rope_half + p]);
                    const double r0 = v[g] * c - v[g + 1] * s, r1 = v[g] * s + v[g + 1] * c;
                    v[g] = r0; v[g + 1] = r1;
                }
                for (int d = 0; d < D; d++)
                    maxd = std::max(maxd, std::abs(b2d(got[size_t(r) * D + d]) - v[d]));
            }
            printf("mla_q_norm_rope D=%-3d mode=%d max diff %.5f (%s)\n", D, mode, maxd, maxd < 2e-2 ? "PASS" : "FAIL");
            rc |= !(maxd < 2e-2);
        }
        cudaFree(dq); cudaFree(dW); cudaFree(dc); cudaFree(ds); cudaFree(dout); cudaFree(dpos);
    }

    // ---- 2. bf16 absorb decode <512,64> vs fp64 oracle ----
    {
        const int B = 2, H = 4, BS = 16, MAXB = 12, LAT = 512, ROPE = 64, QK = LAT + ROPE;
        const int ctx[B] = {57, 130};
        const int nb = B * MAXB + 1;
        std::vector<int> bt(B * MAXB, -1);
        int nxt = 1;
        for (int b = 0; b < B; b++) for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = nxt++;
        std::vector<bf16> cache(size_t(nb) * BS * QK), q(size_t(B) * H * QK);
        for (auto& x : cache) x = bf16(frand() * 0.2f);
        for (auto& x : q) x = bf16(frand() * 0.3f);
        bf16 *dc_, *dq, *dout; int *dbt, *dctx;
        cudaMalloc(&dc_, cache.size() * 2); cudaMalloc(&dq, q.size() * 2);
        cudaMalloc(&dout, size_t(B) * H * LAT * 2);
        cudaMalloc(&dbt, bt.size() * 4); cudaMalloc(&dctx, B * 4);
        cudaMemcpy(dc_, cache.data(), cache.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dctx, ctx, B * 4, cudaMemcpyHostToDevice);
        const float scale = 1.0f / sqrtf(float(QK));
        mla_decode<512, 64><<<dim3(H, B), 32>>>(dq, dc_, dbt, dctx, dout, BS, MAXB, scale, H);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
        std::vector<bf16> got(size_t(B) * H * LAT);
        cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            std::vector<double> sc; std::vector<int64_t> bases;
            for (int t = 0; t < ctx[b]; t++) {
                const int blk = bt[b * MAXB + t / BS];
                if (blk < 0) continue;
                const int64_t base = (int64_t(blk) * BS + t % BS) * QK;
                double s = 0;
                for (int d = 0; d < QK; d++)
                    s += b2d(q[(size_t(b) * H + h) * QK + d]) * b2d(cache[base + d]);
                sc.push_back(s * scale); bases.push_back(base);
            }
            double mx = -1e300;
            for (double s : sc) mx = std::max(mx, s);
            std::vector<double> o(LAT, 0.0); double Z = 0;
            for (size_t i = 0; i < sc.size(); i++) {
                const double w = exp(sc[i] - mx);
                for (int d = 0; d < LAT; d++) o[d] += w * b2d(cache[bases[i] + d]);
                Z += w;
            }
            for (int d = 0; d < LAT; d++)
                maxd = std::max(maxd, std::abs(b2d(got[(size_t(b) * H + h) * LAT + d]) - o[d] / Z));
        }
        printf("mla_decode<512,64> bf16       max diff %.5f (%s)\n", maxd, maxd < 6e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 6e-3);
        cudaFree(dc_); cudaFree(dq); cudaFree(dout); cudaFree(dbt); cudaFree(dctx);
    }

    // ---- 3. fp8 insert -> fp8 decode round trip vs oracle over the dequantized cache ----
    {
        const int B = 2, H = 4, BS = 16, MAXB = 12, LAT = 512, NOPE = 448, PMAX = 64;
        const int ctx[B] = {49, 90};
        const int nb = B * MAXB + 1;
        std::vector<int> bt(B * MAXB, -1);
        int nxt = 1;
        for (int b = 0; b < B; b++) for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = nxt++;
        const int T_ = ctx[0] + ctx[1];
        std::vector<bf16> kv(size_t(T_) * LAT), cb(size_t(PMAX) * 32), sb(cb.size());
        std::vector<int> pos(T_);
        std::vector<int64_t> slots(T_);
        for (auto& x : kv) x = bf16(frand() * 0.4f);
        for (int p = 0; p < PMAX; p++) for (int i = 0; i < 32; i++) {
            const double th = p / pow(10000.0, i / 32.0);
            cb[size_t(p) * 32 + i] = bf16(float(cos(th)));
            sb[size_t(p) * 32 + i] = bf16(float(sin(th)));
        }
        for (int b = 0, i = 0; b < B; b++) for (int t = 0; t < ctx[b]; t++, i++) {
            pos[i] = t % PMAX;
            slots[i] = int64_t(bt[b * MAXB + t / BS]) * BS + t % BS;
        }
        uint8_t *ddata, *dscale; bf16 *dkv, *dcb, *dsb; int* dpos; int64_t* dslot;
        cudaMalloc(&ddata, size_t(nb) * BS * 576); cudaMalloc(&dscale, size_t(nb) * BS * 8);
        cudaMalloc(&dkv, kv.size() * 2); cudaMalloc(&dcb, cb.size() * 2); cudaMalloc(&dsb, sb.size() * 2);
        cudaMalloc(&dpos, T_ * 4); cudaMalloc(&dslot, T_ * 8);
        cudaMemcpy(dkv, kv.data(), kv.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dcb, cb.data(), cb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dsb, sb.data(), sb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dpos, pos.data(), T_ * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dslot, slots.data(), T_ * 8, cudaMemcpyHostToDevice);
        mla_kv_insert_fp8<<<T_, 32>>>(dkv, dcb, dsb, dpos, dslot, ddata, dscale, BS);
        cudaDeviceSynchronize();

        // rebuild the dequantized cache on the host from the DEVICE cache bytes
        std::vector<uint8_t> data(size_t(nb) * BS * 576), scl(size_t(nb) * BS * 8);
        cudaMemcpy(data.data(), ddata, data.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(scl.data(), dscale, scl.size(), cudaMemcpyDeviceToHost);
        auto dec = [](uint8_t v) {
            float mag;
            if (v & 0x78) { int e = (v >> 3) & 0xF, mm = v & 7; mag = std::ldexp(1.0f + mm / 8.0f, e - 7); }
            else mag = float(v & 7) * 0.001953125f;
            return double((v & 0x80) ? -mag : mag);
        };
        std::vector<double> dcache(size_t(nb) * BS * LAT, 0.0);
        for (size_t s = 0; s < size_t(nb) * BS; s++) {
            for (int d = 0; d < NOPE; d++)
                dcache[s * LAT + d] = dec(data[s * 576 + d]) * std::ldexp(1.0, int(scl[s * 8 + d / 64]) - 127);
            const bf16* rope = reinterpret_cast<const bf16*>(&data[s * 576 + NOPE]);
            for (int d = NOPE; d < LAT; d++) dcache[s * LAT + d] = b2d(rope[d - NOPE]);
        }

        std::vector<bf16> q(size_t(B) * H * LAT);
        for (auto& x : q) x = bf16(frand() * 0.3f);
        bf16 *dq, *dout; int *dbt, *dctx;
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dout, q.size() * 2);
        cudaMalloc(&dbt, bt.size() * 4); cudaMalloc(&dctx, B * 4);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dctx, ctx, B * 4, cudaMemcpyHostToDevice);
        const float scale = 1.0f / sqrtf(float(LAT));
        mla_decode_fp8<<<dim3(H, B), 32>>>(dq, ddata, dscale, dbt, dctx, dout, BS, MAXB, scale, H);
        cudaDeviceSynchronize();
        std::vector<bf16> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            std::vector<double> sc; std::vector<size_t> ss_;
            for (int t = 0; t < ctx[b]; t++) {
                const int blk = bt[b * MAXB + t / BS];
                if (blk < 0) continue;
                const size_t s = size_t(blk) * BS + t % BS;
                double sv = 0;
                for (int d = 0; d < LAT; d++)
                    sv += b2d(q[(size_t(b) * H + h) * LAT + d]) * dcache[s * LAT + d];
                sc.push_back(sv * scale); ss_.push_back(s);
            }
            double mx = -1e300;
            for (double s : sc) mx = std::max(mx, s);
            std::vector<double> o(LAT, 0.0); double Z = 0;
            for (size_t i = 0; i < sc.size(); i++) {
                const double w = exp(sc[i] - mx);
                for (int d = 0; d < LAT; d++) o[d] += w * dcache[ss_[i] * LAT + d];
                Z += w;
            }
            for (int d = 0; d < LAT; d++)
                maxd = std::max(maxd, std::abs(b2d(got[(size_t(b) * H + h) * LAT + d]) - o[d] / Z));
        }
        printf("mla fp8 insert+decode (V4)    max diff %.5f (%s)\n", maxd, maxd < 6e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 6e-3);
        cudaFree(ddata); cudaFree(dscale); cudaFree(dkv); cudaFree(dcb); cudaFree(dsb);
        cudaFree(dpos); cudaFree(dslot); cudaFree(dq); cudaFree(dout); cudaFree(dbt); cudaFree(dctx);
    }
    return rc;
}
