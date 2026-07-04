// Partitioned paged decode attention (vLLM v2 shape) + cascade/shared-prefix,
// CUDA/SM86 port of ThunderMittens kernels/paged_attn_v2.
//
// Each (head, batch) query splits across num_partitions KV slices:
//   partition: local online-softmax over [p*PS, min((p+1)*PS, ctx)) ->
//     max_logits (B,H,P), exp_sums (B,H,P), tmp_out (B,H,P,D) locally normalized
//   reduce: m* = max_p m_p; out = sum_p tmp_out_p * S_p e^{m_p-m*} / (sum + 1e-6)
// fp8 partition dequantizes e4m3 codes on read (per-KV-head scales); partials
// stay fp32 so the reduce is format-agnostic. Cascade prefix emits the SAME
// partial layout from a shared contiguous prefix, so prefix ++ suffix partials
// concatenate along P and fold through the same reduce (flashinfer merge_states).
// The reduce is also instantiated at D=512 for MLA.
//
// Build:
//   /usr/local/cuda/bin/nvcc paged_attn_v2.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -I../quant -o paged_attn_v2.out
#include "quant_formats.cuh"   // e4m3_decode
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

namespace tms {

#define NEG_INF (-3.4028234663852886e38f)

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

template <typename T, int D>
__global__ void paged_attention_partition(
        const T* q, const T* key_cache, const T* value_cache,
        const int* block_table, const int* context_lens,
        float* tmp_out, float* max_logits, float* exp_sums,
        int block_size, int bt_stride, float scale,
        int num_heads, int num_kv_heads, int num_partitions, int partition_size, int window) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, part = blockIdx.z, lane = threadIdx.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const int start = part * partition_size;
    const int end = min(start + partition_size, context_len);
    const int t_start = (window > 0) ? max(start, context_len - window) : start;

    const int64_t q_base = (int64_t(batch) * num_heads + head) * D;
    const int64_t stat = (int64_t(batch) * num_heads + head) * num_partitions + part;
    const int64_t out_base = stat * D;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }
    float m = NEG_INF, l = 0.0f;

    for (int t = t_start; t < end; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        const int64_t base = (int64_t(block) * block_size + slot) * num_kv_heads * D + int64_t(kv_head) * D;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) partial += qv[i] * float(key_cache[base + lane + 32 * i]);
        const float score = warp_sum(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++)
            acc[i] = acc[i] * alpha + beta * float(value_cache[base + lane + 32 * i]);
        l = l * alpha + beta;
        m = nm;
    }
    if (lane == 0) {
        max_logits[stat] = (l == 0.0f) ? NEG_INF : m;
        exp_sums[stat] = l;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        tmp_out[out_base + lane + 32 * i] = (l == 0.0f) ? 0.0f : acc[i] / l;
}

// fp8 caches: uint8 e4m3 codes, per-KV-head scales; identical math otherwise
template <typename T, int D>
__global__ void paged_attention_partition_fp8(
        const T* q, const uint8_t* key_cache, const uint8_t* value_cache,
        const int* block_table, const int* context_lens,
        const float* k_scale, const float* v_scale,
        float* tmp_out, float* max_logits, float* exp_sums,
        int block_size, int bt_stride, float scale,
        int num_heads, int num_kv_heads, int num_partitions, int partition_size, int window) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, part = blockIdx.z, lane = threadIdx.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const int start = part * partition_size;
    const int end = min(start + partition_size, context_len);
    const int t_start = (window > 0) ? max(start, context_len - window) : start;
    const float ks = k_scale[kv_head], vs = v_scale[kv_head];

    const int64_t q_base = (int64_t(batch) * num_heads + head) * D;
    const int64_t stat = (int64_t(batch) * num_heads + head) * num_partitions + part;
    const int64_t out_base = stat * D;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }
    float m = NEG_INF, l = 0.0f;

    for (int t = t_start; t < end; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        const int64_t base = (int64_t(block) * block_size + slot) * num_kv_heads * D + int64_t(kv_head) * D;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++)
            partial += qv[i] * ks * tmq::e4m3_decode(key_cache[base + lane + 32 * i]);
        const float score = warp_sum(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++)
            acc[i] = acc[i] * alpha + beta * vs * tmq::e4m3_decode(value_cache[base + lane + 32 * i]);
        l = l * alpha + beta;
        m = nm;
    }
    if (lane == 0) {
        max_logits[stat] = (l == 0.0f) ? NEG_INF : m;
        exp_sums[stat] = l;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        tmp_out[out_base + lane + 32 * i] = (l == 0.0f) ? 0.0f : acc[i] / l;
}

// cascade / shared-prefix: contiguous prefix_k/v (prefix_len, H_KV, D), same partial layout
template <typename T, int D>
__global__ void cascade_prefix_partition(
        const T* q, const T* prefix_k, const T* prefix_v,
        float* tmp_out, float* max_logits, float* exp_sums,
        float scale, int num_heads, int num_kv_heads,
        int prefix_len, int num_partitions, int partition_size) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, part = blockIdx.z, lane = threadIdx.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int start = part * partition_size;
    const int end = min(start + partition_size, prefix_len);

    const int64_t q_base = (int64_t(batch) * num_heads + head) * D;
    const int64_t stat = (int64_t(batch) * num_heads + head) * num_partitions + part;
    const int64_t out_base = stat * D;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }
    float m = NEG_INF, l = 0.0f;

    for (int t = start; t < end; t++) {
        const int64_t base = (int64_t(t) * num_kv_heads + kv_head) * D;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) partial += qv[i] * float(prefix_k[base + lane + 32 * i]);
        const float score = warp_sum(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++)
            acc[i] = acc[i] * alpha + beta * float(prefix_v[base + lane + 32 * i]);
        l = l * alpha + beta;
        m = nm;
    }
    if (lane == 0) {
        max_logits[stat] = (l == 0.0f) ? NEG_INF : m;
        exp_sums[stat] = l;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        tmp_out[out_base + lane + 32 * i] = (l == 0.0f) ? 0.0f : acc[i] / l;
}

template <typename T, int D>
__global__ void paged_attention_reduce(
        const float* tmp_out, const float* max_logits, const float* exp_sums,
        T* out, int num_heads, int num_partitions) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
    const int64_t base = (int64_t(batch) * num_heads + head) * num_partitions;

    float gm = NEG_INF;
    for (int p = 0; p < num_partitions; p++) gm = fmaxf(gm, max_logits[base + p]);
    float gden = 0.0f;
    for (int p = 0; p < num_partitions; p++) {
        const float mp = max_logits[base + p];
        if (mp == NEG_INF) continue;
        gden += exp_sums[base + p] * expf(mp - gm);
    }
    const float inv = 1.0f / (gden + 1e-6f);

    float acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) acc[i] = 0.0f;
    for (int p = 0; p < num_partitions; p++) {
        const float mp = max_logits[base + p];
        if (mp == NEG_INF) continue;
        const float r = exp_sums[base + p] * expf(mp - gm);
        const int64_t ob = (base + p) * D;
        #pragma unroll
        for (int i = 0; i < VPL; i++) acc[i] += tmp_out[ob + lane + 32 * i] * r;
    }
    const int64_t out_base = (int64_t(batch) * num_heads + head) * D;
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        out[out_base + lane + 32 * i] = (gm == NEG_INF) ? T(0) : T(acc[i] * inv);
}

// explicit D=512 reduce instantiation (MLA reuses it)
template __global__ void paged_attention_reduce<__nv_bfloat16, 512>(
    const float*, const float*, const float*, __nv_bfloat16*, int, int);

}  // namespace tms

// ============================== test harness ==============================
#include <cuda_bf16.h>
using namespace tms;

static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }

// nearest e4m3 code by brute force over all 256 (exact nearest; decode is the kernel's own)
static uint8_t host_e4m3_nearest(float x) {
    auto dec = [](uint8_t v) {
        float mag;
        if (v & 0x78) { int e = (v >> 3) & 0xF, mm = v & 7; mag = std::ldexp(1.0f + mm / 8.0f, e - 7); }
        else mag = float(v & 7) * 0.001953125f;
        return (v & 0x80) ? -mag : mag;
    };
    uint8_t best = 0; float bd = 1e30f;
    for (int c = 0; c < 256; c++) {
        if ((c & 0x7F) == 0x7F) continue;   // skip NaN codes
        float d = std::abs(dec(uint8_t(c)) - x);
        if (d < bd) { bd = d; best = uint8_t(c); }
    }
    return best;
}

int main() {
    srand(11);
    const int B = 2, H = 8, HKV = 2, BS = 16, MAXB = 16, D = 128;
    const int ctx[B] = {201, 64};
    const float scale = 1.0f / sqrtf(float(D));
    const int num_blocks = B * MAXB + 1;
    const size_t cache_n = size_t(num_blocks) * BS * HKV * D;

    std::vector<int> bt(B * MAXB, -1);
    int nb = 1;
    for (int b = 0; b < B; b++)
        for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = nb++;
    std::vector<__half> K(cache_n), V(cache_n), q(size_t(B) * H * D);
    for (auto& x : K) x = __float2half(frand() * 0.3f);
    for (auto& x : V) x = __float2half(frand() * 0.3f);
    for (auto& x : q) x = __float2half(frand());

    __half *dK, *dV, *dq, *dout;
    int *dbt, *dctx;
    cudaMalloc(&dK, cache_n * 2); cudaMalloc(&dV, cache_n * 2);
    cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dout, q.size() * 2);
    cudaMalloc(&dbt, bt.size() * 4); cudaMalloc(&dctx, B * 4);
    cudaMemcpy(dK, K.data(), cache_n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), cache_n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dctx, ctx, B * 4, cudaMemcpyHostToDevice);

    // fp64 oracle (full softmax over all mapped tokens; window variants covered by t_start logic)
    auto oracle = [&](int b, int h, int window, std::vector<double>& o) {
        const int kvh = h / (H / HKV);
        const int t0 = (window > 0) ? std::max(0, ctx[b] - window) : 0;
        std::vector<double> sc; std::vector<size_t> bases;
        for (int t = t0; t < ctx[b]; t++) {
            const int blk = bt[b * MAXB + t / BS];
            if (blk < 0) continue;
            const size_t base = (size_t(blk) * BS + t % BS) * HKV * D + size_t(kvh) * D;
            double s = 0;
            for (int d = 0; d < D; d++)
                s += double(__half2float(q[(size_t(b) * H + h) * D + d])) * double(__half2float(K[base + d]));
            sc.push_back(s * scale); bases.push_back(base);
        }
        double mx = -1e300;
        for (double s : sc) mx = std::max(mx, s);
        o.assign(D, 0.0);
        double Z = 0;
        for (size_t i = 0; i < sc.size(); i++) {
            const double w = exp(sc[i] - mx);
            for (int d = 0; d < D; d++) o[d] += w * double(__half2float(V[bases[i] + d]));
            Z += w;
        }
        for (int d = 0; d < D; d++) o[d] = Z ? o[d] / Z : 0.0;
    };

    int rc = 0;
    const int max_ctx = std::max(ctx[0], ctx[1]);
    for (int PS : {16, 48, 512}) for (int window : {0, 40}) {
        const int P = (max_ctx + PS - 1) / PS;
        float *dtmp, *dml, *des;
        cudaMalloc(&dtmp, sizeof(float) * size_t(B) * H * P * D);
        cudaMalloc(&dml, sizeof(float) * B * H * P);
        cudaMalloc(&des, sizeof(float) * B * H * P);
        dim3 pgrid(H, B, P), rgrid(H, B);
        paged_attention_partition<__half, D><<<pgrid, 32>>>(dq, dK, dV, dbt, dctx, dtmp, dml, des, BS, MAXB, scale, H, HKV, P, PS, window);
        paged_attention_reduce<__half, D><<<rgrid, 32>>>(dtmp, dml, des, dout, H, P);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
        std::vector<__half> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        std::vector<double> o;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            oracle(b, h, window, o);
            for (int d = 0; d < D; d++)
                maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(b) * H + h) * D + d])) - o[d]));
        }
        printf("v2 partition/reduce PS=%-3d window=%-2d max diff %.5f (%s)\n", PS, window, maxd, maxd < 5e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 5e-3);
        cudaFree(dtmp); cudaFree(dml); cudaFree(des);
    }

    // fp8 cache variant: quantize K/V per-KV-head, compare vs fp64 oracle on the DEQUANTIZED cache
    {
        std::vector<uint8_t> K8(cache_n), V8(cache_n);
        std::vector<float> ks(HKV), vs(HKV);
        for (int kh = 0; kh < HKV; kh++) {
            float ka = 0, va = 0;
            for (size_t blk = 0; blk < size_t(num_blocks) * BS; blk++)
                for (int d = 0; d < D; d++) {
                    ka = std::max(ka, std::abs(__half2float(K[blk * HKV * D + kh * D + d])));
                    va = std::max(va, std::abs(__half2float(V[blk * HKV * D + kh * D + d])));
                }
            ks[kh] = ka / 448.0f; vs[kh] = va / 448.0f;
        }
        auto decq = [&](uint8_t v) {
            float mag;
            if (v & 0x78) { int e = (v >> 3) & 0xF, mm = v & 7; mag = std::ldexp(1.0f + mm / 8.0f, e - 7); }
            else mag = float(v & 7) * 0.001953125f;
            return (v & 0x80) ? -mag : mag;
        };
        std::vector<__half> Kd(cache_n), Vd(cache_n);
        for (size_t blk = 0; blk < size_t(num_blocks) * BS; blk++)
            for (int kh = 0; kh < HKV; kh++)
                for (int d = 0; d < D; d++) {
                    const size_t i = blk * HKV * D + kh * D + d;
                    K8[i] = host_e4m3_nearest(__half2float(K[i]) / ks[kh]);
                    V8[i] = host_e4m3_nearest(__half2float(V[i]) / vs[kh]);
                    Kd[i] = __float2half(ks[kh] * decq(K8[i]));   // oracle sees the dequantized cache
                    Vd[i] = __float2half(vs[kh] * decq(V8[i]));
                }
        std::vector<__half> Ksave = K, Vsave = V;
        K = Kd; V = Vd;   // oracle closure reads K/V
        uint8_t *dK8, *dV8; float *dks, *dvs;
        cudaMalloc(&dK8, cache_n); cudaMalloc(&dV8, cache_n);
        cudaMalloc(&dks, HKV * 4); cudaMalloc(&dvs, HKV * 4);
        cudaMemcpy(dK8, K8.data(), cache_n, cudaMemcpyHostToDevice);
        cudaMemcpy(dV8, V8.data(), cache_n, cudaMemcpyHostToDevice);
        cudaMemcpy(dks, ks.data(), HKV * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dvs, vs.data(), HKV * 4, cudaMemcpyHostToDevice);
        const int PS = 64, P = (max_ctx + PS - 1) / PS;
        float *dtmp, *dml, *des;
        cudaMalloc(&dtmp, sizeof(float) * size_t(B) * H * P * D);
        cudaMalloc(&dml, sizeof(float) * B * H * P);
        cudaMalloc(&des, sizeof(float) * B * H * P);
        paged_attention_partition_fp8<__half, D><<<dim3(H, B, P), 32>>>(dq, dK8, dV8, dbt, dctx, dks, dvs, dtmp, dml, des, BS, MAXB, scale, H, HKV, P, PS, 0);
        paged_attention_reduce<__half, D><<<dim3(H, B), 32>>>(dtmp, dml, des, dout, H, P);
        cudaDeviceSynchronize();
        std::vector<__half> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        std::vector<double> o;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            oracle(b, h, 0, o);
            for (int d = 0; d < D; d++)
                maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(b) * H + h) * D + d])) - o[d]));
        }
        printf("v2 fp8 cache PS=64            max diff %.5f (%s)\n", maxd, maxd < 5e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 5e-3);
        K = Ksave; V = Vsave;
        cudaFree(dK8); cudaFree(dV8); cudaFree(dks); cudaFree(dvs);
        cudaFree(dtmp); cudaFree(dml); cudaFree(des);
    }

    // cascade: shared prefix (len 96) + per-request paged suffix, concat partials along P
    {
        const int PLEN = 96, PS = 48;
        const int Ppre = (PLEN + PS - 1) / PS, Psuf = (max_ctx + PS - 1) / PS, P = Ppre + Psuf;
        std::vector<__half> pk(size_t(PLEN) * HKV * D), pv(pk.size());
        for (auto& x : pk) x = __float2half(frand() * 0.3f);
        for (auto& x : pv) x = __float2half(frand() * 0.3f);
        __half *dpk, *dpv;
        cudaMalloc(&dpk, pk.size() * 2); cudaMalloc(&dpv, pv.size() * 2);
        cudaMemcpy(dpk, pk.data(), pk.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dpv, pv.data(), pv.size() * 2, cudaMemcpyHostToDevice);
        float *dtmp, *dml, *des;
        cudaMalloc(&dtmp, sizeof(float) * size_t(B) * H * P * D);
        cudaMalloc(&dml, sizeof(float) * B * H * P);
        cudaMalloc(&des, sizeof(float) * B * H * P);
        // prefix partials write partitions [0, Ppre); suffix writes [Ppre, P).
        // Both kernels index stats as (bh*num_partitions + part) with num_partitions = P,
        // so launch prefix with grid.z = Ppre and pass P; suffix needs its stats OFFSET by
        // Ppre - handled by passing (tmp + Ppre*D per bh)? Simplest correct composition:
        // run suffix into its own buffers then interleave on host? Instead: offset pointers -
        // stats are (B,H,P): suffix writes at part' = part + Ppre via pointer arithmetic
        // only if strides match; they do per (b,h) row only when P is the common stride.
        // We keep it simple and correct: launch suffix with num_partitions = P and a
        // part-offset baked in by shifting the grid START - CUDA has no grid offset, so we
        // pass tmp_out/max_logits/exp_sums offset by Ppre in the LAST dim via a wrapper
        // trick: the kernels compute stat = bh*P + part, so passing (ml + Ppre) etc. shifts
        // every (b,h) row by Ppre. Same for tmp_out with (tmp + Ppre*D).
        cascade_prefix_partition<__half, D><<<dim3(H, B, Ppre), 32>>>(dq, dpk, dpv, dtmp, dml, des, scale, H, HKV, PLEN, P, PS);
        paged_attention_partition<__half, D><<<dim3(H, B, Psuf), 32>>>(dq, dK, dV, dbt, dctx,
            dtmp + size_t(Ppre) * D, dml + Ppre, des + Ppre, BS, MAXB, scale, H, HKV, P, PS, 0);
        paged_attention_reduce<__half, D><<<dim3(H, B), 32>>>(dtmp, dml, des, dout, H, P);
        cudaDeviceSynchronize();
        std::vector<__half> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        // oracle: full softmax over prefix ++ mapped suffix
        double maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            const int kvh = h / (H / HKV);
            std::vector<double> sc; std::vector<const __half*> vp;
            for (int t = 0; t < PLEN; t++) {
                const size_t base = (size_t(t) * HKV + kvh) * D;
                double s = 0;
                for (int d = 0; d < D; d++)
                    s += double(__half2float(q[(size_t(b) * H + h) * D + d])) * double(__half2float(pk[base + d]));
                sc.push_back(s * scale); vp.push_back(&pv[base]);
            }
            for (int t = 0; t < ctx[b]; t++) {
                const int blk = bt[b * MAXB + t / BS];
                if (blk < 0) continue;
                const size_t base = (size_t(blk) * BS + t % BS) * HKV * D + size_t(kvh) * D;
                double s = 0;
                for (int d = 0; d < D; d++)
                    s += double(__half2float(q[(size_t(b) * H + h) * D + d])) * double(__half2float(K[base + d]));
                sc.push_back(s * scale); vp.push_back(&V[base]);
            }
            double mx = -1e300;
            for (double s : sc) mx = std::max(mx, s);
            std::vector<double> o(D, 0.0);
            double Z = 0;
            for (size_t i = 0; i < sc.size(); i++) {
                const double w = exp(sc[i] - mx);
                for (int d = 0; d < D; d++) o[d] += w * double(__half2float(vp[i][d]));
                Z += w;
            }
            for (int d = 0; d < D; d++)
                maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(b) * H + h) * D + d])) - o[d] / Z));
        }
        printf("cascade prefix+suffix          max diff %.5f (%s)\n", maxd, maxd < 5e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 5e-3);
        cudaFree(dpk); cudaFree(dpv); cudaFree(dtmp); cudaFree(dml); cudaFree(des);
    }
    return rc;
}
