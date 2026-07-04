// Fused RoPE + paged-KV insert, CUDA/SM86 port of ThunderMittens kernels/rope_kv.
// Split-half / GPT-NeoX RoPE (matches mx.fast.rope(traditional=False)):
//   ko1 = k1*cos - k2*sin;  ko2 = k2*cos + k1*sin   (halves k1=k[:D/2], k2=k[D/2:])
// One warp per (token, kv_head) row; rotated K + unrotated V go straight to the
// paged cache slot (fusing rotary + two scatters). Variants: rope_kv_insert_norm
// (RMSNorm over the K head dim first; gemma flag -> (1+w)) and rope_q (Q path,
// contiguous output, optional norm).
//
// Build:
//   /usr/local/cuda/bin/nvcc rope_kv.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o rope_kv.out
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

namespace tms {

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// optional fused RMSNorm over the full D of one K row (lane-strided), then split-half RoPE,
// then paged insert. norm_weight == nullptr disables the norm. Covers all three TM kernels.
template <typename T, int D, bool NORM>
__global__ void rope_kv_insert(const T* k, const T* v, const T* cosb, const T* sinb,
                               const int* positions, const int64_t* slot_mapping,
                               T* key_cache, T* value_cache, const T* norm_weight,
                               int num_kv_heads, int block_size, int gemma, float eps) {
    constexpr int D2 = D / 2;
    const int row = blockIdx.x;                 // (token, kv_head) flattened
    const int token = row / num_kv_heads;
    const int kv_head = row % num_kv_heads;
    const int lane = threadIdx.x;

    const int64_t slot = slot_mapping[token];
    if (slot < 0) return;
    const int64_t dst = ((slot / block_size) * block_size + slot % block_size) * num_kv_heads + kv_head;
    const int pos = positions[token];

    const T* krow = k + int64_t(row) * D;
    const T* vrow = v + int64_t(row) * D;
    T* kc = key_cache + dst * D;
    T* vc = value_cache + dst * D;

    float kv_[ (D + 31) / 32 ];                 // this lane's K elements (full D)
    #pragma unroll
    for (int i = 0; i < D / 32; i++) kv_[i] = float(krow[lane + 32 * i]);

    if constexpr (NORM) {
        float ss = 0;
        #pragma unroll
        for (int i = 0; i < D / 32; i++) ss += kv_[i] * kv_[i];
        ss = warp_sum(ss);
        const float rms = rsqrtf(ss / D + eps);
        #pragma unroll
        for (int i = 0; i < D / 32; i++) {
            const float w = float(norm_weight[lane + 32 * i]);
            kv_[i] *= rms * (gemma ? (1.0f + w) : w);
        }
    }

    // split-half rotate: lane owns positions (lane + 32i) of each half.
    // kv_[i] holds k[lane+32i]; the first D2/32 entries are half-1 iff lane+32i < D2 -
    // with D2 a multiple of 32, entries i < D2/32 are half 1, the rest half 2.
    constexpr int H = D2 / 32;
    #pragma unroll
    for (int i = 0; i < H; i++) {
        const int d = lane + 32 * i;            // index within the half
        const float c = float(cosb[int64_t(pos) * D2 + d]);
        const float s = float(sinb[int64_t(pos) * D2 + d]);
        const float k1 = kv_[i], k2 = kv_[i + H];
        kc[d]      = T(k1 * c - k2 * s);
        kc[d + D2] = T(k2 * c + k1 * s);
    }
    #pragma unroll
    for (int i = 0; i < D / 32; i++) vc[lane + 32 * i] = vrow[lane + 32 * i];
}

// Q companion: rotate (optionally norm) and write contiguous q_out (Q is not paged)
template <typename T, int D, bool NORM>
__global__ void rope_q(const T* q, const T* cosb, const T* sinb, const int* positions,
                       T* q_out, const T* norm_weight, int num_heads, int gemma, float eps) {
    constexpr int D2 = D / 2;
    const int row = blockIdx.x;                 // (token, head) flattened
    const int token = row / num_heads;
    const int lane = threadIdx.x;
    const int pos = positions[token];
    const T* qrow = q + int64_t(row) * D;
    T* orow = q_out + int64_t(row) * D;

    float qv[(D + 31) / 32];
    #pragma unroll
    for (int i = 0; i < D / 32; i++) qv[i] = float(qrow[lane + 32 * i]);
    if constexpr (NORM) {
        float ss = 0;
        #pragma unroll
        for (int i = 0; i < D / 32; i++) ss += qv[i] * qv[i];
        ss = warp_sum(ss);
        const float rms = rsqrtf(ss / D + eps);
        #pragma unroll
        for (int i = 0; i < D / 32; i++) {
            const float w = float(norm_weight[lane + 32 * i]);
            qv[i] *= rms * (gemma ? (1.0f + w) : w);
        }
    }
    constexpr int H = D2 / 32;
    #pragma unroll
    for (int i = 0; i < H; i++) {
        const int d = lane + 32 * i;
        const float c = float(cosb[int64_t(pos) * D2 + d]);
        const float s = float(sinb[int64_t(pos) * D2 + d]);
        const float k1 = qv[i], k2 = qv[i + H];
        orow[d]      = T(k1 * c - k2 * s);
        orow[d + D2] = T(k2 * c + k1 * s);
    }
}

}  // namespace tms

// ============================== test harness ==============================
using namespace tms;
static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }

template <int D>
int run() {
    srand(21);
    const int T_ = 40, HKV = 4, BS = 16, PMAX = 128;
    const int M = T_ * HKV, D2 = D / 2;
    const int num_blocks = (T_ + BS - 1) / BS + 1;
    const size_t cache_n = size_t(num_blocks) * BS * HKV * D;
    std::vector<__half> K(size_t(M) * D), V(size_t(M) * D), W(D);
    std::vector<__half> cosb(size_t(PMAX) * D2), sinb(size_t(PMAX) * D2);
    std::vector<int> pos(T_);
    std::vector<int64_t> slots(T_);
    for (auto& x : K) x = __float2half(frand());
    for (auto& x : V) x = __float2half(frand());
    for (auto& x : W) x = __float2half(frand() * 0.5f + 1.0f);
    for (int p = 0; p < PMAX; p++) for (int d = 0; d < D2; d++) {
        const double th = p / pow(10000.0, 2.0 * d / D);
        cosb[size_t(p) * D2 + d] = __float2half(float(cos(th)));
        sinb[size_t(p) * D2 + d] = __float2half(float(sin(th)));
    }
    for (int t = 0; t < T_; t++) { pos[t] = (t * 7) % PMAX; slots[t] = (t == 3) ? -1 : t + BS; }

    __half *dK, *dV, *dW, *dc, *ds, *dkc, *dvc;
    int* dpos; int64_t* dslot;
    cudaMalloc(&dK, K.size() * 2); cudaMalloc(&dV, V.size() * 2); cudaMalloc(&dW, D * 2);
    cudaMalloc(&dc, cosb.size() * 2); cudaMalloc(&ds, sinb.size() * 2);
    cudaMalloc(&dkc, cache_n * 2); cudaMalloc(&dvc, cache_n * 2);
    cudaMalloc(&dpos, T_ * 4); cudaMalloc(&dslot, T_ * 8);
    cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, W.data(), D * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, cosb.data(), cosb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, sinb.data(), sinb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, pos.data(), T_ * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dslot, slots.data(), T_ * 8, cudaMemcpyHostToDevice);

    int rc = 0;
    for (int variant = 0; variant < 3; variant++) {   // plain, norm, norm+gemma
        cudaMemset(dkc, 0, cache_n * 2); cudaMemset(dvc, 0, cache_n * 2);
        const bool norm = variant > 0; const int gemma = variant == 2;
        if (norm) rope_kv_insert<__half, D, true><<<M, 32>>>(dK, dV, dc, ds, dpos, dslot, dkc, dvc, dW, HKV, BS, gemma, 1e-6f);
        else      rope_kv_insert<__half, D, false><<<M, 32>>>(dK, dV, dc, ds, dpos, dslot, dkc, dvc, nullptr, HKV, BS, 0, 1e-6f);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
        std::vector<__half> kc(cache_n), vc(cache_n);
        cudaMemcpy(kc.data(), dkc, cache_n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(vc.data(), dvc, cache_n * 2, cudaMemcpyDeviceToHost);

        double maxd = 0;
        for (int t = 0; t < T_; t++) {
            if (slots[t] < 0) continue;
            for (int h = 0; h < HKV; h++) {
                const int row = t * HKV + h;
                // reference row: optional rmsnorm then rotate
                std::vector<double> kr(D);
                for (int d = 0; d < D; d++) kr[d] = __half2float(K[size_t(row) * D + d]);
                if (norm) {
                    double ss = 0;
                    for (int d = 0; d < D; d++) ss += kr[d] * kr[d];
                    const double rms = 1.0 / sqrt(ss / D + 1e-6);
                    for (int d = 0; d < D; d++) {
                        const double w = __half2float(W[d]);
                        kr[d] *= rms * (gemma ? (1.0 + w) : w);
                    }
                }
                const size_t dst = (size_t(slots[t] / BS) * BS + slots[t] % BS) * HKV * D + size_t(h) * D;
                for (int d = 0; d < D2; d++) {
                    const double c = __half2float(cosb[size_t(pos[t]) * D2 + d]);
                    const double s = __half2float(sinb[size_t(pos[t]) * D2 + d]);
                    const double r1 = kr[d] * c - kr[d + D2] * s;
                    const double r2 = kr[d + D2] * c + kr[d] * s;
                    maxd = std::max(maxd, std::abs(double(__half2float(kc[dst + d])) - r1));
                    maxd = std::max(maxd, std::abs(double(__half2float(kc[dst + d + D2])) - r2));
                }
                for (int d = 0; d < D; d++)
                    maxd = std::max(maxd, std::abs(double(__half2float(vc[dst + d])) -
                                                   double(__half2float(V[size_t(row) * D + d]))));
            }
        }
        const char* names[3] = {"plain", "norm", "norm+gemma"};
        printf("rope_kv_insert D=%-3d %-10s max diff %.5f (%s)\n", D, names[variant], maxd, maxd < 4e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 4e-3);
    }

    // rope_q (norm variant)
    {
        const int H = 6, MQ = T_ * H;
        std::vector<__half> q(size_t(MQ) * D);
        for (auto& x : q) x = __float2half(frand());
        __half *dq, *dqo;
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dqo, q.size() * 2);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        rope_q<__half, D, true><<<MQ, 32>>>(dq, dc, ds, dpos, dqo, dW, H, 0, 1e-6f);
        cudaDeviceSynchronize();
        std::vector<__half> qo(q.size());
        cudaMemcpy(qo.data(), dqo, q.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        for (int t = 0; t < T_; t++) for (int h = 0; h < H; h++) {
            const int row = t * H + h;
            std::vector<double> qr(D);
            double ss = 0;
            for (int d = 0; d < D; d++) { qr[d] = __half2float(q[size_t(row) * D + d]); ss += qr[d] * qr[d]; }
            const double rms = 1.0 / sqrt(ss / D + 1e-6);
            for (int d = 0; d < D; d++) qr[d] *= rms * __half2float(W[d]);
            for (int d = 0; d < D / 2; d++) {
                const double c = __half2float(cosb[size_t(pos[t]) * (D / 2) + d]);
                const double s = __half2float(sinb[size_t(pos[t]) * (D / 2) + d]);
                maxd = std::max(maxd, std::abs(double(__half2float(qo[size_t(row) * D + d])) - (qr[d] * c - qr[d + D / 2] * s)));
                maxd = std::max(maxd, std::abs(double(__half2float(qo[size_t(row) * D + d + D / 2])) - (qr[d + D / 2] * c + qr[d] * s)));
            }
        }
        printf("rope_q         D=%-3d norm       max diff %.5f (%s)\n", D, maxd, maxd < 4e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 4e-3);
        cudaFree(dq); cudaFree(dqo);
    }
    cudaFree(dK); cudaFree(dV); cudaFree(dW); cudaFree(dc); cudaFree(ds);
    cudaFree(dkc); cudaFree(dvc); cudaFree(dpos); cudaFree(dslot);
    return rc;
}

int main() {
    int rc = run<64>();
    rc |= run<128>();
    return rc;
}
