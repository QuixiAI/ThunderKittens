// Symmetric (bidirectional) sliding-window attention with banded key-tiling,
// int8 Q.K^T (head-dim 256, GQA) -- CUDA/SM86 serving variant.
//
// Shape-named specialization built on the already-ported attn_q_int8 (int8 Q.K^T,
// per-row amax/127 scale, CUDA_R_8I -> CUDA_R_32I score GEMM, fused dequant +
// softmax, fp16 P.V). The delta vs. every existing box attention kernel (all
// causal / non-causal only) is a SYMMETRIC +-window/2 band mask -- this is the
// EmbeddingGemma alternating-layer *encoder* sliding-window attention, so it is
// NOT causal: query q attends keys [q - window/2, q + window/2] (both sides).
//
// Two things are ported from embeddinggemma.c:
//   (1) the symmetric window mask in `attention_softmax_f16_kernel`
//       (src/engine_cuda.cu): first = q > hw ? q - hw : 0; last = min(T, q+hw+1),
//       applied *inside* the softmax so out-of-band keys get probability 0 -- and
//   (2) the host banded key-tiling in `tensor_core_attention`
//       (swa_tensor_tile_tokens / swa_tensor_banded_min_tokens): when the window
//       is set and T >= banded_min, queries are processed in tiles of `swa_tile`
//       and each tile's score GEMM only spans the key band
//       [tile_start - hw, tile_start + tile_len + hw), so the O(T^2) score/PV
//       matmuls collapse to O(T * (tile + window)). Every query's full band lies
//       within its tile's key band, so banding is exact -- it changes *work*, not
//       the result. The portable fp64 reference is `ei_attention_mha_range` in
//       src/kernels.c (same symmetric first/last band).
//
// Correctness: fp64 oracle recomputes softmax over the symmetric band from the
// int8-quantized Q,K and fp16 V (head 0); the banded kernel output must match the
// repo gate rel < 0.02. A self-parity number reports banded vs. the dense (full
// key range, window masked only in softmax) int8 attention across all heads.
// Perf: banded vs. dense int8 attention -- the win only appears at long T (T >>
// window), where the band is a small fraction of the full key range.
//
// Build:
//   /usr/local/cuda/bin/nvcc attn_swa.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -lcublas -o attn_swa.out
// Run:
//   CUDA_VISIBLE_DEVICES=1 ./attn_swa.out [T] [window] [n_head] [n_head_kv]
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static constexpr int D = 256;          // head dim

// ---- ported from attn_q_int8: per-row symmetric int8 quant (one warp/row) ----
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__global__ void quantize_rows_i8(const __half *input, int8_t *out, float *scales,
                                 uint32_t rows) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= rows) return;
    const size_t base = static_cast<size_t>(row) * D;
    float vals[D / 32]; float amax = 0.f;
#pragma unroll
    for (int i = 0; i < D / 32; i++) { vals[i] = __half2float(input[base + lane + i * 32]); amax = fmaxf(amax, fabsf(vals[i])); }
    amax = warp_max(amax); amax = __shfl_sync(0xffffffffu, amax, 0);
    const float scale = amax / 127.0f, inv = scale > 0.f ? 1.f / scale : 0.f;
#pragma unroll
    for (int i = 0; i < D / 32; i++) {
        int q = __float2int_rn(vals[i] * inv); q = max(-127, min(127, q));
        out[base + lane + i * 32] = static_cast<int8_t>(q);
    }
    if (lane == 0) scales[row] = scale;
}

__device__ __forceinline__ float block_max(float v, float *part) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    v = warp_max(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (blockDim.x / 32) ? part[lane] : -3.4e38f;
        v = warp_max(v);
        if (lane == 0) part[0] = v;
    }
    __syncthreads();
    return part[0];
}
__device__ __forceinline__ float block_sum(float v, float *part) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    v = warp_sum(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (blockDim.x / 32) ? part[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) part[0] = v;
    }
    __syncthreads();
    return part[0];
}

// ---- ported: fused dequant + SYMMETRIC-band mask + row softmax ---------------
// scores are int32 [local_query][local_key] over the tile's key band; dequant by
// qscale[query]*kscale[key]; the +-window/2 symmetric band is masked here (keys
// outside [q-hw, q+hw] get probability 0), exactly attention_softmax_f16_kernel.
// qscale/kscale point at the current head's per-row scale array; query/key are
// GLOBAL token indices (query_start + local_query, key_start + local_key).
__global__ void attention_softmax_swa_i8(const int32_t *scores, __half *probs,
                                         const float *qscale, const float *kscale,
                                         uint32_t query_start, uint32_t key_start,
                                         uint32_t query_count, uint32_t key_count,
                                         uint32_t sequence_tokens, uint32_t window) {
    __shared__ float part[32];
    const uint32_t local_query = blockIdx.x;
    if (local_query >= query_count) return;
    const uint32_t query = query_start + local_query;
    uint32_t first = 0, last = sequence_tokens;
    if (window != 0) {
        const uint32_t hw = window / 2;
        first = query > hw ? query - hw : 0;
        last = min(sequence_tokens, query + hw + 1);
    }
    const uint32_t local_first = first - key_start;   // key_start <= first
    const uint32_t local_last = last - key_start;     // last <= key_stop
    const float qs = qscale[query];
    const size_t row = static_cast<size_t>(local_query) * key_count;
    float lmax = -3.4e38f;
    for (uint32_t k = local_first + threadIdx.x; k < local_last; k += blockDim.x)
        lmax = fmaxf(lmax, static_cast<float>(scores[row + k]) * qs * kscale[key_start + k]);
    const float m = block_max(lmax, part);
    float lsum = 0.f;
    for (uint32_t k = local_first + threadIdx.x; k < local_last; k += blockDim.x)
        lsum += __expf(static_cast<float>(scores[row + k]) * qs * kscale[key_start + k] - m);
    const float denom = block_sum(lsum, part);
    const float invd = denom > 0.f ? 1.f / denom : 0.f;
    for (uint32_t k = threadIdx.x; k < key_count; k += blockDim.x) {
        const bool in_band = (k >= local_first && k < local_last);
        probs[row + k] = in_band
            ? __float2half(__expf(static_cast<float>(scores[row + k]) * qs * kscale[key_start + k] - m) * invd)
            : __float2half(0.0f);
    }
}

// =============================== harness ==============================
static float frand(std::mt19937 &r) { std::normal_distribution<float> nd(0.f, 1.f); return nd(r); }
static uint32_t align_down(uint32_t v, uint32_t a) { return v - v % a; }
static uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

struct SwaAttn {
    cublasHandle_t h;
    int T, H, HKV, window;
    int8_t *dQ8, *dK8; float *dQs, *dKs; __half *dV;
    int32_t *dS; __half *dProbs;
    const int32_t i1 = 1, i0 = 0; const float f1 = 1.f, f0 = 0.f;

    // One SWA attention pass. If `banded` and window!=0 and T>=banded_min, split
    // queries into `swa_tile` tiles and restrict each tile's key band; otherwise
    // dense (full key range, window still masked in softmax). Writes [H][T][D].
    void run(__half *dOut, bool banded, uint32_t swa_tile, uint32_t banded_min) {
        for (int hd = 0; hd < H; hd++) {
            const int kv = hd / (H / HKV);
            const uint32_t query_tile = (banded && window != 0 &&
                (uint32_t)T >= banded_min && swa_tile != 0)
                ? swa_tile : (uint32_t)T;
            for (uint32_t qs = 0; qs < (uint32_t)T; qs += query_tile) {
                const uint32_t qc = min(query_tile, (uint32_t)T - qs);
                uint32_t key_start = 0, key_stop = (uint32_t)T;
                if (window != 0 && query_tile < (uint32_t)T) {
                    const uint32_t hw = window / 2;
                    key_start = qs > hw ? qs - hw : 0;
                    key_stop = min((uint32_t)T, qs + qc + hw);
                    key_start = align_down(key_start, 16);            // int8 GEMM ld
                    key_stop = min((uint32_t)T, align_up(key_stop, 16));
                }
                const uint32_t kc = key_stop - key_start;
                // S(kc,qc) = K_band(kc,D) . Q_tile(qc,D)^T  -> col-major C(kc,qc)
                cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kc, qc, D, &i1,
                             dK8 + (size_t)(kv * T + key_start) * D, CUDA_R_8I, D,
                             dQ8 + (size_t)(hd * T + qs) * D, CUDA_R_8I, D, &i0,
                             dS, CUDA_R_32I, kc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                attention_softmax_swa_i8<<<qc, 256>>>(dS, dProbs,
                    dQs + (size_t)hd * T, dKs + (size_t)kv * T,
                    qs, key_start, qc, kc, (uint32_t)T, (uint32_t)window);
                // O(D,qc) = V_band(D,kc) . P(kc,qc)  (fp16)
                cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, qc, kc, &f1,
                             dV + (size_t)(kv * T + key_start) * D, CUDA_R_16F, D,
                             dProbs, CUDA_R_16F, kc, &f0,
                             dOut + (size_t)(hd * T + qs) * D, CUDA_R_16F, D,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
        }
    }
};

static bool correctness(int T, int window, int H, int HKV, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<__half> Qf((size_t)H * T * D), Kf((size_t)HKV * T * D), Vf((size_t)HKV * T * D);
    for (auto &x : Qf) x = __float2half(frand(rng) * 0.5f);
    for (auto &x : Kf) x = __float2half(frand(rng) * 0.5f);
    for (auto &x : Vf) x = __float2half(frand(rng) * 0.5f);

    SwaAttn a; a.T = T; a.H = H; a.HKV = HKV; a.window = window;
    cublasCreate(&a.h);
    __half *dQ, *dK;
    cudaMalloc(&dQ, Qf.size() * 2); cudaMalloc(&dK, Kf.size() * 2); cudaMalloc(&a.dV, Vf.size() * 2);
    cudaMalloc(&a.dQ8, Qf.size()); cudaMalloc(&a.dK8, Kf.size());
    cudaMalloc(&a.dQs, (size_t)H * T * 4); cudaMalloc(&a.dKs, (size_t)HKV * T * 4);
    cudaMalloc(&a.dS, (size_t)T * T * 4); cudaMalloc(&a.dProbs, (size_t)T * T * 2);
    __half *dOutB, *dOutD;
    cudaMalloc(&dOutB, (size_t)H * T * D * 2); cudaMalloc(&dOutD, (size_t)H * T * D * 2);
    cudaMemcpy(dQ, Qf.data(), Qf.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, Kf.data(), Kf.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(a.dV, Vf.data(), Vf.size() * 2, cudaMemcpyHostToDevice);
    quantize_rows_i8<<<(H * T + 7) / 8, 256>>>(dQ, a.dQ8, a.dQs, H * T);
    quantize_rows_i8<<<(HKV * T + 7) / 8, 256>>>(dK, a.dK8, a.dKs, HKV * T);

    // force banding on (small tile) so the tiling logic is exercised even at moderate T
    a.run(dOutB, /*banded*/true, /*swa_tile*/256, /*banded_min*/256);
    a.run(dOutD, /*banded*/false, 0, 0);   // dense reference (window in softmax)
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { printf("  KERNEL ERROR\n"); return false; }

    std::vector<int8_t> Q8(Qf.size()), K8(Kf.size()); std::vector<float> Qs((size_t)H * T), Ks((size_t)HKV * T);
    cudaMemcpy(Q8.data(), a.dQ8, Q8.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(K8.data(), a.dK8, K8.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(Qs.data(), a.dQs, Qs.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ks.data(), a.dKs, Ks.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<__half> outB((size_t)H * T * D), outD((size_t)H * T * D);
    cudaMemcpy(outB.data(), dOutB, outB.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(outD.data(), dOutD, outD.size() * 2, cudaMemcpyDeviceToHost);

    // fp64 oracle over the SYMMETRIC band (ei_attention_mha_range), head 0 / kv 0.
    const int hw = window > 0 ? window / 2 : 0;
    double gsum = 0, rsum = 0, gmax = 0;
    {
        const int hd = 0, kv = 0;
        std::vector<double> prob(T);
        for (int q = 0; q < T; q++) {
            const int first = window > 0 ? (q - hw > 0 ? q - hw : 0) : 0;
            const int last = window > 0 ? (q + hw + 1 < T ? q + hw + 1 : T) : T;
            double mmax = -1e300;
            for (int k = first; k < last; k++) {
                long dot = 0;
                for (int d = 0; d < D; d++) dot += long(Q8[((size_t)hd * T + q) * D + d]) * long(K8[((size_t)kv * T + k) * D + d]);
                double s = double(dot) * Qs[(size_t)hd * T + q] * Ks[(size_t)kv * T + k];
                prob[k] = s; mmax = std::max(mmax, s);
            }
            double den = 0; for (int k = first; k < last; k++) { prob[k] = std::exp(prob[k] - mmax); den += prob[k]; }
            for (int k = first; k < last; k++) prob[k] /= den;
            for (int d = 0; d < D; d++) {
                double o = 0;
                for (int k = first; k < last; k++) o += prob[k] * double(__half2float(Vf[((size_t)kv * T + k) * D + d]));
                double got = double(__half2float(outB[((size_t)hd * T + q) * D + d]));
                gsum += std::abs(got - o); rsum += std::abs(o); gmax = std::max(gmax, std::abs(got - o));
            }
        }
    }
    // banded vs dense self-parity across ALL heads
    double psum = 0, pref = 0;
    for (size_t i = 0; i < outB.size(); i++) {
        double b = __half2float(outB[i]), d = __half2float(outD[i]);
        psum += std::abs(b - d); pref += std::abs(d);
    }
    double rel = gsum / std::max(rsum, 1e-30);
    double par = psum / std::max(pref, 1e-30);
    bool pass = rel < 0.02 && par < 0.02;
    printf("  T=%-4d window=%-3d GQA %d:%d  banded vs fp64 oracle: rel %.4f%% max %.3g | banded-vs-dense %.4f%%  (%s)\n",
           T, window, H, HKV, 100 * rel, gmax, 100 * par, pass ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(a.dV); cudaFree(a.dQ8); cudaFree(a.dK8);
    cudaFree(a.dQs); cudaFree(a.dKs); cudaFree(a.dS); cudaFree(a.dProbs);
    cudaFree(dOutB); cudaFree(dOutD); cublasDestroy(a.h);
    return pass;
}

static void perf(int T, int window, int H, int HKV, uint32_t swa_tile, uint32_t banded_min, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<__half> Qf((size_t)H * T * D), Kf((size_t)HKV * T * D), Vf((size_t)HKV * T * D);
    for (auto &x : Qf) x = __float2half(frand(rng) * 0.5f);
    for (auto &x : Kf) x = __float2half(frand(rng) * 0.5f);
    for (auto &x : Vf) x = __float2half(frand(rng) * 0.5f);

    SwaAttn a; a.T = T; a.H = H; a.HKV = HKV; a.window = window;
    cublasCreate(&a.h);
    __half *dQ, *dK, *dOut;
    cudaMalloc(&dQ, Qf.size() * 2); cudaMalloc(&dK, Kf.size() * 2); cudaMalloc(&a.dV, Vf.size() * 2);
    cudaMalloc(&a.dQ8, Qf.size()); cudaMalloc(&a.dK8, Kf.size());
    cudaMalloc(&a.dQs, (size_t)H * T * 4); cudaMalloc(&a.dKs, (size_t)HKV * T * 4);
    cudaMalloc(&a.dS, (size_t)T * T * 4); cudaMalloc(&a.dProbs, (size_t)T * T * 2);
    cudaMalloc(&dOut, (size_t)H * T * D * 2);
    cudaMemcpy(dQ, Qf.data(), Qf.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, Kf.data(), Kf.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(a.dV, Vf.data(), Vf.size() * 2, cudaMemcpyHostToDevice);
    // quantize once up front (shared by banded & dense; not part of the attention race)
    quantize_rows_i8<<<(H * T + 7) / 8, 256>>>(dQ, a.dQ8, a.dQs, H * T);
    quantize_rows_i8<<<(HKV * T + 7) / 8, 256>>>(dK, a.dK8, a.dKs, HKV * T);
    cudaDeviceSynchronize();

    auto bench = [&](bool banded) {
        for (int i = 0; i < 5; i++) a.run(dOut, banded, swa_tile, banded_min);
        cudaDeviceSynchronize();
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < 30; i++) a.run(dOut, banded, swa_tile, banded_min);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); return ms / 30;
    };
    float mb = bench(true), md = bench(false);
    const bool engaged = window != 0 && (uint32_t)T >= banded_min && swa_tile != 0 && swa_tile < (uint32_t)T;
    printf("  T=%-4d window=%d tile=%u: banded %.4f ms | dense %.4f ms -> %.2fx  %s\n",
           T, window, swa_tile, mb, md, md / mb, engaged ? "(banded)" : "(banded==dense, no tiling)");
    cudaFree(dQ); cudaFree(dK); cudaFree(a.dV); cudaFree(a.dQ8); cudaFree(a.dK8);
    cudaFree(a.dQs); cudaFree(a.dKs); cudaFree(a.dS); cudaFree(a.dProbs); cudaFree(dOut);
    cublasDestroy(a.h);
}

int main(int argc, char **argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 2048;
    const int window = argc > 2 ? atoi(argv[2]) : 512;
    const int H = argc > 3 ? atoi(argv[3]) : 3;
    const int HKV = argc > 4 ? atoi(argv[4]) : 1;
    printf("== attn_swa (symmetric SWA, banded int8 Q.K^T, head-dim %d, GQA %d:%d)\n", D, H, HKV);

    bool ok = true;
    printf("-- correctness (banding forced on, tile=256; symmetric +-window/2 band) --\n");
    for (int w : {128, 256, 512})
        for (int t : {256, 512, 1024})
            ok &= correctness(t, w, H, HKV, 2025u + t + w);

    printf("-- perf: banded vs dense int8 attention (window=%d, tile=1024, banded_min=1536) --\n", window);
    for (int t : {512, 1024, 2048, 4096})
        perf(t, window, H, HKV, /*swa_tile*/1024, /*banded_min*/1536, 7u + t);

    printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
