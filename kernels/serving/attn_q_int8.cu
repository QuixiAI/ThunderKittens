// int8 Q.K^T attention (head-dim 256, GQA) -- CUDA/SM86 serving variant.
// Shape-named specialization of the attn_q family: instead of quantizing the
// K/V *storage* (attn_q), this quantizes Q and K to int8 with a per-row
// (amax/127) scale and runs the score GEMM S = Q.K^T on the int8 tensor cores
// (CUDA_R_8I -> CUDA_R_32I), then dequantizes the int32 partials *inside* the
// softmax by the product of the two per-row scales -- no extra [query x key]
// pass. P.V stays fp16 (this is "mode 1"; the int8 P.V "mode 2" of the source
// was rejected for accuracy, so it is intentionally NOT ported).
//
// Ported from embeddinggemma.c src/engine_cuda.cu (EI_CUDA_INT8_ATTN=1); named
// by shape (head-dim 256, GQA 3:1), not by model. The two hand-written device
// kernels are quantize_rows_i8 (per-row int8 quant, D=256) and
// attention_softmax_i8 (fused dequant + row softmax); the int8 QK and fp16 PV
// GEMMs use cuBLAS exactly as the source does.
//
// Correctness: fp64 oracle recomputes softmax(qs*ks * (Qi8.Ki8^T)) . Vf from the
// same int8 inputs; the int8-kernel output must match to the repo's rel<2%.
// A second number reports the int8-vs-fp16 attention-output error (the accuracy
// the int8 QK path costs vs a full fp16 attention) -- the justification for
// keeping mode 1 and rejecting mode 2.
//
// Build:
//   /usr/local/cuda/bin/nvcc attn_q_int8.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -lcublas -o attn_q_int8.out
// Run:
//   CUDA_VISIBLE_DEVICES=1 ./attn_q_int8.out [T] [n_head] [n_head_kv]
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static constexpr int D = 256;          // head dim

// ---- ported: per-row symmetric int8 quant of length-D rows (one warp/row) ----
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
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

// ---- helpers for block reductions in the softmax ----
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
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
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

// ---- ported: fused dequant + row softmax (one block per query row) ----
// scores are int32 [query][key]; dequant by qscale[q]*kscale[k]; write fp16 probs.
__global__ void attention_softmax_i8(const int32_t *scores, __half *probs,
                                     const float *qscale, const float *kscale,
                                     uint32_t query_count, uint32_t key_count) {
    __shared__ float part[32];
    const uint32_t q = blockIdx.x;
    if (q >= query_count) return;
    const float qs = qscale[q];
    const size_t row = static_cast<size_t>(q) * key_count;
    float lmax = -3.4e38f;
    for (uint32_t k = threadIdx.x; k < key_count; k += blockDim.x)
        lmax = fmaxf(lmax, static_cast<float>(scores[row + k]) * qs * kscale[k]);
    const float m = block_max(lmax, part);
    float lsum = 0.f;
    for (uint32_t k = threadIdx.x; k < key_count; k += blockDim.x)
        lsum += __expf(static_cast<float>(scores[row + k]) * qs * kscale[k] - m);
    const float denom = block_sum(lsum, part);
    const float invd = denom > 0.f ? 1.f / denom : 0.f;
    for (uint32_t k = threadIdx.x; k < key_count; k += blockDim.x)
        probs[row + k] = __float2half(__expf(static_cast<float>(scores[row + k]) * qs * kscale[k] - m) * invd);
}

// fp16 softmax for the fp16 reference/baseline path
__global__ void attention_softmax_f16(const float *scores, __half *probs,
                                      uint32_t query_count, uint32_t key_count) {
    __shared__ float part[32];
    const uint32_t q = blockIdx.x;
    if (q >= query_count) return;
    const size_t row = static_cast<size_t>(q) * key_count;
    float lmax = -3.4e38f;
    for (uint32_t k = threadIdx.x; k < key_count; k += blockDim.x)
        lmax = fmaxf(lmax, scores[row + k]);
    const float m = block_max(lmax, part);
    float lsum = 0.f;
    for (uint32_t k = threadIdx.x; k < key_count; k += blockDim.x)
        lsum += __expf(scores[row + k] - m);
    const float denom = block_sum(lsum, part);
    const float invd = denom > 0.f ? 1.f / denom : 0.f;
    for (uint32_t k = threadIdx.x; k < key_count; k += blockDim.x)
        probs[row + k] = __float2half(__expf(scores[row + k] - m) * invd);
}

// =============================== harness ==============================
static float frand(std::mt19937 &r) { std::normal_distribution<float> nd(0.f, 1.f); return nd(r); }

int main(int argc, char **argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 512;       // sequence length
    const int H = argc > 2 ? atoi(argv[2]) : 3;         // query heads
    const int HKV = argc > 3 ? atoi(argv[3]) : 1;       // kv heads (GQA)
    printf("== attn_q_int8 (int8 Q.K^T, head-dim %d, GQA %d:%d)  T=%d\n", D, H, HKV, T);

    std::mt19937 rng(2024);
    // Q: [H][T][D]  (per query head), K/V: [HKV][T][D]  (shared, GQA)
    std::vector<__half> Qf(size_t(H) * T * D), Kf(size_t(HKV) * T * D), Vf(size_t(HKV) * T * D);
    for (auto &x : Qf) x = __float2half(frand(rng) * 0.5f);
    for (auto &x : Kf) x = __float2half(frand(rng) * 0.5f);
    for (auto &x : Vf) x = __float2half(frand(rng) * 0.5f);

    // device buffers
    __half *dQ, *dK, *dV, *dProbs, *dOut8, *dOutF;
    int8_t *dQ8, *dK8; float *dQs, *dKs; int32_t *dS_i32; float *dS_f32;
    cudaMalloc(&dQ, Qf.size() * 2); cudaMalloc(&dK, Kf.size() * 2); cudaMalloc(&dV, Vf.size() * 2);
    cudaMalloc(&dQ8, Qf.size()); cudaMalloc(&dK8, Kf.size());
    cudaMalloc(&dQs, size_t(H) * T * 4); cudaMalloc(&dKs, size_t(HKV) * T * 4);
    cudaMalloc(&dS_i32, size_t(T) * T * 4); cudaMalloc(&dS_f32, size_t(T) * T * 4);
    cudaMalloc(&dProbs, size_t(T) * T * 2);
    cudaMalloc(&dOut8, size_t(H) * T * D * 2); cudaMalloc(&dOutF, size_t(H) * T * D * 2);
    cudaMemcpy(dQ, Qf.data(), Qf.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, Kf.data(), Kf.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, Vf.data(), Vf.size() * 2, cudaMemcpyHostToDevice);

    cublasHandle_t h; cublasCreate(&h);
    const int32_t i1 = 1, i0 = 0; const float f1 = 1.f, f0 = 0.f;

    // -------- int8 path (mode 1) --------
    auto run_int8 = [&] {
        quantize_rows_i8<<<(H * T + 7) / 8, 256>>>(dQ, dQ8, dQs, H * T);
        quantize_rows_i8<<<(HKV * T + 7) / 8, 256>>>(dK, dK8, dKs, HKV * T);
        for (int hd = 0; hd < H; hd++) {
            const int kv = hd / (H / HKV);
            // S(T,T) = Q_h(T,D) . K_kv(T,D)^T  -> col-major C(Tk,Tq)
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, T, T, D, &i1,
                         dK8 + size_t(kv) * T * D, CUDA_R_8I, D,
                         dQ8 + size_t(hd) * T * D, CUDA_R_8I, D, &i0,
                         dS_i32, CUDA_R_32I, T, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            attention_softmax_i8<<<T, 256>>>(dS_i32, dProbs,
                                             dQs + size_t(hd) * T, dKs + size_t(kv) * T, T, T);
            // O(T,D) = P(T,T) . V(T,D)  (fp16, mode 1)
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, T, T, &f1,
                         dV + size_t(kv) * T * D, CUDA_R_16F, D,
                         dProbs, CUDA_R_16F, T, &f0,
                         dOut8 + size_t(hd) * T * D, CUDA_R_16F, D,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    };
    // -------- fp16 reference/baseline path --------
    auto run_fp16 = [&] {
        for (int hd = 0; hd < H; hd++) {
            const int kv = hd / (H / HKV);
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, T, T, D, &f1,
                         dK + size_t(kv) * T * D, CUDA_R_16F, D,
                         dQ + size_t(hd) * T * D, CUDA_R_16F, D, &f0,
                         dS_f32, CUDA_R_32F, T, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            attention_softmax_f16<<<T, 256>>>(dS_f32, dProbs, T, T);
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, T, T, &f1,
                         dV + size_t(kv) * T * D, CUDA_R_16F, D,
                         dProbs, CUDA_R_16F, T, &f0,
                         dOutF + size_t(hd) * T * D, CUDA_R_16F, D,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    };

    run_int8(); run_fp16(); cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }

    // ---- fp64 oracle from the int8-quantized Q,K (head 0) ----
    std::vector<int8_t> Q8(Qf.size()), K8(Kf.size()); std::vector<float> Qs(size_t(H) * T), Ks(size_t(HKV) * T);
    cudaMemcpy(Q8.data(), dQ8, Q8.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(K8.data(), dK8, K8.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(Qs.data(), dQs, Qs.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ks.data(), dKs, Ks.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<__half> out8(size_t(H) * T * D), outF(size_t(H) * T * D);
    cudaMemcpy(out8.data(), dOut8, out8.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(outF.data(), dOutF, outF.size() * 2, cudaMemcpyDeviceToHost);

    // oracle for head 0 (kv 0)
    double gsum = 0, rsum = 0, gmax = 0, isum = 0, iref = 0;
    {
        const int hd = 0, kv = 0;
        std::vector<double> prob(T);
        for (int q = 0; q < T; q++) {
            double mmax = -1e300;
            for (int k = 0; k < T; k++) {
                long dot = 0;
                for (int d = 0; d < D; d++) dot += long(Q8[(size_t(hd) * T + q) * D + d]) * long(K8[(size_t(kv) * T + k) * D + d]);
                double s = double(dot) * Qs[size_t(hd) * T + q] * Ks[size_t(kv) * T + k];
                prob[k] = s; mmax = std::max(mmax, s);
            }
            double den = 0; for (int k = 0; k < T; k++) { prob[k] = std::exp(prob[k] - mmax); den += prob[k]; }
            for (int k = 0; k < T; k++) prob[k] /= den;
            for (int d = 0; d < D; d++) {
                double o = 0;
                for (int k = 0; k < T; k++) o += prob[k] * double(__half2float(Vf[(size_t(kv) * T + k) * D + d]));
                double got = double(__half2float(out8[(size_t(hd) * T + q) * D + d]));
                gsum += std::abs(got - o); rsum += std::abs(o); gmax = std::max(gmax, std::abs(got - o));
                double reff = double(__half2float(outF[(size_t(hd) * T + q) * D + d]));
                isum += std::abs(got - reff); iref += std::abs(reff);
            }
        }
    }
    double rel = gsum / std::max(rsum, 1e-30);
    printf("attn_q_int8 vs fp64 oracle (head 0): rel %.4f%% max %.4g  (%s)\n",
           100 * rel, gmax, rel < 0.02 ? "PASS" : "FAIL");
    printf("int8-QK attention vs full-fp16 attention (accuracy of mode 1): rel %.4f%%\n",
           100 * isum / std::max(iref, 1e-30));

    // ---- perf: full attention int8 vs fp16 ----
    auto bench = [&](auto fn) {
        for (int i = 0; i < 10; i++) fn();
        cudaDeviceSynchronize();
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < 50; i++) fn();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); return ms / 50;
    };
    float ms8 = bench(run_int8), msF = bench(run_fp16);
    // QK-only stage isolation
    auto qk8 = [&] {
        quantize_rows_i8<<<(H * T + 7) / 8, 256>>>(dQ, dQ8, dQs, H * T);
        quantize_rows_i8<<<(HKV * T + 7) / 8, 256>>>(dK, dK8, dKs, HKV * T);
        for (int hd = 0; hd < H; hd++) { const int kv = hd / (H / HKV);
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, T, T, D, &i1, dK8 + size_t(kv) * T * D, CUDA_R_8I, D,
                         dQ8 + size_t(hd) * T * D, CUDA_R_8I, D, &i0, dS_i32, CUDA_R_32I, T, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP); } };
    auto qkF = [&] {
        for (int hd = 0; hd < H; hd++) { const int kv = hd / (H / HKV);
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, T, T, D, &f1, dK + size_t(kv) * T * D, CUDA_R_16F, D,
                         dQ + size_t(hd) * T * D, CUDA_R_16F, D, &f0, dS_f32, CUDA_R_32F, T, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); } };
    float qms8 = bench(qk8), qmsF = bench(qkF);
    double qkflop = 2.0 * H * double(T) * T * D / 1e12;
    printf("full attention: int8 %.4f ms  |  fp16 %.4f ms  -> %.2fx\n", ms8, msF, msF / ms8);
    printf("QK stage only : int8 %.4f ms (%.1f TOP/s) | fp16 %.4f ms (%.1f TFLOP/s) -> %.2fx\n",
           qms8, qkflop / (qms8 / 1e3), qmsF, qkflop / (qmsF / 1e3), qmsF / qms8);
    cublasDestroy(h);
    return rel < 0.02 ? 0 : 1;
}
