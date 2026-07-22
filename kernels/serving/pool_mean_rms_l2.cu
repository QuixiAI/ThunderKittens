// Sentence-embedding pooling head: masked mean-pool -> RMSNorm -> L2, per-D warp.
// Shape-named specialization of the serving family, ported from embeddinggemma.c
// (src/engine_cuda.cu `pool_kernel`, ~L1246; portable fp64 reference is the CPU
// `ei_mean_pool_rms_l2` in src/kernels.c, ~L417). Named by the D-vector shape
// (D in {256,512,768,1024}), never by model.
//
// One warp owns a whole sequence's D-vector: lane `l` owns dims l, l+32, ...,
// l+(D-32) (D/32 registers per lane). For each valid token in the sequence's
// [start,stop) offset range: (1) RMS-normalize the token row with the learned
// weight -- ss = sum_d row[d]^2, inv = rsqrt(ss/D + eps), and accumulate
// row[d]*w[d]*inv into the running pool (this is the per-token RMS the reference
// applies BEFORE averaging -- confirmed against both sources). After the loop:
// (2) mean-pool by 1/(stop-start); (3) L2-normalize the pooled vector (guarding
// the all-zero case with unit scale, as the reference does) and store it.
// Weights are applied as plain `w` (the +1 Gemma fold is baked into the exported
// norm weights offline -- both reference kernels use `x*scale*w`, no (1+w) here).
//
// Only the plain mean-pool->RMS->L2 head is ported; the source's fused-singleton
// -pool epilogue is intentionally NOT ported (it fails parity on GEMM shapes).
//
// Correctness: fp64 oracle recomputes the CPU reference math in double precision
// from the same inputs; the fp32 kernel must match to the repo's rel < 0.02
// (pure-fp32 kernel, so it lands far tighter) with cosine ~1.
//
// Build:
//   /usr/local/cuda/bin/nvcc pool_mean_rms_l2.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o pool_mean_rms_l2.out
// Run:
//   CUDA_VISIBLE_DEVICES=1 ./pool_mean_rms_l2.out [tokens_per_seq] [num_seqs]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;
constexpr unsigned kWarpMask = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(kWarpMask, value, offset);
    return value;
}

// ---- ported: mean-pool -> per-token RMSNorm -> L2, one warp per sequence ----
template <int D>
__global__ void pool_mean_rms_l2(const float *input, const float *weight,
                                 float *output, const uint32_t *offsets,
                                 uint32_t batch_size, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t sequence = blockIdx.x * kWarpsPerBlock + warp;
    if (sequence >= batch_size) return;
    const uint32_t start = offsets[sequence];
    const uint32_t stop = offsets[sequence + 1];
    float pooled[D / 32];
#pragma unroll
    for (int item = 0; item < D / 32; item++) pooled[item] = 0.0f;

    for (uint32_t token = start; token < stop; token++) {
        float sum = 0.0f;
#pragma unroll
        for (int item = 0; item < D / 32; item++) {
            const int dim = lane + item * 32;
            const float value = input[static_cast<size_t>(token) * D + dim];
            sum = fmaf(value, value, sum);
        }
        sum = warp_sum(sum);
        sum = __shfl_sync(kWarpMask, sum, 0);
        const float inv = rsqrtf(sum / static_cast<float>(D) + eps);
#pragma unroll
        for (int item = 0; item < D / 32; item++) {
            const int dim = lane + item * 32;
            pooled[item] += input[static_cast<size_t>(token) * D + dim] *
                            weight[dim] * inv;
        }
    }

    const float inv_tokens = 1.0f / static_cast<float>(stop - start);
    float sum = 0.0f;
#pragma unroll
    for (int item = 0; item < D / 32; item++) {
        pooled[item] *= inv_tokens;
        sum = fmaf(pooled[item], pooled[item], sum);
    }
    sum = warp_sum(sum);
    sum = __shfl_sync(kWarpMask, sum, 0);
    const float inv_l2 = sum == 0.0f ? 1.0f : rsqrtf(sum);
#pragma unroll
    for (int item = 0; item < D / 32; item++) {
        const int dim = lane + item * 32;
        output[static_cast<size_t>(sequence) * D + dim] = pooled[item] * inv_l2;
    }
}

// ---- naive baseline: compose the ops (RMS rows -> temp, then mean+L2) --------
// Stage 1: RMS-normalize every token row into a temp [total_tokens][D] buffer.
template <int D>
__global__ void rms_norm_rows(const float *input, const float *weight,
                              float *out, uint32_t total_tokens, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t token = blockIdx.x * kWarpsPerBlock + warp;
    if (token >= total_tokens) return;
    float sum = 0.0f;
#pragma unroll
    for (int item = 0; item < D / 32; item++) {
        const int dim = lane + item * 32;
        const float v = input[static_cast<size_t>(token) * D + dim];
        sum = fmaf(v, v, sum);
    }
    sum = warp_sum(sum);
    sum = __shfl_sync(kWarpMask, sum, 0);
    const float inv = rsqrtf(sum / static_cast<float>(D) + eps);
#pragma unroll
    for (int item = 0; item < D / 32; item++) {
        const int dim = lane + item * 32;
        out[static_cast<size_t>(token) * D + dim] =
            input[static_cast<size_t>(token) * D + dim] * weight[dim] * inv;
    }
}
// Stage 2: mean over each sequence's rows, then L2-normalize.
template <int D>
__global__ void mean_l2_reduce(const float *normed, float *output,
                               const uint32_t *offsets, uint32_t batch_size) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t sequence = blockIdx.x * kWarpsPerBlock + warp;
    if (sequence >= batch_size) return;
    const uint32_t start = offsets[sequence];
    const uint32_t stop = offsets[sequence + 1];
    float pooled[D / 32];
#pragma unroll
    for (int item = 0; item < D / 32; item++) pooled[item] = 0.0f;
    for (uint32_t token = start; token < stop; token++) {
#pragma unroll
        for (int item = 0; item < D / 32; item++) {
            const int dim = lane + item * 32;
            pooled[item] += normed[static_cast<size_t>(token) * D + dim];
        }
    }
    const float inv_tokens = 1.0f / static_cast<float>(stop - start);
    float sum = 0.0f;
#pragma unroll
    for (int item = 0; item < D / 32; item++) {
        pooled[item] *= inv_tokens;
        sum = fmaf(pooled[item], pooled[item], sum);
    }
    sum = warp_sum(sum);
    sum = __shfl_sync(kWarpMask, sum, 0);
    const float inv_l2 = sum == 0.0f ? 1.0f : rsqrtf(sum);
#pragma unroll
    for (int item = 0; item < D / 32; item++) {
        const int dim = lane + item * 32;
        output[static_cast<size_t>(sequence) * D + dim] = pooled[item] * inv_l2;
    }
}

// ============================== harness ==============================
static int row_blocks(size_t rows) {
    return static_cast<int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
}
static float frand(std::mt19937 &r) { std::normal_distribution<float> nd(0.f, 1.f); return nd(r); }

// fp64 oracle mirroring ei_mean_pool_rms_l2 (per-token RMS -> mean -> L2).
static void oracle(const std::vector<float> &x, const std::vector<float> &w,
                   int D, uint32_t start, uint32_t stop, std::vector<double> &out) {
    out.assign(D, 0.0);
    for (uint32_t t = start; t < stop; t++) {
        double ss = 0.0;
        for (int d = 0; d < D; d++) { double v = x[(size_t)t * D + d]; ss += v * v; }
        double scale = 1.0 / std::sqrt(ss / (double)D + 1e-6);
        for (int d = 0; d < D; d++)
            out[d] += (double)x[(size_t)t * D + d] * scale * (double)w[d];
    }
    double inv_tokens = 1.0 / (double)(stop - start);
    double ss2 = 0.0;
    for (int d = 0; d < D; d++) { out[d] *= inv_tokens; ss2 += out[d] * out[d]; }
    if (ss2 != 0.0) { double inv = 1.0 / std::sqrt(ss2); for (int d = 0; d < D; d++) out[d] *= inv; }
}

template <int D>
static bool run_shape(int T, int B) {
    printf("== pool_mean_rms_l2  D=%d\n", D);
    std::mt19937 rng(1234 + D);
    const float eps = 1e-6f;  // model rms_eps

    // ---------- correctness: small packed batch, mixed token counts ----------
    {
        const int token_counts[] = {1, 4, 7, 37, 128, 512};
        const int nseq = sizeof(token_counts) / sizeof(int);
        std::vector<uint32_t> off(nseq + 1, 0);
        for (int s = 0; s < nseq; s++) off[s + 1] = off[s] + token_counts[s];
        const uint32_t total = off[nseq];
        std::vector<float> x((size_t)total * D), w(D);
        for (auto &v : x) v = frand(rng) * 0.7f;
        for (auto &v : w) v = 1.0f + frand(rng) * 0.1f;  // norm weight ~1

        float *dx, *dw, *dout; uint32_t *doff;
        cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dw, w.size() * 4);
        cudaMalloc(&dout, (size_t)nseq * D * 4); cudaMalloc(&doff, off.size() * 4);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(doff, off.data(), off.size() * 4, cudaMemcpyHostToDevice);
        pool_mean_rms_l2<D><<<row_blocks(nseq), kThreads>>>(dx, dw, dout, doff, nseq, eps);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("  KERNEL ERROR\n"); return false; }
        std::vector<float> got((size_t)nseq * D);
        cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);

        double gsum = 0, rsum = 0, gmax = 0, dotp = 0, ng = 0, no = 0;
        std::vector<double> ref;
        for (int s = 0; s < nseq; s++) {
            oracle(x, w, D, off[s], off[s + 1], ref);
            for (int d = 0; d < D; d++) {
                double g = got[(size_t)s * D + d], o = ref[d];
                gsum += std::abs(g - o); rsum += std::abs(o); gmax = std::max(gmax, std::abs(g - o));
                dotp += g * o; ng += g * g; no += o * o;
            }
        }
        double rel = gsum / std::max(rsum, 1e-30);
        double cosine = dotp / std::max(std::sqrt(ng * no), 1e-30);
        bool pass = rel < 0.02 && cosine > 0.9999;
        printf("  correctness (n_tokens 1..512): rel %.6f%% max %.3g  cosine %.8f  (%s)\n",
               100 * rel, gmax, cosine, pass ? "PASS" : "FAIL");
        cudaFree(dx); cudaFree(dw); cudaFree(dout); cudaFree(doff);
        if (!pass) return false;
    }

    // ---------- perf: B sequences of T tokens, fused vs naive-composed ----------
    {
        std::vector<uint32_t> off(B + 1, 0);
        for (int s = 0; s < B; s++) off[s + 1] = off[s] + T;
        const uint32_t total = off[B];
        std::vector<float> x((size_t)total * D), w(D);
        for (auto &v : x) v = frand(rng) * 0.7f;
        for (auto &v : w) v = 1.0f + frand(rng) * 0.1f;

        float *dx, *dw, *dout, *dtmp; uint32_t *doff;
        cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dw, w.size() * 4);
        cudaMalloc(&dout, (size_t)B * D * 4); cudaMalloc(&doff, off.size() * 4);
        cudaMalloc(&dtmp, (size_t)total * D * 4);  // naive stage-1 temp
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(doff, off.data(), off.size() * 4, cudaMemcpyHostToDevice);

        auto fused = [&] {
            pool_mean_rms_l2<D><<<row_blocks(B), kThreads>>>(dx, dw, dout, doff, B, eps);
        };
        auto naive = [&] {
            rms_norm_rows<D><<<row_blocks(total), kThreads>>>(dx, dw, dtmp, total, eps);
            mean_l2_reduce<D><<<row_blocks(B), kThreads>>>(dtmp, dout, doff, B);
        };
        auto bench = [&](auto fn) {
            for (int i = 0; i < 10; i++) fn();
            cudaDeviceSynchronize();
            cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            cudaEventRecord(t0);
            for (int i = 0; i < 50; i++) fn();
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1); return ms / 50;
        };
        float mf = bench(fused), mn = bench(naive);
        if (cudaGetLastError() != cudaSuccess) { printf("  PERF KERNEL ERROR\n"); return false; }
        double gbytes = (double)total * D * 4 / 1e9;  // one read of the token matrix
        printf("  perf (B=%d seqs x T=%d tok): fused %.4f ms (%.1f GB/s) | naive %.4f ms -> %.2fx\n",
               B, T, mf, gbytes / (mf / 1e3), mn, mn / mf);
        cudaFree(dx); cudaFree(dw); cudaFree(dout); cudaFree(doff); cudaFree(dtmp);
    }
    return true;
}

int main(int argc, char **argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 64;    // tokens/seq (perf)
    const int B = argc > 2 ? atoi(argv[2]) : 1024;  // sequences  (perf)
    bool ok = true;
    ok &= run_shape<256>(T, B);
    ok &= run_shape<512>(T, B);
    ok &= run_shape<768>(T, B);
    ok &= run_shape<1024>(T, B);
    printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
