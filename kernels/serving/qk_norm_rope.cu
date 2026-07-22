// Fused per-head QK-RMSNorm + NeoX RoPE with packed-QKV split and f16 cast
// (head-dim 256, GQA) -- CUDA/SM86 serving variant.
//
// Shape-named specialization of the projection family, ported from
// embeddinggemma.c (src/engine_cuda.cu `qkv_norm_rope_f16_kernel` and
// `qk_norm_rope_kernel`; portable fp64 reference is the CPU
// `ei_qk_norm_rope_qk_inplace` in src/kernels.c). Named by the (head-dim 256,
// GQA N_HEAD:N_HEAD_KV) shape, never by model.
//
// One warp owns one (token, head) task. The packed projection output `combined`
// for a token is laid out [Q: N_HEAD*D][K: N_HEAD_KV*D][V: N_HEAD_KV*D] in f32.
// For every Q and K head the warp: (1) RMS-normalizes the D-length head row with
// the learned per-dim norm weight and folds the query logit scale into the
// reciprocal -- inv = rsqrt(ss/D + eps) * scale, with scale = 0.0625 for Q and
// 1.0 for K (the source folds 0.0625 into `inv`, i.e. applies it to x0,x1 BEFORE
// the rotation, which is identical to applying it after because RoPE is a
// rotation); (2) applies NeoX-style rotary using a precomputed (cos,sin) table
// indexed by the token's position -- pairing dim d with dim d+D/2 (split-half);
// (3) casts to f16 into the split Q / K outputs. Every V head is a plain
// f32->f16 copy (no norm, no rope), exactly the source's V task.
//
// Correctness: fp64 oracle recomputes the reference norm->rope->cast math in
// double precision from the same inputs (scale folded into inv, matching the
// kernel); the f16 kernel output must match the repo gate rel<0.02 with cosine~1.
// Swept over several sequence lengths, both rope bases (swa 1e4 / full 1e6), and
// several GQA head-group counts.
//
// Build:
//   /usr/local/cuda/bin/nvcc qk_norm_rope.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o qk_norm_rope.out
// Run:
//   CUDA_VISIBLE_DEVICES=1 ./qk_norm_rope.out [tokens_perf]
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// ---- ported: fused per-head QK-RMSNorm + NeoX RoPE + packed-split + f16 cast ---
template <int N_HEAD, int N_HEAD_KV, int D>
__global__ void qk_norm_rope(const float *combined, __half *q, __half *k,
                             __half *v, const float *q_weight,
                             const float *k_weight, const uint32_t *positions,
                             const float2 *rope_table, uint32_t n_tokens,
                             float eps, float q_scale) {
    constexpr uint32_t Q_DIM = N_HEAD * D;
    constexpr uint32_t KV_DIM = N_HEAD_KV * D;
    constexpr uint32_t COMBINED_STRIDE = Q_DIM + 2 * KV_DIM;
    constexpr uint32_t TASKS = N_HEAD + 2 * N_HEAD_KV;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t task = (size_t)blockIdx.x * kWarpsPerBlock + warp;
    if (task >= (size_t)n_tokens * TASKS) return;
    const uint32_t token = (uint32_t)(task / TASKS);
    const uint32_t ct = (uint32_t)(task - (size_t)token * TASKS);
    const size_t input_base = (size_t)token * COMBINED_STRIDE;

    // V task: plain f32 -> f16 copy.
    if (ct >= N_HEAD + N_HEAD_KV) {
        const uint32_t kv = ct - (N_HEAD + N_HEAD_KV);
        const size_t source = input_base + Q_DIM + KV_DIM + (size_t)kv * D;
        const size_t dest = (size_t)token * KV_DIM + (size_t)kv * D;
        for (uint32_t dim = lane; dim < D; dim += 32)
            v[dest + dim] = __float2half(combined[source + dim]);
        return;
    }

    const bool key_head = ct >= N_HEAD;
    const uint32_t head = key_head ? ct - N_HEAD : ct;
    const size_t source = key_head ? input_base + Q_DIM + (size_t)head * D
                                   : input_base + (size_t)head * D;
    const uint32_t out_stride = key_head ? KV_DIM : Q_DIM;
    const size_t dest = (size_t)token * out_stride + (size_t)head * D;
    const float *weight = key_head ? k_weight : q_weight;
    __half *out = key_head ? k : q;
    const float scale = key_head ? 1.0f : q_scale;

    float sum = 0.0f;
    for (uint32_t dim = lane; dim < D; dim += 32) {
        const float value = combined[source + dim];
        sum = fmaf(value, value, sum);
    }
    sum = warp_sum(sum);
    sum = __shfl_sync(kWarpMask, sum, 0);
    const float inv = rsqrtf(sum / (float)D + eps) * scale;
    const size_t rope_base = (size_t)positions[token] * (D / 2);
    for (uint32_t dim = lane; dim < D / 2; dim += 32) {
        const float2 cs = rope_table[rope_base + dim];
        const float x0 = combined[source + dim] * weight[dim] * inv;
        const float x1 = combined[source + dim + D / 2] *
                         weight[dim + D / 2] * inv;
        out[dest + dim] = __float2half(fmaf(-x1, cs.y, x0 * cs.x));
        out[dest + dim + D / 2] = __float2half(fmaf(x0, cs.y, x1 * cs.x));
    }
}

// ---- naive baseline: split+norm into an f32 temp, then a separate rope+cast ---
template <int N_HEAD, int N_HEAD_KV, int D>
__global__ void split_norm_f32(const float *combined, float *q_tmp, float *k_tmp,
                               __half *v, const float *q_weight,
                               const float *k_weight, uint32_t n_tokens,
                               float eps, float q_scale) {
    constexpr uint32_t Q_DIM = N_HEAD * D;
    constexpr uint32_t KV_DIM = N_HEAD_KV * D;
    constexpr uint32_t COMBINED_STRIDE = Q_DIM + 2 * KV_DIM;
    constexpr uint32_t TASKS = N_HEAD + 2 * N_HEAD_KV;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t task = (size_t)blockIdx.x * kWarpsPerBlock + warp;
    if (task >= (size_t)n_tokens * TASKS) return;
    const uint32_t token = (uint32_t)(task / TASKS);
    const uint32_t ct = (uint32_t)(task - (size_t)token * TASKS);
    const size_t input_base = (size_t)token * COMBINED_STRIDE;
    if (ct >= N_HEAD + N_HEAD_KV) {
        const uint32_t kv = ct - (N_HEAD + N_HEAD_KV);
        const size_t source = input_base + Q_DIM + KV_DIM + (size_t)kv * D;
        const size_t dest = (size_t)token * KV_DIM + (size_t)kv * D;
        for (uint32_t dim = lane; dim < D; dim += 32)
            v[dest + dim] = __float2half(combined[source + dim]);
        return;
    }
    const bool key_head = ct >= N_HEAD;
    const uint32_t head = key_head ? ct - N_HEAD : ct;
    const size_t source = key_head ? input_base + Q_DIM + (size_t)head * D
                                   : input_base + (size_t)head * D;
    const uint32_t out_stride = key_head ? KV_DIM : Q_DIM;
    const size_t dest = (size_t)token * out_stride + (size_t)head * D;
    const float *weight = key_head ? k_weight : q_weight;
    float *out = key_head ? k_tmp : q_tmp;
    const float scale = key_head ? 1.0f : q_scale;
    float sum = 0.0f;
    for (uint32_t dim = lane; dim < D; dim += 32) {
        const float value = combined[source + dim];
        sum = fmaf(value, value, sum);
    }
    sum = warp_sum(sum);
    sum = __shfl_sync(kWarpMask, sum, 0);
    const float inv = rsqrtf(sum / (float)D + eps) * scale;
    for (uint32_t dim = lane; dim < D; dim += 32)
        out[dest + dim] = combined[source + dim] * weight[dim] * inv;
}

template <int N_HEAD, int N_HEAD_KV, int D>
__global__ void rope_cast_f16(const float *q_tmp, const float *k_tmp, __half *q,
                              __half *k, const uint32_t *positions,
                              const float2 *rope_table, uint32_t n_tokens) {
    constexpr uint32_t Q_DIM = N_HEAD * D;
    constexpr uint32_t KV_DIM = N_HEAD_KV * D;
    constexpr uint32_t TASKS = N_HEAD + N_HEAD_KV;   // only Q/K get rope
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t task = (size_t)blockIdx.x * kWarpsPerBlock + warp;
    if (task >= (size_t)n_tokens * TASKS) return;
    const uint32_t token = (uint32_t)(task / TASKS);
    const uint32_t ct = (uint32_t)(task - (size_t)token * TASKS);
    const bool key_head = ct >= N_HEAD;
    const uint32_t head = key_head ? ct - N_HEAD : ct;
    const uint32_t out_stride = key_head ? KV_DIM : Q_DIM;
    const size_t base = (size_t)token * out_stride + (size_t)head * D;
    const float *in = key_head ? k_tmp : q_tmp;
    __half *out = key_head ? k : q;
    const size_t rope_base = (size_t)positions[token] * (D / 2);
    for (uint32_t dim = lane; dim < D / 2; dim += 32) {
        const float2 cs = rope_table[rope_base + dim];
        const float x0 = in[base + dim];
        const float x1 = in[base + dim + D / 2];
        out[base + dim] = __float2half(fmaf(-x1, cs.y, x0 * cs.x));
        out[base + dim + D / 2] = __float2half(fmaf(x0, cs.y, x1 * cs.x));
    }
}

// ============================== harness ==============================
static float frand(std::mt19937 &r) { std::normal_distribution<float> nd(0.f, 1.f); return nd(r); }

static int task_blocks(size_t tasks) {
    return (int)((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

// fp64 oracle: the reference norm(scale-in-inv) -> NeoX rope -> cast, per token.
template <int N_HEAD, int N_HEAD_KV, int D>
static void oracle_token(const std::vector<float> &comb,
                         const std::vector<float> &qw,
                         const std::vector<float> &kw, uint32_t token,
                         uint32_t pos, double base, float q_scale, float eps,
                         std::vector<double> &q_out, std::vector<double> &k_out,
                         std::vector<double> &v_out) {
    constexpr uint32_t Q_DIM = N_HEAD * D, KV_DIM = N_HEAD_KV * D;
    constexpr uint32_t STRIDE = Q_DIM + 2 * KV_DIM;
    const size_t ib = (size_t)token * STRIDE;
    for (int h = 0; h < N_HEAD; h++) {
        const float *src = comb.data() + ib + (size_t)h * D;
        double ss = 0; for (int d = 0; d < D; d++) ss += (double)src[d] * src[d];
        double inv = 1.0 / std::sqrt(ss / (double)D + (double)eps) * (double)q_scale;
        for (int dim = 0; dim < D / 2; dim++) {
            double theta = (double)pos * std::pow(base, -2.0 * dim / (double)D);
            double c = std::cos(theta), s = std::sin(theta);
            double x0 = (double)src[dim] * (double)qw[dim] * inv;
            double x1 = (double)src[dim + D / 2] * (double)qw[dim + D / 2] * inv;
            q_out[(size_t)h * D + dim] = x0 * c - x1 * s;
            q_out[(size_t)h * D + dim + D / 2] = x0 * s + x1 * c;
        }
    }
    for (int h = 0; h < N_HEAD_KV; h++) {
        const float *src = comb.data() + ib + Q_DIM + (size_t)h * D;
        double ss = 0; for (int d = 0; d < D; d++) ss += (double)src[d] * src[d];
        double inv = 1.0 / std::sqrt(ss / (double)D + (double)eps);
        for (int dim = 0; dim < D / 2; dim++) {
            double theta = (double)pos * std::pow(base, -2.0 * dim / (double)D);
            double c = std::cos(theta), s = std::sin(theta);
            double x0 = (double)src[dim] * (double)kw[dim] * inv;
            double x1 = (double)src[dim + D / 2] * (double)kw[dim + D / 2] * inv;
            k_out[(size_t)h * D + dim] = x0 * c - x1 * s;
            k_out[(size_t)h * D + dim + D / 2] = x0 * s + x1 * c;
        }
    }
    for (int h = 0; h < N_HEAD_KV; h++) {
        const float *src = comb.data() + ib + Q_DIM + KV_DIM + (size_t)h * D;
        for (int d = 0; d < D; d++) v_out[(size_t)h * D + d] = (double)src[d];
    }
}

template <int N_HEAD, int N_HEAD_KV, int D>
static bool correctness(int max_tokens, unsigned seed) {
    constexpr uint32_t Q_DIM = N_HEAD * D, KV_DIM = N_HEAD_KV * D;
    constexpr uint32_t STRIDE = Q_DIM + 2 * KV_DIM;
    constexpr uint32_t TASKS = N_HEAD + 2 * N_HEAD_KV;
    const float eps = 1e-6f, q_scale = 0.0625f;
    const int token_counts[] = {1, 7, 37, 128, 512, 2048};
    const double bases[] = {10000.0, 1000000.0};
    std::mt19937 rng(seed);

    std::vector<float> qw(D), kw(D);
    for (auto &x : qw) x = 1.0f + frand(rng) * 0.1f;
    for (auto &x : kw) x = 1.0f + frand(rng) * 0.1f;

    double gsum = 0, rsum = 0, gmax = 0, dotp = 0, ng = 0, no = 0;
    for (double base : bases) {
        for (int T : token_counts) {
            if (T > max_tokens) continue;
            std::vector<float> comb((size_t)T * STRIDE);
            for (auto &x : comb) x = frand(rng) * 0.7f;
            std::vector<uint32_t> pos(T);
            for (int t = 0; t < T; t++) pos[t] = (uint32_t)t;

            std::vector<float2> rope((size_t)T * (D / 2));
            for (int p = 0; p < T; p++)
                for (int dim = 0; dim < D / 2; dim++) {
                    double theta = (double)p * std::pow(base, -2.0 * dim / (double)D);
                    rope[(size_t)p * (D / 2) + dim] =
                        make_float2((float)std::cos(theta), (float)std::sin(theta));
                }

            float *dc, *dqw, *dkw; float2 *drope; uint32_t *dpos;
            __half *dq, *dk, *dv;
            cudaMalloc(&dc, comb.size() * 4);
            cudaMalloc(&dqw, D * 4); cudaMalloc(&dkw, D * 4);
            cudaMalloc(&drope, rope.size() * sizeof(float2));
            cudaMalloc(&dpos, T * 4);
            cudaMalloc(&dq, (size_t)T * Q_DIM * 2);
            cudaMalloc(&dk, (size_t)T * KV_DIM * 2);
            cudaMalloc(&dv, (size_t)T * KV_DIM * 2);
            cudaMemcpy(dc, comb.data(), comb.size() * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dqw, qw.data(), D * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dkw, kw.data(), D * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(drope, rope.data(), rope.size() * sizeof(float2), cudaMemcpyHostToDevice);
            cudaMemcpy(dpos, pos.data(), T * 4, cudaMemcpyHostToDevice);

            qk_norm_rope<N_HEAD, N_HEAD_KV, D><<<task_blocks((size_t)T * TASKS), kThreads>>>(
                dc, dq, dk, dv, dqw, dkw, dpos, drope, T, eps, q_scale);
            cudaDeviceSynchronize();
            if (cudaGetLastError() != cudaSuccess) { printf("  KERNEL ERROR\n"); return false; }

            std::vector<__half> gq((size_t)T * Q_DIM), gk((size_t)T * KV_DIM), gv((size_t)T * KV_DIM);
            cudaMemcpy(gq.data(), dq, gq.size() * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(gk.data(), dk, gk.size() * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(gv.data(), dv, gv.size() * 2, cudaMemcpyDeviceToHost);

            std::vector<double> orq(Q_DIM), ork(KV_DIM), orv(KV_DIM);
            for (int t = 0; t < T; t++) {
                oracle_token<N_HEAD, N_HEAD_KV, D>(comb, qw, kw, t, pos[t], base, q_scale, eps, orq, ork, orv);
                for (uint32_t i = 0; i < Q_DIM; i++) {
                    double g = __half2float(gq[(size_t)t * Q_DIM + i]), o = orq[i];
                    gsum += std::abs(g - o); rsum += std::abs(o); gmax = std::max(gmax, std::abs(g - o));
                    dotp += g * o; ng += g * g; no += o * o;
                }
                for (uint32_t i = 0; i < KV_DIM; i++) {
                    double gkk = __half2float(gk[(size_t)t * KV_DIM + i]), ok = ork[i];
                    gsum += std::abs(gkk - ok); rsum += std::abs(ok); gmax = std::max(gmax, std::abs(gkk - ok));
                    dotp += gkk * ok; ng += gkk * gkk; no += ok * ok;
                    double gvv = __half2float(gv[(size_t)t * KV_DIM + i]), ov = orv[i];
                    gsum += std::abs(gvv - ov); rsum += std::abs(ov); gmax = std::max(gmax, std::abs(gvv - ov));
                    dotp += gvv * ov; ng += gvv * gvv; no += ov * ov;
                }
            }
            cudaFree(dc); cudaFree(dqw); cudaFree(dkw); cudaFree(drope); cudaFree(dpos);
            cudaFree(dq); cudaFree(dk); cudaFree(dv);
        }
    }
    double rel = gsum / std::max(rsum, 1e-30);
    double cosine = dotp / std::max(std::sqrt(ng * no), 1e-30);
    bool pass = rel < 0.02 && cosine > 0.999;
    printf("  GQA %d:%d  correctness (T in {1..2048}, base {1e4,1e6}): rel %.5f%% max %.3g  cosine %.8f  (%s)\n",
           N_HEAD, N_HEAD_KV, 100 * rel, gmax, cosine, pass ? "PASS" : "FAIL");
    return pass;
}

template <int N_HEAD, int N_HEAD_KV, int D>
static void perf(int T, unsigned seed) {
    constexpr uint32_t Q_DIM = N_HEAD * D, KV_DIM = N_HEAD_KV * D;
    constexpr uint32_t STRIDE = Q_DIM + 2 * KV_DIM;
    constexpr uint32_t TASKS = N_HEAD + 2 * N_HEAD_KV;
    constexpr uint32_t QK_TASKS = N_HEAD + N_HEAD_KV;
    const float eps = 1e-6f, q_scale = 0.0625f; const double base = 10000.0;
    std::mt19937 rng(seed);
    std::vector<float> comb((size_t)T * STRIDE), qw(D), kw(D);
    for (auto &x : comb) x = frand(rng) * 0.7f;
    for (auto &x : qw) x = 1.0f + frand(rng) * 0.1f;
    for (auto &x : kw) x = 1.0f + frand(rng) * 0.1f;
    std::vector<uint32_t> pos(T); for (int t = 0; t < T; t++) pos[t] = (uint32_t)t;
    std::vector<float2> rope((size_t)T * (D / 2));
    for (int p = 0; p < T; p++)
        for (int dim = 0; dim < D / 2; dim++) {
            double theta = (double)p * std::pow(base, -2.0 * dim / (double)D);
            rope[(size_t)p * (D / 2) + dim] = make_float2((float)std::cos(theta), (float)std::sin(theta));
        }
    float *dc, *dqw, *dkw, *dqt, *dkt; float2 *drope; uint32_t *dpos; __half *dq, *dk, *dv;
    cudaMalloc(&dc, comb.size() * 4); cudaMalloc(&dqw, D * 4); cudaMalloc(&dkw, D * 4);
    cudaMalloc(&drope, rope.size() * sizeof(float2)); cudaMalloc(&dpos, T * 4);
    cudaMalloc(&dq, (size_t)T * Q_DIM * 2); cudaMalloc(&dk, (size_t)T * KV_DIM * 2); cudaMalloc(&dv, (size_t)T * KV_DIM * 2);
    cudaMalloc(&dqt, (size_t)T * Q_DIM * 4); cudaMalloc(&dkt, (size_t)T * KV_DIM * 4);
    cudaMemcpy(dc, comb.data(), comb.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dqw, qw.data(), D * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dkw, kw.data(), D * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(drope, rope.data(), rope.size() * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, pos.data(), T * 4, cudaMemcpyHostToDevice);

    auto fused = [&] {
        qk_norm_rope<N_HEAD, N_HEAD_KV, D><<<task_blocks((size_t)T * TASKS), kThreads>>>(
            dc, dq, dk, dv, dqw, dkw, dpos, drope, T, eps, q_scale);
    };
    auto naive = [&] {
        split_norm_f32<N_HEAD, N_HEAD_KV, D><<<task_blocks((size_t)T * TASKS), kThreads>>>(
            dc, dqt, dkt, dv, dqw, dkw, T, eps, q_scale);
        rope_cast_f16<N_HEAD, N_HEAD_KV, D><<<task_blocks((size_t)T * QK_TASKS), kThreads>>>(
            dqt, dkt, dq, dk, dpos, drope, T);
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
    if (cudaGetLastError() != cudaSuccess) { printf("  PERF KERNEL ERROR\n"); return; }
    printf("  GQA %d:%d  perf (T=%d): fused %.4f ms | naive split+rope %.4f ms -> %.2fx\n",
           N_HEAD, N_HEAD_KV, T, mf, mn, mn / mf);
    cudaFree(dc); cudaFree(dqw); cudaFree(dkw); cudaFree(drope); cudaFree(dpos);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dqt); cudaFree(dkt);
}

int main(int argc, char **argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 2048;
    printf("== qk_norm_rope (fused QK-RMSNorm + NeoX RoPE + packed split + f16, head-dim 256)\n");
    bool ok = true;
    ok &= correctness<3, 1, 256>(2048, 11);   // EmbeddingGemma shape (reference)
    ok &= correctness<4, 2, 256>(2048, 12);
    ok &= correctness<8, 2, 256>(2048, 13);
    perf<3, 1, 256>(T, 21);
    perf<4, 2, 256>(T, 22);
    perf<8, 2, 256>(T, 23);
    printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
