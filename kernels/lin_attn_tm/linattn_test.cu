/**
 * @file
 * @brief fp64-oracle harness for tm_linattn_kernels.cuh: non-causal
 * linear_attn, serial lin_attn_causal, the 3-kernel chunk-parallel pipeline
 * (must equal the causal oracle exactly up to fp32 rounding), and
 * cmplx_matmul.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        linattn_test.cu -o linattn_test.out
 * Run: CUDA_VISIBLE_DEVICES=6 ./linattn_test.out
 */
#include "tm_linattn_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace tmla;

static int g_fail = 0;
#define CUCHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

template <typename T> T* dnew(const std::vector<T>& h) {
    T* d; CUCHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUCHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> T* dzero(size_t n) {
    T* d; CUCHECK(cudaMalloc(&d, n * sizeof(T)));
    CUCHECK(cudaMemset(d, 0, n * sizeof(T)));
    return d;
}
template <typename T> std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    CUCHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
static void report(const char* name, double err, double tol) {
    const bool ok = err <= tol;
    printf("%-42s %s  (err %.3e, tol %.1e)\n", name, ok ? "PASS" : "FAIL", err, tol);
    if (!ok) ++g_fail;
}

static std::mt19937 g_rng(7);
static std::vector<float> randv(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g_rng);
    return v;
}
static double maxrel(const std::vector<double>& ref, const std::vector<float>& got,
                     double floor_ = 1.0) {
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double s = std::max(std::abs(ref[i]), floor_);
        worst = std::max(worst, std::abs(double(got[i]) - ref[i]) / s);
    }
    return worst;
}

int main() {
    constexpr int B = 2, H = 3, N = 256, D = 64;
    const size_t sz = (size_t)B * H * N * D;
    // scaled inputs keep the O(N*D) sums O(1)-ish for meaningful rel err
    auto q = randv(sz, -0.5f, 0.5f), k = randv(sz, -0.5f, 0.5f), v = randv(sz, -0.5f, 0.5f);

    // fp64 oracles
    std::vector<double> ref_nc(sz), ref_c(sz);
    for (int bh = 0; bh < B * H; ++bh) {
        const size_t base = (size_t)bh * N * D;
        std::vector<double> kv(D * D, 0.0);
        for (int r = 0; r < N; ++r)                             // full KV
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j)
                    kv[i * D + j] += (double)k[base + r * D + i] * v[base + r * D + j];
        for (int r = 0; r < N; ++r)
            for (int j = 0; j < D; ++j) {
                double acc = 0;
                for (int i = 0; i < D; ++i) acc += (double)q[base + r * D + i] * kv[i * D + j];
                ref_nc[base + r * D + j] = acc;
            }
        std::fill(kv.begin(), kv.end(), 0.0);                    // causal inclusive
        for (int r = 0; r < N; ++r) {
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j)
                    kv[i * D + j] += (double)k[base + r * D + i] * v[base + r * D + j];
            for (int j = 0; j < D; ++j) {
                double acc = 0;
                for (int i = 0; i < D; ++i) acc += (double)q[base + r * D + i] * kv[i * D + j];
                ref_c[base + r * D + j] = acc;
            }
        }
    }

    float *dq = dnew(q), *dk = dnew(k), *dv = dnew(v), *dout = dzero<float>(sz);
    dim3 gbh{H, B};
    linear_attn<float, D><<<gbh, D>>>(dq, dk, dv, dout, N);
    CUCHECK(cudaDeviceSynchronize());
    report("linear_attn (non-causal)", maxrel(ref_nc, d2h(dout, sz), 8.0), 2e-6);

    lin_attn_causal<float, D><<<gbh, D>>>(dq, dk, dv, dout, N);
    CUCHECK(cudaDeviceSynchronize());
    report("lin_attn_causal (serial)", maxrel(ref_c, d2h(dout, sz), 8.0), 2e-6);

    // chunk-parallel pipeline
    const int C = N / LIN_CHUNK_L;
    float *dS = dzero<float>((size_t)B * H * C * D * D);
    float *dSex = dzero<float>((size_t)B * H * C * D * D);
    dim3 gc{unsigned(C), H, B};
    lin_chunk_kv<float, D><<<gc, D>>>(dk, dv, dS, N);
    lin_chunk_scan<D><<<B * H, 256>>>(dS, dSex, C);
    lin_chunk_out<float, D><<<gc, D>>>(dq, dk, dv, dSex, dout, N);
    CUCHECK(cudaDeviceSynchronize());
    report("lin_chunk kv->scan->out (parallel)", maxrel(ref_c, d2h(dout, sz), 8.0), 2e-6);

    // cmplx_matmul
    {
        const int Nn = 96, K = 48, M = 64;
        auto A = randv((size_t)2 * Nn * K), Bv = randv((size_t)2 * K * M);
        std::vector<double> ref((size_t)2 * Nn * M, 0.0);
        for (int n = 0; n < Nn; ++n)
            for (int m = 0; m < M; ++m) {
                double dr = 0, di = 0;
                for (int kk = 0; kk < K; ++kk) {
                    const double ar = A[(size_t)n * K + kk], ai = A[(size_t)Nn * K + n * K + kk];
                    const double br = Bv[(size_t)kk * M + m], bi = Bv[(size_t)K * M + kk * M + m];
                    dr += ar * br - ai * bi;
                    di += ar * bi + ai * br;
                }
                ref[(size_t)n * M + m] = dr;
                ref[(size_t)Nn * M + n * M + m] = di;
            }
        float *dA = dnew(A), *dB = dnew(Bv), *dD = dzero<float>((size_t)2 * Nn * M);
        dim3 blk{16, 16};
        dim3 grd{unsigned((M + 15) / 16), unsigned((Nn + 15) / 16)};
        cmplx_matmul<float><<<grd, blk>>>(dA, dB, dD, Nn, K, M);
        CUCHECK(cudaDeviceSynchronize());
        report("cmplx_matmul", maxrel(ref, d2h(dD, (size_t)2 * Nn * M), 4.0), 2e-6);
    }

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
