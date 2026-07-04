/**
 * @file
 * @brief Harness for the Step-4 qgemm variants: qgemm_actorder (GPTQ
 * act-order, in-kernel X gather by perm) and qgemm_blockscale (fp8_raw codes
 * + separate (N/128,K/128) fp16 tile scales). Oracle: dequantize W on-device
 * with the proven dequant_to_fp16 kernel, then fp64 GEMM on the host over
 * those exact values (gathered / tile-scaled).
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        qgemm_variants.cu -o qgemm_variants.out
 * Run: CUDA_VISIBLE_DEVICES=6 ./qgemm_variants.out
 */
#include "tm_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace tmq;

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
    printf("%-42s %s  (rel err %.3e, tol %.1e)\n", name, ok ? "PASS" : "FAIL", err, tol);
    if (!ok) ++g_fail;
}

static std::mt19937 g_rng(31);

int main() {
    const int M = 64, N = 128, K = 256;

    std::vector<half> X((size_t)M * K);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    for (auto& x : X) x = __float2half(nd(g_rng));
    half* dX = dnew(X);

    // ---- actorder over q4_0 (random codes, sane scales) ----
    {
        const int BK = q4_0::block_k, BB = q4_0::block_bytes, bpr = K / BK;
        std::vector<uint8_t> Wq((size_t)N * bpr * BB);
        std::uniform_int_distribution<int> ud(0, 255);
        for (auto& b : Wq) b = uint8_t(ud(g_rng));
        for (int n = 0; n < N; ++n)                       // overwrite scale halves with small values
            for (int b = 0; b < bpr; ++b) {
                const half s = __float2half(nd(g_rng) * 0.1f);
                memcpy(Wq.data() + ((size_t)n * bpr + b) * BB, &s, 2);
            }
        uint8_t* dWq = dnew(Wq);
        half* dWd = dzero<half>((size_t)N * K);
        dequant_to_fp16<q4_0><<<(N * K + 255) / 256, 256>>>(dWd, dWq, N, K);
        CUCHECK(cudaDeviceSynchronize());
        auto Wd = d2h(dWd, (size_t)N * K);

        std::vector<int> perm(K);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), g_rng);
        int* dperm = dnew(perm);

        float* dY = dzero<float>((size_t)M * N);
        dim3 grid{unsigned(N / 16), unsigned(M / 16)};
        qgemm_actorder<q4_0><<<grid, 32>>>(dY, dX, dWq, dperm, M, N, K);
        CUCHECK(cudaDeviceSynchronize());
        auto Y = d2h(dY, (size_t)M * N);

        double worst = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                for (int i = 0; i < K; ++i)
                    acc += (double)__half2float(Wd[(size_t)n * K + i])
                         * __half2float(X[(size_t)m * K + perm[i]]);
                const double s = std::max(1.0, std::abs(acc));
                worst = std::max(worst, std::abs((double)Y[(size_t)m * N + n] - acc) / s);
            }
        report("qgemm_actorder q4_0 (fp64 gathered)", worst, 5e-3);
    }

    // ---- blockscale over fp8_raw ----
    {
        const int BK = fp8_raw::block_k, BB = fp8_raw::block_bytes, bpr = K / BK;
        std::vector<uint8_t> Wq((size_t)N * bpr * BB);
        std::uniform_int_distribution<int> ud(0, 255);
        for (auto& b : Wq) {
            uint8_t v = uint8_t(ud(g_rng));
            if ((v & 0x7F) == 0x7F) v &= 0xFE;            // avoid NaN codes
            b = v;
        }
        const int NT = N / 128, KT = K / 128;
        std::vector<half> sc2d((size_t)NT * KT);
        for (auto& s : sc2d) s = __float2half(0.02f + 0.1f * std::abs(nd(g_rng)));
        uint8_t* dWq = dnew(Wq);
        half* dsc = dnew(sc2d);
        half* dWd = dzero<half>((size_t)N * K);
        dequant_to_fp16<fp8_raw><<<(N * K + 255) / 256, 256>>>(dWd, dWq, N, K);
        CUCHECK(cudaDeviceSynchronize());
        auto Wd = d2h(dWd, (size_t)N * K);

        float* dY = dzero<float>((size_t)M * N);
        dim3 grid{unsigned(N / 16), unsigned(M / 16)};
        qgemm_blockscale<fp8_raw><<<grid, 32>>>(dY, dX, dWq, dsc, M, N, K);
        CUCHECK(cudaDeviceSynchronize());
        auto Y = d2h(dY, (size_t)M * N);

        double worst = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                for (int i = 0; i < K; ++i) {
                    // replicate the kernel's half-precision scale multiply exactly
                    const half w = Wd[(size_t)n * K + i];
                    const half ws = __hmul(w, sc2d[(size_t)(n / 128) * KT + (i / 128)]);
                    acc += (double)__half2float(ws) * __half2float(X[(size_t)m * K + i]);
                }
                const double s = std::max(1.0, std::abs(acc));
                worst = std::max(worst, std::abs((double)Y[(size_t)m * N + n] - acc) / s);
            }
        report("qgemm_blockscale fp8_raw (fp64 tiled)", worst, 5e-3);
    }

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
