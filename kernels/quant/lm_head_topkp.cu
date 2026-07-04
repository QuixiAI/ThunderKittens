/**
 * @file
 * @brief Harness for the fused LM-head top-k / top-p samplers (Step 4 items,
 * tm_kernels.cuh lm_head_topk_* / lm_head_topp_reduce). Oracle: fp64 logits
 * (from fp16 or exactly-dequantized q8_0 weights), exact top-k with
 * smaller-id ties + host rng_gumbel replay (bit-identical RNG); top-p
 * replays the pool/bisection algorithm in fp64. Kernel picks must match the
 * oracle exactly, or (top-p threshold-boundary guard) be a nucleus member
 * whose perturbed logit ties the winner within eps.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        lm_head_topkp.cu -o lm_head_topkp.out
 * Run: CUDA_VISIBLE_DEVICES=6 ./lm_head_topkp.out
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
static void report(const char* name, bool ok, const char* extra = "") {
    printf("%-46s %s%s\n", name, ok ? "PASS" : "FAIL", extra);
    if (!ok) ++g_fail;
}

static std::mt19937 g_rng(99);

int main() {
    const int T = 24, V = 4096, K = 128, TILE_V = 512, TOPK = 16;
    const int num_vtiles = (V + TILE_V - 1) / TILE_V;
    const unsigned seed = 1234;
    const float invtemp = 1.0f / 0.8f;
    const float topp = 0.9f;

    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<half> h((size_t)T * K), W((size_t)V * K);
    for (auto& x : h) x = __float2half(nd(g_rng) * 0.3f);
    for (auto& x : W) x = __float2half(nd(g_rng) * 0.3f);

    // fp64 logits from the fp16 values
    std::vector<double> logits((size_t)T * V);
    for (int t = 0; t < T; ++t)
        for (int v = 0; v < V; ++v) {
            double acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (double)__half2float(W[(size_t)v * K + k]) * __half2float(h[(size_t)t * K + k]);
            logits[(size_t)t * V + v] = acc;
        }

    half *dh = dnew(h), *dW = dnew(W);
    float* dpv = dzero<float>((size_t)T * num_vtiles * TOPK);
    int* dpi = dzero<int>((size_t)T * num_vtiles * TOPK);
    float* dlse = dzero<float>((size_t)T * num_vtiles);
    int* dout = dzero<int>(T);
    dim3 grid{unsigned(num_vtiles), unsigned(T)};

    // ---- top-k ----
    lm_head_topk_partials<0><<<grid, 32>>>(dh, dW, dpv, dpi, nullptr, V, K, TILE_V,
                                           num_vtiles, TOPK, 0, invtemp, nullptr);
    lm_head_topk_reduce<<<T, 32>>>(dpv, dpi, dout, num_vtiles, TOPK, seed, invtemp);
    CUCHECK(cudaDeviceSynchronize());
    {
        auto got = d2h(dout, T);
        int mism = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<int> order(V);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + TOPK, order.end(), [&](int a, int b) {
                const double la = logits[(size_t)t * V + a], lb = logits[(size_t)t * V + b];
                return la > lb || (la == lb && a < b);
            });
            double best = -1e300;
            int bi = order[0];
            for (int kk = 0; kk < TOPK; ++kk) {
                const int id = order[kk];
                const double p = logits[(size_t)t * V + id] * invtemp
                               + (double)rng_gumbel(seed, unsigned(t), unsigned(id));
                if (p > best || (p == best && id < bi)) { best = p; bi = id; }
            }
            if (got[t] != bi) ++mism;
        }
        char extra[64];
        snprintf(extra, sizeof extra, "  (%d/%d mismatched)", mism, T);
        report("lm_head top-k (fp16) vs fp64 oracle", mism == 0, extra);
    }

    // ---- top-p ----
    lm_head_topk_partials<1><<<grid, 32>>>(dh, dW, dpv, dpi, nullptr, V, K, TILE_V,
                                           num_vtiles, TOPK, 0, invtemp, dlse);
    lm_head_topp_reduce<<<T, 32>>>(dpv, dpi, dout, num_vtiles, TOPK, topp, seed, invtemp, dlse);
    CUCHECK(cudaDeviceSynchronize());
    {
        auto got = d2h(dout, T);
        int bad = 0;
        for (int t = 0; t < T; ++t) {
            // replay the pool + bisection in fp64 over the full-vocab data
            std::vector<std::pair<double, int>> pool;   // per-tile top-K
            std::vector<double> tile_lse(num_vtiles);
            for (int vt = 0; vt < num_vtiles; ++vt) {
                const int v0 = vt * TILE_V, v1 = std::min(v0 + TILE_V, V);
                std::vector<int> order(v1 - v0);
                std::iota(order.begin(), order.end(), v0);
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    const double la = logits[(size_t)t * V + a], lb = logits[(size_t)t * V + b];
                    return la > lb || (la == lb && a < b);
                });
                for (int kk = 0; kk < TOPK; ++kk)
                    pool.push_back({logits[(size_t)t * V + order[kk]], order[kk]});
                double mx = -1e300, s = 0;
                for (int v = v0; v < v1; ++v) mx = std::max(mx, logits[(size_t)t * V + v] * invtemp);
                for (int v = v0; v < v1; ++v) s += std::exp(logits[(size_t)t * V + v] * invtemp - mx);
                tile_lse[vt] = mx + std::log(s);
            }
            double mx = -1e300;
            for (auto& c : pool) mx = std::max(mx, c.first * invtemp);
            double Z = 0;
            for (int vt = 0; vt < num_vtiles; ++vt) Z += std::exp(tile_lse[vt] - mx);
            double lo = mx - 40.0, hi = mx;
            for (int it = 0; it < 32; ++it) {
                const double mid = 0.5 * (lo + hi);
                double sm = 0;
                for (auto& c : pool) if (c.first * invtemp >= mid) sm += std::exp(c.first * invtemp - mx);
                if (sm / Z >= topp) lo = mid; else hi = mid;
            }
            const double L = lo;
            double best = -1e300;
            int bi = -1;
            for (auto& c : pool) {
                const double ls = c.first * invtemp;
                if (ls < L) continue;
                const double p = ls + (double)rng_gumbel(seed, unsigned(t), unsigned(c.second));
                if (p > best || (p == best && c.second < bi)) { best = p; bi = c.second; }
            }
            if (got[t] == bi) continue;
            // boundary guard: accept a nucleus member whose perturbed logit ties within eps
            bool okk = false;
            for (auto& c : pool) {
                if (c.second != got[t]) continue;
                const double ls = c.first * invtemp;
                const double p = ls + (double)rng_gumbel(seed, unsigned(t), unsigned(c.second));
                okk = (ls >= L - 1e-3) && (std::abs(p - best) < 1e-3);
            }
            if (!okk) ++bad;
        }
        char extra[64];
        snprintf(extra, sizeof extra, "  (%d/%d outside guard)", bad, T);
        report("lm_head top-p (fp16) vs fp64 replay", bad == 0, extra);
    }

    // ---- quantized (q8_0) top-k: pack W, oracle over exactly-dequantized values ----
    {
        const int BK = 32, BB = 34;                       // q8_0: half scale + 32 int8
        const int bpr = K / BK;
        std::vector<uint8_t> Wq((size_t)V * bpr * BB);
        std::vector<double> wd((size_t)V * K);
        for (int v = 0; v < V; ++v)
            for (int b = 0; b < bpr; ++b) {
                float amax = 0;
                for (int c = 0; c < BK; ++c)
                    amax = std::max(amax, std::abs(__half2float(W[(size_t)v * K + b * BK + c])));
                const float sc = amax / 127.0f;
                uint8_t* base = Wq.data() + ((size_t)v * bpr + b) * BB;
                const half hs = __float2half(sc);
                memcpy(base, &hs, 2);
                const float scf = __half2float(hs);
                for (int c = 0; c < BK; ++c) {
                    const float x = __half2float(W[(size_t)v * K + b * BK + c]);
                    int qq = (int)rintf(scf > 0 ? x / scf : 0.0f);
                    qq = std::max(-127, std::min(127, qq));
                    base[2 + c] = (uint8_t)(int8_t)qq;
                    wd[(size_t)v * K + b * BK + c] = (double)scf * qq;
                }
            }
        std::vector<double> qlogits((size_t)T * V);
        for (int t = 0; t < T; ++t)
            for (int v = 0; v < V; ++v) {
                double acc = 0;
                for (int k = 0; k < K; ++k)
                    acc += wd[(size_t)v * K + k] * (double)__half2float(h[(size_t)t * K + k]);
                qlogits[(size_t)t * V + v] = acc;
            }
        uint8_t* dWq = dnew(Wq);
        lm_head_topk_partials_q<q8_0, 0><<<grid, 32>>>(dh, dWq, dpv, dpi, nullptr, V, K,
                                                       TILE_V, num_vtiles, TOPK, 0, invtemp, nullptr);
        lm_head_topk_reduce<<<T, 32>>>(dpv, dpi, dout, num_vtiles, TOPK, seed, invtemp);
        CUCHECK(cudaDeviceSynchronize());
        auto got = d2h(dout, T);
        int mism = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<int> order(V);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + TOPK, order.end(), [&](int a, int b) {
                const double la = qlogits[(size_t)t * V + a], lb = qlogits[(size_t)t * V + b];
                return la > lb || (la == lb && a < b);
            });
            double best = -1e300;
            int bi = order[0];
            for (int kk = 0; kk < TOPK; ++kk) {
                const int id = order[kk];
                const double p = qlogits[(size_t)t * V + id] * invtemp
                               + (double)rng_gumbel(seed, unsigned(t), unsigned(id));
                if (p > best || (p == best && id < bi)) { best = p; bi = id; }
            }
            if (got[t] != bi) ++mism;
        }
        char extra[64];
        snprintf(extra, sizeof extra, "  (%d/%d mismatched)", mism, T);
        report("lm_head top-k (q8_0) vs fp64 oracle", mism == 0, extra);
    }

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
