/**
 * @file
 * @brief Harness for the MoE family (tm_moe_kernels.cuh): runs the FULL
 * pipeline route -> histogram -> scan -> scatter -> pad -> gather ->
 * swiglu grouped GEMM -> rect grouped GEMM (down-proj) -> finalize on random
 * data and compares the end-to-end MoE MLP output against a dense fp64 CPU
 * reference. Also checks every intermediate invariant (counts, offsets,
 * segment membership, inverse maps, pad sentinels, gathered rows).
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        moe_test.cu -o moe_test.out
 * Run: CUDA_VISIBLE_DEVICES=6 ./moe_test.out
 */
#include "tm_moe_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace tmoe;

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
static void report(const char* name, bool ok, double err = -1.0) {
    if (err >= 0) printf("%-40s %s  (err %.3e)\n", name, ok ? "PASS" : "FAIL", err);
    else printf("%-40s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

static std::mt19937 g_rng(2025);
static std::vector<float> randv(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g_rng);
    return v;
}

int main() {
    const int T = 200, E = 11, K = 4, H = 64, INTER = 96;
    const int TK = T * K;
    const int total_pad_max = ((TK + 31 * E) + 31) / 32 * 32;
    const int max_tiles = total_pad_max / 32;

    auto logits = randv((size_t)T * E, -3, 3);
    auto x = randv((size_t)T * H);
    auto w1 = randv((size_t)E * H * 2 * INTER, -0.3f, 0.3f);   // (E, H, 2*inter) [gate|up]
    auto w2 = randv((size_t)E * INTER * H, -0.3f, 0.3f);       // (E, inter, H)

    // ---------------- CPU fp64 reference ----------------
    std::vector<int> ref_ids(TK);
    std::vector<double> ref_w(TK);
    for (int t = 0; t < T; ++t) {
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const float la = logits[(size_t)t * E + a], lb = logits[(size_t)t * E + b];
            return la > lb || (la == lb && a < b);       // smaller-id ties
        });
        double m = -1e300;
        for (int k = 0; k < K; ++k) m = std::max(m, (double)logits[(size_t)t * E + order[k]]);
        double s = 0;
        for (int k = 0; k < K; ++k) s += std::exp((double)logits[(size_t)t * E + order[k]] - m);
        for (int k = 0; k < K; ++k) {
            ref_ids[t * K + k] = order[k];
            ref_w[t * K + k] = std::exp((double)logits[(size_t)t * E + order[k]] - m) / s;
        }
    }
    // dense MoE MLP: out[t] = sum_k w * ((silu(x@W1g)*(x@W1u)) @ W2)[e]
    std::vector<double> ref_out((size_t)T * H, 0.0);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            const int e = ref_ids[t * K + k];
            const double wgt = ref_w[t * K + k];
            std::vector<double> inter(INTER);
            for (int c = 0; c < INTER; ++c) {
                double g = 0, u = 0;
                for (int i = 0; i < H; ++i) {
                    const double a = x[(size_t)t * H + i];
                    g += a * w1[(size_t)e * H * 2 * INTER + (size_t)i * 2 * INTER + c];
                    u += a * w1[(size_t)e * H * 2 * INTER + (size_t)i * 2 * INTER + INTER + c];
                }
                inter[c] = (g / (1.0 + std::exp(-g))) * u;
            }
            for (int h = 0; h < H; ++h) {
                double acc = 0;
                for (int c = 0; c < INTER; ++c)
                    acc += inter[c] * w2[(size_t)e * INTER * H + (size_t)c * H + h];
                ref_out[(size_t)t * H + h] += wgt * acc;
            }
        }
    }

    // ---------------- GPU pipeline ----------------
    float *dlog = dnew(logits), *dx = dnew(x), *dw1 = dnew(w1), *dw2 = dnew(w2);
    int *dids = dzero<int>(TK);
    float *dwts = dzero<float>(TK);
    moe_route_topk<float><<<T, 32>>>(dlog, dids, dwts, E, K);
    CUCHECK(cudaDeviceSynchronize());
    auto hids = d2h(dids, TK);
    auto hwts = d2h(dwts, TK);
    {
        long mm = 0;
        double werr = 0;
        for (int i = 0; i < TK; ++i) {
            mm += hids[i] != ref_ids[i];
            werr = std::max(werr, std::abs((double)hwts[i] - ref_w[i]));
        }
        report("moe_route_topk ids (exact)", mm == 0);
        report("moe_route_topk weights", werr < 1e-6, werr);
    }

    int *dcnt = dzero<int>(E), *doff = dzero<int>(E + 1), *dcur = dzero<int>(E);
    moe_histogram<<<(TK + 255) / 256, 256>>>(dids, dcnt, TK);
    moe_scan_offsets<<<1, MOE_SCAN_NT>>>(dcnt, doff, dcur, E);
    CUCHECK(cudaDeviceSynchronize());
    auto hoff = d2h(doff, E + 1);
    {
        std::vector<int> cnt(E, 0);
        for (int i = 0; i < TK; ++i) cnt[ref_ids[i]]++;
        bool ok = true;
        int run = 0;
        for (int e = 0; e < E; ++e) { ok &= hoff[e] == run; run += cnt[e]; }
        ok &= hoff[E] == TK;
        report("moe_histogram + scan_offsets", ok);
    }

    int *dsri = dzero<int>(TK), *dinv = dzero<int>(TK);
    moe_scatter<<<(TK + 255) / 256, 256>>>(dids, dcur, dsri, dinv, TK);
    CUCHECK(cudaDeviceSynchronize());
    auto hsri = d2h(dsri, TK);
    auto hinv = d2h(dinv, TK);
    {
        bool ok = true;
        for (int p = 0; p < TK; ++p) {                    // segment membership
            int lo = 0;
            while (lo + 1 < E + 1 && hoff[lo + 1] <= p) ++lo;
            ok &= ref_ids[hsri[p]] == lo;
        }
        for (int r = 0; r < TK; ++r) ok &= hsri[hinv[r]] == r;   // inverse map
        report("moe_scatter (segments + inverse)", ok);
    }

    int *dofp = dzero<int>(E + 1), *deot = dzero<int>(max_tiles),
        *dgix = dzero<int>(total_pad_max), *dinvp = dzero<int>(TK);
    moe_pad_offsets<<<1, MOE_SCAN_NT>>>(doff, dofp, deot, dgix, E, max_tiles, total_pad_max);
    moe_pad_scatter<<<(TK + 255) / 256, 256>>>(dsri, doff, dofp, dgix, dinvp, TK, E, K);
    CUCHECK(cudaDeviceSynchronize());
    auto hofp = d2h(dofp, E + 1);
    auto heot = d2h(deot, max_tiles);
    auto hgix = d2h(dgix, total_pad_max);
    auto hinvp = d2h(dinvp, TK);
    const int total_pad = hofp[E];
    {
        bool ok = true;
        int run = 0;
        for (int e = 0; e < E; ++e) {                     // ceil32 exclusive scan
            ok &= hofp[e] == run;
            run += (hoff[e + 1] - hoff[e] + 31) / 32 * 32;
        }
        ok &= total_pad == run && total_pad % 32 == 0;
        for (int t = 0; t < max_tiles; ++t) {             // expert_of_tile
            const int pos = t * 32;
            if (pos >= total_pad) { ok &= heot[t] == -1; continue; }
            int lo = 0;
            while (lo + 1 < E && hofp[lo + 1] <= pos) ++lo;
            ok &= heot[t] == lo;
        }
        for (int r = 0; r < TK; ++r) {                    // inv_pad consistency
            const int pp = hinvp[r];
            ok &= pp >= 0 && pp < total_pad && hgix[pp] == r / K;
        }
        for (int p = 0; p < total_pad_max; ++p)           // pad rows stay -1
            if (hgix[p] >= 0) ok &= hgix[p] < T;
        report("moe_pad_offsets + pad_scatter", ok);
    }

    float *dperm = dzero<float>((size_t)total_pad_max * H);
    moe_gather<float><<<total_pad_max, 128>>>(dx, dgix, dperm, H);
    CUCHECK(cudaDeviceSynchronize());
    {
        auto hperm = d2h(dperm, (size_t)total_pad_max * H);
        bool ok = true;
        for (int p = 0; p < total_pad_max; ++p) {
            for (int i = 0; i < H; i += 17) {
                const float want = hgix[p] >= 0 ? x[(size_t)hgix[p] * H + i] : 0.0f;
                ok &= hperm[(size_t)p * H + i] == want;
            }
        }
        report("moe_gather (rows + zero pads)", ok);
    }

    // grouped GEMMs over the padded schedule
    float *dinter = dzero<float>((size_t)total_pad_max * INTER);
    float *ddown = dzero<float>((size_t)total_pad_max * H);
    {
        dim3 g1{unsigned(INTER / 32), unsigned(total_pad / 32)};
        moe_grouped_gemm_swiglu<float><<<g1, 256>>>(dinter, dperm, dw1, deot,
                                                    total_pad, H, INTER);
        dim3 g2{unsigned(H / 32), unsigned(total_pad / 32)};
        moe_grouped_gemm_rect<float><<<g2, 256>>>(ddown, dinter, dw2, deot,
                                                  total_pad, INTER, H);
        CUCHECK(cudaDeviceSynchronize());
    }

    float *dout = dzero<float>((size_t)T * H);
    moe_finalize<float><<<T, 32>>>(ddown, dinvp, dwts, dout, K, H);
    CUCHECK(cudaDeviceSynchronize());
    {
        auto hout = d2h(dout, (size_t)T * H);
        double err = 0;
        for (size_t i = 0; i < hout.size(); ++i) {
            const double s = std::max(1.0, std::abs(ref_out[i]));
            err = std::max(err, std::abs((double)hout[i] - ref_out[i]) / s);
        }
        report("END-TO-END MoE MLP vs dense fp64", err < 5e-5, err);
    }

    // square grouped gemm parity (rect with K=N=H) on a small identity-ish case
    {
        std::vector<float> wsq((size_t)E * H * H);
        for (auto& v : wsq) v = std::uniform_real_distribution<float>(-0.3f, 0.3f)(g_rng);
        float* dwsq = dnew(wsq);
        float* dsq = dzero<float>((size_t)total_pad_max * H);
        dim3 g{unsigned(H / 32), unsigned(total_pad / 32)};
        moe_grouped_gemm_rect<float><<<g, 256>>>(dsq, dperm, dwsq, deot, total_pad, H, H);
        CUCHECK(cudaDeviceSynchronize());
        auto hsq = d2h(dsq, (size_t)total_pad_max * H);
        auto hperm = d2h(dperm, (size_t)total_pad_max * H);
        double err = 0;
        for (int p = 0; p < total_pad; p += 7) {
            const int e = heot[p / 32];
            for (int c = 0; c < H; c += 13) {
                double acc = 0;
                for (int k = 0; k < H; ++k)
                    acc += (double)hperm[(size_t)p * H + k] * wsq[(size_t)e * H * H + (size_t)k * H + c];
                const double s = std::max(1.0, std::abs(acc));
                err = std::max(err, std::abs((double)hsq[(size_t)p * H + c] - acc) / s);
            }
        }
        report("moe_grouped_gemm square (spot fp64)", err < 5e-6, err);
    }

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
