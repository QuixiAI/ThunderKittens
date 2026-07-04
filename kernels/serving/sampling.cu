// Harness for sampling kernels (kernels live in sampling_kernels.cuh).
#include "sampling_kernels.cuh"
// ============================== test harness ==============================
using namespace tms;
static float frand() { return (rand() % 20000 - 10000) / 3000.0f; }

int main() {
    srand(81);
    const int T_ = 24, V = 1000;
    const unsigned seed = 555u;
    const float invtemp = 1.0f / 0.8f;
    std::vector<float> logits(size_t(T_) * V);
    for (auto& x : logits) x = frand();
    float* dl; int* dout;
    cudaMalloc(&dl, logits.size() * 4); cudaMalloc(&dout, T_ * 4);
    cudaMemcpy(dl, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice);
    int rc = 0;
    std::vector<int> got(T_);

    auto fetch = [&] { cudaMemcpy(got.data(), dout, T_ * 4, cudaMemcpyDeviceToHost); };
    auto ls = [&](int t, int i) { return logits[size_t(t) * V + i] * invtemp; };

    { // argmax (exact)
        argmax_k<float><<<T_, 32>>>(dl, dout, V);
        cudaDeviceSynchronize(); fetch();
        int bad = 0;
        for (int t = 0; t < T_; t++) {
            int bi = 0;
            for (int i = 1; i < V; i++) if (logits[size_t(t) * V + i] > logits[size_t(t) * V + bi]) bi = i;
            if (got[t] != bi) bad++;
        }
        printf("argmax:        %d/%d exact (%s)\n", T_ - bad, T_, bad ? "FAIL" : "PASS");
        rc |= (bad != 0);
    }
    { // categorical (exact: identical RNG floats + float math)
        sample_categorical<float><<<T_, 32>>>(dl, dout, V, seed, invtemp);
        cudaDeviceSynchronize(); fetch();
        int bad = 0;
        for (int t = 0; t < T_; t++) {
            float best = -3.4e38f; int bi = 0;
            for (int i = 0; i < V; i++) {
                const float p = ls(t, i) + rng_gumbel(seed, t, i);
                if (p > best || (p == best && i < bi)) { best = p; bi = i; }
            }
            if (got[t] != bi) bad++;
        }
        printf("categorical:   %d/%d exact (%s)\n", T_ - bad, T_, bad ? "FAIL" : "PASS");
        rc |= (bad != 0);
    }
    { // top-k (exact: replicate rounds in float)
        const int K = 40;
        top_k_sample<float><<<T_, 32>>>(dl, dout, V, K, seed, invtemp);
        cudaDeviceSynchronize(); fetch();
        int bad = 0;
        for (int t = 0; t < T_; t++) {
            std::vector<int> chosen;
            for (int kk = 0; kk < K; kk++) {
                float best = -3.4e38f; int bi = -1;
                for (int i = 0; i < V; i++) {
                    bool taken = false;
                    for (int c : chosen) if (c == i) taken = true;
                    if (taken) continue;
                    const float v = logits[size_t(t) * V + i];
                    if (v > best || (v == best && i < bi)) { best = v; bi = i; }
                }
                chosen.push_back(bi);
            }
            float best = -3.4e38f; int bi = chosen[0];
            for (int c : chosen) {
                const float p = logits[size_t(t) * V + c] * invtemp + rng_gumbel(seed, t, c);
                if (p > best || (p == best && c < bi)) { best = p; bi = c; }
            }
            if (got[t] != bi) bad++;
        }
        printf("top_k (K=40):  %d/%d exact (%s)\n", T_ - bad, T_, bad ? "FAIL" : "PASS");
        rc |= (bad != 0);
    }
    { // top-p / min-p / typical-p: double oracle of the same algorithm + membership check
        auto check_threshold_sampler = [&](const char* name, int mode, float param) {
            if (mode == 0) top_p_sample<float><<<T_, 32>>>(dl, dout, V, param, seed, invtemp);
            if (mode == 1) min_p_sample<float><<<T_, 32>>>(dl, dout, V, param, seed, invtemp);
            if (mode == 2) typical_p_sample<float><<<T_, 32>>>(dl, dout, V, param, seed, invtemp);
            cudaDeviceSynchronize(); fetch();
            int bad = 0;
            for (int t = 0; t < T_; t++) {
                // double-precision kept-set + gumbel argmax over it
                double mx = -1e300;
                for (int i = 0; i < V; i++) mx = std::max(mx, double(ls(t, i)));
                double Z = 0, S1 = 0;
                for (int i = 0; i < V; i++) { const double e = exp(double(ls(t, i)) - mx); Z += e; S1 += e * ls(t, i); }
                const double H = mx + log(Z) - S1 / Z;
                std::vector<char> kept(V, 0);
                if (mode == 0) {   // smallest nucleus with mass >= p, via double bisection
                    double lo = mx - 40, hi = mx;
                    for (int it = 0; it < 50; it++) {
                        const double mid = (lo + hi) / 2;
                        double sm = 0;
                        for (int i = 0; i < V; i++) if (double(ls(t, i)) >= mid) sm += exp(double(ls(t, i)) - mx);
                        if (sm / Z >= param) lo = mid; else hi = mid;
                    }
                    for (int i = 0; i < V; i++) kept[i] = double(ls(t, i)) >= lo;
                } else if (mode == 1) {
                    const double thr = mx + log(double(param));
                    for (int i = 0; i < V; i++) kept[i] = double(ls(t, i)) >= thr;
                } else {
                    double smax = 0;
                    for (int i = 0; i < V; i++) smax = std::max(smax, std::abs((mx + log(Z) - ls(t, i)) - H));
                    double lo = 0, hi = smax;
                    for (int it = 0; it < 50; it++) {
                        const double mid = (lo + hi) / 2;
                        double mass = 0;
                        for (int i = 0; i < V; i++)
                            if (std::abs((mx + log(Z) - ls(t, i)) - H) <= mid) mass += exp(double(ls(t, i)) - mx);
                        if (mass / Z >= param) hi = mid; else lo = mid;
                    }
                    for (int i = 0; i < V; i++) kept[i] = std::abs((mx + log(Z) - ls(t, i)) - H) <= hi;
                }
                double best = -1e300; int bi = -1;
                for (int i = 0; i < V; i++) {
                    if (!kept[i]) continue;
                    const double p = double(ls(t, i)) + double(rng_gumbel(seed, t, i));
                    if (p > best) { best = p; bi = i; }
                }
                // accept exact match, or near-boundary agreement (float vs double threshold)
                const double gp = double(ls(t, got[t])) + double(rng_gumbel(seed, t, got[t]));
                if (got[t] != bi && !(gp >= best - 1e-3)) bad++;
            }
            printf("%-14s %d/%d agree (%s)\n", name, T_ - bad, T_, bad ? "FAIL" : "PASS");
            rc |= (bad != 0);
        };
        check_threshold_sampler("top_p (0.9):", 0, 0.9f);
        check_threshold_sampler("min_p (0.05):", 1, 0.05f);
        check_threshold_sampler("typical (0.9):", 2, 0.9f);
    }
    { // penalties: histogram + transform, exact elementwise
        const int L = 12;
        std::vector<int> prev(size_t(T_) * L), parent(T_);
        for (int t = 0; t < T_; t++) {
            parent[t] = t;
            for (int j = 0; j < L; j++) prev[size_t(t) * L + j] = (j % 3 == 2) ? -1 : (t * 7 + j * 13) % V;
        }
        std::vector<float> bias(V);
        for (auto& b : bias) b = frand() * 0.1f;
        int *dprev, *dcnt, *dpar; float *dbias, *dout2;
        cudaMalloc(&dprev, prev.size() * 4); cudaMalloc(&dcnt, size_t(T_) * V * 4);
        cudaMalloc(&dpar, T_ * 4); cudaMalloc(&dbias, V * 4);
        cudaMalloc(&dout2, logits.size() * 4);
        cudaMemcpy(dprev, prev.data(), prev.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dpar, parent.data(), T_ * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dbias, bias.data(), V * 4, cudaMemcpyHostToDevice);
        cudaMemset(dcnt, 0, size_t(T_) * V * 4);
        penalty_histogram<<<(T_ * L + 255) / 256, 256>>>(dprev, dcnt, V, L, T_ * L, dpar);
        const float rep = 1.2f, pres = 0.4f, freq = 0.15f;
        const int eos = 7, minlen = 5, genlen = 2;
        apply_penalty<float><<<T_, 32>>>(dl, dcnt, dout2, V, invtemp, rep, pres, freq, dbias, eos, minlen, genlen);
        cudaDeviceSynchronize();
        std::vector<float> outp(logits.size());
        cudaMemcpy(outp.data(), dout2, outp.size() * 4, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int t = 0; t < T_; t++) {
            std::vector<int> cnt(V, 0);
            for (int j = 0; j < L; j++) {
                const int tok = prev[size_t(t) * L + j];
                if (tok >= 0 && tok < V) cnt[tok]++;
            }
            for (int v = 0; v < V; v++) {
                float e = ls(t, v);
                if (cnt[v] > 0) { e = (e < 0) ? e * rep : e / rep; e -= pres; e -= freq * cnt[v]; }
                e += bias[v];
                if (v == eos && genlen < minlen) e = SMP_NEG_INF;
                if (outp[size_t(t) * V + v] != e) bad++;
            }
        }
        printf("penalties:     %d mismatches (%s)\n", bad, bad ? "FAIL" : "PASS");
        rc |= (bad != 0);
        cudaFree(dprev); cudaFree(dcnt); cudaFree(dpar); cudaFree(dbias); cudaFree(dout2);
    }
    { // bitmask + bad words, exact
        const int words = (V + 31) / 32, max_bad = 8;
        std::vector<uint32_t> bm(size_t(T_) * words);
        for (auto& w : bm) w = (uint32_t(rand()) << 16) ^ uint32_t(rand());
        std::vector<int> bad_ids(size_t(T_) * max_bad), bad_lens(T_);
        for (int t = 0; t < T_; t++) {
            bad_lens[t] = t % (max_bad + 1);
            for (int j = 0; j < max_bad; j++) bad_ids[size_t(t) * max_bad + j] = (t * 31 + j * 97) % V;
        }
        uint32_t* dbm; int *dbi, *dblen; float *do3, *do4;
        cudaMalloc(&dbm, bm.size() * 4); cudaMalloc(&dbi, bad_ids.size() * 4);
        cudaMalloc(&dblen, T_ * 4); cudaMalloc(&do3, logits.size() * 4); cudaMalloc(&do4, logits.size() * 4);
        cudaMemcpy(dbm, bm.data(), bm.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dbi, bad_ids.data(), bad_ids.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dblen, bad_lens.data(), T_ * 4, cudaMemcpyHostToDevice);
        apply_token_bitmask<float><<<T_, 32>>>(dl, dbm, do3, V, words);
        apply_bad_words<float><<<T_, 32>>>(dl, do4, dbi, dblen, V, max_bad);
        cudaDeviceSynchronize();
        std::vector<float> o3(logits.size()), o4(logits.size());
        cudaMemcpy(o3.data(), do3, o3.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(o4.data(), do4, o4.size() * 4, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int t = 0; t < T_; t++) for (int v = 0; v < V; v++) {
            const bool allow = (bm[size_t(t) * words + (v >> 5)] >> (v & 31)) & 1;
            const float e3 = allow ? logits[size_t(t) * V + v] : SMP_NEG_INF;
            if (o3[size_t(t) * V + v] != e3) bad++;
            float e4 = logits[size_t(t) * V + v];
            for (int j = 0; j < bad_lens[t]; j++)
                if (bad_ids[size_t(t) * max_bad + j] == v) e4 = SMP_NEG_INF;
            if (o4[size_t(t) * V + v] != e4) bad++;
        }
        printf("bitmask+bad:   %d mismatches (%s)\n", bad, bad ? "FAIL" : "PASS");
        rc |= (bad != 0);
        cudaFree(dbm); cudaFree(dbi); cudaFree(dblen); cudaFree(do3); cudaFree(do4);
    }
    return rc;
}
