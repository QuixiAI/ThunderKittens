// Harness for spec_beam kernels (kernels live in spec_beam_kernels.cuh).
#include "spec_beam_kernels.cuh"
// ============================== test harness ==============================
using namespace tms;
static float frand() { return (rand() % 20000 - 10000) / 3000.0f; }

int main() {
    srand(91);
    int rc = 0;
    const unsigned seed = 777u;

    { // beam advance: partials + select vs double oracle
        const int B = 2, BM = 4, two_bm = 2 * BM, V = 300;
        std::vector<float> logits(size_t(B) * BM * V), cum(B * BM);
        for (auto& x : logits) x = frand();
        for (int i = 0; i < B * BM; i++) cum[i] = frand();
        float *dl, *dcum, *dsc, *dnc; int *dtok, *dnt, *dpar;
        cudaMalloc(&dl, logits.size() * 4); cudaMalloc(&dcum, B * BM * 4);
        cudaMalloc(&dsc, size_t(B) * BM * two_bm * 4); cudaMalloc(&dtok, size_t(B) * BM * two_bm * 4);
        cudaMalloc(&dnt, B * BM * 4); cudaMalloc(&dpar, B * BM * 4); cudaMalloc(&dnc, B * BM * 4);
        cudaMemcpy(dl, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dcum, cum.data(), B * BM * 4, cudaMemcpyHostToDevice);
        beam_topk_partials<float><<<B * BM, 32>>>(dl, dcum, dsc, dtok, V, two_bm);
        beam_select<<<B, 32>>>(dsc, dtok, dnt, dpar, dnc, BM, two_bm);
        cudaDeviceSynchronize();
        std::vector<int> nt(B * BM), par(B * BM);
        std::vector<float> nc(B * BM);
        cudaMemcpy(nt.data(), dnt, B * BM * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(par.data(), dpar, B * BM * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(nc.data(), dnc, B * BM * 4, cudaMemcpyDeviceToHost);
        // oracle
        int bad = 0;
        for (int b = 0; b < B; b++) {
            std::vector<std::pair<double, std::pair<int, int>>> cands;   // (score, (beam, token))
            for (int i = 0; i < BM; i++) {
                const int row = b * BM + i;
                double mx = -1e300;
                for (int v = 0; v < V; v++) mx = std::max(mx, double(logits[size_t(row) * V + v]));
                double Z = 0;
                for (int v = 0; v < V; v++) Z += exp(double(logits[size_t(row) * V + v]) - mx);
                const double lse = mx + log(Z);
                std::vector<int> idx(V);
                for (int v = 0; v < V; v++) idx[v] = v;
                std::partial_sort(idx.begin(), idx.begin() + two_bm, idx.end(), [&](int a, int c) {
                    const float la = logits[size_t(row) * V + a], lc = logits[size_t(row) * V + c];
                    return la > lc || (la == lc && a < c);
                });
                for (int j = 0; j < two_bm; j++)
                    cands.push_back({double(cum[row]) + (double(logits[size_t(row) * V + idx[j]]) - lse),
                                     {i, idx[j]}});
            }
            std::stable_sort(cands.begin(), cands.end(), [](auto& a, auto& c) { return a.first > c.first; });
            for (int k = 0; k < BM; k++) {
                if (nt[b * BM + k] != cands[k].second.second) bad++;
                if (par[b * BM + k] != cands[k].second.first) bad++;
                if (std::abs(double(nc[b * BM + k]) - cands[k].first) > 1e-3) bad++;
            }
        }
        printf("beam advance:      %d mismatches (%s)\n", bad, bad ? "FAIL" : "PASS");
        rc |= (bad != 0);
        cudaFree(dl); cudaFree(dcum); cudaFree(dsc); cudaFree(dtok);
        cudaFree(dnt); cudaFree(dpar); cudaFree(dnc);
    }

    { // spec_verify_linear + compact + kv_meta: exact host replication
        const int B = 4, S = 5, V = 120;
        std::vector<int> draft(B * S), bonus(B), seq(B);
        std::vector<float> dp(size_t(B) * S * V), tp(size_t(B) * (S + 1) * V), u(B * S);
        auto norm = [&](float* p, int n) {
            float s = 0;
            for (int i = 0; i < n; i++) { p[i] = std::abs(p[i]) + 0.01f; s += p[i]; }
            for (int i = 0; i < n; i++) p[i] /= s;
        };
        for (auto& x : dp) x = frand();
        for (auto& x : tp) x = frand();
        for (size_t r = 0; r < size_t(B) * S; r++) norm(&dp[r * V], V);
        for (size_t r = 0; r < size_t(B) * (S + 1); r++) norm(&tp[r * V], V);
        for (int i = 0; i < B * S; i++) { draft[i] = (i * 37) % V; u[i] = (rand() % 1000) / 1000.0f; }
        for (int b = 0; b < B; b++) { bonus[b] = (b * 91) % V; seq[b] = 10 + 7 * b; }
        u[0 * S + 0] = 0.0f;    // force accepts early in request 0
        int *ddr, *dbon, *dout, *dcnt, *dseq, *dpt, *dpp, *dcu, *dnsl;
        float *ddp, *dtp, *du;
        cudaMalloc(&ddr, B * S * 4); cudaMalloc(&dbon, B * 4);
        cudaMalloc(&ddp, dp.size() * 4); cudaMalloc(&dtp, tp.size() * 4); cudaMalloc(&du, B * S * 4);
        cudaMalloc(&dout, B * (S + 1) * 4); cudaMalloc(&dcnt, B * 4); cudaMalloc(&dseq, B * 4);
        cudaMalloc(&dpt, B * (S + 1) * 4); cudaMalloc(&dpp, B * (S + 1) * 4);
        cudaMalloc(&dcu, (B + 1) * 4); cudaMalloc(&dnsl, B * 4);
        cudaMemcpy(ddr, draft.data(), B * S * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dbon, bonus.data(), B * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddp, dp.data(), dp.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtp, tp.data(), tp.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(du, u.data(), B * S * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dseq, seq.data(), B * 4, cudaMemcpyHostToDevice);
        spec_verify_linear<<<B, 32>>>(ddr, ddp, dtp, dbon, du, dout, dcnt, S, V, seed);
        spec_compact<<<1, 256>>>(dout, dcnt, dseq, dpt, dpp, dcu, B, S + 1);
        spec_update_kv_meta<<<1, 256>>>(dseq, dcnt, dnsl, B);
        cudaDeviceSynchronize();
        std::vector<int> out(B * (S + 1)), cnt(B), pt(B * (S + 1)), pp(B * (S + 1)), cu2(B + 1), nsl(B);
        cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(cnt.data(), dcnt, B * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(pt.data(), dpt, pt.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(pp.data(), dpp, pp.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(cu2.data(), dcu, (B + 1) * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(nsl.data(), dnsl, B * 4, cudaMemcpyDeviceToHost);
        int bad = 0;
        std::vector<int> eout(B * (S + 1), SPEC_PLACEHOLDER), ecnt(B);
        for (int b = 0; b < B; b++) {
            int rej = S;
            for (int i = 0; i < S; i++) {
                const int dt = draft[b * S + i];
                const float p_t = tp[(size_t(b) * (S + 1) + i) * V + dt];
                const float p_d = dp[(size_t(b) * S + i) * V + dt];
                if (p_d <= 0.0f || u[b * S + i] * p_d <= p_t) { eout[b * (S + 1) + i] = dt; continue; }
                float best = -3.4e38f; int bi = 0;
                for (int v = 0; v < V; v++) {
                    const float r = std::max(0.0f, tp[(size_t(b) * (S + 1) + i) * V + v] - dp[(size_t(b) * S + i) * V + v]);
                    const float g = ((r > 0) ? logf(r) : -3.4028234663852886e38f) + rng_gumbel(seed, b * S + i, v);
                    if (g > best || (g == best && v < bi)) { best = g; bi = v; }
                }
                eout[b * (S + 1) + i] = bi;
                rej = i;
                break;
            }
            if (rej == S) eout[b * (S + 1) + S] = bonus[b];
            ecnt[b] = rej;
        }
        for (int b = 0; b < B; b++) {
            if (cnt[b] != ecnt[b]) bad++;
            for (int j = 0; j <= ecnt[b] && j <= S; j++)
                if (j < ecnt[b] + 1 && out[b * (S + 1) + j] != eout[b * (S + 1) + j]) bad++;
            if (nsl[b] != seq[b] + ecnt[b] + 1) bad++;
        }
        int run = 0;
        for (int b = 0; b < B; b++) {
            if (cu2[b] != run) bad++;
            for (int j = 0; j < ecnt[b] + 1; j++) {
                if (pt[run + j] != eout[b * (S + 1) + j]) bad++;
                if (pp[run + j] != seq[b] + j) bad++;
            }
            run += ecnt[b] + 1;
        }
        if (cu2[B] != run) bad++;
        printf("spec linear+compact: %d mismatches (%s)\n", bad, bad ? "FAIL" : "PASS");
        rc |= (bad != 0);
        cudaFree(ddr); cudaFree(dbon); cudaFree(ddp); cudaFree(dtp); cudaFree(du);
        cudaFree(dout); cudaFree(dcnt); cudaFree(dseq); cudaFree(dpt); cudaFree(dpp);
        cudaFree(dcu); cudaFree(dnsl);
    }

    { // build_dynamic_tree + spec_verify_tree: exact host replication
        const int B = 3, N = 7, V = 90;
        // tree: 0 -> {1,2}; 1 -> {3,4}; 2 -> {5}; 3 -> {6}
        std::vector<int> parents = {-1, 0, 0, 1, 1, 2, 3};
        std::vector<int> par_all(B * N);
        for (int b = 0; b < B; b++) for (int i = 0; i < N; i++) par_all[b * N + i] = parents[i];
        std::vector<int> draft(size_t(B) * (N - 1)), valid(B, 1);
        valid[2] = 0;   // request 2: no tree
        for (size_t i = 0; i < draft.size(); i++) draft[i] = int(i * 13 + 5) % V;
        std::vector<float> tp(size_t(B) * N * V);
        for (auto& x : tp) x = frand();
        for (size_t r = 0; r < size_t(B) * N; r++) {
            float s = 0;
            for (int v = 0; v < V; v++) { tp[r * V + v] = std::abs(tp[r * V + v]) + 0.01f; s += tp[r * V + v]; }
            for (int v = 0; v < V; v++) tp[r * V + v] /= s;
        }
        int *dpar, *dnt2, *dns, *dpos, *ddr, *dai, *dat, *dan, *dval;
        float* dtp2;
        cudaMalloc(&dpar, B * N * 4); cudaMalloc(&dnt2, B * N * 4); cudaMalloc(&dns, B * N * 4);
        cudaMalloc(&dpos, B * N * 4); cudaMalloc(&ddr, draft.size() * 4);
        cudaMalloc(&dai, B * N * 4); cudaMalloc(&dat, B * N * 4); cudaMalloc(&dan, B * 4);
        cudaMalloc(&dval, B * 4); cudaMalloc(&dtp2, tp.size() * 4);
        cudaMemcpy(dpar, par_all.data(), B * N * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddr, draft.data(), draft.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dval, valid.data(), B * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtp2, tp.data(), tp.size() * 4, cudaMemcpyHostToDevice);
        build_dynamic_tree<<<B, 32>>>(dpar, dnt2, dns, dpos, N);
        spec_verify_tree<<<B, 32>>>(ddr, dtp2, dnt2, dns, dai, dat, dan, N, V, seed, dval);
        cudaDeviceSynchronize();
        std::vector<int> nt2(B * N), ns(B * N), pos(B * N), ai(B * N), at(B * N), an(B);
        cudaMemcpy(nt2.data(), dnt2, B * N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ns.data(), dns, B * N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(pos.data(), dpos, B * N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ai.data(), dai, B * N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(at.data(), dat, B * N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(an.data(), dan, B * 4, cudaMemcpyDeviceToHost);
        // host: tree pointers
        int bad = 0;
        std::vector<int> ent(N, -1), ens(N, -1), epos(N);
        for (int c = 0; c < N; c++) {
            int d = 0, x = c;
            while (parents[x] >= 0) { x = parents[x]; d++; }
            epos[c] = d;
            const int p = parents[c];
            if (c == 0 || p < 0) continue;
            for (int c2 = c + 1; c2 < N; c2++) if (parents[c2] == p) { ens[c] = c2; break; }
            bool first = true;
            for (int c2 = 1; c2 < c; c2++) if (parents[c2] == p) first = false;
            if (first) ent[p] = c;
        }
        for (int b = 0; b < B; b++) for (int i = 0; i < N; i++) {
            if (nt2[b * N + i] != ent[i] || ns[b * N + i] != ens[i] || pos[b * N + i] != epos[i]) bad++;
        }
        // host: verify walk
        for (int b = 0; b < B; b++) {
            std::vector<int> eai(N, -1), eat(N, -1);
            int last = 0, num = 0, term = 0;
            if (valid[b] == 0) { eai[0] = 0; term = 1; }
            else {
                eai[0] = 0;
                for (int j = 1; j < N; j++) {
                    const int fc = ent[last];
                    if (fc == -1) { term = 1; break; }
                    const float coin = rng_uniform(seed, b, j);
                    float acc = 0;
                    bool okd = false;
                    for (int c = fc; c != -1; c = ens[c]) {
                        const int tok = draft[size_t(b) * (N - 1) + (c - 1)];
                        acc += tp[(size_t(b) * N + last) * V + tok];
                        if (coin <= acc) { eat[num] = tok; num++; eai[num] = c; last = c; okd = true; break; }
                    }
                    if (!okd) { term = 2; break; }
                }
            }
            if (term != 0) {
                float best = -3.4e38f; int bi = -1;
                const int fc = (term == 2) ? ent[last] : -1;
                for (int v = 0; v < V; v++) {
                    const float pv = tp[(size_t(b) * N + last) * V + v];
                    if (pv <= 0) continue;
                    if (term == 2) {
                        bool tried = false;
                        for (int c = fc; c != -1; c = ens[c])
                            if (draft[size_t(b) * (N - 1) + (c - 1)] == v) { tried = true; break; }
                        if (tried) continue;
                    }
                    const float g = logf(pv) + rng_gumbel(seed + 0x2545F491u, b, v);
                    if (g > best || (g == best && v < bi)) { best = g; bi = v; }
                }
                eat[num] = bi;
            }
            if (an[b] != num) bad++;
            for (int i = 0; i < N; i++)
                if (ai[b * N + i] != eai[i] || at[b * N + i] != eat[i]) bad++;
        }
        printf("tree build+verify:  %d mismatches (%s)\n", bad, bad ? "FAIL" : "PASS");
        rc |= (bad != 0);
        cudaFree(dpar); cudaFree(dnt2); cudaFree(dns); cudaFree(dpos); cudaFree(ddr);
        cudaFree(dai); cudaFree(dat); cudaFree(dan); cudaFree(dval); cudaFree(dtp2);
    }
    return rc;
}
