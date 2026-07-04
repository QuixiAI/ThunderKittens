// Harness for beam_xcache kernels (kernels live in beam_xcache_kernels.cuh).
#include "beam_xcache_kernels.cuh"
// ============================== test harness ==============================
using namespace tms;
static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }

int main() {
    srand(61);
    int rc = 0;

    // ---- beam builders vs CPU reference ----
    {
        const int B = 2, BM = 4, MAXB = 6, BS = 8;
        std::vector<int> parent = {0, 0, 2, 1,   3, 1, 1, 2};   // (B, BM)
        std::vector<int> bt(B * BM * MAXB), seq(B * BM);
        for (size_t i = 0; i < bt.size(); i++) bt[i] = (i % 5 == 4) ? -1 : int(i);
        for (int i = 0; i < B * BM; i++) seq[i] = 7 + 5 * i;
        const int n_slots = B * BM * MAXB;
        int *dp, *dbt, *dseq, *dnbt; int64_t* dpairs;
        cudaMalloc(&dp, parent.size() * 4); cudaMalloc(&dbt, bt.size() * 4);
        cudaMalloc(&dseq, seq.size() * 4); cudaMalloc(&dpairs, size_t(n_slots) * 16);
        cudaMalloc(&dnbt, bt.size() * 4);
        cudaMemcpy(dp, parent.data(), parent.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dseq, seq.data(), seq.size() * 4, cudaMemcpyHostToDevice);
        beam_build_copy_pairs<<<(n_slots + 255) / 256, 256>>>(dp, dbt, dseq, dpairs, BM, MAXB, BS, n_slots);
        beam_remap_block_table<<<B * BM, 32>>>(dbt, dp, dnbt, BM, MAXB);
        cudaDeviceSynchronize();
        std::vector<int64_t> pairs(size_t(n_slots) * 2);
        std::vector<int> nbt(bt.size());
        cudaMemcpy(pairs.data(), dpairs, pairs.size() * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(nbt.data(), dnbt, nbt.size() * 4, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int gid = 0; gid < n_slots; gid++) {
            const int gb = gid / MAXB, c = gid % MAXB, b = gb / BM, k = gb % BM;
            int64_t es = -1, ed = -1;
            const int p = parent[b * BM + k];
            if (p != k && c < (seq[gb] + BS - 1) / BS) {
                const int s = bt[(b * BM + p) * MAXB + c], d = bt[gb * MAXB + c];
                if (s >= 0 && d >= 0) { es = s; ed = d; }
            }
            if (pairs[2 * gid] != es || pairs[2 * gid + 1] != ed) bad++;
        }
        for (int r = 0; r < B * BM; r++) {
            const int b = r / BM, p = parent[r];
            for (int c = 0; c < MAXB; c++)
                if (nbt[r * MAXB + c] != bt[(b * BM + p) * MAXB + c]) bad++;
        }
        printf("beam_build_copy_pairs + remap: %d mismatches (%s)\n", bad, bad ? "FAIL" : "PASS");
        rc |= (bad != 0);
        cudaFree(dp); cudaFree(dbt); cudaFree(dseq); cudaFree(dpairs); cudaFree(dnbt);
    }

    // ---- kv_cache_scales ----
    {
        const size_t n = 100003;
        std::vector<__half> k(n), v(n);
        float km = 0, vm = 0;
        for (size_t i = 0; i < n; i++) {
            k[i] = __float2half(frand() * 3);
            v[i] = __float2half(frand() * 5);
            km = std::max(km, std::abs(__half2float(k[i])));
            vm = std::max(vm, std::abs(__half2float(v[i])));
        }
        __half *dk, *dv; float *dks, *dvs;
        cudaMalloc(&dk, n * 2); cudaMalloc(&dv, n * 2);
        cudaMalloc(&dks, 4); cudaMalloc(&dvs, 4);
        cudaMemcpy(dk, k.data(), n * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v.data(), n * 2, cudaMemcpyHostToDevice);
        kv_cache_scales<__half><<<1, 256>>>(dk, dv, dks, dvs, n);
        cudaDeviceSynchronize();
        float ks, vs;
        cudaMemcpy(&ks, dks, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&vs, dvs, 4, cudaMemcpyDeviceToHost);
        const bool ok = (ks == km / 240.0f) && (vs == vm / 240.0f);
        printf("kv_cache_scales: %s\n", ok ? "PASS" : "FAIL");
        rc |= !ok;
        cudaFree(dk); cudaFree(dv); cudaFree(dks); cudaFree(dvs);
    }

    // ---- xcache decode vs fp64 oracle (x = 16/sizeof(half) = 8) ----
    {
        const int B = 2, H = 8, HKV = 2, BS = 16, MAXB = 8, D = 128, x = 8;
        const int ctx[B] = {45, 100};
        const int nb = B * MAXB + 1;
        std::vector<int> bt(B * MAXB, -1);
        int nxt = 1;
        for (int b = 0; b < B; b++) for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = nxt++;
        // x-packed key (nb, HKV, D/x, BS, x); value (nb, HKV, D, BS)
        std::vector<__half> K(size_t(nb) * HKV * D * BS), V(K.size()), q(size_t(B) * H * D);
        for (auto& e : K) e = __float2half(frand() * 0.4f);
        for (auto& e : V) e = __float2half(frand() * 0.4f);
        for (auto& e : q) e = __float2half(frand());
        __half *dK, *dV, *dq, *dout; int *dbt, *dctx;
        cudaMalloc(&dK, K.size() * 2); cudaMalloc(&dV, V.size() * 2);
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dout, q.size() * 2);
        cudaMalloc(&dbt, bt.size() * 4); cudaMalloc(&dctx, B * 4);
        cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dctx, ctx, B * 4, cudaMemcpyHostToDevice);
        const float scale = 1.0f / sqrtf(float(D));
        paged_attention_xcache<__half, D><<<dim3(H, B), 32>>>(dq, dK, dV, dbt, dctx, dout, BS, MAXB, scale, H, HKV, x);
        cudaDeviceSynchronize();
        std::vector<__half> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        const int dh = D / x;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            const int kvh = h / (H / HKV);
            std::vector<double> sc; std::vector<int> ts;
            for (int t = 0; t < ctx[b]; t++) {
                const int blk = bt[b * MAXB + t / BS];
                if (blk < 0) continue;
                const int64_t hb = int64_t(blk) * HKV + kvh;
                double s = 0;
                for (int d = 0; d < D; d++) {
                    const int64_t kidx = ((hb * dh + d / x) * BS + t % BS) * x + d % x;
                    s += double(__half2float(q[(size_t(b) * H + h) * D + d])) * double(__half2float(K[kidx]));
                }
                sc.push_back(s * scale); ts.push_back(t);
            }
            double mx = -1e300;
            for (double s : sc) mx = std::max(mx, s);
            std::vector<double> o(D, 0.0); double Z = 0;
            for (size_t i = 0; i < sc.size(); i++) {
                const int t = ts[i], blk = bt[b * MAXB + t / BS];
                const int64_t hb = int64_t(blk) * HKV + kvh;
                const double w = exp(sc[i] - mx);
                for (int d = 0; d < D; d++)
                    o[d] += w * double(__half2float(V[(hb * D + d) * BS + t % BS]));
                Z += w;
            }
            for (int d = 0; d < D; d++)
                maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(b) * H + h) * D + d])) - o[d] / Z));
        }
        printf("paged_attention_xcache D=128 x=8: max diff %.5f (%s)\n", maxd, maxd < 5e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 5e-3);
        cudaFree(dK); cudaFree(dV); cudaFree(dq); cudaFree(dout); cudaFree(dbt); cudaFree(dctx);
    }
    return rc;
}
