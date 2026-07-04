// Harness for attn_varlen kernels (kernels live in attn_varlen_kernels.cuh).
#include "attn_varlen_kernels.cuh"
// ============================== test harness ==============================
using namespace tms;
static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }

int main() {
    srand(71);
    const int B = 3, H = 4, HKV = 2, BS = 16, MAXB = 12, D = 64;
    const int qlen[B] = {13, 40, 1};              // ragged; batch 2 is a decode-like step
    const int ctx[B] = {29, 40, 90};              // context >= qlen (prefixes on 0 and 2)
    std::vector<int> cu = {0};
    for (int b = 0; b < B; b++) cu.push_back(cu.back() + qlen[b]);
    const int total_q = cu[B];
    const int max_tiles = 32;

    // worklist
    int *dcu, *dqlens, *dpad, *dts, *dtl, *dnt;
    cudaMalloc(&dcu, (B + 1) * 4); cudaMalloc(&dqlens, B * 4); cudaMalloc(&dpad, (B + 1) * 4);
    cudaMalloc(&dts, max_tiles * 4); cudaMalloc(&dtl, max_tiles * 4); cudaMalloc(&dnt, 4);
    cudaMemcpy(dcu, cu.data(), (B + 1) * 4, cudaMemcpyHostToDevice);
    varlen_build_worklist<<<1, 256>>>(dcu, dqlens, dpad, dts, dtl, dnt, B, max_tiles);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
    std::vector<int> ts(max_tiles), tl(max_tiles), pad(B + 1);
    int nt;
    cudaMemcpy(ts.data(), dts, max_tiles * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(tl.data(), dtl, max_tiles * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(pad.data(), dpad, (B + 1) * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&nt, dnt, 4, cudaMemcpyDeviceToHost);
    {   // reference worklist
        int bad = 0, rt = 0, rp = 0;
        std::vector<int> ets(max_tiles, -1), etl(max_tiles, 0), epad(B + 1);
        for (int b = 0; b < B; b++) {
            epad[b] = rp;
            const int n = (qlen[b] + 7) / 8;
            for (int t = 0; t < n; t++) { ets[rt + t] = b; etl[rt + t] = t * 8; }
            rt += n; rp += n * 8;
        }
        epad[B] = rp;
        if (nt != rt) bad++;
        for (int i = 0; i < max_tiles; i++) if (ts[i] != ets[i] || tl[i] != etl[i]) bad++;
        for (int b = 0; b <= B; b++) if (pad[b] != epad[b]) bad++;
        printf("varlen_build_worklist: n_tiles=%d, %d mismatches (%s)\n", nt, bad, bad ? "FAIL" : "PASS");
        if (bad) return 1;
    }

    // pad/gather round trip
    const int total_padded = pad[B];
    {
        std::vector<__half> q(size_t(total_q) * H * D);
        for (auto& x : q) x = __float2half(frand());
        __half *dq, *dhm, *dback;
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dhm, size_t(H) * total_padded * D * 2);
        cudaMalloc(&dback, q.size() * 2);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemset(dback, 0, q.size() * 2);
        varlen_q_pad_gather<__half, D><<<total_padded, 128>>>(dq, dhm, dcu, dpad, B, H, total_padded);
        varlen_o_regather<__half, D><<<total_padded, 128>>>(dhm, dback, dcu, dpad, B, H, total_padded);
        cudaDeviceSynchronize();
        std::vector<__half> back(q.size());
        cudaMemcpy(back.data(), dback, q.size() * 2, cudaMemcpyDeviceToHost);
        size_t bad = 0;
        for (size_t i = 0; i < q.size(); i++)
            if (__half2float(back[i]) != __half2float(q[i])) bad++;
        printf("pad_gather/regather round trip: %zu mismatches (%s)\n", bad, bad ? "FAIL" : "PASS");
        if (bad) return 1;
        cudaFree(dq); cudaFree(dhm); cudaFree(dback);
    }

    // prefill attention over paged KV with prefixes
    {
        const int nb = B * MAXB + 1;
        const size_t cache_n = size_t(nb) * BS * HKV * D;
        std::vector<int> bt(B * MAXB, -1);
        int nxt = 1;
        for (int b = 0; b < B; b++) for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = nxt++;
        std::vector<__half> K(cache_n), V(cache_n), q(size_t(total_q) * H * D);
        for (auto& x : K) x = __float2half(frand() * 0.4f);
        for (auto& x : V) x = __float2half(frand() * 0.4f);
        for (auto& x : q) x = __float2half(frand());
        __half *dK, *dV, *dq, *dout;
        int *dbt, *dctx;
        cudaMalloc(&dK, cache_n * 2); cudaMalloc(&dV, cache_n * 2);
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dout, q.size() * 2);
        cudaMalloc(&dbt, bt.size() * 4); cudaMalloc(&dctx, B * 4);
        cudaMemcpy(dK, K.data(), cache_n * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dV, V.data(), cache_n * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dctx, ctx, B * 4, cudaMemcpyHostToDevice);
        const float scale = 1.0f / sqrtf(float(D));
        attn_varlen_prefill<__half, D><<<dim3(max_tiles, H), 256>>>(
            dq, dK, dV, dbt, dctx, dcu, dqlens, dts, dtl, dout, BS, MAXB, scale, H, HKV);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
        std::vector<__half> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);

        double maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) for (int ql = 0; ql < qlen[b]; ql++) {
            const int kvh = h / (H / HKV);
            const int t_end = (ctx[b] - qlen[b]) + ql + 1;
            std::vector<double> sc; std::vector<size_t> bases;
            for (int t = 0; t < t_end; t++) {
                const int blk = bt[b * MAXB + t / BS];
                if (blk < 0) continue;
                const size_t base = (size_t(blk) * BS + t % BS) * HKV * D + size_t(kvh) * D;
                double s = 0;
                for (int d = 0; d < D; d++)
                    s += double(__half2float(q[(size_t(cu[b] + ql) * H + h) * D + d])) *
                         double(__half2float(K[base + d]));
                sc.push_back(s * scale); bases.push_back(base);
            }
            double mx = -1e300;
            for (double s : sc) mx = std::max(mx, s);
            std::vector<double> o(D, 0.0); double Z = 0;
            for (size_t i = 0; i < sc.size(); i++) {
                const double w = exp(sc[i] - mx);
                for (int d = 0; d < D; d++) o[d] += w * double(__half2float(V[bases[i] + d]));
                Z += w;
            }
            for (int d = 0; d < D; d++)
                maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(cu[b] + ql) * H + h) * D + d])) - o[d] / Z));
        }
        printf("attn_varlen_prefill (ragged + prefix + GQA): max diff %.5f (%s)\n",
               maxd, maxd < 5e-3 ? "PASS" : "FAIL");
        if (!(maxd < 5e-3)) return 1;
        cudaFree(dK); cudaFree(dV); cudaFree(dq); cudaFree(dout); cudaFree(dbt); cudaFree(dctx);
    }
    return 0;
}
