// Harness for paged_attn_v2 kernels (kernels live in paged_attn_v2_kernels.cuh).
#include "paged_attn_v2_kernels.cuh"
// ============================== test harness ==============================
#include <cuda_bf16.h>
using namespace tms;

static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }

// nearest e4m3 code by brute force over all 256 (exact nearest; decode is the kernel's own)
static uint8_t host_e4m3_nearest(float x) {
    auto dec = [](uint8_t v) {
        float mag;
        if (v & 0x78) { int e = (v >> 3) & 0xF, mm = v & 7; mag = std::ldexp(1.0f + mm / 8.0f, e - 7); }
        else mag = float(v & 7) * 0.001953125f;
        return (v & 0x80) ? -mag : mag;
    };
    uint8_t best = 0; float bd = 1e30f;
    for (int c = 0; c < 256; c++) {
        if ((c & 0x7F) == 0x7F) continue;   // skip NaN codes
        float d = std::abs(dec(uint8_t(c)) - x);
        if (d < bd) { bd = d; best = uint8_t(c); }
    }
    return best;
}

int main() {
    srand(11);
    const int B = 2, H = 8, HKV = 2, BS = 16, MAXB = 16, D = 128;
    const int ctx[B] = {201, 64};
    const float scale = 1.0f / sqrtf(float(D));
    const int num_blocks = B * MAXB + 1;
    const size_t cache_n = size_t(num_blocks) * BS * HKV * D;

    std::vector<int> bt(B * MAXB, -1);
    int nb = 1;
    for (int b = 0; b < B; b++)
        for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = nb++;
    std::vector<__half> K(cache_n), V(cache_n), q(size_t(B) * H * D);
    for (auto& x : K) x = __float2half(frand() * 0.3f);
    for (auto& x : V) x = __float2half(frand() * 0.3f);
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

    // fp64 oracle (full softmax over all mapped tokens; window variants covered by t_start logic)
    auto oracle = [&](int b, int h, int window, std::vector<double>& o) {
        const int kvh = h / (H / HKV);
        const int t0 = (window > 0) ? std::max(0, ctx[b] - window) : 0;
        std::vector<double> sc; std::vector<size_t> bases;
        for (int t = t0; t < ctx[b]; t++) {
            const int blk = bt[b * MAXB + t / BS];
            if (blk < 0) continue;
            const size_t base = (size_t(blk) * BS + t % BS) * HKV * D + size_t(kvh) * D;
            double s = 0;
            for (int d = 0; d < D; d++)
                s += double(__half2float(q[(size_t(b) * H + h) * D + d])) * double(__half2float(K[base + d]));
            sc.push_back(s * scale); bases.push_back(base);
        }
        double mx = -1e300;
        for (double s : sc) mx = std::max(mx, s);
        o.assign(D, 0.0);
        double Z = 0;
        for (size_t i = 0; i < sc.size(); i++) {
            const double w = exp(sc[i] - mx);
            for (int d = 0; d < D; d++) o[d] += w * double(__half2float(V[bases[i] + d]));
            Z += w;
        }
        for (int d = 0; d < D; d++) o[d] = Z ? o[d] / Z : 0.0;
    };

    int rc = 0;
    const int max_ctx = std::max(ctx[0], ctx[1]);
    for (int PS : {16, 48, 512}) for (int window : {0, 40}) {
        const int P = (max_ctx + PS - 1) / PS;
        float *dtmp, *dml, *des;
        cudaMalloc(&dtmp, sizeof(float) * size_t(B) * H * P * D);
        cudaMalloc(&dml, sizeof(float) * B * H * P);
        cudaMalloc(&des, sizeof(float) * B * H * P);
        dim3 pgrid(H, B, P), rgrid(H, B);
        paged_attention_partition<__half, D><<<pgrid, 32>>>(dq, dK, dV, dbt, dctx, dtmp, dml, des, BS, MAXB, scale, H, HKV, P, PS, window);
        paged_attention_reduce<__half, D><<<rgrid, 32>>>(dtmp, dml, des, dout, H, P);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
        std::vector<__half> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        std::vector<double> o;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            oracle(b, h, window, o);
            for (int d = 0; d < D; d++)
                maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(b) * H + h) * D + d])) - o[d]));
        }
        printf("v2 partition/reduce PS=%-3d window=%-2d max diff %.5f (%s)\n", PS, window, maxd, maxd < 5e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 5e-3);
        cudaFree(dtmp); cudaFree(dml); cudaFree(des);
    }

    // fp8 cache variant: quantize K/V per-KV-head, compare vs fp64 oracle on the DEQUANTIZED cache
    {
        std::vector<uint8_t> K8(cache_n), V8(cache_n);
        std::vector<float> ks(HKV), vs(HKV);
        for (int kh = 0; kh < HKV; kh++) {
            float ka = 0, va = 0;
            for (size_t blk = 0; blk < size_t(num_blocks) * BS; blk++)
                for (int d = 0; d < D; d++) {
                    ka = std::max(ka, std::abs(__half2float(K[blk * HKV * D + kh * D + d])));
                    va = std::max(va, std::abs(__half2float(V[blk * HKV * D + kh * D + d])));
                }
            ks[kh] = ka / 448.0f; vs[kh] = va / 448.0f;
        }
        auto decq = [&](uint8_t v) {
            float mag;
            if (v & 0x78) { int e = (v >> 3) & 0xF, mm = v & 7; mag = std::ldexp(1.0f + mm / 8.0f, e - 7); }
            else mag = float(v & 7) * 0.001953125f;
            return (v & 0x80) ? -mag : mag;
        };
        std::vector<__half> Kd(cache_n), Vd(cache_n);
        for (size_t blk = 0; blk < size_t(num_blocks) * BS; blk++)
            for (int kh = 0; kh < HKV; kh++)
                for (int d = 0; d < D; d++) {
                    const size_t i = blk * HKV * D + kh * D + d;
                    K8[i] = host_e4m3_nearest(__half2float(K[i]) / ks[kh]);
                    V8[i] = host_e4m3_nearest(__half2float(V[i]) / vs[kh]);
                    Kd[i] = __float2half(ks[kh] * decq(K8[i]));   // oracle sees the dequantized cache
                    Vd[i] = __float2half(vs[kh] * decq(V8[i]));
                }
        std::vector<__half> Ksave = K, Vsave = V;
        K = Kd; V = Vd;   // oracle closure reads K/V
        uint8_t *dK8, *dV8; float *dks, *dvs;
        cudaMalloc(&dK8, cache_n); cudaMalloc(&dV8, cache_n);
        cudaMalloc(&dks, HKV * 4); cudaMalloc(&dvs, HKV * 4);
        cudaMemcpy(dK8, K8.data(), cache_n, cudaMemcpyHostToDevice);
        cudaMemcpy(dV8, V8.data(), cache_n, cudaMemcpyHostToDevice);
        cudaMemcpy(dks, ks.data(), HKV * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dvs, vs.data(), HKV * 4, cudaMemcpyHostToDevice);
        const int PS = 64, P = (max_ctx + PS - 1) / PS;
        float *dtmp, *dml, *des;
        cudaMalloc(&dtmp, sizeof(float) * size_t(B) * H * P * D);
        cudaMalloc(&dml, sizeof(float) * B * H * P);
        cudaMalloc(&des, sizeof(float) * B * H * P);
        paged_attention_partition_fp8<__half, D><<<dim3(H, B, P), 32>>>(dq, dK8, dV8, dbt, dctx, dks, dvs, dtmp, dml, des, BS, MAXB, scale, H, HKV, P, PS, 0);
        paged_attention_reduce<__half, D><<<dim3(H, B), 32>>>(dtmp, dml, des, dout, H, P);
        cudaDeviceSynchronize();
        std::vector<__half> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        std::vector<double> o;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            oracle(b, h, 0, o);
            for (int d = 0; d < D; d++)
                maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(b) * H + h) * D + d])) - o[d]));
        }
        printf("v2 fp8 cache PS=64            max diff %.5f (%s)\n", maxd, maxd < 5e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 5e-3);
        K = Ksave; V = Vsave;
        cudaFree(dK8); cudaFree(dV8); cudaFree(dks); cudaFree(dvs);
        cudaFree(dtmp); cudaFree(dml); cudaFree(des);
    }

    // cascade: shared prefix (len 96) + per-request paged suffix, concat partials along P
    {
        const int PLEN = 96, PS = 48;
        const int Ppre = (PLEN + PS - 1) / PS, Psuf = (max_ctx + PS - 1) / PS, P = Ppre + Psuf;
        std::vector<__half> pk(size_t(PLEN) * HKV * D), pv(pk.size());
        for (auto& x : pk) x = __float2half(frand() * 0.3f);
        for (auto& x : pv) x = __float2half(frand() * 0.3f);
        __half *dpk, *dpv;
        cudaMalloc(&dpk, pk.size() * 2); cudaMalloc(&dpv, pv.size() * 2);
        cudaMemcpy(dpk, pk.data(), pk.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dpv, pv.data(), pv.size() * 2, cudaMemcpyHostToDevice);
        float *dtmp, *dml, *des;
        cudaMalloc(&dtmp, sizeof(float) * size_t(B) * H * P * D);
        cudaMalloc(&dml, sizeof(float) * B * H * P);
        cudaMalloc(&des, sizeof(float) * B * H * P);
        // prefix partials write partitions [0, Ppre); suffix writes [Ppre, P).
        // Both kernels index stats as (bh*num_partitions + part) with num_partitions = P,
        // so launch prefix with grid.z = Ppre and pass P; suffix needs its stats OFFSET by
        // Ppre - handled by passing (tmp + Ppre*D per bh)? Simplest correct composition:
        // run suffix into its own buffers then interleave on host? Instead: offset pointers -
        // stats are (B,H,P): suffix writes at part' = part + Ppre via pointer arithmetic
        // only if strides match; they do per (b,h) row only when P is the common stride.
        // We keep it simple and correct: launch suffix with num_partitions = P and a
        // part-offset baked in by shifting the grid START - CUDA has no grid offset, so we
        // pass tmp_out/max_logits/exp_sums offset by Ppre in the LAST dim via a wrapper
        // trick: the kernels compute stat = bh*P + part, so passing (ml + Ppre) etc. shifts
        // every (b,h) row by Ppre. Same for tmp_out with (tmp + Ppre*D).
        cascade_prefix_partition<__half, D><<<dim3(H, B, Ppre), 32>>>(dq, dpk, dpv, dtmp, dml, des, scale, H, HKV, PLEN, P, PS);
        paged_attention_partition<__half, D><<<dim3(H, B, Psuf), 32>>>(dq, dK, dV, dbt, dctx,
            dtmp + size_t(Ppre) * D, dml + Ppre, des + Ppre, BS, MAXB, scale, H, HKV, P, PS, 0);
        paged_attention_reduce<__half, D><<<dim3(H, B), 32>>>(dtmp, dml, des, dout, H, P);
        cudaDeviceSynchronize();
        std::vector<__half> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        // oracle: full softmax over prefix ++ mapped suffix
        double maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            const int kvh = h / (H / HKV);
            std::vector<double> sc; std::vector<const __half*> vp;
            for (int t = 0; t < PLEN; t++) {
                const size_t base = (size_t(t) * HKV + kvh) * D;
                double s = 0;
                for (int d = 0; d < D; d++)
                    s += double(__half2float(q[(size_t(b) * H + h) * D + d])) * double(__half2float(pk[base + d]));
                sc.push_back(s * scale); vp.push_back(&pv[base]);
            }
            for (int t = 0; t < ctx[b]; t++) {
                const int blk = bt[b * MAXB + t / BS];
                if (blk < 0) continue;
                const size_t base = (size_t(blk) * BS + t % BS) * HKV * D + size_t(kvh) * D;
                double s = 0;
                for (int d = 0; d < D; d++)
                    s += double(__half2float(q[(size_t(b) * H + h) * D + d])) * double(__half2float(K[base + d]));
                sc.push_back(s * scale); vp.push_back(&V[base]);
            }
            double mx = -1e300;
            for (double s : sc) mx = std::max(mx, s);
            std::vector<double> o(D, 0.0);
            double Z = 0;
            for (size_t i = 0; i < sc.size(); i++) {
                const double w = exp(sc[i] - mx);
                for (int d = 0; d < D; d++) o[d] += w * double(__half2float(vp[i][d]));
                Z += w;
            }
            for (int d = 0; d < D; d++)
                maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(b) * H + h) * D + d])) - o[d] / Z));
        }
        printf("cascade prefix+suffix          max diff %.5f (%s)\n", maxd, maxd < 5e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 5e-3);
        cudaFree(dpk); cudaFree(dpv); cudaFree(dtmp); cudaFree(dml); cudaFree(des);
    }
    return rc;
}
