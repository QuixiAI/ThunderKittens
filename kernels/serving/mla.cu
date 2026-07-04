// Harness for mla kernels (kernels live in mla_kernels.cuh).
#include "mla_kernels.cuh"
#include "paged_attn_v2_kernels.cuh"   // paged_attention_reduce<bf16,512>
// ============================== test harness ==============================
using namespace tms;
static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }
static double b2d(bf16 x) { return double(__bfloat162float(x)); }

int main() {
    srand(41);
    int rc = 0;

    // ---- 1. mla_q_norm_rope: D 192 (128+64) and 512 (448+64), norm modes 0/1/2 ----
    for (int D : {192, 512}) {
        const int T_ = 12, H = 4, nope = D - 64, rope_dim = 64, rope_half = 32, PMAX = 64;
        const int M = T_ * H;
        std::vector<bf16> q(size_t(M) * D), W(D), cb(size_t(PMAX) * rope_half), sb(cb.size());
        std::vector<int> pos(T_);
        for (auto& x : q) x = bf16(frand());
        for (auto& x : W) x = bf16(frand() * 0.5f + 1.0f);
        for (int p = 0; p < PMAX; p++) for (int i = 0; i < rope_half; i++) {
            const double th = p / pow(10000.0, 2.0 * i / rope_dim);
            cb[size_t(p) * rope_half + i] = bf16(float(cos(th)));
            sb[size_t(p) * rope_half + i] = bf16(float(sin(th)));
        }
        for (int t = 0; t < T_; t++) pos[t] = (t * 5) % PMAX;
        bf16 *dq, *dW, *dc, *ds, *dout; int* dpos;
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dW, D * 2);
        cudaMalloc(&dc, cb.size() * 2); cudaMalloc(&ds, sb.size() * 2);
        cudaMalloc(&dout, q.size() * 2); cudaMalloc(&dpos, T_ * 4);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dW, W.data(), D * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dc, cb.data(), cb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, sb.data(), sb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dpos, pos.data(), T_ * 4, cudaMemcpyHostToDevice);
        for (int mode = 0; mode < 3; mode++) {
            if (D == 192) mla_q_norm_rope<192><<<M, 32>>>(dq, dc, ds, dpos, dout, H, nope, rope_dim, mode, 1e-6f, dW);
            else          mla_q_norm_rope<512><<<M, 32>>>(dq, dc, ds, dpos, dout, H, nope, rope_dim, mode, 1e-6f, dW);
            cudaDeviceSynchronize();
            std::vector<bf16> got(q.size());
            cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
            double maxd = 0;
            for (int r = 0; r < M; r++) {
                const int t = r / H;
                std::vector<double> v(D);
                double ss = 0;
                for (int d = 0; d < D; d++) { v[d] = b2d(q[size_t(r) * D + d]); ss += v[d] * v[d]; }
                if (mode) {
                    const double rms = 1.0 / sqrt(ss / D + 1e-6);
                    for (int d = 0; d < D; d++) v[d] *= rms * (mode == 2 ? b2d(W[d]) : 1.0);
                }
                for (int g = nope; g < D; g += 2) {
                    const int p = (g - nope) / 2;
                    const double c = b2d(cb[size_t(pos[t]) * rope_half + p]);
                    const double s = b2d(sb[size_t(pos[t]) * rope_half + p]);
                    const double r0 = v[g] * c - v[g + 1] * s, r1 = v[g] * s + v[g + 1] * c;
                    v[g] = r0; v[g + 1] = r1;
                }
                for (int d = 0; d < D; d++)
                    maxd = std::max(maxd, std::abs(b2d(got[size_t(r) * D + d]) - v[d]));
            }
            printf("mla_q_norm_rope D=%-3d mode=%d max diff %.5f (%s)\n", D, mode, maxd, maxd < 2e-2 ? "PASS" : "FAIL");
            rc |= !(maxd < 2e-2);
        }
        cudaFree(dq); cudaFree(dW); cudaFree(dc); cudaFree(ds); cudaFree(dout); cudaFree(dpos);
    }

    // ---- 2. bf16 absorb decode <512,64> vs fp64 oracle ----
    {
        const int B = 2, H = 4, BS = 16, MAXB = 12, LAT = 512, ROPE = 64, QK = LAT + ROPE;
        const int ctx[B] = {57, 130};
        const int nb = B * MAXB + 1;
        std::vector<int> bt(B * MAXB, -1);
        int nxt = 1;
        for (int b = 0; b < B; b++) for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = nxt++;
        std::vector<bf16> cache(size_t(nb) * BS * QK), q(size_t(B) * H * QK);
        for (auto& x : cache) x = bf16(frand() * 0.2f);
        for (auto& x : q) x = bf16(frand() * 0.3f);
        bf16 *dc_, *dq, *dout; int *dbt, *dctx;
        cudaMalloc(&dc_, cache.size() * 2); cudaMalloc(&dq, q.size() * 2);
        cudaMalloc(&dout, size_t(B) * H * LAT * 2);
        cudaMalloc(&dbt, bt.size() * 4); cudaMalloc(&dctx, B * 4);
        cudaMemcpy(dc_, cache.data(), cache.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dctx, ctx, B * 4, cudaMemcpyHostToDevice);
        const float scale = 1.0f / sqrtf(float(QK));
        mla_decode<512, 64><<<dim3(H, B), 32>>>(dq, dc_, dbt, dctx, dout, BS, MAXB, scale, H);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
        std::vector<bf16> got(size_t(B) * H * LAT);
        cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            std::vector<double> sc; std::vector<int64_t> bases;
            for (int t = 0; t < ctx[b]; t++) {
                const int blk = bt[b * MAXB + t / BS];
                if (blk < 0) continue;
                const int64_t base = (int64_t(blk) * BS + t % BS) * QK;
                double s = 0;
                for (int d = 0; d < QK; d++)
                    s += b2d(q[(size_t(b) * H + h) * QK + d]) * b2d(cache[base + d]);
                sc.push_back(s * scale); bases.push_back(base);
            }
            double mx = -1e300;
            for (double s : sc) mx = std::max(mx, s);
            std::vector<double> o(LAT, 0.0); double Z = 0;
            for (size_t i = 0; i < sc.size(); i++) {
                const double w = exp(sc[i] - mx);
                for (int d = 0; d < LAT; d++) o[d] += w * b2d(cache[bases[i] + d]);
                Z += w;
            }
            for (int d = 0; d < LAT; d++)
                maxd = std::max(maxd, std::abs(b2d(got[(size_t(b) * H + h) * LAT + d]) - o[d] / Z));
        }
        printf("mla_decode<512,64> bf16       max diff %.5f (%s)\n", maxd, maxd < 6e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 6e-3);
        cudaFree(dc_); cudaFree(dq); cudaFree(dout); cudaFree(dbt); cudaFree(dctx);
    }

    // ---- 3. fp8 insert -> fp8 decode round trip vs oracle over the dequantized cache ----
    {
        const int B = 2, H = 4, BS = 16, MAXB = 12, LAT = 512, NOPE = 448, PMAX = 64;
        const int ctx[B] = {49, 90};
        const int nb = B * MAXB + 1;
        std::vector<int> bt(B * MAXB, -1);
        int nxt = 1;
        for (int b = 0; b < B; b++) for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = nxt++;
        const int T_ = ctx[0] + ctx[1];
        std::vector<bf16> kv(size_t(T_) * LAT), cb(size_t(PMAX) * 32), sb(cb.size());
        std::vector<int> pos(T_);
        std::vector<int64_t> slots(T_);
        for (auto& x : kv) x = bf16(frand() * 0.4f);
        for (int p = 0; p < PMAX; p++) for (int i = 0; i < 32; i++) {
            const double th = p / pow(10000.0, i / 32.0);
            cb[size_t(p) * 32 + i] = bf16(float(cos(th)));
            sb[size_t(p) * 32 + i] = bf16(float(sin(th)));
        }
        for (int b = 0, i = 0; b < B; b++) for (int t = 0; t < ctx[b]; t++, i++) {
            pos[i] = t % PMAX;
            slots[i] = int64_t(bt[b * MAXB + t / BS]) * BS + t % BS;
        }
        uint8_t *ddata, *dscale; bf16 *dkv, *dcb, *dsb; int* dpos; int64_t* dslot;
        cudaMalloc(&ddata, size_t(nb) * BS * 576); cudaMalloc(&dscale, size_t(nb) * BS * 8);
        cudaMalloc(&dkv, kv.size() * 2); cudaMalloc(&dcb, cb.size() * 2); cudaMalloc(&dsb, sb.size() * 2);
        cudaMalloc(&dpos, T_ * 4); cudaMalloc(&dslot, T_ * 8);
        cudaMemcpy(dkv, kv.data(), kv.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dcb, cb.data(), cb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dsb, sb.data(), sb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dpos, pos.data(), T_ * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dslot, slots.data(), T_ * 8, cudaMemcpyHostToDevice);
        mla_kv_insert_fp8<<<T_, 32>>>(dkv, dcb, dsb, dpos, dslot, ddata, dscale, BS);
        cudaDeviceSynchronize();

        // rebuild the dequantized cache on the host from the DEVICE cache bytes
        std::vector<uint8_t> data(size_t(nb) * BS * 576), scl(size_t(nb) * BS * 8);
        cudaMemcpy(data.data(), ddata, data.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(scl.data(), dscale, scl.size(), cudaMemcpyDeviceToHost);
        auto dec = [](uint8_t v) {
            float mag;
            if (v & 0x78) { int e = (v >> 3) & 0xF, mm = v & 7; mag = std::ldexp(1.0f + mm / 8.0f, e - 7); }
            else mag = float(v & 7) * 0.001953125f;
            return double((v & 0x80) ? -mag : mag);
        };
        std::vector<double> dcache(size_t(nb) * BS * LAT, 0.0);
        for (size_t s = 0; s < size_t(nb) * BS; s++) {
            for (int d = 0; d < NOPE; d++)
                dcache[s * LAT + d] = dec(data[s * 576 + d]) * std::ldexp(1.0, int(scl[s * 8 + d / 64]) - 127);
            const bf16* rope = reinterpret_cast<const bf16*>(&data[s * 576 + NOPE]);
            for (int d = NOPE; d < LAT; d++) dcache[s * LAT + d] = b2d(rope[d - NOPE]);
        }

        std::vector<bf16> q(size_t(B) * H * LAT);
        for (auto& x : q) x = bf16(frand() * 0.3f);
        bf16 *dq, *dout; int *dbt, *dctx;
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dout, q.size() * 2);
        cudaMalloc(&dbt, bt.size() * 4); cudaMalloc(&dctx, B * 4);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dctx, ctx, B * 4, cudaMemcpyHostToDevice);
        const float scale = 1.0f / sqrtf(float(LAT));
        mla_decode_fp8<<<dim3(H, B), 32>>>(dq, ddata, dscale, dbt, dctx, dout, BS, MAXB, scale, H);
        cudaDeviceSynchronize();
        std::vector<bf16> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            std::vector<double> sc; std::vector<size_t> ss_;
            for (int t = 0; t < ctx[b]; t++) {
                const int blk = bt[b * MAXB + t / BS];
                if (blk < 0) continue;
                const size_t s = size_t(blk) * BS + t % BS;
                double sv = 0;
                for (int d = 0; d < LAT; d++)
                    sv += b2d(q[(size_t(b) * H + h) * LAT + d]) * dcache[s * LAT + d];
                sc.push_back(sv * scale); ss_.push_back(s);
            }
            double mx = -1e300;
            for (double s : sc) mx = std::max(mx, s);
            std::vector<double> o(LAT, 0.0); double Z = 0;
            for (size_t i = 0; i < sc.size(); i++) {
                const double w = exp(sc[i] - mx);
                for (int d = 0; d < LAT; d++) o[d] += w * dcache[ss_[i] * LAT + d];
                Z += w;
            }
            for (int d = 0; d < LAT; d++)
                maxd = std::max(maxd, std::abs(b2d(got[(size_t(b) * H + h) * LAT + d]) - o[d] / Z));
        }
        printf("mla fp8 insert+decode (V4)    max diff %.5f (%s)\n", maxd, maxd < 6e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 6e-3);
        cudaFree(ddata); cudaFree(dscale); cudaFree(dkv); cudaFree(dcb); cudaFree(dsb);
        cudaFree(dpos); cudaFree(dslot); cudaFree(dq); cudaFree(dout); cudaFree(dbt); cudaFree(dctx);
    }
    // ---- 4. bf16 mla_kv_insert (norm modes 0/2, neg slot skip) vs fp64 replay ----
    {
        const int T_ = 37, LAT = 512, ROPE = 64, RH = 32, BS = 16, PMAX = 64;
        const int nslots = (T_ / BS + 2) * BS;
        std::vector<bf16> kvc(size_t(T_) * LAT), kpe(size_t(T_) * ROPE), W(LAT),
                          cb(size_t(PMAX) * RH), sb(cb.size());
        std::vector<int> pos(T_);
        std::vector<int64_t> slots(T_);
        for (auto& x : kvc) x = bf16(frand());
        for (auto& x : kpe) x = bf16(frand());
        for (auto& x : W) x = bf16(frand() * 0.5f + 1.0f);
        for (int p = 0; p < PMAX; p++) for (int i = 0; i < RH; i++) {
            const double th = p / pow(10000.0, double(i) / RH);
            cb[size_t(p) * RH + i] = bf16(float(cos(th)));
            sb[size_t(p) * RH + i] = bf16(float(sin(th)));
        }
        for (int t = 0; t < T_; t++) {
            pos[t] = (t * 3) % PMAX;
            slots[t] = (t == 5) ? -1 : (t * 7) % nslots;    // one padding token
        }
        // distinct slots except the -1 (t*7 mod nslots collides only if nslots%7==0; 48? T_=37, nslots=48: 7*t mod 48 cycles length 48 -> distinct)
        bf16 *dkvc, *dkpe, *dW, *dcb, *dsb, *dcache; int* dpos; int64_t* dslot;
        const size_t cache_n = size_t(nslots) * (LAT + ROPE);
        cudaMalloc(&dkvc, kvc.size() * 2); cudaMalloc(&dkpe, kpe.size() * 2);
        cudaMalloc(&dW, LAT * 2); cudaMalloc(&dcb, cb.size() * 2); cudaMalloc(&dsb, sb.size() * 2);
        cudaMalloc(&dcache, cache_n * 2); cudaMalloc(&dpos, T_ * 4); cudaMalloc(&dslot, T_ * 8);
        cudaMemcpy(dkvc, kvc.data(), kvc.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dkpe, kpe.data(), kpe.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dW, W.data(), LAT * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dcb, cb.data(), cb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dsb, sb.data(), sb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dpos, pos.data(), T_ * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dslot, slots.data(), T_ * 8, cudaMemcpyHostToDevice);
        for (int mode : {0, 2}) {
            const bf16 sentinel = bf16(-7.5f);
            std::vector<bf16> init(cache_n, sentinel);
            cudaMemcpy(dcache, init.data(), cache_n * 2, cudaMemcpyHostToDevice);
            mla_kv_insert<512><<<T_, 32>>>(dkvc, dkpe, dcb, dsb, dpos, dslot, dcache,
                                           BS, ROPE, mode, 1e-6f, dW);
            cudaDeviceSynchronize();
            std::vector<bf16> got(cache_n);
            cudaMemcpy(got.data(), dcache, cache_n * 2, cudaMemcpyDeviceToHost);
            double maxd = 0;
            bool skip_ok = true;
            std::vector<bool> written(nslots, false);
            for (int t = 0; t < T_; t++) {
                if (slots[t] < 0) continue;
                written[slots[t]] = true;
                const size_t dst = size_t(slots[t]) * (LAT + ROPE);
                double ss = 0;
                for (int d = 0; d < LAT; d++) { const double v = b2d(kvc[size_t(t) * LAT + d]); ss += v * v; }
                const double rms = mode ? 1.0 / sqrt(ss / LAT + 1e-6) : 1.0;
                for (int d = 0; d < LAT; d++) {
                    double v = b2d(kvc[size_t(t) * LAT + d]) * rms;
                    if (mode == 2) v *= b2d(W[d]);
                    maxd = std::max(maxd, std::abs(b2d(got[dst + d]) - v));
                }
                for (int i = 0; i < RH; i++) {
                    const double e = b2d(kpe[size_t(t) * ROPE + 2 * i]);
                    const double o = b2d(kpe[size_t(t) * ROPE + 2 * i + 1]);
                    const double c = b2d(cb[size_t(pos[t]) * RH + i]);
                    const double s = b2d(sb[size_t(pos[t]) * RH + i]);
                    maxd = std::max(maxd, std::abs(b2d(got[dst + LAT + 2 * i]) - (e * c - o * s)));
                    maxd = std::max(maxd, std::abs(b2d(got[dst + LAT + 2 * i + 1]) - (e * s + o * c)));
                }
            }
            for (int s = 0; s < nslots; s++) {              // untouched slots keep the sentinel
                if (written[s]) continue;
                for (int d = 0; d < LAT + ROPE; d += 97)
                    skip_ok &= b2d(got[size_t(s) * (LAT + ROPE) + d]) == b2d(sentinel);
            }
            printf("mla_kv_insert<512> bf16 mode=%d max diff %.5f skip=%s (%s)\n", mode, maxd,
                   skip_ok ? "ok" : "BAD", (maxd < 2e-2 && skip_ok) ? "PASS" : "FAIL");
            rc |= !(maxd < 2e-2 && skip_ok);
        }
        cudaFree(dkvc); cudaFree(dkpe); cudaFree(dW); cudaFree(dcb); cudaFree(dsb);
        cudaFree(dcache); cudaFree(dpos); cudaFree(dslot);
    }

    // ---- 5. partitioned + sparse decode variants vs fp64 oracles ----
    {
        const int B = 2, H = 4, BS = 16, MAXB = 12, LAT = 512, NOPE = 448, ROPE = 64, QK = LAT + ROPE;
        const int ctx[B] = {49, 90};
        const int nb = B * MAXB + 1;
        std::vector<int> bt(B * MAXB, -1);
        int nxt = 1;
        for (int b = 0; b < B; b++) for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = nxt++;

        // fp8 cache: fill via random kv (identity rope: pos 0) using the proven insert
        const int T_ = ctx[0] + ctx[1];
        std::vector<bf16> kv(size_t(T_) * LAT), cb(size_t(64) * 32), sb(cb.size());
        std::vector<int> pos(T_, 0);
        std::vector<int64_t> slots(T_);
        for (auto& x : kv) x = bf16(frand() * 0.4f);
        for (int p = 0; p < 64; p++) for (int i = 0; i < 32; i++) {
            const double th = p / pow(10000.0, i / 32.0);
            cb[size_t(p) * 32 + i] = bf16(float(cos(th)));
            sb[size_t(p) * 32 + i] = bf16(float(sin(th)));
        }
        for (int b = 0, i = 0; b < B; b++) for (int t = 0; t < ctx[b]; t++, i++) {
            pos[i] = t % 64;
            slots[i] = int64_t(bt[b * MAXB + t / BS]) * BS + t % BS;
        }
        uint8_t *ddata, *dscale; bf16 *dkv, *dcb, *dsb; int* dpos; int64_t* dslot;
        cudaMalloc(&ddata, size_t(nb) * BS * 576); cudaMalloc(&dscale, size_t(nb) * BS * 8);
        cudaMalloc(&dkv, kv.size() * 2); cudaMalloc(&dcb, cb.size() * 2); cudaMalloc(&dsb, sb.size() * 2);
        cudaMalloc(&dpos, T_ * 4); cudaMalloc(&dslot, T_ * 8);
        cudaMemcpy(dkv, kv.data(), kv.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dcb, cb.data(), cb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dsb, sb.data(), sb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dpos, pos.data(), T_ * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dslot, slots.data(), T_ * 8, cudaMemcpyHostToDevice);
        mla_kv_insert_fp8<<<T_, 32>>>(dkv, dcb, dsb, dpos, dslot, ddata, dscale, BS);
        cudaDeviceSynchronize();
        std::vector<uint8_t> data(size_t(nb) * BS * 576), scl(size_t(nb) * BS * 8);
        cudaMemcpy(data.data(), ddata, data.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(scl.data(), dscale, scl.size(), cudaMemcpyDeviceToHost);
        auto dec = [](uint8_t v) {
            float mag;
            if (v & 0x78) { int e = (v >> 3) & 0xF, mm = v & 7; mag = std::ldexp(1.0f + mm / 8.0f, e - 7); }
            else mag = float(v & 7) * 0.001953125f;
            return double((v & 0x80) ? -mag : mag);
        };
        std::vector<double> dcache(size_t(nb) * BS * LAT, 0.0);
        for (size_t s = 0; s < size_t(nb) * BS; s++) {
            for (int d = 0; d < NOPE; d++)
                dcache[s * LAT + d] = dec(data[s * 576 + d]) * std::ldexp(1.0, int(scl[s * 8 + d / 64]) - 127);
            const bf16* rope = reinterpret_cast<const bf16*>(&data[s * 576 + NOPE]);
            for (int d = NOPE; d < LAT; d++) dcache[s * LAT + d] = b2d(rope[d - NOPE]);
        }

        std::vector<bf16> q(size_t(B) * H * LAT);
        for (auto& x : q) x = bf16(frand() * 0.3f);
        bf16 *dq, *dout; int *dbt, *dctx;
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dout, q.size() * 2);
        cudaMalloc(&dbt, bt.size() * 4); cudaMalloc(&dctx, B * 4);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dctx, ctx, B * 4, cudaMemcpyHostToDevice);
        const float scale = 1.0f / sqrtf(float(LAT));

        // fp64 oracle over an arbitrary token set per (b,h)
        auto oracle = [&](int b, int h, const std::vector<int>& toks, std::vector<double>& o) {
            std::vector<double> sc; std::vector<size_t> ss_;
            for (int t : toks) {
                const int blk = bt[b * MAXB + t / BS];
                if (blk < 0) continue;
                const size_t s = size_t(blk) * BS + t % BS;
                double sv = 0;
                for (int d = 0; d < LAT; d++)
                    sv += b2d(q[(size_t(b) * H + h) * LAT + d]) * dcache[s * LAT + d];
                sc.push_back(sv * scale); ss_.push_back(s);
            }
            double mx = -1e300;
            for (double s : sc) mx = std::max(mx, s);
            o.assign(LAT, 0.0);
            double Z = 0;
            for (size_t i = 0; i < sc.size(); i++) {
                const double w = exp(sc[i] - mx);
                for (int d = 0; d < LAT; d++) o[d] += w * dcache[ss_[i] * LAT + d];
                Z += w;
            }
            for (int d = 0; d < LAT; d++) o[d] /= Z;
        };

        // sparse index lists: every 3rd token + one -1 padding entry
        const int max_topk = 40;
        std::vector<int> idx(B * max_topk, -1), tlen(B);
        std::vector<std::vector<int>> sel(B);
        for (int b = 0; b < B; b++) {
            int n = 0;
            for (int t = 0; t < ctx[b] && n < max_topk - 1; t += 3) { idx[b * max_topk + n++] = t; sel[b].push_back(t); }
            idx[b * max_topk + n++] = -1;                   // skipped entry inside the list
            tlen[b] = n;
        }
        int *didx, *dtlen;
        cudaMalloc(&didx, idx.size() * 4); cudaMalloc(&dtlen, B * 4);
        cudaMemcpy(didx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtlen, tlen.data(), B * 4, cudaMemcpyHostToDevice);

        // (a) fp8 sparse, full
        mla_decode_fp8_v<true, false><<<dim3(H, B), 32>>>(dq, ddata, dscale, dbt, nullptr,
            didx, dtlen, max_topk, dout, nullptr, nullptr, nullptr, BS, MAXB, scale, H, 1, 0);
        cudaDeviceSynchronize();
        std::vector<bf16> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            std::vector<double> o;
            oracle(b, h, sel[b], o);
            for (int d = 0; d < LAT; d++)
                maxd = std::max(maxd, std::abs(b2d(got[(size_t(b) * H + h) * LAT + d]) - o[d]));
        }
        printf("mla_decode_fp8_sparse         max diff %.5f (%s)\n", maxd, maxd < 6e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 6e-3);

        // (b) fp8 dense partition + reduce
        const int PS = 32, P = (90 + PS - 1) / PS;
        float *dtmp, *dml, *des;
        cudaMalloc(&dtmp, size_t(B) * H * P * LAT * 4);
        cudaMalloc(&dml, size_t(B) * H * P * 4); cudaMalloc(&des, size_t(B) * H * P * 4);
        mla_decode_fp8_v<false, true><<<dim3(H, B, P), 32>>>(dq, ddata, dscale, dbt, dctx,
            nullptr, nullptr, 0, nullptr, dtmp, dml, des, BS, MAXB, scale, H, P, PS);
        paged_attention_reduce<bf16, 512><<<dim3(H, B), 32>>>(dtmp, dml, des, dout, H, P);
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            std::vector<int> all(ctx[b]);
            for (int t = 0; t < ctx[b]; t++) all[t] = t;
            std::vector<double> o;
            oracle(b, h, all, o);
            for (int d = 0; d < LAT; d++)
                maxd = std::max(maxd, std::abs(b2d(got[(size_t(b) * H + h) * LAT + d]) - o[d]));
        }
        printf("mla_decode_fp8_partition+red  max diff %.5f (%s)\n", maxd, maxd < 6e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 6e-3);

        // (c) fp8 sparse partition (PS 8 over the index list) + reduce
        const int PS2 = 8, P2 = (max_topk + PS2 - 1) / PS2;
        float *dtmp2, *dml2, *des2;
        cudaMalloc(&dtmp2, size_t(B) * H * P2 * LAT * 4);
        cudaMalloc(&dml2, size_t(B) * H * P2 * 4); cudaMalloc(&des2, size_t(B) * H * P2 * 4);
        mla_decode_fp8_v<true, true><<<dim3(H, B, P2), 32>>>(dq, ddata, dscale, dbt, nullptr,
            didx, dtlen, max_topk, nullptr, dtmp2, dml2, des2, BS, MAXB, scale, H, P2, PS2);
        paged_attention_reduce<bf16, 512><<<dim3(H, B), 32>>>(dtmp2, dml2, des2, dout, H, P2);
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);
        maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            std::vector<double> o;
            oracle(b, h, sel[b], o);
            for (int d = 0; d < LAT; d++)
                maxd = std::max(maxd, std::abs(b2d(got[(size_t(b) * H + h) * LAT + d]) - o[d]));
        }
        printf("mla_decode_fp8_sparse_part+red max diff %.5f (%s)\n", maxd, maxd < 6e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 6e-3);

        // (d) bf16 partitioned decode + reduce vs the serial bf16 kernel's oracle path
        {
            std::vector<bf16> cache(size_t(nb) * BS * QK), qq(size_t(B) * H * QK);
            for (auto& x : cache) x = bf16(frand() * 0.2f);
            for (auto& x : qq) x = bf16(frand() * 0.3f);
            bf16 *dc_, *dq2, *dout2;
            cudaMalloc(&dc_, cache.size() * 2); cudaMalloc(&dq2, qq.size() * 2);
            cudaMalloc(&dout2, size_t(B) * H * LAT * 2);
            cudaMemcpy(dc_, cache.data(), cache.size() * 2, cudaMemcpyHostToDevice);
            cudaMemcpy(dq2, qq.data(), qq.size() * 2, cudaMemcpyHostToDevice);
            const float sc2 = 1.0f / sqrtf(float(QK));
            mla_decode_partition<512, 64><<<dim3(H, B, P), 32>>>(dq2, dc_, dbt, dctx,
                dtmp, dml, des, BS, MAXB, sc2, H, P, PS);
            paged_attention_reduce<bf16, 512><<<dim3(H, B), 32>>>(dtmp, dml, des, dout2, H, P);
            cudaDeviceSynchronize();
            std::vector<bf16> got2(size_t(B) * H * LAT);
            cudaMemcpy(got2.data(), dout2, got2.size() * 2, cudaMemcpyDeviceToHost);
            double md2 = 0;
            for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
                std::vector<double> sc; std::vector<int64_t> bases;
                for (int t = 0; t < ctx[b]; t++) {
                    const int blk = bt[b * MAXB + t / BS];
                    if (blk < 0) continue;
                    const int64_t base = (int64_t(blk) * BS + t % BS) * QK;
                    double s = 0;
                    for (int d = 0; d < QK; d++)
                        s += b2d(qq[(size_t(b) * H + h) * QK + d]) * b2d(cache[base + d]);
                    sc.push_back(s * sc2); bases.push_back(base);
                }
                double mx = -1e300;
                for (double s : sc) mx = std::max(mx, s);
                std::vector<double> o(LAT, 0.0); double Z = 0;
                for (size_t i = 0; i < sc.size(); i++) {
                    const double w = exp(sc[i] - mx);
                    for (int d = 0; d < LAT; d++) o[d] += w * b2d(cache[bases[i] + d]);
                    Z += w;
                }
                for (int d = 0; d < LAT; d++)
                    md2 = std::max(md2, std::abs(b2d(got2[(size_t(b) * H + h) * LAT + d]) - o[d] / Z));
            }
            printf("mla_decode_partition bf16+red max diff %.5f (%s)\n", md2, md2 < 6e-3 ? "PASS" : "FAIL");
            rc |= !(md2 < 6e-3);
            cudaFree(dc_); cudaFree(dq2); cudaFree(dout2);
        }
        cudaFree(ddata); cudaFree(dscale); cudaFree(dkv); cudaFree(dcb); cudaFree(dsb);
        cudaFree(dpos); cudaFree(dslot); cudaFree(dq); cudaFree(dout); cudaFree(dbt); cudaFree(dctx);
        cudaFree(didx); cudaFree(dtlen); cudaFree(dtmp); cudaFree(dml); cudaFree(des);
        cudaFree(dtmp2); cudaFree(dml2); cudaFree(des2);
    }
    return rc;
}
