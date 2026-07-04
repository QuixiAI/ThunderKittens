// Harness for rope_kv kernels (kernels live in rope_kv_kernels.cuh).
#include "rope_kv_kernels.cuh"
// ============================== test harness ==============================
using namespace tms;
static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }

template <int D>
int run() {
    srand(21);
    const int T_ = 40, HKV = 4, BS = 16, PMAX = 128;
    const int M = T_ * HKV, D2 = D / 2;
    const int num_blocks = (T_ + BS - 1) / BS + 1;
    const size_t cache_n = size_t(num_blocks) * BS * HKV * D;
    std::vector<__half> K(size_t(M) * D), V(size_t(M) * D), W(D);
    std::vector<__half> cosb(size_t(PMAX) * D2), sinb(size_t(PMAX) * D2);
    std::vector<int> pos(T_);
    std::vector<int64_t> slots(T_);
    for (auto& x : K) x = __float2half(frand());
    for (auto& x : V) x = __float2half(frand());
    for (auto& x : W) x = __float2half(frand() * 0.5f + 1.0f);
    for (int p = 0; p < PMAX; p++) for (int d = 0; d < D2; d++) {
        const double th = p / pow(10000.0, 2.0 * d / D);
        cosb[size_t(p) * D2 + d] = __float2half(float(cos(th)));
        sinb[size_t(p) * D2 + d] = __float2half(float(sin(th)));
    }
    for (int t = 0; t < T_; t++) { pos[t] = (t * 7) % PMAX; slots[t] = (t == 3) ? -1 : t + BS; }

    __half *dK, *dV, *dW, *dc, *ds, *dkc, *dvc;
    int* dpos; int64_t* dslot;
    cudaMalloc(&dK, K.size() * 2); cudaMalloc(&dV, V.size() * 2); cudaMalloc(&dW, D * 2);
    cudaMalloc(&dc, cosb.size() * 2); cudaMalloc(&ds, sinb.size() * 2);
    cudaMalloc(&dkc, cache_n * 2); cudaMalloc(&dvc, cache_n * 2);
    cudaMalloc(&dpos, T_ * 4); cudaMalloc(&dslot, T_ * 8);
    cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, W.data(), D * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, cosb.data(), cosb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, sinb.data(), sinb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, pos.data(), T_ * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dslot, slots.data(), T_ * 8, cudaMemcpyHostToDevice);

    int rc = 0;
    for (int variant = 0; variant < 3; variant++) {   // plain, norm, norm+gemma
        cudaMemset(dkc, 0, cache_n * 2); cudaMemset(dvc, 0, cache_n * 2);
        const bool norm = variant > 0; const int gemma = variant == 2;
        if (norm) rope_kv_insert<__half, D, true><<<M, 32>>>(dK, dV, dc, ds, dpos, dslot, dkc, dvc, dW, HKV, BS, gemma, 1e-6f);
        else      rope_kv_insert<__half, D, false><<<M, 32>>>(dK, dV, dc, ds, dpos, dslot, dkc, dvc, nullptr, HKV, BS, 0, 1e-6f);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
        std::vector<__half> kc(cache_n), vc(cache_n);
        cudaMemcpy(kc.data(), dkc, cache_n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(vc.data(), dvc, cache_n * 2, cudaMemcpyDeviceToHost);

        double maxd = 0;
        for (int t = 0; t < T_; t++) {
            if (slots[t] < 0) continue;
            for (int h = 0; h < HKV; h++) {
                const int row = t * HKV + h;
                // reference row: optional rmsnorm then rotate
                std::vector<double> kr(D);
                for (int d = 0; d < D; d++) kr[d] = __half2float(K[size_t(row) * D + d]);
                if (norm) {
                    double ss = 0;
                    for (int d = 0; d < D; d++) ss += kr[d] * kr[d];
                    const double rms = 1.0 / sqrt(ss / D + 1e-6);
                    for (int d = 0; d < D; d++) {
                        const double w = __half2float(W[d]);
                        kr[d] *= rms * (gemma ? (1.0 + w) : w);
                    }
                }
                const size_t dst = (size_t(slots[t] / BS) * BS + slots[t] % BS) * HKV * D + size_t(h) * D;
                for (int d = 0; d < D2; d++) {
                    const double c = __half2float(cosb[size_t(pos[t]) * D2 + d]);
                    const double s = __half2float(sinb[size_t(pos[t]) * D2 + d]);
                    const double r1 = kr[d] * c - kr[d + D2] * s;
                    const double r2 = kr[d + D2] * c + kr[d] * s;
                    maxd = std::max(maxd, std::abs(double(__half2float(kc[dst + d])) - r1));
                    maxd = std::max(maxd, std::abs(double(__half2float(kc[dst + d + D2])) - r2));
                }
                for (int d = 0; d < D; d++)
                    maxd = std::max(maxd, std::abs(double(__half2float(vc[dst + d])) -
                                                   double(__half2float(V[size_t(row) * D + d]))));
            }
        }
        const char* names[3] = {"plain", "norm", "norm+gemma"};
        printf("rope_kv_insert D=%-3d %-10s max diff %.5f (%s)\n", D, names[variant], maxd, maxd < 4e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 4e-3);
    }

    // rope_q (norm variant)
    {
        const int H = 6, MQ = T_ * H;
        std::vector<__half> q(size_t(MQ) * D);
        for (auto& x : q) x = __float2half(frand());
        __half *dq, *dqo;
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dqo, q.size() * 2);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        rope_q<__half, D, true><<<MQ, 32>>>(dq, dc, ds, dpos, dqo, dW, H, 0, 1e-6f);
        cudaDeviceSynchronize();
        std::vector<__half> qo(q.size());
        cudaMemcpy(qo.data(), dqo, q.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        for (int t = 0; t < T_; t++) for (int h = 0; h < H; h++) {
            const int row = t * H + h;
            std::vector<double> qr(D);
            double ss = 0;
            for (int d = 0; d < D; d++) { qr[d] = __half2float(q[size_t(row) * D + d]); ss += qr[d] * qr[d]; }
            const double rms = 1.0 / sqrt(ss / D + 1e-6);
            for (int d = 0; d < D; d++) qr[d] *= rms * __half2float(W[d]);
            for (int d = 0; d < D / 2; d++) {
                const double c = __half2float(cosb[size_t(pos[t]) * (D / 2) + d]);
                const double s = __half2float(sinb[size_t(pos[t]) * (D / 2) + d]);
                maxd = std::max(maxd, std::abs(double(__half2float(qo[size_t(row) * D + d])) - (qr[d] * c - qr[d + D / 2] * s)));
                maxd = std::max(maxd, std::abs(double(__half2float(qo[size_t(row) * D + d + D / 2])) - (qr[d + D / 2] * c + qr[d] * s)));
            }
        }
        printf("rope_q         D=%-3d norm       max diff %.5f (%s)\n", D, maxd, maxd < 4e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 4e-3);
        cudaFree(dq); cudaFree(dqo);
    }
    cudaFree(dK); cudaFree(dV); cudaFree(dW); cudaFree(dc); cudaFree(ds);
    cudaFree(dkc); cudaFree(dvc); cudaFree(dpos); cudaFree(dslot);
    return rc;
}

int main() {
    int rc = run<64>();
    rc |= run<128>();
    return rc;
}
