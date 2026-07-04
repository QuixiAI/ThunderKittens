// Harness for attn_q kernels (kernels live in attn_q_kernels.cuh).
#include "attn_q_kernels.cuh"
// ============================== test harness ==============================
using namespace tms;
static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }

// host-side simple packers in the documented layouts (kernel decode already proven
// bit-exact vs quant.py; the harness only needs valid bytes + a matching host decode)
static void pack_q8_0(const float* x, uint8_t* out) {          // 32 vals -> 34 bytes
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::abs(x[i]));
    const float d = amax / 127.0f, inv = d ? 1.0f / d : 0.0f;
    *reinterpret_cast<__half*>(out) = __float2half_rn(d);
    for (int i = 0; i < 32; i++) {
        int v = int(std::rint(x[i] * inv));
        out[2 + i] = uint8_t(int8_t(std::max(-127, std::min(127, v))));
    }
}
static void pack_q4_0(const float* x, uint8_t* out) {          // 32 vals -> 18 bytes
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::abs(x[i]));
    const float d = amax / 7.0f, inv = d ? 1.0f / d : 0.0f;    // simple symmetric fit
    *reinterpret_cast<__half*>(out) = __float2half_rn(d);
    uint8_t nib[32];
    for (int i = 0; i < 32; i++) {
        int v = int(std::rint(x[i] * inv)) + 8;
        nib[i] = uint8_t(std::max(0, std::min(15, v)));
    }
    for (int i = 0; i < 16; i++) out[2 + i] = nib[i] | (nib[i + 16] << 4);
}
static double dec_q8_0(const uint8_t* b, int c) {
    return double(__half2float(*reinterpret_cast<const __half*>(b))) * double(int8_t(b[2 + c]));
}
static double dec_q4_0(const uint8_t* b, int c) {
    const int nib = (c < 16) ? (b[2 + c] & 0xF) : (b[2 + c - 16] >> 4);
    return double(__half2float(*reinterpret_cast<const __half*>(b))) * double(nib - 8);
}

template <typename FMT, int D>
int run(const char* name, void (*pack)(const float*, uint8_t*), double (*dec)(const uint8_t*, int)) {
    const int B = 2, H = 3, N = 96;
    const int BPR = D / FMT::block_k;
    const size_t rows = size_t(B) * H * N;
    std::vector<half> q(rows * D);
    std::vector<float> Kf(rows * D), Vf(rows * D);
    for (auto& x : q) x = __float2half(frand());
    for (auto& x : Kf) x = frand() * 0.5f;
    for (auto& x : Vf) x = frand() * 0.5f;
    std::vector<uint8_t> Kq(rows * BPR * FMT::block_bytes), Vq(Kq.size());
    for (size_t r = 0; r < rows; r++)
        for (int b = 0; b < BPR; b++) {
            pack(&Kf[r * D + b * FMT::block_k], &Kq[(r * BPR + b) * FMT::block_bytes]);
            pack(&Vf[r * D + b * FMT::block_k], &Vq[(r * BPR + b) * FMT::block_bytes]);
        }

    half *dq, *dout; uint8_t *dK, *dV;
    cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dout, q.size() * 2);
    cudaMalloc(&dK, Kq.size()); cudaMalloc(&dV, Vq.size());
    cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, Kq.data(), Kq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, Vq.data(), Vq.size(), cudaMemcpyHostToDevice);
    const float scale = 1.0f / sqrtf(float(D));

    int rc = 0;
    for (int causal = 0; causal < 2; causal++) {
        dim3 grid(N, H, B);
        if (causal) attn_q<FMT, D, true><<<grid, 32>>>(dq, dK, dV, dout, N, H, scale);
        else        attn_q<FMT, D, false><<<grid, 32>>>(dq, dK, dV, dout, N, H, scale);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
        std::vector<half> got(q.size());
        cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);

        double maxd = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) for (int qi = 0; qi < N; qi++) {
            const size_t bh = (size_t(b) * H + h);
            const int t_end = causal ? qi + 1 : N;
            std::vector<double> sc(t_end);
            double mx = -1e300;
            for (int t = 0; t < t_end; t++) {
                double s = 0;
                for (int d = 0; d < D; d++)
                    s += double(__half2float(q[(bh * N + qi) * D + d])) *
                         dec(&Kq[((bh * N + t) * BPR + d / FMT::block_k) * FMT::block_bytes], d % FMT::block_k);
                sc[t] = s * scale;
                mx = std::max(mx, sc[t]);
            }
            std::vector<double> o(D, 0.0);
            double Z = 0;
            for (int t = 0; t < t_end; t++) {
                const double w = exp(sc[t] - mx);
                for (int d = 0; d < D; d++)
                    o[d] += w * dec(&Vq[((bh * N + t) * BPR + d / FMT::block_k) * FMT::block_bytes], d % FMT::block_k);
                Z += w;
            }
            for (int d = 0; d < D; d++)
                maxd = std::max(maxd, std::abs(double(__half2float(got[(bh * N + qi) * D + d])) - o[d] / Z));
        }
        printf("attn_q %-9s D=%-3d %-10s max diff %.5f (%s)\n", name, D,
               causal ? "causal" : "non-causal", maxd, maxd < 5e-3 ? "PASS" : "FAIL");
        rc |= !(maxd < 5e-3);
    }
    cudaFree(dq); cudaFree(dout); cudaFree(dK); cudaFree(dV);
    return rc;
}

int main() {
    srand(31);
    int rc = 0;
    rc |= run<q8_0, 64>("q8_0", pack_q8_0, dec_q8_0);
    rc |= run<q8_0, 128>("q8_0", pack_q8_0, dec_q8_0);
    rc |= run<q4_0, 64>("q4_0", pack_q4_0, dec_q4_0);
    rc |= run<q4_0, 128>("q4_0", pack_q4_0, dec_q4_0);
    return rc;
}
