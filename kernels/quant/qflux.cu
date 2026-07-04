// Fused quantized linear + GELU, CUDA/SM86 port of ThunderMittens kernels/qflux:
//   Y(M,N) = gelu_tanh( X(M,K) @ dequant(Wq(N,K))^T + bias(N) )
// Marlin zero-shuffle fragment path (tm_qmm.cuh) with the epilogue in registers.
// Superblock formats route through dequant-to-fp16 + fp16_raw like qgemm.
//
// Build:
//   /usr/local/cuda/bin/nvcc qflux.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o qflux.out
#include "tm_qmm.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

using namespace tmq;

// tanh-approx GELU, matching TM's substrate gelu() / F.gelu(approximate='tanh')
__device__ __forceinline__ float gelu_tanh(float x) {
    const float c = 0.7978845608028654f;   // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

template<typename FMT>
__global__ void qflux_gelu(float* Y, const half* X, const uint8_t* Wq, const float* bias,
                           int M, int N, int K) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;

    float acc0[4] = {0, 0, 0, 0};
    float acc1[4] = {0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half2 a[4], b[4];
        load_xfrag(a, X, K, m0, k0);
        load_wfrag<FMT>(b, Wq, bpr, n0, k0);
        half2 b0[2] = {b[0], b[2]};
        half2 b1[2] = {b[1], b[3]};
        mma16816(acc0, a, b0);
        mma16816(acc1, a, b1);
    }
    const int lane = threadIdx.x & 31;
    const int r = m0 + lane / 4, c0 = n0 + (lane % 4) * 2;
    const float bA = bias[c0], bB = bias[c0 + 1], bC = bias[c0 + 8], bD = bias[c0 + 9];
    if (r < M) {
        Y[size_t(r) * N + c0]     = gelu_tanh(acc0[0] + bA);
        Y[size_t(r) * N + c0 + 1] = gelu_tanh(acc0[1] + bB);
        Y[size_t(r) * N + c0 + 8] = gelu_tanh(acc1[0] + bC);
        Y[size_t(r) * N + c0 + 9] = gelu_tanh(acc1[1] + bD);
    }
    if (r + 8 < M) {
        Y[size_t(r + 8) * N + c0]     = gelu_tanh(acc0[2] + bA);
        Y[size_t(r + 8) * N + c0 + 1] = gelu_tanh(acc0[3] + bB);
        Y[size_t(r + 8) * N + c0 + 8] = gelu_tanh(acc1[2] + bC);
        Y[size_t(r + 8) * N + c0 + 9] = gelu_tanh(acc1[3] + bD);
    }
}

template<typename FMT>
__global__ void dequant_to_fp16_k(half* out, const uint8_t* Wq, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int row = idx / K, col = idx % K;
    const uint8_t* base = Wq + (size_t(row) * (K / FMT::block_k) + col / FMT::block_k) * FMT::block_bytes;
    out[idx] = __float2half(FMT::dequant(base, col % FMT::block_k));
}

static std::vector<uint8_t> read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != size_t(n)) exit(2);
    fclose(f);
    return v;
}

template<typename FMT>
int run(const std::string& dir, int N, int K) {
    constexpr bool SUPERBLOCK = FMT::block_k > 64;
    const int M = 64;
    auto Wq_h = read_file(dir + "/Wq.bin");
    auto X_h  = read_file(dir + "/X2.bin");
    auto B_h  = read_file(dir + "/bias.bin");
    auto Y_h  = read_file(dir + "/Yflux_ref.bin");

    uint8_t* dWq; half* dX; float *dY, *dB; half* dWf = nullptr;
    cudaMalloc(&dWq, Wq_h.size());
    cudaMalloc(&dX, sizeof(half) * M * K);
    cudaMalloc(&dB, sizeof(float) * N);
    cudaMalloc(&dY, sizeof(float) * size_t(M) * N);
    cudaMemcpy(dWq, Wq_h.data(), Wq_h.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X_h.data(), X_h.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_h.data(), B_h.size(), cudaMemcpyHostToDevice);
    if (SUPERBLOCK) cudaMalloc(&dWf, sizeof(half) * size_t(N) * K);

    dim3 grid(N / 16, (M + 15) / 16);
    if constexpr (SUPERBLOCK) {
        dequant_to_fp16_k<FMT><<<(N * K + 255) / 256, 256>>>(dWf, dWq, N, K);
        qflux_gelu<fp16_raw><<<grid, 32>>>(dY, dX, reinterpret_cast<const uint8_t*>(dWf), dB, M, N, K);
    } else {
        qflux_gelu<FMT><<<grid, 32>>>(dY, dX, dWq, dB, M, N, K);
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }

    std::vector<float> got(size_t(M) * N);
    cudaMemcpy(got.data(), dY, sizeof(float) * got.size(), cudaMemcpyDeviceToHost);
    const float* ref = reinterpret_cast<const float*>(Y_h.data());
    double gsum = 0, rsum = 0, gmax = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = std::abs(double(got[i]) - double(ref[i]));
        gmax = std::max(gmax, d); gsum += d; rsum += std::abs(double(ref[i]));
    }
    double rel = gsum / std::max(rsum, 1e-30);
    printf("qflux%s: rel %.4f%% max %.4g  (%s)\n",
           SUPERBLOCK ? "[dequant-route]" : "", 100 * rel, gmax, rel < 0.03 ? "PASS" : "FAIL");
    return rel < 0.03 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <golden_dir>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    char fmt[64]; int N, K;
    FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
    if (!f || fscanf(f, "%63s %d %d", fmt, &N, &K) != 3) return 2;
    fclose(f);
    printf("== qflux %s  N=%d K=%d M=64\n", fmt, N, K);
    std::string s(fmt);
    if (s == "q8_0")       return run<q8_0>(dir, N, K);
    if (s == "q4_0")       return run<q4_0>(dir, N, K);
    if (s == "q4_1")       return run<q4_1>(dir, N, K);
    if (s == "q5_0")       return run<q5_0>(dir, N, K);
    if (s == "q5_1")       return run<q5_1>(dir, N, K);
    if (s == "kU4B8")      return run<kU4B8>(dir, N, K);
    if (s == "kU4")        return run<kU4>(dir, N, K);
    if (s == "hqq")        return run<hqq>(dir, N, K);
    if (s == "fp8_e4m3")   return run<fp8_e4m3>(dir, N, K);
    if (s == "e5m2")       return run<e5m2>(dir, N, K);
    if (s == "fp8_block")  return run<fp8_block>(dir, N, K);
    if (s == "fp4_e2m1")   return run<fp4_e2m1>(dir, N, K);
    if (s == "mxfp8")      return run<mxfp8>(dir, N, K);
    if (s == "mxfp4")      return run<mxfp4>(dir, N, K);
    if (s == "nvfp4")      return run<nvfp4>(dir, N, K);
    if (s == "mxfp6_e3m2") return run<mxfp6_e3m2>(dir, N, K);
    if (s == "mxfp6_e2m3") return run<mxfp6_e2m3>(dir, N, K);
    if (s == "bitnet")     return run<bitnet>(dir, N, K);
    if (s == "q2_K")       return run<q2_K>(dir, N, K);
    if (s == "q3_K")       return run<q3_K>(dir, N, K);
    if (s == "q4_K")       return run<q4_K>(dir, N, K);
    if (s == "q5_K")       return run<q5_K>(dir, N, K);
    if (s == "q6_K")       return run<q6_K>(dir, N, K);
    if (s == "iq4_nl")     return run<iq4_nl>(dir, N, K);
    if (s == "iq4_xs")     return run<iq4_xs>(dir, N, K);
    if (s == "iq2_xxs")    return run<iq2_xxs>(dir, N, K);
    if (s == "iq2_xs")     return run<iq2_xs>(dir, N, K);
    if (s == "iq3_xxs")    return run<iq3_xxs>(dir, N, K);
    if (s == "iq1_s")      return run<iq1_s>(dir, N, K);
    fprintf(stderr, "unknown format %s\n", fmt);
    return 2;
}
