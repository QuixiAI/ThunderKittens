// Quantized GEMV (batch-1 decode) + format-exactness checker, CUDA/SM86 port of
// ThunderMittens kernels/qgemv. One warp per output row; each lane owns an
// 8-contiguous-column span inside a block (block-major walk keeps the block-scale
// reads uniform per lane-group and the X loads vectorizable). This is the
// memory-bound decode path where shrinking weight bytes 4-8x is the whole win.
//
// Harness reads golden data produced by gen_golden.py (which runs ThunderMittens'
// quant.py - byte-identical packing across Metal and CUDA):
//   ./qgemv.out <golden_dir>/<format>   (expects W.bin/Wq.bin/X.bin/D_ref.bin/meta.txt)
//
// Build:
//   /usr/local/cuda/bin/nvcc qgemv.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o qgemv.out
#include "quant_formats.cuh"
#include "quant_formats_tables.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

using namespace tmq;

// ---- exactness kernel: dequantize every weight on the GPU ----
template<typename FMT>
__global__ void dequant_all(float* out, const uint8_t* Wq, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int row = idx / K, col = idx % K;
    int bpr = K / FMT::block_k;
    const uint8_t* base = Wq + (size_t(row) * bpr + col / FMT::block_k) * FMT::block_bytes;
    out[idx] = FMT::dequant(base, col % FMT::block_k);
}

// ---- qgemv: D = dequant(Wq) @ X, warp per row ----
template<typename FMT>
__global__ void qgemv(half* D, const uint8_t* Wq, const half* X, int N, int K) {
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;   // 32 threads
    const int bpr  = K / FMT::block_k;
    const uint8_t* row_base = Wq + size_t(row) * bpr * FMT::block_bytes;

    constexpr int CPL = 8;                        // contiguous cols per lane
    constexpr int LPB = FMT::block_k / CPL;       // lanes per block (2..32 for block_k 16..256)
    constexpr int BPI = (32 / LPB) > 0 ? (32 / LPB) : 1;  // blocks per warp iteration
    const int b_off = lane / LPB;
    const int col0  = (lane % LPB) * CPL;

    float acc = 0.0f;
    if constexpr (FMT::block_k <= 256) {
        for (int kb = b_off; kb < bpr; kb += BPI) {
            const uint8_t* base = Wq + (size_t(row) * bpr + kb) * FMT::block_bytes;
            const half* xp = X + kb * FMT::block_k + col0;
            float w[8];
            dequant8<FMT>(base, col0, w);
            #pragma unroll
            for (int i = 0; i < 8; i++) acc += w[i] * __half2float(xp[i]);
        }
    }
    (void)row_base;
    // warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) D[row] = __float2half(acc);
}

// nvfp4 has block_k=16 < 32 lanes: LPB=2, BPI=16 - covered by the generic path above.

// ---- host harness ----
static std::vector<uint8_t> read_file(const std::string& p, size_t* sz = nullptr) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != size_t(n)) { fprintf(stderr, "short read %s\n", p.c_str()); exit(2); }
    fclose(f);
    if (sz) *sz = n;
    return v;
}

template<typename FMT>
int run(const std::string& dir, int N, int K) {
    auto Wq_h   = read_file(dir + "/Wq.bin");
    auto Wref_h = read_file(dir + "/W_deq.bin");   // quant.py dequantize_* output, float32 (N,K)
    auto X_h    = read_file(dir + "/X.bin");       // float16 (K)
    auto D_h    = read_file(dir + "/D_ref.bin");   // float32 (N) = W_deq @ X

    uint8_t* dWq; float* dWdeq; half* dX; half* dD;
    cudaMalloc(&dWq, Wq_h.size());
    cudaMalloc(&dWdeq, sizeof(float) * N * K);
    cudaMalloc(&dX, sizeof(half) * K);
    cudaMalloc(&dD, sizeof(half) * N);
    cudaMemcpy(dWq, Wq_h.data(), Wq_h.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X_h.data(), X_h.size(), cudaMemcpyHostToDevice);

    // 1. dequant exactness vs quant.py
    dequant_all<FMT><<<(N * K + 255) / 256, 256>>>(dWdeq, dWq, N, K);
    std::vector<float> wgot(N * K);
    cudaMemcpy(wgot.data(), dWdeq, sizeof(float) * N * K, cudaMemcpyDeviceToHost);
    const float* wref = reinterpret_cast<const float*>(Wref_h.data());
    size_t bad = 0; double maxd = 0;
    for (size_t i = 0; i < size_t(N) * K; i++) {
        double d = std::abs(double(wgot[i]) - double(wref[i]));
        maxd = std::max(maxd, d);
        if (d > 1e-6 * std::max(1.0, std::abs(double(wref[i])))) bad++;
    }
    printf("dequant: %s (bad=%zu, max diff %.3g)\n", bad ? "FAIL" : "EXACT", bad, maxd);

    // 2. gemv vs float reference
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    qgemv<FMT><<<N, 32>>>(dD, dWq, dX, N, K);            // warmup + correctness
    cudaDeviceSynchronize();
    int iters = 200;
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) qgemv<FMT><<<N, 32>>>(dD, dWq, dX, N, K);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= iters;

    std::vector<half> dg(N);
    cudaMemcpy(dg.data(), dD, sizeof(half) * N, cudaMemcpyDeviceToHost);
    const float* dref = reinterpret_cast<const float*>(D_h.data());
    double gmax = 0, gsum = 0, rsum = 0;
    for (int i = 0; i < N; i++) {
        double d = std::abs(double(__half2float(dg[i])) - double(dref[i]));
        gmax = std::max(gmax, d); gsum += d; rsum += std::abs(double(dref[i]));
    }
    double rel = gsum / std::max(rsum, 1e-30);
    double gb = (double(N) * (size_t(K) / FMT::block_k) * FMT::block_bytes + 2.0 * K + 2.0 * N) / 1e9;
    printf("gemv:    rel %.4f%% max %.4g | %.3f ms  %.1f GB/s  (%s)\n",
           100 * rel, gmax, ms, gb / (ms / 1e3), rel < 0.005 ? "PASS" : "FAIL");
    return (bad || rel >= 0.005) ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <golden_dir_with_format_suffix>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    // meta: "fmt N K"
    char fmt[64]; int N, K;
    {
        FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
        if (!f || fscanf(f, "%63s %d %d", fmt, &N, &K) != 3) { fprintf(stderr, "bad meta\n"); return 2; }
        fclose(f);
    }
    printf("== %s  N=%d K=%d\n", fmt, N, K);
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
