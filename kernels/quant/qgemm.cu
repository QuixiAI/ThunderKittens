// Weight-only quantized GEMM, CUDA/SM86 port of ThunderMittens kernels/qgemm.
// Torch-linear semantics: Y(M,N) = X(M,K) @ dequant(Wq(N,K))^T, fp16 X, fp32 accum.
//
// Two paths, mirroring TM:
//  - fragment path (Marlin zero-shuffle): FMT::dequant straight into the mma B
//    fragment (load_wfrag<FMT>), no shared staging, no barriers. Used for
//    block_k <= 64 formats (all the bit-arithmetic ones incl. mxfp8/mxfp4/nvfp4).
//  - full-dequant route for the branchy 256-superblock k/i-quants: dequant the
//    whole W to fp16 once (dequant_all), then run the SAME kernel with the
//    fp16_raw passthrough format. (TM measured dequant-then-GEMM 2-2.3x faster
//    than in-GEMM branchy dequant for these formats at M>=64.)
//
// Correctness-first: W fragments read straight from global (L2-cached; the
// cp.async ring + wider tiles are the perf pass). One warp per 16x16 output tile.
//
// Build:
//   /usr/local/cuda/bin/nvcc qgemm.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o qgemm.out
#include "tm_qmm.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

using namespace tmq;

// ---- qgemm: one warp per 16x16 output tile ----
// Y(M,N) = X(M,K) @ W(N,K)^T. B fragment for mma.ABt: W rows are the mma "columns";
// the W fragment layout above IS the col-major B operand of m16n8k16 when split
// into its two 8-col (= 8 W-row) halves: b0 = {data[0], data[1]}, b1 = {data[2], data[3]}.
template<typename FMT>
__global__ void qgemm(float* Y, const half* X, const uint8_t* Wq, int M, int N, int K) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;

    float acc0[4] = {0, 0, 0, 0};   // rows m0..+16, W-rows n0..n0+8
    float acc1[4] = {0, 0, 0, 0};   // rows m0..+16, W-rows n0+8..n0+16
    for (int k0 = 0; k0 < K; k0 += 16) {
        half2 a[4], b[4];
        load_xfrag(a, X, K, m0, k0);
        load_wfrag<FMT>(b, Wq, bpr, n0, k0);
        // mma B operand wants (k-major pairs per W-row group); our frag has
        // data[0,1] = W rows n0..(+8 in data[1]? no: rows +8*(k%2)) —
        // data[0]=rows n0+0..7? lane r = n0+lane/4 (+8 for k%2=1).
        // b half 0 (W rows n0..n0+7 x k0..k0+15): needs {cols 0-7, cols 8-15} = {data[0], data[2]}
        half2 b0[2] = {b[0], b[2]};
        half2 b1[2] = {b[1], b[3]};
        mma16816(acc0, a, b0);
        mma16816(acc1, a, b1);
    }
    // store: acc layout of m16n8: d[0,1] = (row lane/4, col (lane%4)*2 +0/1), d[2,3] = row+8
    const int lane = threadIdx.x & 31;
    const int r = m0 + lane / 4, c0 = n0 + (lane % 4) * 2;
    if (r < M) {
        Y[size_t(r) * N + c0]     = acc0[0];
        Y[size_t(r) * N + c0 + 1] = acc0[1];
        Y[size_t(r) * N + c0 + 8]     = acc1[0];
        Y[size_t(r) * N + c0 + 8 + 1] = acc1[1];
    }
    if (r + 8 < M) {
        Y[size_t(r + 8) * N + c0]     = acc0[2];
        Y[size_t(r + 8) * N + c0 + 1] = acc0[3];
        Y[size_t(r + 8) * N + c0 + 8]     = acc1[2];
        Y[size_t(r + 8) * N + c0 + 8 + 1] = acc1[3];
    }
}

// full-dequant kernel (route for 256-superblock formats)
template<typename FMT>
__global__ void dequant_to_fp16(half* out, const uint8_t* Wq, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int row = idx / K, col = idx % K;
    const uint8_t* base = Wq + (size_t(row) * (K / FMT::block_k) + col / FMT::block_k) * FMT::block_bytes;
    out[idx] = __float2half(FMT::dequant(base, col % FMT::block_k));
}

// ---- ksplit variant (perf pass): at decode shapes the warp-per-tile grid is
// only (N/16)*(M/16) warps — half the SMs idle at M=64,N=512. Slice K across
// blockIdx.z and atomicAdd fp32 partials into a zeroed Y. Same fragment math.
// (Also consolidated in tm_kernels.cuh as tmq::qgemm_ksplit.) ----
template<typename FMT>
__global__ void qgemm_ksplit(float* Y, const half* X, const uint8_t* Wq,
                             int M, int N, int K, int k_chunk) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int k_beg = blockIdx.z * k_chunk;
    const int k_end = min(K, k_beg + k_chunk);
    const int bpr = K / FMT::block_k;

    float acc0[4] = {0, 0, 0, 0};
    float acc1[4] = {0, 0, 0, 0};
    for (int k0 = k_beg; k0 < k_end; k0 += 16) {
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
    if (r < M) {
        atomicAdd(&Y[size_t(r) * N + c0],         acc0[0]);
        atomicAdd(&Y[size_t(r) * N + c0 + 1],     acc0[1]);
        atomicAdd(&Y[size_t(r) * N + c0 + 8],     acc1[0]);
        atomicAdd(&Y[size_t(r) * N + c0 + 8 + 1], acc1[1]);
    }
    if (r + 8 < M) {
        atomicAdd(&Y[size_t(r + 8) * N + c0],         acc0[2]);
        atomicAdd(&Y[size_t(r + 8) * N + c0 + 1],     acc0[3]);
        atomicAdd(&Y[size_t(r + 8) * N + c0 + 8],     acc1[2]);
        atomicAdd(&Y[size_t(r + 8) * N + c0 + 8 + 1], acc1[3]);
    }
}

static inline int qgemm_pick_kchunk(int M, int N, int K, int block_k) {
    const long tiles = long((N + 15) / 16) * ((M + 15) / 16);
    const int target = 1664;                      // ~82 SMs x 20 warps
    int splits = int((target + tiles - 1) / tiles);
    const int align = (block_k > 16) ? block_k : 16;
    int chunk = ((K / (splits > 0 ? splits : 1)) + align - 1) / align * align;
    if (chunk < align) chunk = align;
    return chunk;
}

// ---- harness ----
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
    constexpr bool SUPERBLOCK = FMT::block_k > 64;   // route k/i-quants via full dequant
    const int M = 64;
    auto Wq_h = read_file(dir + "/Wq.bin");
    auto X_h  = read_file(dir + "/X2.bin");     // fp16 (M,K)
    auto Y_h  = read_file(dir + "/Y_ref.bin");  // fp32 (M,N) = X @ Wdeq^T

    uint8_t* dWq; half* dX; float* dY; half* dWf = nullptr;
    cudaMalloc(&dWq, Wq_h.size());
    cudaMalloc(&dX, sizeof(half) * M * K);
    cudaMalloc(&dY, sizeof(float) * size_t(M) * N);
    cudaMemcpy(dWq, Wq_h.data(), Wq_h.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X_h.data(), X_h.size(), cudaMemcpyHostToDevice);

    dim3 grid(N / 16, (M + 15) / 16);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    auto launch = [&] {
        if constexpr (SUPERBLOCK) {
            dequant_to_fp16<FMT><<<(N * K + 255) / 256, 256>>>(dWf, dWq, N, K);
            qgemm<fp16_raw><<<grid, 32>>>(dY, dX, reinterpret_cast<const uint8_t*>(dWf), M, N, K);
        } else {
            qgemm<FMT><<<grid, 32>>>(dY, dX, dWq, M, N, K);
        }
    };
    if (SUPERBLOCK) cudaMalloc(&dWf, sizeof(half) * size_t(N) * K);
    launch();
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
    int iters = 50;
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) launch();
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= iters;

    std::vector<float> got(size_t(M) * N);
    cudaMemcpy(got.data(), dY, sizeof(float) * got.size(), cudaMemcpyDeviceToHost);
    const float* ref = reinterpret_cast<const float*>(Y_h.data());
    double gsum = 0, rsum = 0, gmax = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = std::abs(double(got[i]) - double(ref[i]));
        gmax = std::max(gmax, d); gsum += d; rsum += std::abs(double(ref[i]));
    }
    double rel = gsum / std::max(rsum, 1e-30);
    double tflop = 2.0 * M * N * K / 1e12;
    printf("qgemm%s: rel %.4f%% max %.4g | %.3f ms  %.2f TFLOP/s  (%s)\n",
           SUPERBLOCK ? "[dequant-route]" : "", 100 * rel, gmax, ms, tflop / (ms / 1e3),
           rel < 0.02 ? "PASS" : "FAIL");
    int rc = rel < 0.02 ? 0 : 1;

    // ---- ksplit variant: K sliced across blockIdx.z + fp32 atomic combine ----
    {
        const int chunk = qgemm_pick_kchunk(M, N, K, SUPERBLOCK ? 16 : FMT::block_k);
        const int splits = (K + chunk - 1) / chunk;
        dim3 gridz(N / 16, (M + 15) / 16, splits);
        auto launch2 = [&] {
            cudaMemsetAsync(dY, 0, sizeof(float) * size_t(M) * N);
            if constexpr (SUPERBLOCK) {
                dequant_to_fp16<FMT><<<(N * K + 255) / 256, 256>>>(dWf, dWq, N, K);
                qgemm_ksplit<fp16_raw><<<gridz, 32>>>(dY, dX,
                    reinterpret_cast<const uint8_t*>(dWf), M, N, K, chunk);
            } else {
                qgemm_ksplit<FMT><<<gridz, 32>>>(dY, dX, dWq, M, N, K, chunk);
            }
        };
        launch2();
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("KSPLIT KERNEL ERROR\n"); return 1; }
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) launch2();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1); ms /= iters;
        cudaMemcpy(got.data(), dY, sizeof(float) * got.size(), cudaMemcpyDeviceToHost);
        gsum = 0; rsum = 0; gmax = 0;
        for (size_t i = 0; i < got.size(); i++) {
            double d = std::abs(double(got[i]) - double(ref[i]));
            gmax = std::max(gmax, d); gsum += d; rsum += std::abs(double(ref[i]));
        }
        rel = gsum / std::max(rsum, 1e-30);
        printf("qgemm-ksplit(x%d)%s: rel %.4f%% max %.4g | %.3f ms  %.2f TFLOP/s  (%s)\n",
               splits, SUPERBLOCK ? "[dequant-route]" : "", 100 * rel, gmax, ms,
               tflop / (ms / 1e3), rel < 0.02 ? "PASS" : "FAIL");
        rc |= !(rel < 0.02);
    }
    return rc;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <golden_dir>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    char fmt[64]; int N, K;
    FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
    if (!f || fscanf(f, "%63s %d %d", fmt, &N, &K) != 3) return 2;
    fclose(f);
    printf("== qgemm %s  N=%d K=%d M=64\n", fmt, N, K);
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
