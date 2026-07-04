// Integer-path quantized GEMV (batch-1 decode), CUDA/SM86 port of ThunderMittens
// kernels/qgemv_int. int8 x int8 -> int32 accumulate, then scale. Where Metal
// emulated idot4 in ALU ops, this uses native __dp4a.
//
//   qgemv_w8a8: int8 W (per-channel scale) x int8 X (per-token scale), 2 rows/warp.
//   qgemv_w2a8: BitNet ternary 2-bit W (per-group absmean scale) x int8 X.
//
// Oracle (TM contract): out = w_scale[n] * a_scale * (Wq_i32 @ Xq_i32)  — integer-exact
// sums, so the test tolerance covers only the final float multiply.
//
// Build:
//   /usr/local/cuda/bin/nvcc qgemv_int.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o qgemv_int.out
#include "quant_formats.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

using namespace tmq;

__device__ __forceinline__ int warp_sum_i32(int v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}
__device__ __forceinline__ float warp_sum_f32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---- W8A8: two rows per warp, X loaded once (uint4 = 16 int8), dp4a x4 per row ----
__global__ void qgemv_w8a8(half* D, const int8_t* Wq, const int8_t* Xq,
                           const half* w_scale, const half* a_scale, int N, int K) {
    const int row0 = blockIdx.x * 2;
    const bool two = row0 + 1 < N;
    const int lane = threadIdx.x;
    const uint4* w0 = reinterpret_cast<const uint4*>(Wq + size_t(row0) * K);
    const uint4* w1 = reinterpret_cast<const uint4*>(Wq + size_t(row0 + (two ? 1 : 0)) * K);
    const uint4* xv = reinterpret_cast<const uint4*>(Xq);
    int acc0 = 0, acc1 = 0;
    for (int u = lane; u < K / 16; u += 32) {
        const uint4 x = xv[u];
        const uint4 a = w0[u];
        acc0 = idot4(a.x, x.x, acc0); acc0 = idot4(a.y, x.y, acc0);
        acc0 = idot4(a.z, x.z, acc0); acc0 = idot4(a.w, x.w, acc0);
        const uint4 b = w1[u];
        acc1 = idot4(b.x, x.x, acc1); acc1 = idot4(b.y, x.y, acc1);
        acc1 = idot4(b.z, x.z, acc1); acc1 = idot4(b.w, x.w, acc1);
    }
    for (int k = (K & ~15) + lane * 4; k + 4 <= K; k += 128) {   // K%16 tail (K%4==0)
        const unsigned x = reinterpret_cast<const unsigned*>(Xq)[k / 4];
        acc0 = idot4(reinterpret_cast<const unsigned*>(Wq)[(size_t(row0) * K + k) / 4], x, acc0);
        acc1 = idot4(reinterpret_cast<const unsigned*>(Wq)[(size_t(row0 + (two ? 1 : 0)) * K + k) / 4], x, acc1);
    }
    acc0 = warp_sum_i32(acc0);
    acc1 = warp_sum_i32(acc1);
    if (lane == 0) {
        D[row0] = __float2half(float(acc0) * __half2float(w_scale[row0]) * __half2float(a_scale[0]));
        if (two) D[row0 + 1] = __float2half(float(acc1) * __half2float(w_scale[row0 + 1]) * __half2float(a_scale[0]));
    }
}

// ---- W2A8 BitNet: block-major walk, per-group scale applied once per 8-code span ----
__global__ void qgemv_w2a8(half* D, const uint8_t* Wq, const int8_t* Xq,
                           const half* a_scale, int N, int K) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int bpr = K / bitnet::block_k;
    const uint8_t* row_base = Wq + size_t(row) * bpr * bitnet::block_bytes;
    constexpr int CPL = 8;                          // codes per lane
    constexpr int LPB = bitnet::block_k / CPL;      // 4 lanes per block
    constexpr int BPI = 32 / LPB;                   // 8 blocks per iteration
    const int b_off = lane / LPB;
    const int col0  = (lane % LPB) * CPL;
    float lane_acc = 0.0f;
    for (int g = b_off; g < bpr; g += BPI) {
        const uint8_t* base = row_base + size_t(g) * bitnet::block_bytes;
        const uint16_t codes = *reinterpret_cast<const uint16_t*>(base + 2 + (col0 >> 2));
        const int8_t* x = Xq + g * bitnet::block_k + col0;
        int isum = 0, ixsum = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            isum += int((codes >> (2 * i)) & 3) * int(x[i]);   // code in {0,1,2}
            ixsum += int(x[i]);                                // subtract the -1 bias
        }
        lane_acc += float(isum - ixsum) * bitnet::gscale(base);
    }
    const float facc = warp_sum_f32(lane_acc);
    if (lane == 0) D[row] = __float2half(facc * __half2float(a_scale[0]));
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

static int check(const char* name, const std::vector<half>& got, const uint8_t* ref_bytes, int N, float ms, double gb) {
    const float* ref = reinterpret_cast<const float*>(ref_bytes);
    double gsum = 0, rsum = 0, gmax = 0;
    for (int i = 0; i < N; i++) {
        double d = std::abs(double(__half2float(got[i])) - double(ref[i]));
        gmax = std::max(gmax, d); gsum += d; rsum += std::abs(double(ref[i]));
    }
    double rel = gsum / std::max(rsum, 1e-30);
    printf("%s: rel %.4f%% max %.4g | %.3f ms  %.1f GB/s  (%s)\n",
           name, 100 * rel, gmax, ms, gb / (ms / 1e3), rel < 0.02 ? "PASS" : "FAIL");
    return rel < 0.02 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <golden_int_dir>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    int N, K;
    FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
    if (!f || fscanf(f, "%d %d", &N, &K) != 2) { fprintf(stderr, "bad meta\n"); return 2; }
    fclose(f);
    printf("== qgemv_int  N=%d K=%d\n", N, K);
    int rc = 0;

    { // W8A8
        auto Wq = read_file(dir + "/Wq8.bin"), Xq = read_file(dir + "/Xq8.bin");
        auto ws = read_file(dir + "/w_scale.bin"), as = read_file(dir + "/a_scale.bin");
        auto Dr = read_file(dir + "/D_w8a8_ref.bin");
        int8_t *dW, *dX; half *dws, *das, *dD;
        cudaMalloc(&dW, Wq.size()); cudaMalloc(&dX, Xq.size());
        cudaMalloc(&dws, ws.size()); cudaMalloc(&das, as.size()); cudaMalloc(&dD, sizeof(half) * N);
        cudaMemcpy(dW, Wq.data(), Wq.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dX, Xq.data(), Xq.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dws, ws.data(), ws.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(das, as.data(), as.size(), cudaMemcpyHostToDevice);
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        qgemv_w8a8<<<(N + 1) / 2, 32>>>(dD, dW, dX, dws, das, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int i = 0; i < 200; i++) qgemv_w8a8<<<(N + 1) / 2, 32>>>(dD, dW, dX, dws, das, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 200;
        std::vector<half> got(N);
        cudaMemcpy(got.data(), dD, sizeof(half) * N, cudaMemcpyDeviceToHost);
        rc |= check("w8a8", got, Dr.data(), N, ms, (double(N) * K + K + 2.0 * N) / 1e9);
    }
    { // W2A8 bitnet
        auto Wq = read_file(dir + "/Wq2.bin"), Xq = read_file(dir + "/Xq8.bin");
        auto as = read_file(dir + "/a_scale.bin");
        auto Dr = read_file(dir + "/D_w2a8_ref.bin");
        uint8_t* dW; int8_t* dX; half *das, *dD;
        cudaMalloc(&dW, Wq.size()); cudaMalloc(&dX, Xq.size());
        cudaMalloc(&das, as.size()); cudaMalloc(&dD, sizeof(half) * N);
        cudaMemcpy(dW, Wq.data(), Wq.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dX, Xq.data(), Xq.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(das, as.data(), as.size(), cudaMemcpyHostToDevice);
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        qgemv_w2a8<<<N, 32>>>(dD, dW, dX, das, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int i = 0; i < 200; i++) qgemv_w2a8<<<N, 32>>>(dD, dW, dX, das, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 200;
        std::vector<half> got(N);
        cudaMemcpy(got.data(), dD, sizeof(half) * N, cudaMemcpyDeviceToHost);
        rc |= check("w2a8", got, Dr.data(), N, ms, (double(N) * K / 32 * 10 + K + 2.0 * N) / 1e9);
    }
    return rc;
}
