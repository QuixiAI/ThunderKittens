// Runtime (GPU-side) activation quantizers, CUDA/SM86 port of ThunderMittens
// kernels/quant_rt. The only quantizers that run on-device (weights quantize
// offline in quant.py):
//   quantize_per_token_{fp8,int8}: one warp per row, absmax -> scale=absmax/QMAX,
//                                  RNE encode. For W8A8 / fp8-KV ingest.
//   quantize_per_tensor_{fp8,int8}: two-pass; pass 1 = global absmax via atomicMax
//                                   on an order-preserving float->uint mapping.
//
// Encoders are RNE ports of TM's tk_e4m3_encode / tk_int8_encode (numpy-rint
// compatible: rintf = round-half-even, matching np.rint).
//
// Build:
//   /usr/local/cuda/bin/nvcc quant_rt.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o quant_rt.out
#include "quant_formats.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

using namespace tmq;

// float -> e4m3 code, round-to-nearest-even, clamp to +-448 (0x7E). Software, exact.
__device__ __forceinline__ uint8_t e4m3_encode(float x) {
    const unsigned sign = (x < 0.0f) ? 0x80u : 0x00u;
    float a = fabsf(x);
    if (!(a < 448.0f)) return uint8_t(sign | 0x7Eu);       // clamp (also catches NaN)
    if (a < 0.0009765625f) {                               // < 2^-10: rounds to 0 or smallest sub
        int mant = int(rintf(a * 512.0f));                 // a / 2^-9
        return uint8_t(sign | unsigned(mant));             // 0 or 1
    }
    int e;
    frexpf(a, &e);
    int E = e - 1;                                          // a in [2^E, 2^(E+1))
    if (E < -6) {                                           // subnormal range
        int mant = int(rintf(a * 512.0f));
        if (mant >= 8) return uint8_t(sign | (1u << 3));    // promote to smallest normal
        return uint8_t(sign | unsigned(mant));
    }
    float two_m = a / exp2f(float(E));                      // [1,2)
    int mant = int(rintf((two_m - 1.0f) * 8.0f));
    int exp = E + 7;
    if (mant >= 8) { mant = 0; exp += 1; }
    if (exp > 15 || (exp == 15 && mant > 6)) return uint8_t(sign | 0x7Eu);
    return uint8_t(sign | (unsigned(exp) << 3) | unsigned(mant));
}
// float -> symmetric int8 [-127,127], RNE (matches np.rint).
__device__ __forceinline__ int8_t int8_encode(float x) {
    float r = rintf(x);
    r = fminf(fmaxf(r, -127.0f), 127.0f);
    return int8_t(int(r));
}

__device__ __forceinline__ float warp_max_f32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// ---- per-token (per row of A (T,D)): one warp per row ----
template<bool FP8>
__global__ void quantize_per_token(uint8_t* codes, float* scale, const float* A, int T, int D) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* row = A + size_t(t) * D;
    float amax = 0.0f;
    for (int d = lane; d < D; d += 32) amax = fmaxf(amax, fabsf(row[d]));
    amax = warp_max_f32(amax);
    const float QMAX = FP8 ? 448.0f : 127.0f;
    float s = amax / QMAX;
    float inv = (s == 0.0f) ? 1.0f : 1.0f / s;
    if (lane == 0) scale[t] = s;
    for (int d = lane; d < D; d += 32) {
        float v = row[d] * inv;
        codes[size_t(t) * D + d] = FP8 ? e4m3_encode(v) : uint8_t(int8_encode(v));
    }
}

// ---- per-tensor: pass 1 absmax via atomicMax on order-preserving uint mapping ----
__device__ __forceinline__ unsigned float_to_ordered(float f) {
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ordered_to_float(unsigned u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}
__global__ void tensor_absmax(unsigned* omax, const float* A, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    float m = 0.0f;
    for (size_t j = i * 16; j < min(n, (i + 1) * 16); j++) m = fmaxf(m, fabsf(A[j]));
    m = warp_max_f32(m);
    if ((threadIdx.x & 31) == 0) atomicMax(omax, float_to_ordered(m));
}
template<bool FP8>
__global__ void tensor_encode(uint8_t* codes, float* scale, const unsigned* omax, const float* A, size_t n) {
    const float QMAX = FP8 ? 448.0f : 127.0f;
    float s = ordered_to_float(*omax) / QMAX;
    float inv = (s == 0.0f) ? 1.0f : 1.0f / s;
    if (blockIdx.x == 0 && threadIdx.x == 0) *scale = s;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (size_t j = i * 16; j < min(n, (i + 1) * 16); j++) {
        float v = A[j] * inv;
        codes[j] = FP8 ? e4m3_encode(v) : uint8_t(int8_encode(v));
    }
}

// ---- harness: random A, check scale exactness + round-trip reconstruction ----
int main() {
    const int T = 129, D = 513;                    // deliberately non-multiples of 32
    const size_t n = size_t(T) * D;
    std::vector<float> A(n);
    srand(9);
    for (auto& x : A) x = (rand() % 20000 - 10000) / 700.0f;
    float* dA; uint8_t* dC; float* dS; unsigned* dM;
    cudaMalloc(&dA, n * 4); cudaMalloc(&dC, n); cudaMalloc(&dS, sizeof(float) * T); cudaMalloc(&dM, 4);
    cudaMemcpy(dA, A.data(), n * 4, cudaMemcpyHostToDevice);

    int rc = 0;
    auto run = [&](bool fp8, bool per_tensor) {
        if (per_tensor) {
            cudaMemset(dM, 0, 4);
            int blocks = int((n + 16 * 256 - 1) / (16 * 256));
            tensor_absmax<<<blocks, 256>>>(dM, dA, n);
            if (fp8) tensor_encode<true><<<blocks, 256>>>(dC, dS, dM, dA, n);
            else     tensor_encode<false><<<blocks, 256>>>(dC, dS, dM, dA, n);
        } else {
            if (fp8) quantize_per_token<true><<<T, 32>>>(dC, dS, dA, T, D);
            else     quantize_per_token<false><<<T, 32>>>(dC, dS, dA, T, D);
        }
        cudaDeviceSynchronize();
        std::vector<uint8_t> C(n); std::vector<float> S(T);
        cudaMemcpy(C.data(), dC, n, cudaMemcpyDeviceToHost);
        cudaMemcpy(S.data(), dS, sizeof(float) * T, cudaMemcpyDeviceToHost);
        const float QMAX = fp8 ? 448.0f : 127.0f;
        double maxerr = 0; int badscale = 0;
        int rows = per_tensor ? 1 : T;
        for (int t = 0; t < rows; t++) {
            // scale exactness: absmax/QMAX in fp32
            float amax = 0;
            size_t lo = per_tensor ? 0 : size_t(t) * D, hi = per_tensor ? n : lo + D;
            for (size_t j = lo; j < hi; j++) amax = std::max(amax, std::abs(A[j]));
            if (S[t] != amax / QMAX) badscale++;
            float s = S[t] == 0 ? 1.0f : S[t];
            for (size_t j = lo; j < hi; j++) {
                float dec = fp8 ? [&]{ // decode e4m3 on host
                        uint8_t v = C[j]; float mag;
                        if (v & 0x78) { int e = (v >> 3) & 0xF, m = v & 7; mag = std::ldexp(1.0f + m / 8.0f, e - 7); }
                        else mag = (v & 7) * 0.001953125f;
                        return (v & 0x80) ? -mag : mag; }()
                    : float(int8_t(C[j]));
                // quantization error bound: half a code step
                float q = std::abs(A[j]) / s;
                float step = fp8 ? std::max(std::ldexp(1.0f / 8.0f, std::max(int(std::floor(std::log2(std::max(q, 1e-9f)))), -6)), 0.001953125f)
                                 : 1.0f;
                maxerr = std::max(maxerr, double(std::abs(dec * s - A[j]) / (s * step)));
            }
        }
        const char* name = fp8 ? (per_tensor ? "per_tensor_fp8" : "per_token_fp8")
                               : (per_tensor ? "per_tensor_int8" : "per_token_int8");
        printf("%s: scale %s, max err %.3f half-steps (%s)\n", name,
               badscale ? "MISMATCH" : "exact", maxerr, (badscale == 0 && maxerr <= 0.5001) ? "PASS" : "FAIL");
        rc |= !(badscale == 0 && maxerr <= 0.5001);
    };
    run(false, false); run(true, false); run(false, true); run(true, true);
    return rc;
}
