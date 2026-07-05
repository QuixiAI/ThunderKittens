/**
 * @file
 * @brief M0 shared-primitive test: the MetalForge-parity encoders
 * (e2m1_encode / e8m0_encode) round-trip through our decoders, and the
 * nvfp4_sf_offset swizzle matches a CPU replica of MetalForge's formula.
 * Also exercises warp_min_f and block_inclusive_scan_f.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        mf_primitives_test.cu -o mf_primitives_test.out -I../serving
 * Run: CUDA_VISIBLE_DEVICES=0 ./mf_primitives_test.out
 */
#include "quant_formats.cuh"
#include "../serving/tm_warp.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>

using namespace tmq;

static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

// ---- fp4/e8m0 encode round-trip on device ----
__global__ void fp4_roundtrip(const float* in, float* deq_out, uint8_t* code_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t c = e2m1_encode(in[i]);
    code_out[i] = c;
    deq_out[i] = e2m1_decode(c);
}
__global__ void e8m0_roundtrip(const float* in, float* deq_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    deq_out[i] = e8m0_decode(e8m0_encode(in[i]));
}
// ---- warp_min + block inclusive scan ----
__global__ void warpmin_k(const float* in, float* out) {
    float v = in[threadIdx.x];
    v = tms::warp_min_f(v);
    if (threadIdx.x == 0) *out = v;
}
__global__ void scan_k(const float* in, float* out, int n) {
    __shared__ float sg[8];
    float total;
    const float r = tms::block_inclusive_scan_f(threadIdx.x < n ? in[threadIdx.x] : 0.0f,
                                                threadIdx.x, blockDim.x, sg, total);
    if (threadIdx.x < n) out[threadIdx.x] = r;
}

// CPU replica of MetalForge nvfp4_quant_swizzled_sf_offset
static long ref_swizzle(int row, int group, int num_k_tiles) {
    const int m_tile = row >> 7, outer_m = row & 31, inner_m = (row >> 5) & 3;
    const int k_tile = group >> 2, inner_k = group & 3;
    return ((long)(m_tile) * num_k_tiles + k_tile) * 512L + ((outer_m << 4) | (inner_m << 2) | inner_k);
}
__global__ void swizzle_k(long* out, int rows, int groups, int num_k_tiles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * groups) return;
    out[idx] = nvfp4_sf_offset(idx / groups, idx % groups, num_k_tiles);
}

int main() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-8.0f, 8.0f);

    // fp4 round-trip: e2m1_decode(e2m1_encode(x)) must land on the nearest codebook value
    const int N = 100000;
    std::vector<float> in(N);
    for (auto& x : in) x = d(rng);
    float *din, *ddeq; uint8_t* dcode;
    CK(cudaMalloc(&din, N*4)); CK(cudaMalloc(&ddeq, N*4)); CK(cudaMalloc(&dcode, N));
    CK(cudaMemcpy(din, in.data(), N*4, cudaMemcpyHostToDevice));
    fp4_roundtrip<<<(N+255)/256,256>>>(din, ddeq, dcode, N);
    CK(cudaDeviceSynchronize());
    std::vector<float> deq(N); CK(cudaMemcpy(deq.data(), ddeq, N*4, cudaMemcpyDeviceToHost));
    const float cb[8] = {0,0.5f,1,1.5f,2,3,4,6};
    long bad = 0;
    for (int i = 0; i < N; i++) {
        // nearest codebook magnitude to |clamp(x,6)|
        float ax = std::min(std::fabs(in[i]), 6.0f), best = 1e9f, bv = 0;
        for (float c : cb) if (std::fabs(ax - c) < best) { best = std::fabs(ax - c); bv = c; }
        float want = in[i] < 0 ? -bv : bv;
        if (deq[i] != want) bad++;
    }
    printf("e2m1 encode round-trip: %ld/%d nearest-codebook mismatches (%s)\n", bad, N, bad?"FAIL":"PASS");
    g_fail += bad != 0;

    // e8m0: encode(2^k)->decode must reproduce 2^k exactly for normal exponents
    std::vector<float> pin(60);
    for (int i = 0; i < 60; i++) pin[i] = ldexpf(1.0f, i - 30);   // 2^-30 .. 2^29
    float *dp, *dpo; CK(cudaMalloc(&dp, 60*4)); CK(cudaMalloc(&dpo, 60*4));
    CK(cudaMemcpy(dp, pin.data(), 60*4, cudaMemcpyHostToDevice));
    e8m0_roundtrip<<<1,64>>>(dp, dpo, 60);
    CK(cudaDeviceSynchronize());
    std::vector<float> po(60); CK(cudaMemcpy(po.data(), dpo, 60*4, cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < 60; i++) if (po[i] != pin[i]) bad++;
    printf("e8m0 encode round-trip (powers of 2): %ld/60 mismatches (%s)\n", bad, bad?"FAIL":"PASS");
    g_fail += bad != 0;

    // warp_min
    std::vector<float> wv(32);
    for (auto& x : wv) x = d(rng);
    float *dwv, *dwo; CK(cudaMalloc(&dwv, 32*4)); CK(cudaMalloc(&dwo, 4));
    CK(cudaMemcpy(dwv, wv.data(), 32*4, cudaMemcpyHostToDevice));
    warpmin_k<<<1,32>>>(dwv, dwo);
    CK(cudaDeviceSynchronize());
    float wmin; CK(cudaMemcpy(&wmin, dwo, 4, cudaMemcpyDeviceToHost));
    float rmin = wv[0]; for (float x : wv) rmin = std::min(rmin, x);
    printf("warp_min_f: %s\n", wmin == rmin ? "PASS" : "FAIL");
    g_fail += wmin != rmin;

    // block inclusive scan (200 elems, 256-thread block)
    const int S = 200;
    std::vector<float> sv(S);
    for (auto& x : sv) x = d(rng);
    float *dsv, *dso; CK(cudaMalloc(&dsv, S*4)); CK(cudaMalloc(&dso, S*4));
    CK(cudaMemcpy(dsv, sv.data(), S*4, cudaMemcpyHostToDevice));
    scan_k<<<1,256>>>(dsv, dso, S);
    CK(cudaDeviceSynchronize());
    std::vector<float> so(S); CK(cudaMemcpy(so.data(), dso, S*4, cudaMemcpyDeviceToHost));
    double run = 0, worst = 0;
    for (int i = 0; i < S; i++) { run += sv[i]; worst = std::max(worst, std::fabs(so[i] - run)); }
    printf("block_inclusive_scan_f: max abs err %.3e (%s)\n", worst, worst < 1e-3 ? "PASS" : "FAIL");
    g_fail += !(worst < 1e-3);

    // swizzle vs CPU replica
    const int rows = 256, groups = 32, num_k_tiles = groups / 4;
    long* dsw; CK(cudaMalloc(&dsw, (long)rows*groups*8));
    swizzle_k<<<(rows*groups+255)/256,256>>>(dsw, rows, groups, num_k_tiles);
    CK(cudaDeviceSynchronize());
    std::vector<long> sw((size_t)rows*groups);
    CK(cudaMemcpy(sw.data(), dsw, (long)rows*groups*8, cudaMemcpyDeviceToHost));
    bad = 0;
    for (int r = 0; r < rows; r++) for (int gp = 0; gp < groups; gp++)
        if (sw[(size_t)r*groups+gp] != ref_swizzle(r, gp, num_k_tiles)) bad++;
    printf("nvfp4_sf_offset swizzle: %ld/%d mismatches (%s)\n", bad, rows*groups, bad?"FAIL":"PASS");
    g_fail += bad != 0;

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
