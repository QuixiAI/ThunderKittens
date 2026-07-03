// Exact int8 x int8 -> int32 warpgroup mma_ABt emulation test (SM80/86).
// Build: nvcc sm80_int8_smoke.cu -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 \
//        -std=c++20 -O2 --expt-extended-lambda --expt-relaxed-constexpr \
//        -I../../include -o sm80_int8_smoke && ./sm80_int8_smoke

#include "kittens.cuh"
#include <cstdio>
#include <random>
#include <vector>

using namespace kittens;

constexpr int M = 64, N = 64, K = 64;
using st_a = st_int8<M, K>;
using st_b = st_int8<N, K>; // B is N x K for ABt
using d_rt = rt<int, 16, N>;

using a_gl = gl<int8, 1, 1, -1, -1, st_a>;
using b_gl = gl<int8, 1, 1, -1, -1, st_b>;
using st_d = st_int<M, N>;
using d_gl = gl<int, 1, 1, -1, -1, st_d>;
__global__ __launch_bounds__(128, 1) void int8_smoke(const __grid_constant__ a_gl ag, const __grid_constant__ b_gl bg, const __grid_constant__ d_gl dg, const int8 *A, const int8 *B, int *D) {
    extern __shared__ int __shm[];
    shared_allocator al(&__shm[0]);
    st_a &a = al.allocate<st_a>();
    st_b &b = al.allocate<st_b>();

    using wg = kittens::group<4>;
    const int warp = kittens::warpid();

#ifdef USE_LOAD_ASYNC
    if(warp == 0) {
        kittens::group<1>::load_async(a, ag, {0, 0});
        kittens::group<1>::load_async(b, bg, {0, 0});
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
#else
    for(int i = threadIdx.x; i < M*K; i += blockDim.x) a[{i/K, i%K}] = A[i];
    for(int i = threadIdx.x; i < N*K; i += blockDim.x) b[{i/K, i%K}] = B[i];
    __syncthreads();
#endif

    d_rt d;
    wg::mm_ABt(d, a, b);

#ifdef USE_SMEM_WRITEBACK
    // mirror the int8 GEMM kernel's writeback: rt -> st_int -> global
    __syncthreads();
    st_d &dsm = al.allocate<st_d>();
    wg::store(dsm, d);
    wg::sync(2);
    if(warp == 0) kittens::group<1>::store(dg, dsm, {0, 0});
    __syncthreads();
#else
    int lane = kittens::laneid();
    #pragma unroll
    for(int j = 0; j < d_rt::width; j++) {
        #pragma unroll
        for(int k = 0; k < d_rt::packed_per_tile; k++) {
            int row = warp*16 + (lane / 4) + (k % 2) * 8;
            int col = j*16 + (lane % 4)*2 + (k / 2)*8;
            D[row*N + col]     = d.tiles[0][j].data[k].x;
            D[row*N + col + 1] = d.tiles[0][j].data[k].y;
        }
    }
#endif
}

int main() {
    std::mt19937 gen(7);
    std::uniform_int_distribution<int> dis(-128, 127);
    std::vector<int8_t> A(M*K), B(N*K);
    for(auto &x : A) x = (int8_t)dis(gen);
    for(auto &x : B) x = (int8_t)dis(gen);

    std::vector<int> ref(M*N);
    for(int i = 0; i < M; i++) for(int j = 0; j < N; j++) {
        int acc = 0;
        for(int k = 0; k < K; k++) acc += int(A[i*K+k]) * int(B[j*K+k]);
        ref[i*N+j] = acc;
    }

    int8 *dA, *dB; int *dD;
    cudaMalloc(&dA, M*K); cudaMalloc(&dB, N*K); cudaMalloc(&dD, M*N*4);
    cudaMemcpy(dA, A.data(), M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N*K, cudaMemcpyHostToDevice);

    constexpr int smem = sizeof(st_a) + sizeof(st_b) + sizeof(st_d) + 1024;
    cudaFuncSetAttribute(int8_smoke, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    a_gl ag{dA, nullptr, nullptr, M, K};
    b_gl bg{dB, nullptr, nullptr, N, K};
    d_gl dg{dD, nullptr, nullptr, M, N};
    int8_smoke<<<1, 128, smem>>>(ag, bg, dg, dA, dB, dD);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { printf("kernel error: %s\n", cudaGetErrorString(err)); return 1; }

    std::vector<int> got(M*N);
    cudaMemcpy(got.data(), dD, M*N*4, cudaMemcpyDeviceToHost);
    int bad = 0, first = -1;
    for(int i = 0; i < M*N; i++) if(got[i] != ref[i]) { if(first < 0) first = i; bad++; }
    printf("int8 mm_ABt: %d/%d mismatched", bad, M*N);
    if(bad) printf(" (first at r%d c%d: got %d want %d)", first/N, first%N, got[first], ref[first]);
    printf("\n%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
