// Reproduces the int8_ampere GEMM's exact warp topology at 1 iteration:
// warpgroup 2's warp 0 cp.asyncs A[2] + B tiles, arrives on an mbarrier (32
// lanes); consumer warpgroups 0/1 wait, run the emulated warpgroup::mma_ABt,
// and write results straight to global via fragment math (no smem writeback,
// no store warp) — isolating the producer->consumer input path.
//
// Build: nvcc sm80_int8_pipeline_smoke.cu -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 \
//        -std=c++20 -O2 --expt-extended-lambda --expt-relaxed-constexpr \
//        -I../../include -o sm80_int8_pipeline_smoke && ./sm80_int8_pipeline_smoke

#include "kittens.cuh"
#include <cstdio>
#include <random>
#include <vector>

using namespace kittens;

constexpr int M = 128, N = 64, K = 64; // two 64-row A tiles, one B tile
using st_a = st_int8<64, K>;
using st_b = st_int8<N, K>;
using a_gl = gl<int8, 1, 1, -1, -1, st_a>;
using b_gl = gl<int8, 1, 1, -1, -1, st_b>;
using d_rt = rt<int, 16, N>;
using d_gl = gl<int, 1, 1, -1, -1, st_int<64, N>>;

__global__ __launch_bounds__(384, 1) void pipe_smoke(const __grid_constant__ a_gl ag, const __grid_constant__ b_gl bg, const __grid_constant__ d_gl dg, int *D, int *D2) {
    extern __shared__ int __shm[];
    shared_allocator al(&__shm[0]);
    st_a (&a)[2] = al.allocate<st_a, 2>();
    st_b &b = al.allocate<st_b>();
    __shared__ semaphore arrived;

    const int wgid = warpgroup::groupid();
    if(threadIdx.x == 0) init_semaphore(arrived, WARP_THREADS, 0);
    __syncthreads();

    using st_d = st_int<64, N>;
    st_d (&dsm)[2] = al.allocate<st_d, 2>();
    __shared__ semaphore outputs_arrived;
    if(threadIdx.x == 0) init_semaphore(outputs_arrived, 0, 2);
    __syncthreads();

    if(wgid == 2) {
        if(warpgroup::warpid() == 0) {
            warp::load_async(a[0], ag, {0, 0});
            warp::load_async(a[1], ag, {1, 0});
            warp::load_async(b, bg, {0, 0});
            kittens::load_async_arrive(arrived);
        } else if(warpgroup::warpid() == 1) { // store warp
            wait(outputs_arrived, 0);
#ifdef USE_FENCE
            __threadfence_block();
#endif
            warp::store(dg, dsm[0], {0, 0});
            warp::store(dg, dsm[1], {1, 0});
        }
    } else {
        wait(arrived, 0);
        d_rt d;
        warp::zero(d);
#ifdef PATTERN_FILL
        { // fill d with a position formula instead of mma: value = row*1000 + col
            int lane = kittens::laneid();
            int wrow = wgid*64 + warpgroup::warpid()*16;
            #pragma unroll
            for(int j = 0; j < d_rt::width; j++) {
                #pragma unroll
                for(int k = 0; k < d_rt::packed_per_tile; k++) {
                    int row = wrow + (lane / 4) + (k % 2) * 8;
                    int col = j*16 + (lane % 4)*2 + (k / 2)*8;
                    d.tiles[0][j].data[k].x = row*1000 + col;
                    d.tiles[0][j].data[k].y = row*1000 + col + 1;
                }
            }
        }
#else
        warpgroup::mma_ABt(d, a[wgid], b);
#endif
        warpgroup::store(dsm[wgid], d);
#ifdef USE_FENCE
        __threadfence_block();
#endif
        warpgroup::sync(wgid+1);
        warpgroup::arrive(outputs_arrived);
    }
    // dump dsm via swizzle-aware element reads into D2 (all threads join here)
    __syncthreads();
    for(int i = threadIdx.x; i < M*N; i += blockDim.x) {
        int t = i / (64*N), r = (i / N) % 64, c = i % N;
        D2[i] = dsm[t][{r, c}];
    }
}

int main() {
    std::mt19937 gen(11);
    std::uniform_int_distribution<int> dis(-128, 127);
    std::vector<int8_t> A(M*K), B(N*K);
    for(auto &x : A) x = (int8_t)dis(gen);
    for(auto &x : B) x = (int8_t)dis(gen);

    std::vector<int> ref(M*N);
#ifdef PATTERN_FILL
    for(int i = 0; i < M; i++) for(int j = 0; j < N; j++) ref[i*N+j] = i*1000 + j;
#else
    for(int i = 0; i < M; i++) for(int j = 0; j < N; j++) {
        int acc = 0;
        for(int k = 0; k < K; k++) acc += int(A[i*K+k]) * int(B[j*K+k]);
        ref[i*N+j] = acc;
    }
#endif

    int8 *dA, *dB; int *dD, *dD2;
    cudaMalloc(&dA, M*K); cudaMalloc(&dB, N*K); cudaMalloc(&dD, M*N*4); cudaMalloc(&dD2, M*N*4);
    cudaMemcpy(dA, A.data(), M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N*K, cudaMemcpyHostToDevice);

    constexpr int smem = 2*sizeof(st_a) + sizeof(st_b) + 2*sizeof(st_int<64, N>) + 1024;
    cudaFuncSetAttribute(pipe_smoke, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    a_gl ag{dA, nullptr, nullptr, M, K};
    b_gl bg{dB, nullptr, nullptr, N, K};
    d_gl dg{dD, nullptr, nullptr, M, N};

    for(int trial = 0; trial < 3; trial++) {
        cudaMemset(dD, 0, M*N*4);
        cudaMemset(dD2, 0, M*N*4);
        pipe_smoke<<<1, 384, smem>>>(ag, bg, dg, dD, dD2);
        cudaError_t err = cudaDeviceSynchronize();
        if(err != cudaSuccess) { printf("kernel error: %s\n", cudaGetErrorString(err)); return 1; }
        std::vector<int> got(M*N);
        cudaMemcpy(got.data(), dD, M*N*4, cudaMemcpyDeviceToHost);
        int bad = 0, first = -1;
        for(int i = 0; i < M*N; i++) if(got[i] != ref[i]) { if(first < 0) first = i; bad++; }
        printf("trial %d (via store warp): %d/%d mismatched", trial, bad, M*N);
        if(bad) printf(" (first r%d c%d: got %d want %d)", first/N, first%N, got[first], ref[first]);
        printf("\n");
        std::vector<int> got2(M*N);
        cudaMemcpy(got2.data(), dD2, M*N*4, cudaMemcpyDeviceToHost);
        bad = 0; first = -1;
        for(int i = 0; i < M*N; i++) if(got2[i] != ref[i]) { if(first < 0) first = i; bad++; }
        printf("trial %d (dsm dump):      %d/%d mismatched", trial, bad, M*N);
        if(bad) printf(" (first r%d c%d: got %d want %d)", first/N, first%N, got2[first], ref[first]);
        printf("\n");
        if(trial == 0 && bad) { // where did (0,28)'s value 28 land? and what's around it?
            for(int i = 0; i < M*N; i++) if(got2[i] == 28) printf("value 28 found at r%d c%d\n", i/N, i%N);
            printf("row 0 got:  "); for(int c = 24; c < 36; c++) printf("%d ", got2[c]); printf("\n");
            printf("row 0 want: "); for(int c = 24; c < 36; c++) printf("%d ", ref[c]); printf("\n");
        }
        if(trial == 0 && bad) { // hole map: rows 0..15, mark mismatched cols
            for(int r = 0; r < 16; r++) {
                printf("r%02d: ", r);
                for(int c = 0; c < N; c++) putchar(got2[r*N+c] != ref[r*N+c] ? 'X' : '.');
                printf("\n");
            }
        }
    }
    return 0;
}
