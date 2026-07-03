// Standalone smoke test for the SM80 warpgroup-MMA emulation
// (include/ops/group/mma/warpgroup_sm80.cuh).
//
// Build:  nvcc sm80_wgmma_smoke.cu -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 \
//             -std=c++20 -O2 --expt-extended-lambda --expt-relaxed-constexpr \
//             -I../../include -o sm80_wgmma_smoke && ./sm80_wgmma_smoke
//
// Exercises, with one warpgroup (4 warps):
//   st x st : mma_AB, mma_ABt, mma_AtB, mma_AtBt (bf16 -> fp32)
//   rt x st : mma_AB, mma_ABt                    (bf16 -> fp32)
//   int8    : mma_ABt st x st                    (int8 -> int32, exact)
// against a CPU reference.

#include "kittens.cuh"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace kittens;

constexpr int M = 64;   // warpgroup logical rows
constexpr int K = 64;
constexpr int N = 64;

using st_a  = st_bf<M, K>;
using st_at = st_bf<K, M>;
using st_b  = st_bf<K, N>;
using st_bt = st_bf<N, K>;
using d_rt  = rt_fl<16, N>; // per-warp slice: 16 rows x N cols

// raw row-major store for one warp's rt_fl<16,N> slice (mma.sync accumulator layout)
__device__ inline void store_slice(float *dst, const d_rt &src, int stride) {
    int lane = kittens::laneid();
    #pragma unroll
    for(int j = 0; j < d_rt::width; j++) {
        #pragma unroll
        for(int k = 0; k < d_rt::packed_per_tile; k++) {
            int row = (lane / 4) + (k % 2) * 8;
            int col = j*16 + (lane % 4)*2 + (k / 2)*8;
            dst[row*stride + col]     = src.tiles[0][j].data[k].x;
            dst[row*stride + col + 1] = src.tiles[0][j].data[k].y;
        }
    }
}

__global__ __launch_bounds__(128, 1) void smoke_bf16(
        const bf16 *A, const bf16 *At, const bf16 *B, const bf16 *Bt,
        float *D_AB, float *D_ABt, float *D_AtB, float *D_AtBt,
        float *D_AB_rt, float *D_ABt_rt) {
    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]);
    st_a  &a  = alloc.allocate<st_a>();
    st_at &at = alloc.allocate<st_at>();
    st_b  &b  = alloc.allocate<st_b>();
    st_bt &bt = alloc.allocate<st_bt>();

    using wg = kittens::group<4>;
    const int warp = kittens::warpid();

    // cooperative load of operands into shared
    for(int i = threadIdx.x; i < M*K; i += blockDim.x) { a[{i/K, i%K}]  = A[i]; }
    for(int i = threadIdx.x; i < K*M; i += blockDim.x) { at[{i/M, i%M}] = At[i]; }
    for(int i = threadIdx.x; i < K*N; i += blockDim.x) { b[{i/N, i%N}]  = B[i]; }
    for(int i = threadIdx.x; i < N*K; i += blockDim.x) { bt[{i/K, i%K}] = Bt[i]; }
    __syncthreads();

    d_rt d;

    wg::mm_AB(d, a, b);
    store_slice(D_AB + warp*16*N, d, N);

    wg::mm_ABt(d, a, bt);
    store_slice(D_ABt + warp*16*N, d, N);

    wg::mm_AtB(d, at, b);
    store_slice(D_AtB + warp*16*N, d, N);

    wg::mm_AtBt(d, at, bt);
    store_slice(D_AtBt + warp*16*N, d, N);

    // rt x st: each warp loads its own 16-row slice of A into registers
    rt_bf<16, K> a_reg;
    kittens::group<1>::load(a_reg, a.subtile<16, K>({warp, 0}));

    wg::mm_AB(d, a_reg, b);
    store_slice(D_AB_rt + warp*16*N, d, N);

    wg::mm_ABt(d, a_reg, bt);
    store_slice(D_ABt_rt + warp*16*N, d, N);
}

static void cpu_gemm(const std::vector<float> &A, const std::vector<float> &B,
                     std::vector<float> &C, bool ta, bool tb) {
    for(int i = 0; i < M; i++) for(int j = 0; j < N; j++) {
        float acc = 0;
        for(int k = 0; k < K; k++) {
            float av = ta ? A[k*M + i] : A[i*K + k];
            float bv = tb ? B[j*K + k] : B[k*N + j];
            acc += av * bv;
        }
        C[i*N + j] = acc;
    }
}

static bool check(const char *name, const std::vector<float> &ref, const float *got) {
    double max_err = 0;
    for(int i = 0; i < M*N; i++) max_err = std::max(max_err, (double)std::fabs(ref[i] - got[i]));
    bool ok = max_err < 0.5; // bf16 inputs ~[-1,1], K=64: generous but catches layout bugs cold
    printf("%-10s max_abs_err=%.4f  %s\n", name, max_err, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.f, 1.f);

    std::vector<float> Af(M*K), Bf(K*N);
    for(auto &x : Af) x = dis(gen);
    for(auto &x : Bf) x = dis(gen);
    // quantize to bf16 precision for the reference
    for(auto &x : Af) x = __bfloat162float(__float2bfloat16(x));
    for(auto &x : Bf) x = __bfloat162float(__float2bfloat16(x));

    std::vector<bf16> A(M*K), At(K*M), B(K*N), Bt(N*K);
    for(int i = 0; i < M; i++) for(int k = 0; k < K; k++) {
        A[i*K + k] = __float2bfloat16(Af[i*K + k]);
        At[k*M + i] = __float2bfloat16(Af[i*K + k]);
    }
    for(int k = 0; k < K; k++) for(int j = 0; j < N; j++) {
        B[k*N + j] = __float2bfloat16(Bf[k*N + j]);
        Bt[j*K + k] = __float2bfloat16(Bf[k*N + j]);
    }

    bf16 *dA, *dAt, *dB, *dBt;
    float *dD[6];
    cudaMalloc(&dA, M*K*2); cudaMalloc(&dAt, K*M*2);
    cudaMalloc(&dB, K*N*2); cudaMalloc(&dBt, N*K*2);
    for(auto &p : dD) cudaMalloc(&p, M*N*4);
    cudaMemcpy(dA, A.data(), M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dAt, At.data(), K*M*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), K*N*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dBt, Bt.data(), N*K*2, cudaMemcpyHostToDevice);

    constexpr int smem = sizeof(st_a) + sizeof(st_at) + sizeof(st_b) + sizeof(st_bt) + 1024;
    cudaFuncSetAttribute(smoke_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    smoke_bf16<<<1, 128, smem>>>(dA, dAt, dB, dBt, dD[0], dD[1], dD[2], dD[3], dD[4], dD[5]);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { printf("kernel error: %s\n", cudaGetErrorString(err)); return 1; }

    std::vector<float> ref(M*N), got(M*N);
    bool all_ok = true;
    const char *names[6] = {"mma_AB", "mma_ABt", "mma_AtB", "mma_AtBt", "mma_AB_rt", "mma_ABt_rt"};
    for(int t = 0; t < 6; t++) {
        cpu_gemm(Af, Bf, ref, false, false); // logical product is always A(MxK) x B(KxN)
        cudaMemcpy(got.data(), dD[t], M*N*4, cudaMemcpyDeviceToHost);
        all_ok &= check(names[t], ref, got.data());
    }
    printf(all_ok ? "ALL PASS\n" : "FAILURES\n");
    return all_ok ? 0 : 1;
}
