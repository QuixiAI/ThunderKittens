// Smoke test: based lin_attn accumulate_a0 on SM86.
// o[t] should become the running column-cumsum of v across 8 tiles x 16 rows, over 2 "blocks".
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++20 --extended-lambda --expt-relaxed-constexpr \
//     -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 -I../../include \
//     -o sm80_a0_smoke.out sm80_a0_smoke.cu
#include "kittens.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace kittens;
#define ACTIVE_TILES 8

using vgl = gl<bf16, 1, 1, 256, 64>;   // 2 blocks x 8 tiles x 16 rows
using ogl = gl<bf16, 1, 1, 256, 64>;

template<kittens::ducks::st::all ST, int N_TILES>
__device__ void accumulate_a0(ST (&o)[N_TILES], sv_fl<ST::cols> &running_sum, const ST (&v)[N_TILES]) {
    float acc;
    if(threadIdx.x < ST::cols) {
        int col = threadIdx.x;
        acc = running_sum[col];
        #pragma unroll
        for(int t = 0; t < N_TILES; t++) {
            #pragma unroll
            for(int i = 0; i < ST::rows; i++) {
                acc += __bfloat162float(v[t][int2{i, col}]);
                o[t][int2{i, col}] += __float2bfloat16(acc);
            }
        }
        running_sum[col] = acc;
    }
}

__global__ void smoke(const __grid_constant__ vgl vg, const __grid_constant__ ogl og) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<16,64> (&v_s)[ACTIVE_TILES] = al.allocate<st_bf<16,64>, ACTIVE_TILES>();
    st_bf<16,64> (&o_s)[ACTIVE_TILES] = al.allocate<st_bf<16,64>, ACTIVE_TILES>();
    sv_fl<64> &a0_total = al.allocate<sv_fl<64>>();

    int warpid = kittens::warpid();
    if(warpid == 0) kittens::warp::zero(a0_total);
    __syncthreads();

    for(int block = 0; block < 2; block++) {
        if(warpid >= 8) {
            kittens::warp::load(v_s[warpid-8], vg, {0, 0, block*ACTIVE_TILES + warpid - 8, 0});
        }
        if(warpid < 8) {
            rt_fl<16,64> o;
            kittens::warp::zero(o);
            kittens::warp::store(o_s[warpid], o);
        }
        __syncthreads();
        accumulate_a0(o_s, a0_total, v_s);
        __syncthreads();
        if(warpid < 8) {
            kittens::warp::store(og, o_s[warpid], {0, 0, block*ACTIVE_TILES + warpid, 0});
        }
        __syncthreads();
    }
}

int main() {
    srand(3);
    const int NV = 256*64;
    float vh[NV];
    for(auto &x : vh) x = (rand()%1000 - 500)/1000.f;
    bf16 vb[NV];
    for(int i=0;i<NV;i++){ vb[i]=__float2bfloat16(vh[i]); vh[i]=__bfloat162float(vb[i]); }

    bf16 *vd, *od;
    cudaMalloc(&vd, sizeof(vb)); cudaMalloc(&od, sizeof(vb));
    cudaMemcpy(vd, vb, sizeof(vb), cudaMemcpyHostToDevice);
    cudaMemset(od, 0, sizeof(vb));

    vgl vg{vd, nullptr, nullptr, nullptr, nullptr};
    ogl og{od, nullptr, nullptr, nullptr, nullptr};
    int smem = 16*2048 + 512;
    cudaFuncSetAttribute(smoke, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    smoke<<<1, 16*32, smem>>>(vg, og);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ printf("KERNEL ERR %s\n", cudaGetErrorString(err)); return 1; }

    bf16 ob[NV];
    cudaMemcpy(ob, od, sizeof(ob), cudaMemcpyDeviceToHost);

    int bad=0; double sd=0;
    float acc[64] = {0};
    for(int r=0;r<256;r++) {
        for(int c=0;c<64;c++) {
            acc[c] += vh[r*64+c];
            float got = __bfloat162float(ob[r*64+c]);
            float diff = fabsf(got-acc[c]);
            sd += diff;
            if(diff > 0.06f*fmaxf(0.5f,fabsf(acc[c]))) {
                if(bad < 8) printf("o[%3d][%2d] got %9.5f want %9.5f\n", r, c, got, acc[c]);
                bad++;
            }
        }
    }
    printf("accumulate_a0: %s (%d bad, avg diff %.5f)\n", bad? "FAIL":"PASS", bad, sd/(256*64.0));
    return bad ? 1 : 0;
}
