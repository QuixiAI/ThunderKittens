// Mirror of based lin_attn_ampere only_t0 flow: exact smem layout + warp roles,
// a0 path only. Bisect which surrounding machinery breaks accumulate_a0.
// Build with -DSTAGE=<n>:
//   0: allocations only (q_s/k_s/a1_s/a2_o_accum present but untouched)
//   1: + q/k loads by warps 0-7
//   2: + local q/k/v register loads + transpose/swap (no stores)
//   3: + a1 store to ring + cumsum_inplace
//   /usr/local/cuda/bin/nvcc -std=c++20 --extended-lambda --expt-relaxed-constexpr \
//     -DKITTENS_SM86 -DSTAGE=3 -gencode arch=compute_86,code=sm_86 -I../../include \
//     -o sm80_a0_mirror_smoke.out sm80_a0_mirror_smoke.cu
#include "kittens.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace kittens;
#define NUM_WORKERS  (16)
#define ACTIVE_TILES (8)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

using qgl = gl<bf16, 1, 1, -1, 16>;
using vgl = gl<bf16, 1, 1, -1, 64>;
using ogl = gl<bf16, 1, 1, -1, 64>;

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

template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void cumsum_inplace(ST (&x)[N_TILES], int total_block_idx) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;
    for(int i = 1; i < N_TILES; i++) {
        #pragma unroll
        for(int j = threadIdx.x; j < ST::num_elements; j+=STRIDE) {
            x[(total_block_idx+i)%N_TILES].data[j] += x[(total_block_idx+i-1)%N_TILES].data[j];
        }
    }
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void smoke(const __grid_constant__ qgl qg, const __grid_constant__ qgl kg,
           const __grid_constant__ vgl vg, const __grid_constant__ ogl og, int n) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<16,16> (&q_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,16>, ACTIVE_TILES>();
    st_bf<16,16> (&k_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,16>, ACTIVE_TILES>();
    st_bf<16,64> (&v_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,64>, ACTIVE_TILES>();
    st_bf<16,64> (&o_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,64>, ACTIVE_TILES>();
    st_bf<16,64> (&a1_s)[ACTIVE_TILES + 1]  = al.allocate<st_bf<16,64>, ACTIVE_TILES + 1>();
    st_bf<16,64> (&a2_o_accum)[NUM_WORKERS] = al.allocate<st_bf<16,64>, NUM_WORKERS>();
    int total_block_idx = 0;
    sv_fl<64> &a0_total = al.allocate<sv_fl<64>>();

    int warpid = kittens::warpid();
    if (warpid == 0) kittens::warp::zero(a0_total);
    if (warpid < ACTIVE_TILES + 1) kittens::warp::zero(a1_s[warpid]);

    int n_blocks = n / (ACTIVE_TILES * 16);
    for (int block = 0; block < n_blocks; block++) {
        rt_bf<16, 16> q, k;
        rt_bf<16, 64> v;
        rt_fl<16, 16> local_attn;
        rt_fl<16, 64> o, accum;
        int cur_idx;
        if(warpid < ACTIVE_TILES) {
            cur_idx = block*ACTIVE_TILES + warpid;
#if STAGE >= 1
            kittens::warp::load(q_s[warpid], qg, {0, 0, cur_idx, 0});
            kittens::warp::load(k_s[warpid], kg, {0, 0, cur_idx, 0});
#endif
        } else {
            cur_idx = block*ACTIVE_TILES + warpid - ACTIVE_TILES;
            kittens::warp::load(v_s[warpid-8], vg, {0, 0, cur_idx, 0});
        }
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
#if STAGE >= 2
            kittens::warp::load(q, q_s[warpid]);
            kittens::warp::load(k, k_s[warpid]);
            kittens::warp::zero(local_attn);
            kittens::warp::mma_ABt(local_attn, q, k, local_attn);
            kittens::warp::load(v, v_s[warpid]);
            auto &v_col = kittens::warp::swap_layout_inplace(v);
#endif
            kittens::warp::zero(o);
#if STAGE >= 3
            kittens::warp::zero(accum);
            auto &kt = kittens::warp::transpose_inplace(k);
            kittens::warp::mma_AB(accum, kt, v_col, accum);
            kittens::warp::store(a1_s[(total_block_idx+warpid+1)%(ACTIVE_TILES+1)], accum);
#endif
        }
        __syncthreads();
#if STAGE >= 3
        cumsum_inplace<NUM_WORKERS>(a1_s, total_block_idx);
        __syncthreads();
#endif
        if(warpid < ACTIVE_TILES) {
            kittens::warp::store(o_s[warpid], o);
        }
        total_block_idx = (total_block_idx+ACTIVE_TILES)%(ACTIVE_TILES+1);
        __syncthreads();

        accumulate_a0(o_s, a0_total, v_s);
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            kittens::warp::store(og, o_s[warpid], {0, 0, cur_idx, 0});
        }
        __syncthreads();
    }
}

int main() {
    srand(3);
    const int N = 1024;
    const int NQ = N*16, NV = N*64;
    static float qh[NQ], kh[NQ], vh[NV];
    for(auto &x : qh) x = (rand()%1000 - 500)/1000.f;
    for(auto &x : kh) x = (rand()%1000 - 500)/1000.f;
    for(auto &x : vh) x = (rand()%1000 - 500)/1000.f;
    static bf16 qb[NQ], kb[NQ], vb[NV];
    for(int i=0;i<NQ;i++){ qb[i]=__float2bfloat16(qh[i]); qh[i]=__bfloat162float(qb[i]); }
    for(int i=0;i<NQ;i++) kb[i]=__float2bfloat16(kh[i]);
    for(int i=0;i<NV;i++){ vb[i]=__float2bfloat16(vh[i]); vh[i]=__bfloat162float(vb[i]); }

    bf16 *qd, *kd, *vd, *od;
    cudaMalloc(&qd, sizeof(qb)); cudaMalloc(&kd, sizeof(kb));
    cudaMalloc(&vd, sizeof(vb)); cudaMalloc(&od, sizeof(vb));
    cudaMemcpy(qd, qb, sizeof(qb), cudaMemcpyHostToDevice);
    cudaMemcpy(kd, kb, sizeof(kb), cudaMemcpyHostToDevice);
    cudaMemcpy(vd, vb, sizeof(vb), cudaMemcpyHostToDevice);
    cudaMemset(od, 0, sizeof(vb));

    qgl qg{qd, nullptr, nullptr, N, nullptr};
    qgl kg{kd, nullptr, nullptr, N, nullptr};
    vgl vg{vd, nullptr, nullptr, N, nullptr};
    ogl og{od, nullptr, nullptr, N, nullptr};
    unsigned long smem = kittens::MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(smoke, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    smoke<<<1, NUM_THREADS, smem>>>(qg, kg, vg, og, N);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ printf("KERNEL ERR %s\n", cudaGetErrorString(err)); return 1; }

    static bf16 ob[NV];
    cudaMemcpy(ob, od, sizeof(ob), cudaMemcpyDeviceToHost);

    int bad=0; double sd=0;
    float acc[64] = {0};
    for(int r=0;r<N;r++) {
        for(int c=0;c<64;c++) {
            acc[c] += vh[r*64+c];
            float got = __bfloat162float(ob[r*64+c]);
            float diff = fabsf(got-acc[c]);
            sd += diff;
            if(diff > 0.08f*fmaxf(1.0f,fabsf(acc[c]))) {
                if(bad < 5) printf("o[%3d][%2d] got %9.5f want %9.5f\n", r, c, got, acc[c]);
                bad++;
            }
        }
    }
    printf("STAGE %d a0 mirror: %s (%d bad, avg diff %.5f)\n", STAGE, bad? "FAIL":"PASS", bad, sd/(double)(N*64));
    return bad ? 1 : 0;
}
