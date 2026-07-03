// Smoke test: based lin_attn LOCAL term per-warp, mirroring kernel structure exactly.
// 16 warps: warps 0-7 load q/k tiles, warps 8-15 load v tiles; each of warps 0-7 computes
// o_w = causal(qk^T + 0.5(qk^T)^2) @ v for ITS tile and stores to global.
// Checks whether per-warp (warpid>0) load/store paths are broken.
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++20 --extended-lambda --expt-relaxed-constexpr \
//     -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 -I../../include \
//     -o sm80_based_local_smoke.out sm80_based_local_smoke.cu
#include "kittens.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace kittens;
#define ACTIVE_TILES 8

using qgl = gl<bf16, 1, 1, 128, 16>;   // 8 tiles of 16x16
using vgl = gl<bf16, 1, 1, 128, 64>;   // 8 tiles of 16x64
using ogl = gl<bf16, 1, 1, 128, 64>;

__global__ void smoke(const __grid_constant__ qgl qg, const __grid_constant__ qgl kg,
                      const __grid_constant__ vgl vg, const __grid_constant__ ogl og) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<16,16> (&q_s)[ACTIVE_TILES] = al.allocate<st_bf<16,16>, ACTIVE_TILES>();
    st_bf<16,16> (&k_s)[ACTIVE_TILES] = al.allocate<st_bf<16,16>, ACTIVE_TILES>();
    st_bf<16,64> (&v_s)[ACTIVE_TILES] = al.allocate<st_bf<16,64>, ACTIVE_TILES>();
    st_bf<16,64> (&o_s)[ACTIVE_TILES] = al.allocate<st_bf<16,64>, ACTIVE_TILES>();

    int warpid = kittens::warpid();
    if(warpid < ACTIVE_TILES) {
        kittens::warp::load(q_s[warpid], qg, {0, 0, warpid, 0});
        kittens::warp::load(k_s[warpid], kg, {0, 0, warpid, 0});
    } else {
        kittens::warp::load(v_s[warpid-8], vg, {0, 0, warpid-ACTIVE_TILES, 0});
    }
    __syncthreads();

    if(warpid < ACTIVE_TILES) {
        rt_bf<16,16> q, k, local_attn_bf;
        rt_fl<16,16> local_attn, temp_attn_accum;
        rt_bf<16,64> v;
        rt_fl<16,64> o;

        kittens::warp::load(q, q_s[warpid]);
        kittens::warp::load(k, k_s[warpid]);
        kittens::warp::zero(local_attn);
        kittens::warp::mma_ABt(local_attn, q, k, local_attn);
        kittens::warp::copy(temp_attn_accum, local_attn);
        kittens::warp::mul(temp_attn_accum, temp_attn_accum, temp_attn_accum);
        kittens::warp::mul(temp_attn_accum, temp_attn_accum, 0.5f);
        kittens::warp::add(temp_attn_accum, temp_attn_accum, local_attn);
        kittens::warp::copy(local_attn_bf, temp_attn_accum);
        kittens::warp::apply(local_attn_bf, local_attn_bf, [](int row, int col, float val) { return row >= col ? val : 0.0f; });
        kittens::warp::load(v, v_s[warpid]);
        auto &v_col = kittens::warp::swap_layout_inplace(v);
        kittens::warp::zero(o);
        kittens::warp::mma_AB(o, local_attn_bf, v_col, o);
        kittens::warp::store(o_s[warpid], o);       // the rt_fl -> st_bf mixed store
        __syncwarp();
        kittens::warp::store(og, o_s[warpid], {0, 0, warpid, 0});
    }
}

int main() {
    srand(11);
    const int NQ = 128*16, NV = 128*64;
    float qh[NQ], kh[NQ], vh[NV];
    for(auto &x : qh) x = (rand()%1000 - 500)/1000.f;
    for(auto &x : kh) x = (rand()%1000 - 500)/1000.f;
    for(auto &x : vh) x = (rand()%1000 - 500)/1000.f;
    bf16 qb[NQ], kb[NQ], vb[NV];
    for(int i=0;i<NQ;i++){ qb[i]=__float2bfloat16(qh[i]); qh[i]=__bfloat162float(qb[i]); }
    for(int i=0;i<NQ;i++){ kb[i]=__float2bfloat16(kh[i]); kh[i]=__bfloat162float(kb[i]); }
    for(int i=0;i<NV;i++){ vb[i]=__float2bfloat16(vh[i]); vh[i]=__bfloat162float(vb[i]); }

    bf16 *qd, *kd, *vd, *od;
    cudaMalloc(&qd, sizeof(qb)); cudaMalloc(&kd, sizeof(kb));
    cudaMalloc(&vd, sizeof(vb)); cudaMalloc(&od, sizeof(vb));
    cudaMemcpy(qd, qb, sizeof(qb), cudaMemcpyHostToDevice);
    cudaMemcpy(kd, kb, sizeof(kb), cudaMemcpyHostToDevice);
    cudaMemcpy(vd, vb, sizeof(vb), cudaMemcpyHostToDevice);
    cudaMemset(od, 0, sizeof(vb));

    qgl qg{qd, nullptr, nullptr, nullptr, nullptr};
    qgl kg{kd, nullptr, nullptr, nullptr, nullptr};
    vgl vg{vd, nullptr, nullptr, nullptr, nullptr};
    ogl og{od, nullptr, nullptr, nullptr, nullptr};

    int smem = 8*(512+512+2048+2048) + 4096;
    cudaFuncSetAttribute(smoke, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    smoke<<<1, 16*32, smem>>>(qg, kg, vg, og);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ printf("KERNEL ERR %s\n", cudaGetErrorString(err)); return 1; }

    bf16 ob[NV];
    cudaMemcpy(ob, od, sizeof(ob), cudaMemcpyDeviceToHost);

    int totalbad = 0;
    for(int t = 0; t < 8; t++) {
        // CPU ref for tile t
        float S[16*16];
        for(int i=0;i<16;i++) for(int j=0;j<16;j++) {
            float s=0;
            for(int d=0;d<16;d++) s += qh[(t*16+i)*16+d]*kh[(t*16+j)*16+d];
            s = s + 0.5f*s*s;
            S[i*16+j] = (i>=j) ? __bfloat162float(__float2bfloat16(s)) : 0.f;
        }
        int bad=0; double sd=0;
        for(int i=0;i<16;i++) for(int c=0;c<64;c++) {
            float want=0;
            for(int j=0;j<16;j++) want += S[i*16+j]*vh[(t*16+j)*64+c];
            float got = __bfloat162float(ob[(t*16+i)*64+c]);
            float diff = fabsf(got-want);
            sd += diff;
            if(diff > 0.05f*fmaxf(0.5f,fabsf(want))) {
                if(bad < 3) printf("  tile %d o[%2d][%2d] got %9.5f want %9.5f\n", t, i, c, got, want);
                bad++;
            }
        }
        printf("tile %d (warp %d): %s (%d bad, avg diff %.5f)\n", t, t, bad? "FAIL":"PASS", bad, sd/(16*64));
        totalbad += bad;
    }
    return totalbad ? 1 : 0;
}
