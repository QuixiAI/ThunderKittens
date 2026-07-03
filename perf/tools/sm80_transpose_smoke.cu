// Smoke test: rt transpose_inplace + swap_layout_inplace + mma chains on SM80/SM86.
// Repro for based lin_attn_ampere debugging.
// Build:
//   nvcc -std=c++20 -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 \
//     -I../../include -o sm80_transpose_smoke.out sm80_transpose_smoke.cu
#include "kittens.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace kittens;

using kgl  = gl<bf16, 1, 1, 16, 16>;
using vgl  = gl<bf16, 1, 1, 16, 64>;
using o0gl = gl<float, 1, 1, 16, 16>;
using o1gl = gl<float, 1, 1, 16, 64>;

// out0 = k^T (transpose_inplace check)
// out1 = k^T @ v (the based a1 chain: transpose_inplace + swap_layout_inplace + mma_AB)
__global__ void smoke(const __grid_constant__ kgl kg, const __grid_constant__ vgl vg,
                      const __grid_constant__ o0gl o0g, const __grid_constant__ o1gl o1g) {
    __shared__ alignas(1024) st_bf<16,16> k_s;
    __shared__ alignas(1024) st_bf<16,64> v_s;
    __shared__ alignas(1024) st_fl<16,16> kt_s;
    __shared__ alignas(1024) st_fl<16,64> o_s;

    kittens::warp::load(k_s, kg, {0,0,0,0});
    kittens::warp::load(v_s, vg, {0,0,0,0});
    __syncwarp();

    rt_bf<16,16> k;
    rt_bf<16,64> v;
    rt_fl<16,16> ktf;
    rt_fl<16,64> acc;

    kittens::warp::load(k, k_s);
    auto &kt = kittens::warp::transpose_inplace(k);
    kittens::warp::copy(ktf, kt);
    kittens::warp::store(kt_s, ktf);

    kittens::warp::load(v, v_s);
    auto &v_col = kittens::warp::swap_layout_inplace(v);

    kittens::warp::zero(acc);
    kittens::warp::mma_AB(acc, kt, v_col, acc);
    kittens::warp::store(o_s, acc);
    __syncwarp();

    kittens::warp::store(o0g, kt_s, {0,0,0,0});
    kittens::warp::store(o1g, o_s, {0,0,0,0});
}

int main() {
    srand(42);
    float kh[16*16], vh[16*64];
    for(auto &x : kh) x = (rand()%1000 - 500)/250.f;
    for(auto &x : vh) x = (rand()%1000 - 500)/250.f;
    // round through bf16 for exact comparison
    bf16 kb[16*16], vb[16*64];
    for(int i=0;i<16*16;i++){ kb[i]=__float2bfloat16(kh[i]); kh[i]=__bfloat162float(kb[i]); }
    for(int i=0;i<16*64;i++){ vb[i]=__float2bfloat16(vh[i]); vh[i]=__bfloat162float(vb[i]); }

    bf16 *kd, *vd; float *o0d, *o1d;
    cudaMalloc(&kd, sizeof(kb)); cudaMalloc(&vd, sizeof(vb));
    cudaMalloc(&o0d, 16*16*4); cudaMalloc(&o1d, 16*64*4);
    cudaMemcpy(kd, kb, sizeof(kb), cudaMemcpyHostToDevice);
    cudaMemcpy(vd, vb, sizeof(vb), cudaMemcpyHostToDevice);

    kgl  kg{kd, nullptr, nullptr, nullptr, nullptr};
    vgl  vg{vd, nullptr, nullptr, nullptr, nullptr};
    o0gl o0g{o0d, nullptr, nullptr, nullptr, nullptr};
    o1gl o1g{o1d, nullptr, nullptr, nullptr, nullptr};
    smoke<<<1,32>>>(kg, vg, o0g, o1g);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ printf("KERNEL ERR %s\n", cudaGetErrorString(err)); return 1; }

    float o0[16*16], o1[16*64];
    cudaMemcpy(o0, o0d, sizeof(o0), cudaMemcpyDeviceToHost);
    cudaMemcpy(o1, o1d, sizeof(o1), cudaMemcpyDeviceToHost);

    // check transpose
    int bad0 = 0;
    for(int r=0;r<16;r++) for(int c=0;c<16;c++) {
        float want = kh[c*16+r];
        if(fabsf(o0[r*16+c]-want) > 1e-6f) {
            if(bad0 < 5) printf("kt[%d][%d] got %f want %f\n", r, c, o0[r*16+c], want);
            bad0++;
        }
    }
    printf("transpose_inplace: %s (%d bad)\n", bad0? "FAIL":"PASS", bad0);

    // check k^T @ v
    int bad1 = 0;
    for(int r=0;r<16;r++) for(int c=0;c<64;c++) {
        float want = 0;
        for(int i=0;i<16;i++) want += kh[i*16+r]*vh[i*64+c];
        if(fabsf(o1[r*64+c]-want) > 0.05f*fmaxf(1.f,fabsf(want))) {
            if(bad1 < 5) printf("ktv[%d][%d] got %f want %f\n", r, c, o1[r*64+c], want);
            bad1++;
        }
    }
    printf("kt @ v chain:      %s (%d bad)\n", bad1? "FAIL":"PASS", bad1);
    return (bad0||bad1) ? 1 : 0;
}
