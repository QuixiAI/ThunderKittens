// Smoke test: based lin_attn cross-tile terms on SM86.
// Computes o_cross = q1 @ (k0^T v0) + sum_w (q1 . q1[:,w]/sqrt2) @ ((k0 . k0[:,w]/sqrt2)^T v0)
// with the kernel's exact register ops, vs CPU reference q1 k0^T v0 + 0.5 (q1 k0^T)^2 v0.
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++20 --extended-lambda --expt-relaxed-constexpr \
//     -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 -I../../include \
//     -o sm80_based_cross_smoke.out sm80_based_cross_smoke.cu
#include "kittens.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace kittens;

using qgl = gl<bf16, 1, 1, 16, 16>;
using vgl = gl<bf16, 1, 1, 16, 64>;
using ogl = gl<float, 1, 1, 16, 64>;

__device__ static void mul_slice(rt_bf<16, 16> &reg) {
    const int target_col = kittens::warpid();
    const int lane       = kittens::laneid();
    #pragma unroll
    for(int row_offset = 0; row_offset < 2; row_offset++) {
        const int src_thread = (lane / 4)*4 + (target_col%8)/2;
        const int col_offset = target_col >= 8;
        bf16_2 src_val = reg.tiles[0][0].data[2*col_offset + row_offset];
        bf16 val = __shfl_sync(kittens::MASK_ALL, (target_col%2 == 0) ? src_val.x : src_val.y, src_thread);
        val *= __float2bfloat16(0.70710678118);
        reg.tiles[0][0].data[row_offset] *= bf16_2{val, val};
        reg.tiles[0][0].data[row_offset+2] *= bf16_2{val, val};
    }
}

__global__ void smoke(const __grid_constant__ qgl qg, const __grid_constant__ qgl kg,
                      const __grid_constant__ vgl vg, const __grid_constant__ ogl og) {
    __shared__ alignas(1024) st_bf<16,16> q_s, k_s;
    __shared__ alignas(1024) st_bf<16,64> v_s;
    __shared__ alignas(1024) st_bf<16,64> partial[16];
    __shared__ alignas(1024) st_fl<16,64> o_s;

    int warpid = kittens::warpid();
    if(warpid == 0) {
        kittens::warp::load(q_s, qg, {0,0,0,0});
        kittens::warp::load(k_s, kg, {0,0,0,0});
    }
    if(warpid == 1) kittens::warp::load(v_s, vg, {0,0,0,0});
    __syncthreads();

    rt_bf<16,16> q, k;
    rt_bf<16,64> v;
    rt_fl<16,64> o, a1, a2;

    // a1 term (warp 0 computes it alone into partial[0..], others contribute a2 slices)
    // Each warp: a2 slice for feature col = warpid.
    kittens::warp::load(k, k_s);
    mul_slice(k);
    auto &kt = kittens::warp::transpose_inplace(k);
    kittens::warp::load(v, v_s);
    auto &v_col = kittens::warp::swap_layout_inplace(v);
    kittens::warp::zero(a2);
    kittens::warp::mma_AB(a2, kt, v_col, a2);        // a2 state slice for this feature col

    kittens::warp::load(q, q_s);
    mul_slice(q);
    rt_bf<16,64> a2_bf;
    kittens::warp::copy(a2_bf, a2);
    auto &a2_bf_col = kittens::warp::swap_layout_inplace(a2_bf);
    kittens::warp::zero(o);
    kittens::warp::mma_AB(o, q, a2_bf_col, o);
    kittens::warp::store(partial[warpid], o);
    __syncthreads();

    // warp 0: a1 term + reduce all partials
    if(warpid == 0) {
        rt_bf<16,16> k2;
        rt_bf<16,64> v2;
        kittens::warp::load(k2, k_s);
        auto &k2t = kittens::warp::transpose_inplace(k2);
        kittens::warp::load(v2, v_s);
        auto &v2_col = kittens::warp::swap_layout_inplace(v2);
        kittens::warp::zero(a1);
        kittens::warp::mma_AB(a1, k2t, v2_col, a1);  // k0^T v0 in fp32
        rt_bf<16,64> a1_bf;
        kittens::warp::copy(a1_bf, a1);              // mimic the bf16 shared ring precision
        auto &a1_col = kittens::warp::swap_layout_inplace(a1_bf);

        rt_bf<16,16> q2;
        kittens::warp::load(q2, q_s);
        kittens::warp::zero(o);
        kittens::warp::mma_AB(o, q2, a1_col, o);     // q1 @ a1

        for(int w = 0; w < 16; w++) {
            rt_bf<16,64> p;
            kittens::warp::load(p, partial[w]);
            rt_fl<16,64> pf;
            kittens::warp::copy(pf, p);
            kittens::warp::add(o, o, pf);
        }
        kittens::warp::store(o_s, o);
        __syncwarp();
        kittens::warp::store(og, o_s, {0,0,0,0});
    }
}

int main() {
    srand(7);
    float qh[16*16], kh[16*16], vh[16*64];
    for(auto &x : qh) x = (rand()%1000 - 500)/1000.f;
    for(auto &x : kh) x = (rand()%1000 - 500)/1000.f;
    for(auto &x : vh) x = (rand()%1000 - 500)/1000.f;
    bf16 qb[16*16], kb[16*16], vb[16*64];
    for(int i=0;i<16*16;i++){ qb[i]=__float2bfloat16(qh[i]); qh[i]=__bfloat162float(qb[i]); }
    for(int i=0;i<16*16;i++){ kb[i]=__float2bfloat16(kh[i]); kh[i]=__bfloat162float(kb[i]); }
    for(int i=0;i<16*64;i++){ vb[i]=__float2bfloat16(vh[i]); vh[i]=__bfloat162float(vb[i]); }

    bf16 *qd, *kd, *vd; float *od;
    cudaMalloc(&qd, sizeof(qb)); cudaMalloc(&kd, sizeof(kb)); cudaMalloc(&vd, sizeof(vb));
    cudaMalloc(&od, 16*64*4);
    cudaMemcpy(qd, qb, sizeof(qb), cudaMemcpyHostToDevice);
    cudaMemcpy(kd, kb, sizeof(kb), cudaMemcpyHostToDevice);
    cudaMemcpy(vd, vb, sizeof(vb), cudaMemcpyHostToDevice);

    qgl qg{qd, nullptr, nullptr, nullptr, nullptr};
    qgl kg{kd, nullptr, nullptr, nullptr, nullptr};
    vgl vg{vd, nullptr, nullptr, nullptr, nullptr};
    ogl og{od, nullptr, nullptr, nullptr, nullptr};
    smoke<<<1,16*32>>>(qg, kg, vg, og);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ printf("KERNEL ERR %s\n", cudaGetErrorString(err)); return 1; }

    float o[16*64];
    cudaMemcpy(o, od, sizeof(o), cudaMemcpyDeviceToHost);

    // CPU reference: S = q k^T (16x16); o = S v + 0.5 S^2. v
    float S[16*16];
    for(int i=0;i<16;i++) for(int j=0;j<16;j++) {
        float s=0; for(int d=0;d<16;d++) s += qh[i*16+d]*kh[j*16+d];
        S[i*16+j]=s;
    }
    int bad=0; double sumdiff=0;
    for(int i=0;i<16;i++) for(int c=0;c<64;c++) {
        float want=0;
        for(int j=0;j<16;j++) want += (S[i*16+j] + 0.5f*S[i*16+j]*S[i*16+j]) * vh[j*64+c];
        float diff = fabsf(o[i*64+c]-want);
        sumdiff += diff;
        if(diff > 0.05f*fmaxf(1.f,fabsf(want))) {
            if(bad < 8) printf("o[%2d][%2d] got %9.5f want %9.5f\n", i, c, o[i*64+c], want);
            bad++;
        }
    }
    printf("cross-tile a1+a2: %s (%d bad, avg diff %.5f)\n", bad? "FAIL":"PASS", bad, sumdiff/(16*64));
    return bad ? 1 : 0;
}
