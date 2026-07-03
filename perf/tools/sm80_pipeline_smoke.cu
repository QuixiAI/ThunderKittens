// Standalone smoke test for the SM80 producer primitives used by the lcf/lcsf
// template ports: group<1>::load_async coordinates + cp.async.mbarrier.arrive.
//
// Build:  nvcc sm80_pipeline_smoke.cu -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 \
//             -std=c++20 -O2 --expt-extended-lambda --expt-relaxed-constexpr \
//             -I../../include -o sm80_pipeline_smoke && ./sm80_pipeline_smoke

#include "kittens.cuh"
#include <cstdio>
#include <vector>

using namespace kittens;

constexpr int ROWS = 16, COLS = 64;
using tile = st_bf<ROWS, COLS>;
using layout = gl<bf16, -1, -1, -1, COLS>;

// Test A: one warp cp.async-loads 4 tiles at different (b,d,r) coords, waits via
// mbarrier + load_async_arrive, then copies them back out to distinct slots.
__global__ __launch_bounds__(32, 1) void pipeline_smoke(const __grid_constant__ layout g, bf16 *out, int *hang_flag) {
    extern __shared__ int __shm[];
    shared_allocator al(&__shm[0]);
    tile (&t)[4] = al.allocate<tile, 4>();
    __shared__ semaphore sem, sem_empty;

    if(threadIdx.x == 0) {
        init_semaphore(sem, 32, 0);        // one warp's cp.async completions
        init_semaphore(sem_empty, 32, 0);  // zero-pending arrive test
    }
    __syncwarp();

    // 4 tiles from different coordinates: {0,0,0,0}, {0,0,1,0}, {0,1,2,0}, {1,0,3,0}
    warp::load_async(t[0], g, {0, 0, 0, 0});
    warp::load_async(t[1], g, {0, 0, 1, 0});
    warp::load_async(t[2], g, {0, 1, 2, 0});
    warp::load_async(t[3], g, {1, 0, 3, 0});
    kittens::load_async_arrive(sem);
    wait(sem, 0);

    // write back
    for(int i = 0; i < 4; i++) {
        for(int e = threadIdx.x; e < ROWS*COLS; e += 32) {
            out[i*ROWS*COLS + e] = t[i][{e/COLS, e%COLS}];
        }
    }
    __syncwarp();

    // Test B: load_async_arrive with ZERO pending cp.asyncs must still complete.
    kittens::load_async_arrive(sem_empty);
    wait(sem_empty, 0);
    if(threadIdx.x == 0) *hang_flag = 1; // reached: no hang
}

// --- Test C: global -> register load with row-tile coords (the rotary sin/cos path) ---
using rope_layout = gl<bf16, 1, 1, -1, 32>;
__global__ __launch_bounds__(32, 1) void g2r_smoke(const __grid_constant__ rope_layout g, float *out) {
    rt_fl<16, 32> r;
    for(int tile = 0; tile < 3; tile++) {
        warp::load(r, g, {tile, 0});
        // store via standard fragment layout math (row-major out)
        int lane = kittens::laneid();
        #pragma unroll
        for(int j = 0; j < r.width; j++) {
            #pragma unroll
            for(int k = 0; k < r.packed_per_tile; k++) {
                int row = (lane / 4) + (k % 2) * 8;
                int col = j*16 + (lane % 4)*2 + (k / 2)*8;
                out[tile*16*32 + row*32 + col]     = r.tiles[0][j].data[k].x;
                out[tile*16*32 + row*32 + col + 1] = r.tiles[0][j].data[k].y;
            }
        }
    }
}

int main() {
    // globals: B=2, D=2, R=64 rows, C=64 -> value = linear index
    const int B = 2, D = 2, R = 64, C = COLS;
    size_t n = (size_t)B*D*R*C;
    std::vector<bf16> host(n);
    for(size_t i = 0; i < n; i++) host[i] = __float2bfloat16((float)(i % 8192));

    bf16 *dbuf, *dout;
    int *dflag;
    cudaMalloc(&dbuf, n*2);
    cudaMalloc(&dout, 4*ROWS*COLS*2);
    cudaMalloc(&dflag, 4);
    cudaMemset(dflag, 0, 4);
    cudaMemcpy(dbuf, host.data(), n*2, cudaMemcpyHostToDevice);

    layout g{dbuf, B, D, R, nullptr};
    constexpr int smem = 4*sizeof(tile) + 1024;
    cudaFuncSetAttribute(pipeline_smoke, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    pipeline_smoke<<<1, 32, smem>>>(g, dout, dflag);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { printf("kernel error: %s\n", cudaGetErrorString(err)); return 1; }

    std::vector<bf16> got(4*ROWS*COLS);
    int flag = 0;
    cudaMemcpy(got.data(), dout, got.size()*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(&flag, dflag, 4, cudaMemcpyDeviceToHost);

    // expected tile origins (element offsets): coord {b,d,r,0} with r in tile units
    auto offset = [&](int b, int d, int r_tile) {
        return ((size_t)b*D + d)*R*C + (size_t)r_tile*ROWS*C;
    };
    size_t offs[4] = { offset(0,0,0), offset(0,0,1), offset(0,1,2), offset(1,0,3) };
    bool all_ok = true;
    for(int i = 0; i < 4; i++) {
        int bad = 0;
        for(int e = 0; e < ROWS*COLS; e++) {
            float want = __bfloat162float(host[offs[i] + (e/COLS)*C + (e%COLS)]);
            float have = __bfloat162float(got[i*ROWS*COLS + e]);
            if(want != have) bad++;
        }
        printf("tile %d (coord set %d): %s (%d/%d mismatched)\n", i, i, bad ? "FAIL" : "PASS", bad, ROWS*COLS);
        all_ok &= (bad == 0);
    }
    printf("zero-pending load_async_arrive: %s\n", flag ? "PASS (no hang)" : "FAIL");
    all_ok &= (flag == 1);

    // Test C
    {
        const int RR = 64, CC = 32;
        std::vector<bf16> rope(RR*CC);
        for(int i = 0; i < RR*CC; i++) rope[i] = __float2bfloat16((float)(i%1000));
        bf16 *drope; float *dr_out;
        cudaMalloc(&drope, RR*CC*2); cudaMalloc(&dr_out, 3*16*CC*4);
        cudaMemcpy(drope, rope.data(), RR*CC*2, cudaMemcpyHostToDevice);
        rope_layout rg{drope, nullptr, nullptr, RR, nullptr};
        g2r_smoke<<<1, 32>>>(rg, dr_out);
        cudaError_t e2 = cudaDeviceSynchronize();
        if(e2 != cudaSuccess) { printf("g2r kernel error: %s\n", cudaGetErrorString(e2)); return 1; }
        std::vector<float> rgot(3*16*CC);
        cudaMemcpy(rgot.data(), dr_out, rgot.size()*4, cudaMemcpyDeviceToHost);
        for(int t = 0; t < 3; t++) {
            int bad = 0;
            for(int e = 0; e < 16*CC; e++) {
                float want = __bfloat162float(rope[(t*16 + e/CC)*CC + e%CC]);
                if(want != rgot[t*16*CC + e]) bad++;
            }
            printf("g2r row-tile %d: %s (%d/%d mismatched)\n", t, bad ? "FAIL" : "PASS", bad, 16*CC);
            all_ok &= (bad == 0);
        }
    }
    printf(all_ok ? "ALL PASS\n" : "FAILURES\n");
    return all_ok ? 0 : 1;
}

