// Minimal lcsf-template identity-copy kernel for the SM80 producer pattern:
// producer cp.asyncs input tiles + load_async_arrive; consumer copies
// input->output through registers; producer stores synchronously.
// Output must equal input exactly.
//
// Build:  nvcc sm80_lcsf_smoke.cu -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 \
//             -std=c++20 -O2 --expt-extended-lambda --expt-relaxed-constexpr \
//             -I../../include -I../../prototype -o sm80_lcsf_smoke && ./sm80_lcsf_smoke

#include "kittens.cuh"
#include "prototype.cuh"
#include <cstdio>
#include <vector>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;

struct copy_layout {
    static constexpr int warps = 8;
    using seq_tile   = st_bf<16, 64>;
    using seq_global = gl<bf16, -1, -1, -1, 64, seq_tile>;
    struct globals { seq_global o, x; };
    struct input_block    { seq_tile x[warps]; };
    struct output_block   { seq_tile o[warps]; };
    struct producer_state {};
    struct consumer_state {};
};
struct copy_template {
    static constexpr int NUM_CONSUMER_WARPS=8, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=3, INPUT_PIPE_STAGES=3;
#if !defined(KITTENS_SM90) && !defined(KITTENS_SM10X) && !defined(KITTENS_SM120)
    static constexpr int PRODUCER_BARRIER_ARRIVALS=32;
#endif
    using layout = copy_layout;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if(args.task_iter == 0) args.num_iters = args.globals.x.depth(); // one iter per depth slice
        else args.num_iters = -1;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                for(int i = 0; i < layout::warps; i++) {
                    warp::load_async(args.input.x[i], args.globals.x, {0, args.iter, i, 0});
                }
                kittens::load_async_arrive(args.inputs_arrived);
                __syncwarp();
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                for(int i = 0; i < layout::warps; i++) {
                    warp::store(args.globals.o, args.output.o[i], {0, args.iter, i, 0});
                }
                __syncwarp();
                if(laneid() == 0) arrive(args.outputs_finished, 32);
                __syncwarp();
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>();
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            rt_bf<16, 64> x;
            warp::load(x, args.input.x[warpid()]);
            if(laneid() == 0) arrive(args.inputs_finished);
            __syncwarp();
            warp::store(args.output.o[warpid()], x);
            __syncwarp();
            if(laneid() == 0) arrive(args.outputs_arrived);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

int main() {
    const int DEPTH = 12, R = 8*16, C = 64; // 12 iterations, 8 tiles each
    size_t n = (size_t)DEPTH*R*C;
    std::vector<bf16> host(n), got(n);
    for(size_t i = 0; i < n; i++) host[i] = __float2bfloat16((float)(i % 4093));

    bf16 *dx, *d_out;
    cudaMalloc(&dx, n*2); cudaMalloc(&d_out, n*2);
    cudaMemcpy(dx, host.data(), n*2, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, n*2);

    typename copy_layout::seq_global Xg{dx, 1, DEPTH, R, nullptr};
    typename copy_layout::seq_global Og{d_out, 1, DEPTH, R, nullptr};
    typename copy_layout::globals g{Og, Xg};

    unsigned long mem_size = MAX_SHARED_MEMORY - 2048;
    cudaFuncSetAttribute(prototype::lcsf::kernel<copy_template>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<copy_template>);
    prototype::lcsf::kernel<copy_template><<<dim3(1,1), block, mem_size>>>(g);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { printf("kernel error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(got.data(), d_out, n*2, cudaMemcpyDeviceToHost);
    size_t bad = 0; size_t first_bad = n;
    for(size_t i = 0; i < n; i++) {
        if(__bfloat162float(got[i]) != __bfloat162float(host[i])) { if(bad==0) first_bad = i; bad++; }
    }
    printf("identity copy through lcsf: %zu/%zu mismatched", bad, n);
    if(bad) {
        size_t i = first_bad;
        printf(" (first at %zu: depth=%zu tile=%zu row=%zu col=%zu, got %.1f want %.1f)",
               i, i/(R*C), (i/(16*C))%8, (i/C)%16, i%C,
               __bfloat162float(got[i]), __bfloat162float(host[i]));
    }
    printf("\n%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
