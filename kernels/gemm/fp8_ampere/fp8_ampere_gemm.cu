// Ampere (SM80/SM86) fp8-STORAGE GEMM, the Marlin approach: Ampere has no fp8
// tensor cores (that needs SM89+), but fp8e4m3 as a storage format still halves
// global bandwidth. Tiles stream in as raw fp8 through the same cp.async ring as
// the int8_ampere kernel; fragments are dequantized fp8e4m3 -> fp16 in registers
// during the shared->register load (e4m3 is exactly representable in fp16) and
// the math runs on fp16 tensor cores (m16n8k16, fp32 accumulate). D is fp32.
//
// Structure is a copy of int8_ampere_gemm.cu: producer warp 0 cp.asyncs a/b
// tiles, warp 1 stores results synchronously, 2 consumer warpgroups.
//
// Build:
//   /usr/local/cuda/bin/nvcc fp8_ampere_gemm.cu -std=c++20 --extended-lambda \
//     --expt-relaxed-constexpr -I../../../include -I.. -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -lcuda -lcudart -O2 -o fp8_ampere_gemm.out

#include "kittens.cuh"
#include "../common.cuh"
#include <cuda_fp8.h>

using namespace kittens;

template <int _Mb, int _Nb, int _Kb, int _SUPERGROUP_SIZE, int _LOAD_PIPE_DEPTH>
struct config {
    static_assert(_Mb == 128, "Mb must be 128");
    static_assert(_Nb >= 16 && _Nb <= 256 && _Nb % 16 == 0, "Nb must be 16, 32, ..., 256");
    static_assert(_Kb >= 16 && _Kb % 16 == 0, "Kb must be a multiple of 16");
    static_assert(_SUPERGROUP_SIZE >= 1 && _SUPERGROUP_SIZE <= 16, "SUPERGROUP_SIZE must be 1-16");
    static_assert(_LOAD_PIPE_DEPTH >= 1 && _LOAD_PIPE_DEPTH <= 16, "LOAD_PIPE_DEPTH must be 1-16");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;

    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

// TK's fp8 tile types are gated to SM90+, but this kernel only needs the BYTES:
// tiles are st_int8 (same 1-byte swizzled layout, proven on SM86 by int8_ampere)
// and the fp8e4m3 semantics live entirely in load_fp8_frag / the host helpers.
template <typename C>
struct globals {
    using a_tile = st_int8<C::Mb/2, C::Kb>;
    using b_tile = st_int8<C::Nb, C::Kb>;
    using d_tile = st_fl<C::Mb/2, C::Nb>;

    using a_gl = gl<int8, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<int8, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<float,   1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;

    __host__ __inline__ dim3 grid() {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        return dim3(sms);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() {
        constexpr size_t _dynamic_shared_memory = sizeof(a_tile) * C::LOAD_PIPE_DEPTH * 2 +
                                                  sizeof(b_tile) * C::LOAD_PIPE_DEPTH +
                                                  sizeof(d_tile) * 2 + 1024;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

// Marlin-style fast e4m3 -> fp16 pair dequant: realign the fp8 bit pattern into the
// fp16 fields (sign kept, exp+mant shifted into place) and fix the bias gap with a
// single multiply by 2^(15-7)=256. Exact for all finite e4m3 values incl. denormals.
__device__ static inline half_2 dequant_fp8x2(uint16_t raw) {
    uint32_t v = (((uint32_t)raw & 0xFF00u) << 16) | (((uint32_t)raw & 0x00FFu) << 8);
    uint32_t h = (v & 0x80008000u) | ((v & 0x7F007F00u) >> 1);
    half_2 r = *reinterpret_cast<half_2*>(&h);
    return __hmul2(r, half_2{__float2half(256.f), __float2half(256.f)});
}

// dequant-load one 16x16 fp8e4m3 chunk of a shared tile into an fp16 mma fragment.
// fragment convention (TK 2.0 row layout): data[k] holds rows +8*(k%2), cols +8*(k/2);
// each lane owns the element pair (lane/4, (lane%4)*2) of its 8x8 quadrant, so a
// single u16 read grabs both fp8 bytes of a pair (pairs never straddle a 16B swizzle atom).
template<ducks::st::all ST>
__device__ static inline void load_fp8_frag(rt_hf<16,16> &dst, const ST &src, int row0, int col0) {
    const int lane = kittens::laneid();
    const int r = row0 + lane/4;
    const int c = col0 + (lane%4)*2;
    #pragma unroll
    for(int k = 0; k < 4; k++) {
        const int rr = r + (k%2)*8;
        const int cc = c + (k/2)*8;
        uint16_t raw = *reinterpret_cast<const uint16_t*>(
            (const typename ST::dtype*)src.idx((typename ST::dtype*)src.data, int2{rr, cc}));
        dst.tiles[0][0].data[k] = dequant_fp8x2(raw);
    }
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ globals<C> g) {
    using G = globals<C>;

    const int iters_per_task = g.a.cols() / C::Kb;
    const int rblks = g.d.rows() / C::Mb;
    const int cblks = g.d.cols() / C::Nb;
    const int num_blks = rblks * cblks;
    const int warpgroup_id = warpgroup::groupid();
    int input_ring = 0;

    extern __shared__ int __shm[];
    shared_allocator allocator((int*)&__shm[0]);

    typename G::a_tile (&a_smem)[C::LOAD_PIPE_DEPTH][2] = allocator.allocate<typename G::a_tile, C::LOAD_PIPE_DEPTH, 2>();
    typename G::b_tile (&b_smem)[C::LOAD_PIPE_DEPTH]    = allocator.allocate<typename G::b_tile, C::LOAD_PIPE_DEPTH>();
    typename G::d_tile (&d_smem)[2]                     = allocator.allocate<typename G::d_tile, 2>();

    __shared__ semaphore inputs_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(inputs_arrived[i],  WARP_THREADS, 0); // producer warp's cp.async lanes
            init_semaphore(inputs_finished[i], 0, C::NUM_CONSUMERS * WARPGROUP_WARPS);
        }
        init_semaphore(outputs_arrived,  0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    if (warpgroup_id == C::NUM_CONSUMERS) {
        if (warpgroup::warpid() == 0) { // load warp: cp.async ring
            for (int task_id = blockIdx.x; task_id < num_blks; task_id += gridDim.x) {
                int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, task_id);
                for (int idx = 0; idx < iters_per_task; idx++) {
                    wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    #pragma unroll
                    for (int i = 0; i < 2; i++)
                        warp::load_async(a_smem[input_ring][i], g.a, {tile_coord.x*2+i, idx});
                    warp::load_async(b_smem[input_ring], g.b, {tile_coord.y, idx});
                    kittens::load_async_arrive(inputs_arrived[input_ring]);
                    update_phasebit<1>(bitfield, input_ring);
                    input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
                }
            }
        } else if (warpgroup::warpid() == 1) { // store warp: synchronous writeback
            for (int task_id = blockIdx.x; task_id < num_blks; task_id += gridDim.x) {
                int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, task_id);
                wait(outputs_arrived, get_phasebit<0>(bitfield, 0));
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    warp::store(g.d, d_smem[i], {tile_coord.x*2+i, tile_coord.y});
                __syncwarp();
                if (laneid() == 0) arrive(outputs_finished);
                update_phasebit<0>(bitfield, 0);
            }
        }
    } else {
        const int warp_row0 = warpgroup::warpid() * 16; // this warp's 16-row slice of the 64-row a tile
        for (int task_id = blockIdx.x; task_id < num_blks; task_id += gridDim.x) {
            rt_fl<16, C::Nb> d_reg;
            warp::zero(d_reg);

            for (int idx = 0; idx < iters_per_task; idx++) {
                wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                // dequant fp8 -> fp16 fragments in registers, mma on fp16 tensor cores
                #pragma unroll
                for (int kk = 0; kk < C::Kb/16; kk++) {
                    rt_hf<16,16> a_frag;
                    load_fp8_frag(a_frag, a_smem[input_ring][warpgroup_id], warp_row0, kk*16);
                    #pragma unroll
                    for (int j = 0; j < C::Nb/16; j++) {
                        rt_hf<16,16> b_frag;
                        load_fp8_frag(b_frag, b_smem[input_ring], j*16, kk*16);
                        auto &d_chunk = reinterpret_cast<rt_fl<16,16>&>(d_reg.tiles[0][j]);
                        warp::mma_ABt(d_chunk, a_frag, b_frag, d_chunk);
                    }
                }
                warp::arrive(inputs_finished[input_ring]); // mma.sync is synchronous
                update_phasebit<0>(bitfield, input_ring);
                input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
            }

            wait(outputs_finished, get_phasebit<1>(bitfield, 0));
            warpgroup::store(d_smem[warpgroup_id], d_reg);
            warpgroup::sync(warpgroup_id+1);
            warpgroup::arrive(outputs_arrived);
            update_phasebit<1>(bitfield, 0);
        }
    }
}

// host-side fp8 helpers, self-contained because TK's fp8 types are SM90-gated:
// fill a byte buffer with random floats quantized to fp8e4m3, and an fp32
// reference GEMM reading fp8 bytes directly (A row-major MxK, B col-major NxK).
__global__ void fill_fp8_kernel(int8* data, size_t count, uint64_t seed, float min_val, float max_val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint64_t x = seed + idx;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        float u = (float)(x >> 40) * (1.0f / 16777216.0f);
        float val = u * (max_val - min_val) + min_val;
        data[idx] = (int8)__nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
    }
}
static inline void fill_fp8(int8* data, size_t count, uint64_t seed, float min_val, float max_val) {
    fill_fp8_kernel<<<(int)((count + 255) / 256), 256>>>(data, count, seed, min_val, max_val);
}

__global__ void reference_fp8_gemm_kernel(float* D, const int8* A, const int8* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            __half_raw a = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(uint8_t)A[row * K + k], __NV_E4M3);
            __half_raw b = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(uint8_t)B[col * K + k], __NV_E4M3);
            acc += __half2float(a) * __half2float(b);
        }
        D[row * N + col] = acc;
    }
}
static inline void reference_fp8_gemm(float* D, const int8* A, const int8* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    reference_fp8_gemm_kernel<<<grid, block>>>(D, A, B, M, N, K);
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE
              << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH << "\n";
    std::cout << "Number of iterations per task: " << (K / C::Kb) << "\n";

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = size_t(M) * K + size_t(N) * K + 4 * size_t(M) * N;
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory (fp8 bytes, typed int8 for the TK side)
    std::vector<int8*> d_A(arg_group_count);
    std::vector<int8*> d_B(arg_group_count);
    std::vector<float*> d_D(arg_group_count);
    float* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M*K));
        CUDACHECK(cudaMalloc(&d_B[i], K*N));
        CUDACHECK(cudaMalloc(&d_D[i], M*N*sizeof(float)));
    }
    CUDACHECK(cudaMalloc(&d_D_ref, M*N*sizeof(float)));
    std::cout << "Allocated device memory" << std::endl;

    // Initialize matrices on device (small values: fp8 dynamic range is tiny)
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill_fp8(d_A[i], M*K, seed + i*100, -0.5f, 0.5f);
        fill_fp8(d_B[i], K*N, seed + i*100 + 1, -0.5f, 0.5f);
        fill<float, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<float, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Initialized matrices on device" << std::endl;

    // Compute reference GEMM on device (fp8 in, fp32 accumulate/out)
    reference_fp8_gemm(d_D_ref, d_A[0], d_B[0], M, N, K);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Computed reference GEMM on device" << std::endl;

    // Prepare kernel inputs
    std::vector<globals<C>> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename globals<C>::a_gl Ag{d_A[i], nullptr, nullptr, M, K};
        typename globals<C>::b_gl Bg{d_B[i], nullptr, nullptr, N, K};
        typename globals<C>::d_gl Dg{d_D[i], nullptr, nullptr, M, N};
        g.push_back(globals<C>{Ag, Bg, Dg});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));

    // Number of iterations
    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

    // Warmup
    for(int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        kernel<C><<<g[idx].grid(), g[idx].block(), g[idx].dynamic_shared_memory()>>>(g[idx]);
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for(int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        kernel<C><<<g[idx].grid(), g[idx].block(), g[idx].dynamic_shared_memory()>>>(g[idx]);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double ops = double(2.0) * M * N * K;
    double tflops = (ops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Verify results: fp16 mma with fp32 accumulate vs fp32 reference on fp8 inputs.
    // The mma accumulation order differs from the serial reference; tolerance is relative.
    {
        std::vector<float> got(M*N), ref(M*N);
        CUDACHECK(cudaMemcpy(got.data(), d_D[0], M*N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaMemcpy(ref.data(), d_D_ref, M*N*sizeof(float), cudaMemcpyDeviceToHost));
        double max_err = 0, sum_err = 0, sum_ref = 0;
        size_t bad = 0;
        for (size_t i = 0; i < M*N; i++) {
            double err = std::abs(double(got[i]) - double(ref[i]));
            max_err = std::max(max_err, err);
            sum_err += err;
            sum_ref += std::abs(double(ref[i]));
            if (err > 1e-2 * std::max(1.0, std::abs(double(ref[i])))) bad++;
        }
        std::cout << "Max error: " << max_err << "\n";
        std::cout << "Avg error: " << sum_err / (M*N) << " (avg |ref| " << sum_ref / (M*N) << ")\n";
        std::cout << "Error count (>1% rel): " << bad << (bad == 0 ? "  -- PASS" : "  -- FAIL") << "\n";
    }

    // Clean up
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_D[i]);
    }
    cudaFree(d_D_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

__host__ int main() {
    bool ncu = false;
    int N;

    // SM86 configs mirror int8_ampere: a 2x(64x64)=8KB + b 4KB per stage in fp8,
    // d 2x16KB fp32 -> 80KB total at depth 4
    N = 1024;
    run_benchmark<config<128, 64, 64, 1, 4>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<config<128, 64, 64, 4, 4>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<config<128, 64, 64, 8, 4>>(N, N, N, ncu);

    return 0;
}
