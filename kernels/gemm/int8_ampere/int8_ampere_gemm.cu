// Ampere (SM80/SM86) port of kernels/gemm/int8_h100: exact int8 x int8 -> int32
// GEMM. The H100 version's TMA ring + wgmma become a cp.async ring + warp-mma
// (m16n8k32 IMMA via the warpgroup emulation); PDL and register reallocation
// drop out. Producer warp 0 issues cp.async for a/b tiles and its 32 lanes
// arrive on completion. Two writeback modes, selected per problem size:
//  - staged (DIRECT_STORE=false): producer warp 1 drains d_smem to global with
//    coalesced stores. Best at small sizes where the store tail dominates.
//  - direct (DIRECT_STORE=true): consumers store accumulator fragments straight
//    to global. Uncoalesced, but frees 64KB of smem for Nb=128 tiles whose
//    higher arithmetic intensity wins at large sizes (34 -> 56 TOPS @8192).

#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

template <int _Mb, int _Nb, int _Kb, int _SUPERGROUP_SIZE, int _LOAD_PIPE_DEPTH, bool _DIRECT_STORE = false>
struct config {
    static_assert(_Mb == 128, "Mb must be 128");
    static_assert(_Nb >= 16 && _Nb <= 256 && _Nb % 16 == 0, "Nb must be 16, 32, ..., 256");
    static_assert(_Kb >= 32 && _Kb % 32 == 0, "Kb must be a multiple of 32");
    static_assert(_SUPERGROUP_SIZE >= 1 && _SUPERGROUP_SIZE <= 16, "SUPERGROUP_SIZE must be 1-16");
    static_assert(_LOAD_PIPE_DEPTH >= 1 && _LOAD_PIPE_DEPTH <= 16, "LOAD_PIPE_DEPTH must be 1-16");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr bool DIRECT_STORE = _DIRECT_STORE;

    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

template <typename C>
struct globals {
    using a_tile = st_int8<C::Mb/2, C::Kb>;
    using b_tile = st_int8<C::Nb, C::Kb>;
    using d_tile = st_int<C::Mb/2, C::Nb>;

    using a_gl = gl<int8, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<int8, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<int,  1, 1, -1, -1, d_tile>;

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
                                                  (C::DIRECT_STORE ? 0 : sizeof(d_tile) * 2) + 1024;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

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
    typename G::d_tile (*d_smem) = nullptr; // [2], staged mode only
    if constexpr (!C::DIRECT_STORE) d_smem = &allocator.allocate<typename G::d_tile, 2>()[0];

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
        } else if (!C::DIRECT_STORE && warpgroup::warpid() == 1) { // store warp (staged mode)
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
        for (int task_id = blockIdx.x; task_id < num_blks; task_id += gridDim.x) {
            rt<int, C::Mb/8, C::Nb> d_reg;
            warp::zero(d_reg);

            for (int idx = 0; idx < iters_per_task; idx++) {
                wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                warpgroup::mma_ABt(d_reg, a_smem[input_ring][warpgroup_id], b_smem[input_ring]);
                warp::arrive(inputs_finished[input_ring]); // mma.sync is synchronous
                update_phasebit<0>(bitfield, input_ring);
                input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
            }

            if constexpr (C::DIRECT_STORE) {
                // each warp writes its 16-row slice straight to global
                // (coord units are the rt tile shape, 16 x Nb)
                int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, task_id);
                warp::store(g.d, d_reg, {(tile_coord.x*2 + warpgroup_id)*4 + warpgroup::warpid(), tile_coord.y});
            } else {
                wait(outputs_finished, get_phasebit<1>(bitfield, 0));
                warpgroup::store(d_smem[warpgroup_id], d_reg);
                warpgroup::sync(warpgroup_id+1);
                warpgroup::arrive(outputs_arrived);
                update_phasebit<1>(bitfield, 0);
            }
        }
    }
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

    // Allocate device memory
    std::vector<int8*> d_A(arg_group_count);
    std::vector<int8*> d_B(arg_group_count);
    std::vector<int*> d_D(arg_group_count);
    int* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M*K*sizeof(int8)));
        CUDACHECK(cudaMalloc(&d_B[i], K*N*sizeof(int8)));
        CUDACHECK(cudaMalloc(&d_D[i], M*N*sizeof(int)));
    }
    CUDACHECK(cudaMalloc(&d_D_ref, M*N*sizeof(int)));
    std::cout << "Allocated device memory" << std::endl;

    // Initialize matrices on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<int8, FillMode::RANDOM>(d_A[i], M*K, seed + i*100, -128.0f, 127.0f);
        fill<int8, FillMode::RANDOM>(d_B[i], K*N, seed + i*100 + 1, -128.0f, 127.0f);
        fill<int, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<int, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Initialized matrices on device" << std::endl;

    // Compute reference GEMM on device
    reference_gemm<int8, int>(d_D_ref, d_A[0], d_B[0], M, N, K);
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

    // Calculate duration and TOPS
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double ops = double(2.0) * M * N * K;
    double tops = (ops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tops << " TOPS\n";

    // Verify results
    check_correctness(d_D[0], d_D_ref, M * N);
    if (getenv("INT8_DEBUG_MAP")) { // per-64x64-tile mismatch counts
        std::vector<int> got(M*N), ref(M*N);
        cudaMemcpy(got.data(), d_D[0], M*N*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ref.data(), d_D_ref, M*N*4, cudaMemcpyDeviceToHost);
        for (size_t tr = 0; tr < M/64; tr++) {
            printf("tile row %2zu: ", tr);
            for (size_t tc = 0; tc < N/64; tc++) {
                int bad = 0;
                for (size_t r = 0; r < 64; r++) for (size_t c = 0; c < 64; c++) {
                    size_t i = (tr*64+r)*N + tc*64+c;
                    if (got[i] != ref[i]) bad++;
                }
                printf("%5d ", bad);
            }
            printf("\n");
        }
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

    return tops;
}

__host__ int main() {
    bool ncu = false;
    int N;

    // SM86 configs. Small sizes: staged coalesced writeback, Nb=64 for occupancy.
    // Large sizes: direct register writeback frees smem for Nb=128 (higher intensity).
    N = 1024;
    run_benchmark<config<128, 64, 64, 1, 4, false>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<config<128, 128, 64, 4, 4, true>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<config<128, 128, 64, 8, 4, true>>(N, N, N, ncu);

    return 0;
}
