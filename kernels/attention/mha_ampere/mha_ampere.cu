// Ampere (SM80/SM86) multi-head attention forward, ported from the deleted
// 4090 demo kernel (git 50ee1f0a) and brought to parity with mha_h100:
// (B,H,N,D) layout, causal option, GQA, and the L (logsumexp) output the
// backward pass consumes. Pipeline: 4 workers, 3-stage cp.async ring for K/V,
// warp-scope mma; each warp owns ROWS<D> query rows.

#include "kittens.cuh"

using namespace kittens;

constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 3;

template<int D> constexpr size_t ROWS = 16*(128/D); // height of each worker tile (rows)
template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, ROWS<D>, D, L>;
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>;
template<int D> using global_layout = gl<bf16, -1, -1, -1, D, shared_tile<D>>; // B, H, N, D
template<int D> using l_vec_t = sv_fl<ROWS<D>>;
template<int D> using l_layout = gl<float, -1, -1, -1, -1, l_vec_t<D>>; // B, H, 1, N

template<int D> struct fwd_globals {
    global_layout<D> Qg, Kg, Vg, Og;
    l_layout<D> Lg;
    const int hr; // q heads per kv head (GQA ratio)
};

template<int D, bool is_causal> __launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void attend_ker(const __grid_constant__ fwd_globals<D> g) {

    using load_group = kittens::group<2>; // pairs of workers collaboratively load k, v tiles
    int loadid = load_group::groupid(), workerid = kittens::warpid(); // which worker am I?
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;
    const int kv_head = head / g.hr;
    const int N = g.Qg.rows();

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    shared_tile<D> (&k_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();

    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);
    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg, k_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16, col_l> v_reg; // V is column layout, as we use mma_AB.
    qkvo_tile<D, float> o_reg; // Output tile.
    attn_tile<D, float> att_block; // attention tile, in float. (We want to use float wherever possible.)
    attn_tile<D, bf16> att_block_mma; // bf16 attention tile for the second mma_AB. We cast right before that op.
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec; // these are column vectors for the in-place softmax.
    // each warp loads its own Q tile
    if (q_seq*ROWS<D> < N) {
        warp::load(qo_smem[workerid], g.Qg, {batch, head, q_seq, 0});  // going through shared memory improves coalescing of dram reads.
        __syncwarp();
        warp::load(q_reg, qo_smem[workerid]);
    }
    __syncthreads();

    // Pre-scale Q so QK^T lands in exp2-ready units (softmax_scale * log2(e)).
    if constexpr(D == 64) q_reg *= __float2bfloat16(0.125f * 1.44269504089f);
    else if constexpr(D == 128) q_reg *= __float2bfloat16(0.08838834764f * 1.44269504089f);

    max_vec = base_types::constants<float>::neg_infty();
    norm_vec = 0.f;
    o_reg = 0.f;
    // launch the load of the first k, v tiles
    int kv_blocks = (N + LOAD_BLOCKS*ROWS<D>-1) / (LOAD_BLOCKS*ROWS<D>), tic = 0;
    // for causal attention only KV blocks up to this worker-block's diagonal
    // are needed; every warp shares the same bound so the loads stay uniform.
    if constexpr (is_causal) {
        int last_q_row = (blockIdx.x * NUM_WORKERS + NUM_WORKERS) * ROWS<D>;
        int needed = (last_q_row + LOAD_BLOCKS*ROWS<D> - 1) / (LOAD_BLOCKS*ROWS<D>);
        kv_blocks = kv_blocks < needed ? kv_blocks : needed;
    }
    load_group::load_async(k_smem[loadid][0], g.Kg, {batch, kv_head, loadid, 0});
    load_group::load_async(v_smem[loadid][0], g.Vg, {batch, kv_head, loadid, 0});
    // iterate over k, v for these q's that have been loaded
    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=(tic+1)%3) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        if(next_load_idx*ROWS<D> < N && kv_idx+1 < kv_blocks) {
            int next_tic = (tic+1)%3;
            load_group::load_async(k_smem[loadid][next_tic], g.Kg, {batch, kv_head, next_load_idx, 0});
            load_group::load_async(v_smem[loadid][next_tic], g.Vg, {batch, kv_head, next_load_idx, 0});
            load_async_wait<1>(); // next k, v can stay in flight.
        }
        else load_async_wait();
        __syncthreads();

        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D> < N; subtile++) {
            warp::load(k_reg, k_smem[subtile][tic]); // load k from shared into registers
            att_block = 0.f; // zero attention tile
            warp::mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block); // Q@K.T
            int first_index = (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D>;
            int start_fill = N-first_index < ROWS<D> ? N-first_index : ROWS<D>;
            warp::right_fill(att_block, att_block, start_fill, base_types::constants<float>::neg_infty());
            if constexpr (is_causal) {
                // mask 16x16 subtiles above the diagonal (global 16-row block indices)
                constexpr int SUB = ROWS<D>/16;
                int q16 = q_seq*SUB, k16 = (kv_idx*LOAD_BLOCKS + subtile)*SUB;
                #pragma unroll
                for(int i = 0; i < SUB; i++) {
                    #pragma unroll
                    for(int j = 0; j < SUB; j++) {
                        auto &attn_subtile = reinterpret_cast<rt_fl<16,16>&>(att_block.tiles[i][j]);
                        if      (k16+j >  q16+i) { warp::neg_infty  (attn_subtile); }
                        else if (k16+j == q16+i) { warp::make_causal(attn_subtile, attn_subtile, base_types::constants<float>::neg_infty()); }
                    }
                }
                __syncwarp();
            }
            max_vec_last = max_vec;
            max_vec = warp::max<axis::COL>(att_block, max_vec);
            att_block = warp::exp2(att_block - max_vec);
            max_vec_last = warp::exp2(max_vec_last - max_vec);
            norm_vec *= max_vec_last;
            norm_vec = warp::sum<axis::COL>(att_block, norm_vec);
            att_block_mma = att_block; // copy to bf16 tile
            warp::load(v_reg, v_smem[subtile][tic]);
            o_reg *= max_vec_last;
            warp::mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

    o_reg /= norm_vec;
    __syncthreads();
    if (q_seq*ROWS<D> < N) { // write out o and L.
        warp::store(qo_smem[workerid], o_reg); // going through shared memory improves coalescing of dram writes.
        __syncwarp();
        warp::store(g.Og, qo_smem[workerid], {batch, head, q_seq, 0});
        // L = -(ln(norm) + max*ln2) / softmax_scale, matching mha_h100's
        // convention (max_vec here is already in log2(e)*scale units).
        typename attn_tile<D, float>::col_vec l_vec;
        warp::log(l_vec, norm_vec);
        warp::mul(max_vec, max_vec, 0.69314718056f);
        warp::add(l_vec, l_vec, max_vec);
        if constexpr (D == 64) { warp::mul(l_vec, l_vec, -8.0f); }
        else                   { warp::mul(l_vec, l_vec, -11.313708499f); }
        warp::store(g.Lg, l_vec, {batch, head, 0, q_seq});
    }
}


// ---------------------------------------------------------------------------
// Backward pass (Ampere): KV-outer FA2 mirroring mha_h100's dataflow, with
// synchronous warp-mma, cp.async loads, and fp32 atomicAdd gradient writes
// (host must zero-initialize qg/kg/vg). D=64 uses two consumer warpgroups per
// block; D=128 uses one to fit the 99KB shared-memory budget.
// ---------------------------------------------------------------------------

template<int D> constexpr int BWD_WGS = (D == 64) ? 2 : 1; // consumer warpgroups per block
constexpr int BWD_TILE = 64; // KV rows per warpgroup and QO rows per iteration

template<int D> struct bwd_globals {
    using k_tile  = st_bf<BWD_TILE, D>;
    using q_tile  = st_bf<BWD_TILE, D>;
    using l_tile  = sv_fl<BWD_TILE>;

    using qkv_gl  = gl<bf16,  -1, -1, -1, D, q_tile>;
    using grad_gl = gl<float, -1, -1, -1, D>;
    using vec_gl  = gl<float, -1, -1, -1, -1, l_tile>;

    qkv_gl  q, k, v, og;
    grad_gl qg, kg, vg;
    vec_gl  l, d;
    const int hr;
};

// prep kernel: D[b,h,n] = rowsum(dO * O), registers only.
template<int D> struct bwd_prep_globals {
    using io_gl  = gl<bf16, -1, -1, -1, D, st_bf<16, D>>;
    using vec_gl = gl<float, -1, -1, -1, -1, sv_fl<16>>;
    io_gl og, o;
    vec_gl d;
};

template<int D> __launch_bounds__(4*WARP_THREADS, 2)
__global__ void bwd_prep_ker(const __grid_constant__ bwd_prep_globals<D> g) {
    const int batch = blockIdx.z, head = blockIdx.y;
    const int row16 = blockIdx.x * 4 + kittens::warpid();
    rt_fl<16, D> og_reg, o_reg;
    typename rt_fl<16, D>::col_vec d_reg;
    warp::load(o_reg,  g.o,  {batch, head, row16, 0});
    warp::load(og_reg, g.og, {batch, head, row16, 0});
    warp::mul(og_reg, og_reg, o_reg);
    warp::row_sum(d_reg, og_reg);
    warp::store(g.d, d_reg, {batch, head, 0, row16});
}

// broadcast a 64-long fp32 shared vector across the columns of an rt_fl<16,64>
__device__ static inline void bwd_stream_tile(rt_fl<16, BWD_TILE> &reg_tile, const sv_fl<BWD_TILE> &smem_vec) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[base_col + 0];
        reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[base_col + 0];
        reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[base_col + 8];
        reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[base_col + 8];
    }
}
__device__ static inline void bwd_stream_sub_tile(rt_fl<16, BWD_TILE> &reg_tile, const sv_fl<BWD_TILE> &smem_vec) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[base_col + 0]);
        reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[base_col + 0]);
        reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[base_col + 8]);
        reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[base_col + 8]);
    }
}

// causal mask on the TRANSPOSED S block (rows = kv, cols = qo)
template<int WGS>
__device__ static inline void bwd_causal_mask(rt_fl<16, BWD_TILE> &reg_tile, int qo_idx) {
    int q_blk = qo_idx * (BWD_TILE/16);
    int k_blk = (blockIdx.x * WGS * (BWD_TILE/16))
                + ((kittens::warpid()/kittens::WARPGROUP_WARPS) * (BWD_TILE/16))
                + (kittens::warpid() % kittens::WARPGROUP_WARPS);
    #pragma unroll
    for (int j = 0; j < (BWD_TILE/16); j++) {
        int q_idx = q_blk + j;
        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(reg_tile.tiles[0][j]);
        if      (q_idx  < k_blk) { warp::neg_infty(attn_subtile); }
        else if (q_idx == k_blk) { warp::make_causal_t(attn_subtile, attn_subtile, base_types::constants<float>::neg_infty()); }
    }
}

// fp32 atomicAdd of a warp's rt_fl<16,D> accumulator fragment into a
// (B,H,N,D) float global at 16-row granularity.
template<int D, typename GL>
__device__ static inline void atomic_add_tile(const GL &gl_dst, const rt_fl<16, D> &src, int b, int h, int row16) {
    float *base = (float*)gl_dst.raw_ptr;
    const long H = gl_dst.depth(), N = gl_dst.rows();
    const long head_off = ((long)b*H + h) * N * D;
    int lane = kittens::laneid();
    #pragma unroll
    for(int j = 0; j < rt_fl<16, D>::width; j++) {
        #pragma unroll
        for(int k = 0; k < rt_fl<16, D>::packed_per_tile; k++) {
            int row = row16*16 + (lane / 4) + (k % 2) * 8;
            int col = j*16 + (lane % 4)*2 + (k / 2)*8;
            atomicAdd(&base[head_off + (long)row*D + col],     src.tiles[0][j].data[k].x);
            atomicAdd(&base[head_off + (long)row*D + col + 1], src.tiles[0][j].data[k].y);
        }
    }
}

template<int D, bool is_causal> __launch_bounds__(BWD_WGS<D>*4*WARP_THREADS, 1)
__global__ void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g) {
    constexpr int WGS = BWD_WGS<D>;
    using all_warps = kittens::group<4*WGS>;
    using k_tile  = st_bf<BWD_TILE, D>;
    using q_tile  = st_bf<BWD_TILE, D>;
    using ds_tile = st_bf<BWD_TILE, BWD_TILE>;
    using l_tile  = sv_fl<BWD_TILE>;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    k_tile  (&k_smem)[WGS]   = al.allocate<k_tile, WGS>();
    k_tile  (&v_smem)[WGS]   = al.allocate<k_tile, WGS>();
    q_tile  (&q_smem)        = al.allocate<q_tile>();
    q_tile  (&og_smem)       = al.allocate<q_tile>();
    ds_tile (&ds_smem)[WGS]  = al.allocate<ds_tile, WGS>();
    l_tile  (&l_smem)        = al.allocate<l_tile>();
    l_tile  (&d_smem)        = al.allocate<l_tile>();

    const int batch = blockIdx.z, head = blockIdx.y;
    const int kv_head = head / g.hr;
    const int N = g.q.rows();
    const int warpid = kittens::warpid();
    const int wgid = warpid / kittens::WARPGROUP_WARPS;
    const int qo_blocks = N / BWD_TILE;
    const int q_start = is_causal ? (blockIdx.x * WGS) : 0;

    // K/V tiles for this block (one 64-row tile per warpgroup)
    all_warps::load_async(k_smem[0], g.k, {batch, kv_head, blockIdx.x*WGS + 0, 0});
    all_warps::load_async(v_smem[0], g.v, {batch, kv_head, blockIdx.x*WGS + 0, 0});
    if constexpr (WGS == 2) {
        all_warps::load_async(k_smem[1], g.k, {batch, kv_head, blockIdx.x*WGS + 1, 0});
        all_warps::load_async(v_smem[1], g.v, {batch, kv_head, blockIdx.x*WGS + 1, 0});
    }

    rt_fl<16, D> kg_reg, vg_reg;
    warp::zero(kg_reg);
    warp::zero(vg_reg);

    rt_fl<16, BWD_TILE> s_block_t, dp_block_t, ds_block_t;
    rt_bf<16, BWD_TILE> p_block_t_mma, ds_block_t_mma;

    for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++) {
        // load this iteration's Q, dO, L, D (single-buffered, block-wide)
        all_warps::load_async(q_smem,  g.q,  {batch, head, qo_idx, 0});
        all_warps::load_async(og_smem, g.og, {batch, head, qo_idx, 0});
        if (warpid == 0) {
            warp::load_async(l_smem, g.l, {batch, head, 0, qo_idx});
            warp::load_async(d_smem, g.d, {batch, head, 0, qo_idx});
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncthreads();

        // S^T = K @ Q^T, seeded with the L vector so exp2 gives P directly
        bwd_stream_tile(s_block_t, l_smem);
        warpgroup::mma_ABt(s_block_t, k_smem[wgid], q_smem);
        warpgroup::mm_ABt(dp_block_t, v_smem[wgid], og_smem);
        if constexpr (D == 64) { warp::mul(s_block_t, s_block_t, 1.44269504089f*0.125f); }
        else                   { warp::mul(s_block_t, s_block_t, 1.44269504089f*0.08838834764f); }
        if constexpr (is_causal) { bwd_causal_mask<WGS>(s_block_t, qo_idx); }
        warp::exp2(s_block_t, s_block_t);          // now P^T
        warp::copy(p_block_t_mma, s_block_t);
        bwd_stream_sub_tile(dp_block_t, d_smem);   // dP^T - D
        warp::mul(ds_block_t, s_block_t, dp_block_t);
        if constexpr (D == 64) { warp::mul(ds_block_t, ds_block_t, 0.125f); }
        else                   { warp::mul(ds_block_t, ds_block_t, 0.08838834764f); }
        warp::copy(ds_block_t_mma, ds_block_t);

        warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem);  // dV += P^T @ dO
        warpgroup::store(ds_smem[wgid], ds_block_t);
        warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem);  // dK += dS^T @ Q
        __syncthreads();

        // dQ for this tile = sum over kv warpgroups of dS @ K; warpgroup 0
        // computes it from the shared dS^T tiles, then atomically accumulates.
        if (wgid == 0) {
            rt_fl<16, D> qg_reg;
            warpgroup::mm_AtB(qg_reg, ds_smem[0], k_smem[0]);
            if constexpr (WGS == 2) { warpgroup::mma_AtB(qg_reg, ds_smem[1], k_smem[1]); }
            atomic_add_tile<D>(g.qg, qg_reg, batch, head, qo_idx*4 + (warpid % 4));
        }
        __syncthreads(); // ds_smem/q_smem/og_smem reusable next iteration
    }

    // accumulate this block's dK/dV (atomic: hr query heads share a KV head)
    int kv_row16 = (blockIdx.x*WGS + wgid)*4 + (warpid % 4);
    atomic_add_tile<D>(g.kg, kg_reg, batch, kv_head, kv_row16);
    atomic_add_tile<D>(g.vg, vg_reg, batch, kv_head, kv_row16);
}

#ifdef TORCH_COMPILE

#include "pyutils/torchutils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <iostream>
#include <vector>

template<int D, bool is_causal>
static void launch_fwd(bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o, float *d_l,
                       long batch, long qo_heads, long kv_heads, long seq_len, int hr,
                       cudaStream_t stream) {
    fwd_globals<D> g{
        global_layout<D>{d_q, (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, nullptr},
        global_layout<D>{d_k, (unsigned)batch, (unsigned)kv_heads, (unsigned)seq_len, nullptr},
        global_layout<D>{d_v, (unsigned)batch, (unsigned)kv_heads, (unsigned)seq_len, nullptr},
        global_layout<D>{d_o, (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, nullptr},
        l_layout<D>{d_l, (unsigned)batch, (unsigned)qo_heads, 1U, (unsigned)seq_len},
        hr
    };
    unsigned long mem_size = 2*(NUM_WORKERS/2)*PIPE_STAGES*sizeof(shared_tile<D>) + 2048;
    dim3 grid(seq_len/(NUM_WORKERS*ROWS<D>), qo_heads, batch);
    cudaFuncSetAttribute(attend_ker<D, is_causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<D, is_causal><<<grid, NUM_WORKERS*WARP_THREADS, mem_size, stream>>>(g);
}

std::vector<at::Tensor>
attention_forward(at::Tensor q, at::Tensor k, at::Tensor v, bool causal)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    TORCH_CHECK(k.size(0) == batch && v.size(0) == batch, "Batch dimension must match for all inputs");
    TORCH_CHECK(k.size(2) == seq_len && v.size(2) == seq_len, "Sequence length must match for all inputs");
    TORCH_CHECK(k.size(3) == head_dim && v.size(3) == head_dim, "Head dimension must match for all inputs");
    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(qo_heads >= kv_heads && qo_heads % kv_heads == 0, "QO heads must be a multiple of KV heads");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension must match for K and V");

    auto hr = qo_heads / kv_heads;

    bf16 *d_q = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16 *d_k = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16 *d_v = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());

    at::Tensor o = at::empty({batch, qo_heads, seq_len, head_dim}, v.options());
    at::Tensor l_vec = at::empty({batch, qo_heads, seq_len, 1}, q.options().dtype(at::kFloat));

    bf16  *d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    float *d_l = l_vec.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (head_dim == 64) {
        TORCH_CHECK(seq_len % (NUM_WORKERS*ROWS<64>) == 0, "sequence length must be divisible by 128 for D=64");
        if (causal) launch_fwd<64, true >(d_q, d_k, d_v, d_o, d_l, batch, qo_heads, kv_heads, seq_len, hr, stream);
        else        launch_fwd<64, false>(d_q, d_k, d_v, d_o, d_l, batch, qo_heads, kv_heads, seq_len, hr, stream);
    } else {
        TORCH_CHECK(seq_len % (NUM_WORKERS*ROWS<128>) == 0, "sequence length must be divisible by 64 for D=128");
        if (causal) launch_fwd<128, true >(d_q, d_k, d_v, d_o, d_l, batch, qo_heads, kv_heads, seq_len, hr, stream);
        else        launch_fwd<128, false>(d_q, d_k, d_v, d_o, d_l, batch, qo_heads, kv_heads, seq_len, hr, stream);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);
    return {o, l_vec};
}

template<int D, bool is_causal>
static void launch_bwd(bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_og,
                       float *d_qg, float *d_kg, float *d_vg, float *d_l, float *d_dvec,
                       long batch, long qo_heads, long kv_heads, long seq_len, int hr,
                       cudaStream_t stream) {
    using BG = bwd_globals<D>;
    BG g{
        typename BG::qkv_gl{d_q,  (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, nullptr},
        typename BG::qkv_gl{d_k,  (unsigned)batch, (unsigned)kv_heads, (unsigned)seq_len, nullptr},
        typename BG::qkv_gl{d_v,  (unsigned)batch, (unsigned)kv_heads, (unsigned)seq_len, nullptr},
        typename BG::qkv_gl{d_og, (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, nullptr},
        typename BG::grad_gl{d_qg, (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, nullptr},
        typename BG::grad_gl{d_kg, (unsigned)batch, (unsigned)kv_heads, (unsigned)seq_len, nullptr},
        typename BG::grad_gl{d_vg, (unsigned)batch, (unsigned)kv_heads, (unsigned)seq_len, nullptr},
        typename BG::vec_gl{d_l,    (unsigned)batch, (unsigned)qo_heads, 1U, (unsigned)seq_len},
        typename BG::vec_gl{d_dvec, (unsigned)batch, (unsigned)qo_heads, 1U, (unsigned)seq_len},
        hr
    };
    constexpr int WGS = BWD_WGS<D>;
    unsigned long mem_size = 4*WGS*sizeof(st_bf<BWD_TILE, D>)/2 /*k+v*/ + 2*sizeof(st_bf<BWD_TILE, D>)
                           + WGS*sizeof(st_bf<BWD_TILE, BWD_TILE>) + 2*sizeof(sv_fl<BWD_TILE>) + 4096;
    mem_size = 2*WGS*sizeof(st_bf<BWD_TILE, D>) + 2*sizeof(st_bf<BWD_TILE, D>)
             + WGS*sizeof(st_bf<BWD_TILE, BWD_TILE>) + 2*sizeof(sv_fl<BWD_TILE>) + 4096;
    dim3 grid(seq_len/(WGS*BWD_TILE), qo_heads, batch);
    cudaFuncSetAttribute(bwd_attend_ker<D, is_causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    bwd_attend_ker<D, is_causal><<<grid, WGS*4*WARP_THREADS, mem_size, stream>>>(g);
}

std::vector<at::Tensor>
attention_backward(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
                   at::Tensor l_vec, at::Tensor og, bool causal)
{
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    CHECK_INPUT(o); CHECK_INPUT(l_vec); CHECK_INPUT(og);

    auto batch    = q.size(0);
    auto qo_heads = q.size(1);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);
    auto kv_heads = k.size(1);
    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(qo_heads >= kv_heads && qo_heads % kv_heads == 0, "QO heads must be a multiple of KV heads");
    auto hr = qo_heads / kv_heads;

    bf16 *d_q  = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16 *d_k  = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16 *d_v  = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16 *d_o  = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16 *d_og = reinterpret_cast<bf16*>(og.data_ptr<c10::BFloat16>());
    float *d_l = l_vec.data_ptr<float>();

    // gradient buffers are accumulated with atomicAdd -> must start zeroed
    at::Tensor qg = at::zeros_like(q, q.options().dtype(at::kFloat));
    at::Tensor kg = at::zeros_like(k, k.options().dtype(at::kFloat));
    at::Tensor vg = at::zeros_like(v, v.options().dtype(at::kFloat));
    at::Tensor d_vec = at::empty({batch, qo_heads, seq_len, 1}, q.options().dtype(at::kFloat));

    float *d_qg = qg.data_ptr<float>();
    float *d_kg = kg.data_ptr<float>();
    float *d_vg = vg.data_ptr<float>();
    float *d_dv = d_vec.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto run_prep = [&](auto d_const) {
        constexpr int D = decltype(d_const)::value;
        bwd_prep_globals<D> pg{
            typename bwd_prep_globals<D>::io_gl{d_og, (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, nullptr},
            typename bwd_prep_globals<D>::io_gl{d_o,  (unsigned)batch, (unsigned)qo_heads, (unsigned)seq_len, nullptr},
            typename bwd_prep_globals<D>::vec_gl{d_dv, (unsigned)batch, (unsigned)qo_heads, 1U, (unsigned)seq_len}
        };
        dim3 grid(seq_len/64, qo_heads, batch);
        bwd_prep_ker<D><<<grid, 4*WARP_THREADS, 0, stream>>>(pg);
    };

    if (head_dim == 64) {
        TORCH_CHECK(seq_len % (BWD_WGS<64>*BWD_TILE) == 0, "sequence length must be divisible by 128 for D=64");
        run_prep(std::integral_constant<int, 64>{});
        if (causal) launch_bwd<64, true >(d_q, d_k, d_v, d_og, d_qg, d_kg, d_vg, d_l, d_dv, batch, qo_heads, kv_heads, seq_len, hr, stream);
        else        launch_bwd<64, false>(d_q, d_k, d_v, d_og, d_qg, d_kg, d_vg, d_l, d_dv, batch, qo_heads, kv_heads, seq_len, hr, stream);
    } else {
        TORCH_CHECK(seq_len % (BWD_WGS<128>*BWD_TILE) == 0, "sequence length must be divisible by 64 for D=128");
        run_prep(std::integral_constant<int, 128>{});
        if (causal) launch_bwd<128, true >(d_q, d_k, d_v, d_og, d_qg, d_kg, d_vg, d_l, d_dv, batch, qo_heads, kv_heads, seq_len, hr, stream);
        else        launch_bwd<128, false>(d_q, d_k, d_v, d_og, d_qg, d_kg, d_vg, d_l, d_dv, batch, qo_heads, kv_heads, seq_len, hr, stream);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);
    return {qg, kg, vg};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mha_forward", attention_forward, "MHA forward on Ampere. Takes Q,K,V in (B,H,N,D), D in {64,128}; returns (O, L). Supports causal and GQA.");
    m.def("mha_backward", attention_backward, "MHA backward on Ampere. Takes Q,K,V,O,L,dO in (B,H,N,D); returns (dQ, dK, dV) in fp32. Supports causal and GQA.");
}

#else
#include "harness.impl"
#endif
