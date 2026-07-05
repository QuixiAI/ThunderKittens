/**
 * @file
 * @brief CUDA/SM86 port of MetalForge's variable-length Mamba-1 selective scan
 * (selective_scan.metal *_varlen). Scalar per-(dim, state) recurrence — distinct
 * from our tile-SSD kernels/mamba2 (that's the chunked Mamba-2 form); this is the
 * vLLM ragged-batch selective_scan_fwd with paged recurrent state.
 *
 *   delta_t = softplus(delta + delta_bias)                (optional softplus)
 *   S       = exp(delta_t * A) * S + B * delta_t * u      (per state)
 *   out_t   = D*u + sum_state(S * C)   (* silu(z) if has_z)
 *
 * Layout (varlen, packed by total_tokens): u,delta,out,z (dim, total_tokens);
 * A (dim, dstate); B,C (n_groups, dstate, total_tokens); D,delta_bias (dim);
 * state (num_blocks, dim, dstate) paged via cache_indices[batch] (null_block
 * skips), has_initial_state[batch] gates loading prior state. Ragged token
 * ranges from query_start_loc. grid (dim, batch); block round_up(dstate,32).
 * apc (mid-sequence chunk checkpointing) is the remaining M3 follow-up.
 */
#pragma once
#include "../serving/tm_warp.cuh"        // warp_sum_f
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tmss {

template <typename T> __device__ __forceinline__ float sf(T v);
template <> __device__ __forceinline__ float sf<float>(float v)                 { return v; }
template <> __device__ __forceinline__ float sf<__half>(__half v)               { return __half2float(v); }
template <> __device__ __forceinline__ float sf<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T fs(float v);
template <> __device__ __forceinline__ float          fs<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half         fs<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16  fs<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float ss_softplus(float x) {
    return x <= 20.0f ? logf(1.0f + expf(x)) : x;
}

// Block reduce of a per-thread contribution over the first `dstate` threads.
__device__ __forceinline__ float ss_block_reduce(float v, int tid, int dstate, float* sh) {
    v = (tid < dstate) ? v : 0.0f;
    v = tms::warp_sum_f(v);
    if ((tid & 31) == 0) sh[tid >> 5] = v;
    __syncthreads();
    float s = 0.0f;
    if (tid == 0) { const int nw = (blockDim.x + 31) / 32; for (int i = 0; i < nw; ++i) s += sh[i]; }
    return s;   // valid on thread 0
}

template <typename T>
__global__ void selective_scan_fwd_varlen(
        const T* __restrict__ u, const T* __restrict__ delta, const float* __restrict__ A,
        const T* __restrict__ B, const T* __restrict__ C, const float* __restrict__ D,
        const float* __restrict__ delta_bias, const T* __restrict__ z,
        const int* __restrict__ query_start_loc, const int* __restrict__ cache_indices,
        const uint8_t* __restrict__ has_initial_state, T* __restrict__ out, float* __restrict__ state,
        int batch, int dim, int total_tokens, int dstate, int n_groups,
        int has_d, int has_delta_bias, int has_z, int delta_softplus,
        int use_cache_indices, int use_has_initial_state, int null_block_id) {
    extern __shared__ float sh[];
    const int b = blockIdx.y, d = blockIdx.x, tid = threadIdx.x;
    if (b >= batch || d >= dim) return;
    const int group = d / (dim / n_groups);
    const int state_idx = tid;
    const bool active = state_idx < dstate;

    const int cache_idx = use_cache_indices ? cache_indices[b] : b;
    if (use_cache_indices && cache_idx == null_block_id) return;
    const int seq_start = query_start_loc[b], seq_end = query_start_loc[b + 1];

    const long ud_base = (long)d * total_tokens;
    const long bc_base = (long)group * dstate * total_tokens;
    const long state_base = ((long)cache_idx * dim + d) * dstate;
    const float d_val = has_d ? D[d] : 0.0f;
    const float bias = has_delta_bias ? delta_bias[d] : 0.0f;
    const float a_val = active ? A[(long)d * dstate + state_idx] : 0.0f;
    const bool load_init = use_has_initial_state && has_initial_state[b];
    float running = (active && load_init) ? state[state_base + state_idx] : 0.0f;

    for (int t = seq_start; t < seq_end; ++t) {
        const long tok = ud_base + t;
        const float u_val = sf(u[tok]);
        float dv = sf(delta[tok]) + bias;
        if (delta_softplus) dv = ss_softplus(dv);
        float contribution = 0.0f;
        if (active) {
            const long bc = bc_base + (long)state_idx * total_tokens + t;
            running = expf(dv * a_val) * running + sf(B[bc]) * dv * u_val;
            contribution = running * sf(C[bc]);
        }
        const float red = ss_block_reduce(contribution, tid, dstate, sh);
        if (tid == 0) {
            float sum = d_val * u_val + red;
            if (has_z) { const float zv = sf(z[tok]); sum *= zv / (1.0f + expf(-zv)); }
            out[tok] = fs<T>(sum);
        }
        __syncthreads();
    }
    if (active) state[state_base + state_idx] = running;
}

} // namespace tmss
