/**
 * @file
 * @brief CUDA/SM86 port of MetalForge's M6 misc kernels: the random-sign
 * Fast Walsh-Hadamard rotation at the core of TurboQuant (turboquant.metal),
 * GPTQ/act-order column permute (layout.metal), and per-LoRA MoE alignment
 * (moe.metal moe_lora_align).
 *
 * The FWHT rotation is the reusable QuaRot/SpinQuant-style transform: with a
 * per-index sign vector S and the FWHT matrix H, forward(x) = (1/sqrt(D)) H S x
 * and inverse(y) = S (1/sqrt(D)) H y, so inverse(forward(x)) == x (H^2 = D*I,
 * S^2 = I). One warp per row; lane owns D/32 contiguous elements (low index bits
 * butterflied in registers, the 5 lane bits via __shfl_xor). The sign vector is
 * a runtime buffer (D floats, +-1) rather than MetalForge's embedded 960-entry
 * FWHT_SIGNS tables; the DeepSeek sub-8-bit K-uniform / V-centroid cache codec
 * (tq_encode / tq_load_k_vec) that wraps this rotation is the remaining piece.
 */
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace tm6 {

template <typename T> __device__ __forceinline__ float t6(T v);
template <> __device__ __forceinline__ float t6<float>(float v)                 { return v; }
template <> __device__ __forceinline__ float t6<__half>(__half v)               { return __half2float(v); }
template <> __device__ __forceinline__ float t6<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T f6(float v);
template <> __device__ __forceinline__ float          f6<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half         f6<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16  f6<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// Random-sign FWHT rotation, one warp per row. D in {64,128,256,512}; lane owns
// E=D/32 contiguous elements at [lane*E, lane*E+E). INVERSE=false applies the
// sign before the transform, true after (fused with the 1/sqrt(D) norm).
template <typename T, int D, bool INVERSE>
__global__ void fwht_rotate(const T* __restrict__ x, T* __restrict__ out,
                            const float* __restrict__ sign, int nrows) {
    constexpr int E = D / 32;
    const long row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= nrows) return;
    const long base = row * D + (long)lane * E;
    const float inv_sqrt = rsqrtf(float(D));

    float v[E];
    #pragma unroll
    for (int i = 0; i < E; ++i) {
        v[i] = t6(x[base + i]);
        if (!INVERSE) v[i] *= sign[lane * E + i];          // sign before (forward)
    }
    // in-lane butterflies over the low log2(E) index bits
    #pragma unroll
    for (int h = 1; h < E; h <<= 1) {
        #pragma unroll
        for (int i = 0; i < E; ++i) if ((i & h) == 0) {
            const float a = v[i], b = v[i + h];
            v[i] = a + b; v[i + h] = a - b;
        }
    }
    // cross-lane butterflies over the 5 lane bits (32 lanes = 2^5)
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool upper = (lane & m) != 0;
        #pragma unroll
        for (int i = 0; i < E; ++i) {
            const float p = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = upper ? (p - v[i]) : (v[i] + p);
        }
    }
    #pragma unroll
    for (int i = 0; i < E; ++i) {
        v[i] *= inv_sqrt;
        if (INVERSE) v[i] *= sign[lane * E + i];           // sign after (inverse)
        out[base + i] = f6<T>(v[i]);
    }
}

// GPTQ/act-order column permute (MetalForge permute_cols_16bit): 16-bit gather
// output[r, c] = input[r, perm[c]]. grid (cols, rows); one thread per element.
template <typename T>
__global__ void permute_cols(const T* __restrict__ input, T* __restrict__ output,
                             const int* __restrict__ perm, int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || col >= cols) return;
    output[(long)row * cols + col] = input[(long)row * cols + perm[col]];
}

// Per-LoRA MoE alignment (MetalForge moe_lora_align_int32): one block per
// active LoRA adapter. Histograms token->expert (filtered by token_lora_mapping
// == lora_id), block-pads the per-expert segments, scatters routing-row indices
// into sorted_base, and fills expert_base with the owning expert per padded
// block. sorted/expert bases are per-LoRA slices. num_experts <= 256.
__global__ void moe_lora_align(const int* __restrict__ topk_ids,
        const int* __restrict__ token_lora_mapping, const int* __restrict__ lora_ids,
        const uint8_t* __restrict__ adapter_enabled, int* __restrict__ sorted_token_ids,
        int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_pad,
        int max_loras, int num_experts, int assignments, int topk, int block_size,
        int sorted_capacity_per_lora, int expert_capacity_per_lora) {
    const int lora_idx = blockIdx.x;
    const int tid = threadIdx.x, nt = blockDim.x;
    if (lora_idx >= max_loras || num_experts <= 0 || num_experts > 256) return;
    const int lora_id = lora_ids[lora_idx];
    if (lora_id < 0 || !adapter_enabled[lora_id]) return;

    int* sorted_base = sorted_token_ids + (long)lora_id * sorted_capacity_per_lora;
    int* expert_base = expert_ids + (long)lora_id * expert_capacity_per_lora;
    __shared__ int counts[257];
    __shared__ int offsets[257];

    for (int i = tid; i < sorted_capacity_per_lora; i += nt) sorted_base[i] = assignments;
    for (int i = tid; i < expert_capacity_per_lora; i += nt) expert_base[i] = -1;
    if (tid == 0) num_tokens_post_pad[lora_id] = 0;
    for (int e = tid; e <= num_experts; e += nt) counts[e] = 0;
    __syncthreads();

    for (int i = tid; i < assignments; i += nt) {
        if (token_lora_mapping[i / topk] != lora_id) continue;
        int e = topk_ids[i];
        if (e < 0 || e >= num_experts) e = num_experts;
        atomicAdd(&counts[e], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int pad = 0;
        for (int e = 0; e < num_experts; ++e) {
            offsets[e] = pad;
            pad += ((counts[e] + block_size - 1) / block_size) * block_size;
        }
        offsets[num_experts] = pad;
        num_tokens_post_pad[lora_id] = pad;
    }
    __syncthreads();
    for (int e = tid; e <= num_experts; e += nt) counts[e] = 0;
    __syncthreads();

    for (int i = tid; i < assignments; i += nt) {
        if (token_lora_mapping[i / topk] != lora_id) continue;
        int e = topk_ids[i];
        if (e < 0 || e >= num_experts) e = num_experts;
        const int dst = offsets[e] + atomicAdd(&counts[e], 1);
        if (dst >= 0 && dst < sorted_capacity_per_lora) sorted_base[dst] = i;
    }
    __syncthreads();

    for (int e = tid; e < num_experts; e += nt) {
        const int sb = offsets[e] / block_size, nb = (offsets[e+1] - offsets[e]) / block_size;
        for (int b = 0; b < nb && sb + b < expert_capacity_per_lora; ++b) expert_base[sb + b] = e;
    }
}

} // namespace tm6
