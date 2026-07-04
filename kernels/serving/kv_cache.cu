// Paged KV cache machinery + fused v1 decode attention, CUDA/SM86 port of
// ThunderMittens kernels/kv_cache (core set; fp8 variants + beam kernels follow).
//
// Paged layout (the canonical one all serving kernels share, vLLM-compatible):
//   key_cache/value_cache: (num_blocks, block_size, num_kv_heads, D) contiguous
//   block_table: (batch, max_blocks) int32, block < 0 = unmapped (skipped)
//   slot_mapping: (num_tokens,) int64, slot < 0 = padding (skipped)
//   slot -> address: ((block*block_size + offset)*num_kv_heads + kv_head)*D
//
// paged_attention: one warp per (query head, batch); lane owns D/32 elements;
// online softmax over the context; fused GQA + ALiBi + sliding window +
// block-sparse mask (mask shares the block_table layout).
//
// Build (test harness):
//   /usr/local/cuda/bin/nvcc kv_cache.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o kv_cache.out
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

namespace tms {

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

template <typename T>
__global__ void kv_cache_zero(T* key_cache, T* value_cache, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    key_cache[i] = T(0);
    value_cache[i] = T(0);
}

// one block per token; threads stride the H_KV*D row
template <typename T>
__global__ void kv_cache_scatter(const T* key, const T* value, const int64_t* slot_mapping,
                                 T* key_cache, T* value_cache,
                                 int num_heads, int head_size, int block_size) {
    const int token = blockIdx.x;
    const int64_t slot = slot_mapping[token];
    if (slot < 0) return;
    const int64_t block = slot / block_size, off = slot % block_size;
    const int row_elems = num_heads * head_size;
    const int64_t src = int64_t(token) * row_elems;
    const int64_t dst = (block * block_size + off) * row_elems;
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        key_cache[dst + i] = key[src + i];
        value_cache[dst + i] = value[src + i];
    }
}

// paged -> packed; binary search cu_seq_lens for the owning batch
template <typename T>
__global__ void kv_cache_gather(const T* key_cache, const T* value_cache,
                                T* key_out, T* value_out,
                                const int* block_table, const int* cu_seq_lens,
                                int num_tokens, int num_seqs, int block_size,
                                int bt_stride, int num_heads, int head_size) {
    const int token = blockIdx.x;
    if (token >= num_tokens) return;
    int lo = 0, hi = num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (cu_seq_lens[mid] <= token) lo = mid; else hi = mid - 1;
    }
    const int batch = lo, local = token - cu_seq_lens[batch];
    const int col = local / block_size, slot = local % block_size;
    const int block = block_table[batch * bt_stride + col];
    const int row_elems = num_heads * head_size;
    const int64_t out = int64_t(token) * row_elems;
    if (block < 0) {
        for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
            key_out[out + i] = T(0);
            value_out[out + i] = T(0);
        }
        return;
    }
    const int64_t src = (int64_t(block) * block_size + slot) * row_elems;
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        key_out[out + i] = key_cache[src + i];
        value_out[out + i] = value_cache[src + i];
    }
}

template <typename T>
__global__ void kv_cache_clone(const T* key_cache, const T* value_cache,
                               T* key_out, T* value_out, size_t n) {
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i + 4 <= n) {
        *reinterpret_cast<float2*>(key_out + i)   = *reinterpret_cast<const float2*>(key_cache + i);
        *reinterpret_cast<float2*>(value_out + i) = *reinterpret_cast<const float2*>(value_cache + i);
    } else {
        for (size_t j = i; j < n; j++) { key_out[j] = key_cache[j]; value_out[j] = value_cache[j]; }
    }
}

// src->dst block copies: reads ORIGINAL, writes CLONE (race-free beam-reorder chains)
template <typename T>
__global__ void kv_cache_copy_blocks(const T* key_src, const T* value_src,
                                     T* key_dst, T* value_dst,
                                     const int64_t* block_mapping,   // (num_pairs, 2), -1 sentinel
                                     int block_elems) {
    const int pair = blockIdx.x;
    const int64_t src = block_mapping[2 * pair], dst = block_mapping[2 * pair + 1];
    if (src < 0 || dst < 0) return;
    const int64_t s = src * block_elems, d = dst * block_elems;
    for (int i = threadIdx.x; i < block_elems; i += blockDim.x) {
        key_dst[d + i] = key_src[s + i];
        value_dst[d + i] = value_src[s + i];
    }
}

// ---- fused v1 decode: one warp per (head, batch); GQA + ALiBi + window + block-sparse ----
template <typename T, int D>
__global__ void paged_attention(const T* q, const T* key_cache, const T* value_cache,
                                const int* block_table, const int* context_lens, T* out,
                                int block_size, int bt_stride, float scale,
                                int num_heads, int num_kv_heads,
                                const float* alibi_slopes, int use_alibi,
                                const int* block_mask, int use_mask, int window) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const int64_t row = (int64_t(batch) * num_heads + head) * D;
    const int t_start = (window > 0) ? max(0, context_len - window) : 0;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) {
        qv[i] = float(q[row + lane + 32 * i]);
        acc[i] = 0.0f;
    }
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = t_start; t < context_len; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        if (use_mask && block_mask[batch * bt_stride + col] == 0) continue;
        const int64_t base = (int64_t(block) * block_size + slot) * num_kv_heads * D + int64_t(kv_head) * D;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) partial += qv[i] * float(key_cache[base + lane + 32 * i]);
        float score = warp_sum(partial) * scale;
        if (use_alibi) score += alibi_slopes[head] * float(t - context_len + 1);
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++)
            acc[i] = acc[i] * alpha + beta * float(value_cache[base + lane + 32 * i]);
        l = l * alpha + beta;
        m = nm;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        out[row + lane + 32 * i] = (l == 0.0f) ? T(0) : T(acc[i] / l);
}

}  // namespace tms

// ============================== test harness ==============================
using namespace tms;

static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }

int main() {
    srand(5);
    // scenario: B=3, H=8, H_KV in {8 (MHA), 2 (GQA)}, D in {64,128}, block_size=16
    const int B = 3, H = 8, BS = 16, MAXB = 16;
    const int ctx[B] = {37, 128, 5};
    int rc = 0;
    for (int D : {64, 128}) for (int HKV : {8, 2}) {
        const int num_blocks = B * MAXB + 1;
        const size_t cache_n = size_t(num_blocks) * BS * HKV * D;
        // host paged cache + block table (identity-ish mapping with a shuffled block order)
        std::vector<int> bt(B * MAXB, -1);
        int next_block = 1;   // block 0 left unmapped to exercise skipping
        for (int b = 0; b < B; b++)
            for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = next_block++;
        std::vector<__half> K(cache_n), V(cache_n);
        for (auto& x : K) x = __float2half(frand() * 0.3f);
        for (auto& x : V) x = __float2half(frand() * 0.3f);
        std::vector<__half> q(size_t(B) * H * D);
        for (auto& x : q) x = __float2half(frand());
        std::vector<float> slopes(H);
        for (int h = 0; h < H; h++) slopes[h] = -0.05f * (h + 1);
        std::vector<int> mask(B * MAXB, 1);
        mask[0 * MAXB + 1] = 0;   // knock out one block of batch 0

        __half *dK, *dV, *dq, *dout;
        int *dbt, *dctx, *dmask; float* dsl;
        cudaMalloc(&dK, cache_n * 2); cudaMalloc(&dV, cache_n * 2);
        cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dout, q.size() * 2);
        cudaMalloc(&dbt, bt.size() * 4); cudaMalloc(&dctx, B * 4);
        cudaMalloc(&dmask, mask.size() * 4); cudaMalloc(&dsl, H * 4);
        cudaMemcpy(dK, K.data(), cache_n * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dV, V.data(), cache_n * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dctx, ctx, B * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dmask, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dsl, slopes.data(), H * 4, cudaMemcpyHostToDevice);

        const float scale = 1.0f / sqrtf(float(D));
        for (int variant = 0; variant < 4; variant++) {   // dense, alibi, window, mask
            const int use_alibi = variant == 1, window = variant == 2 ? 24 : 0, use_mask = variant == 3;
            dim3 grid(H, B);
            if (D == 64) paged_attention<__half, 64><<<grid, 32>>>(dq, dK, dV, dbt, dctx, dout, BS, MAXB, scale, H, HKV, dsl, use_alibi, dmask, use_mask, window);
            else         paged_attention<__half, 128><<<grid, 32>>>(dq, dK, dV, dbt, dctx, dout, BS, MAXB, scale, H, HKV, dsl, use_alibi, dmask, use_mask, window);
            cudaDeviceSynchronize();
            if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
            std::vector<__half> got(q.size());
            cudaMemcpy(got.data(), dout, q.size() * 2, cudaMemcpyDeviceToHost);

            // CPU reference (double, full softmax)
            double maxd = 0;
            for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
                const int kvh = h / (H / HKV);
                const int t0 = (window > 0) ? std::max(0, ctx[b] - window) : 0;
                std::vector<double> sc;
                std::vector<int> ts;
                for (int t = t0; t < ctx[b]; t++) {
                    const int col = t / BS, blk = bt[b * MAXB + col];
                    if (blk < 0) continue;
                    if (use_mask && mask[b * MAXB + col] == 0) continue;
                    double s = 0;
                    const size_t base = (size_t(blk) * BS + t % BS) * HKV * D + size_t(kvh) * D;
                    for (int d = 0; d < D; d++)
                        s += double(__half2float(q[(size_t(b) * H + h) * D + d])) * double(__half2float(K[base + d]));
                    s *= scale;
                    if (use_alibi) s += slopes[h] * double(t - ctx[b] + 1);
                    sc.push_back(s); ts.push_back(t);
                }
                double mx = -1e300;
                for (double s : sc) mx = std::max(mx, s);
                std::vector<double> o(D, 0.0);
                double Z = 0;
                for (size_t i = 0; i < sc.size(); i++) {
                    const int t = ts[i], col = t / BS, blk = bt[b * MAXB + col];
                    const double w = exp(sc[i] - mx);
                    const size_t base = (size_t(blk) * BS + t % BS) * HKV * D + size_t(kvh) * D;
                    for (int d = 0; d < D; d++) o[d] += w * double(__half2float(V[base + d]));
                    Z += w;
                }
                for (int d = 0; d < D; d++) {
                    const double ref = Z ? o[d] / Z : 0.0;
                    maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(b) * H + h) * D + d])) - ref));
                }
            }
            const char* names[4] = {"dense", "alibi", "window", "blocksparse"};
            printf("paged_attention D=%d HKV=%d %-11s max diff %.5f (%s)\n", D, HKV, names[variant],
                   maxd, maxd < 5e-3 ? "PASS" : "FAIL");
            rc |= !(maxd < 5e-3);
        }

        // scatter -> gather round trip
        {
            const int T = ctx[0] + ctx[1] + ctx[2];
            std::vector<__half> nk(size_t(T) * HKV * D), nv(nk.size());
            for (auto& x : nk) x = __float2half(frand());
            for (auto& x : nv) x = __float2half(frand());
            std::vector<int64_t> slots(T);
            std::vector<int> cu = {0, ctx[0], ctx[0] + ctx[1], T};
            for (int b = 0, i = 0; b < B; b++)
                for (int t = 0; t < ctx[b]; t++, i++)
                    slots[i] = int64_t(bt[b * MAXB + t / BS]) * BS + t % BS;
            __half *dnk, *dnv, *dgk, *dgv; int64_t* dslot; int* dcu;
            cudaMalloc(&dnk, nk.size() * 2); cudaMalloc(&dnv, nv.size() * 2);
            cudaMalloc(&dgk, nk.size() * 2); cudaMalloc(&dgv, nv.size() * 2);
            cudaMalloc(&dslot, T * 8); cudaMalloc(&dcu, 4 * 4);
            cudaMemcpy(dnk, nk.data(), nk.size() * 2, cudaMemcpyHostToDevice);
            cudaMemcpy(dnv, nv.data(), nv.size() * 2, cudaMemcpyHostToDevice);
            cudaMemcpy(dslot, slots.data(), T * 8, cudaMemcpyHostToDevice);
            cudaMemcpy(dcu, cu.data(), 4 * 4, cudaMemcpyHostToDevice);
            kv_cache_zero<__half><<<int((cache_n + 255) / 256), 256>>>(dK, dV, cache_n);
            kv_cache_scatter<__half><<<T, 128>>>(dnk, dnv, dslot, dK, dV, HKV, D, BS);
            kv_cache_gather<__half><<<T, 128>>>(dK, dV, dgk, dgv, dbt, dcu, T, B, BS, MAXB, HKV, D);
            cudaDeviceSynchronize();
            std::vector<__half> gk(nk.size());
            cudaMemcpy(gk.data(), dgk, nk.size() * 2, cudaMemcpyDeviceToHost);
            size_t bad = 0;
            for (size_t i = 0; i < nk.size(); i++)
                if (__half2float(gk[i]) != __half2float(nk[i])) bad++;
            printf("scatter/gather D=%d HKV=%d round-trip: %zu mismatches (%s)\n", D, HKV, bad, bad ? "FAIL" : "PASS");
            rc |= (bad != 0);
            cudaFree(dnk); cudaFree(dnv); cudaFree(dgk); cudaFree(dgv); cudaFree(dslot); cudaFree(dcu);
        }
        cudaFree(dK); cudaFree(dV); cudaFree(dq); cudaFree(dout);
        cudaFree(dbt); cudaFree(dctx); cudaFree(dmask); cudaFree(dsl);
    }
    return rc;
}
