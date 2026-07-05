/**
 * @file
 * @brief CUDA/SM86 port of MetalForge's EAGLE speculative-decode core
 * (spec_decode.metal): rejection verification (greedy + random) with the
 * externalized-uniform / precomputed-recovered convention, the residual
 * recovered-token argmax, and the eagle_prepare_inputs index math. These
 * complement our spec_verify_linear (which draws its own Gumbel); the EAGLE
 * variants take pre-drawn uniforms and a separate recovered-token pass so a
 * host reference is exact. Metadata kernels are 1 thread/request; the recovered
 * sampler is one warp per draft token (argmax over the residual distribution,
 * smaller-id ties — reuses warp_argmax).
 */
#pragma once
#include "tm_warp.cuh"                   // warp_argmax
#include <cstdint>

namespace tmeg {

// Greedy verify: write target_argmax per draft pos until draft != target, then
// (if none rejected) the bonus token. cu_num_draft_tokens is a prefix sum.
__global__ void rejection_greedy_sample(int64_t* out, const int64_t* target_argmax,
        const int64_t* draft_token_ids, const int64_t* bonus_token_ids,
        const int* cu_num_draft_tokens, const uint8_t* is_greedy, int has_is_greedy,
        int batch, int out_stride, int bonus_stride) {
    const int req = blockIdx.x * blockDim.x + threadIdx.x;
    if (req >= batch) return;
    if (has_is_greedy && !is_greedy[req]) return;
    const long start = req == 0 ? 0 : cu_num_draft_tokens[req - 1];
    const long end = cu_num_draft_tokens[req];
    const long nd = end - start;
    int64_t* row = out + (long)req * out_stride;
    bool rejected = false;
    for (long p = 0; p < nd; ++p) {
        const int64_t tgt = target_argmax[start + p];
        row[p] = tgt;
        if (draft_token_ids[start + p] != tgt) { rejected = true; break; }
    }
    if (!rejected) row[nd] = bonus_token_ids[(long)req * bonus_stride];
}

// Random verify: accept draft iff p_target/q_draft >= uniform_probs[token]
// (q=1 if no_draft_probs); else emit the precomputed recovered token and stop.
__global__ void rejection_random_sample(int64_t* out, const float* target_probs,
        const float* draft_probs, const int64_t* draft_token_ids, const int64_t* bonus_token_ids,
        const int64_t* recovered_token_ids, const float* uniform_probs,
        const int* cu_num_draft_tokens, const uint8_t* is_greedy, int has_is_greedy,
        int no_draft_probs, int batch, int out_stride, int bonus_stride,
        int target_stride, int draft_stride) {
    const int req = blockIdx.x * blockDim.x + threadIdx.x;
    if (req >= batch) return;
    if (has_is_greedy && is_greedy[req]) return;
    const long start = req == 0 ? 0 : cu_num_draft_tokens[req - 1];
    const long end = cu_num_draft_tokens[req];
    const long nd = end - start;
    int64_t* row = out + (long)req * out_stride;
    bool rejected = false;
    for (long p = 0; p < nd; ++p) {
        const long tok = start + p;
        const int64_t did = draft_token_ids[tok];
        const float pt = target_probs[tok * target_stride + did];
        const float q = no_draft_probs ? 1.0f : draft_probs[tok * draft_stride + did];
        const float ratio = q > 0.0f ? pt / q : 0.0f;
        if (ratio >= uniform_probs[tok]) row[p] = did;
        else { row[p] = recovered_token_ids[tok]; rejected = true; break; }
    }
    if (!rejected) row[nd] = bonus_token_ids[(long)req * bonus_stride];
}

// Residual recovered-token argmax: per draft token, over vocab
// prob = (no_draft ? (v==draft?0:p_t) : max(p_t-p_d,0)) * inv_q[v]; argmax
// (smaller id ties). One warp per draft token; binary-search the request.
__global__ void sample_recovered_tokens(int64_t* out, const float* target_probs,
        const float* draft_probs, const int64_t* draft_token_ids, const float* inv_q,
        const int* cu_num_draft_tokens, int no_draft_probs, int batch, int total_draft,
        int vocab, int target_stride, int draft_stride, int inv_q_stride) {
    const int tok = blockIdx.x, lane = threadIdx.x;
    if (tok >= total_draft) return;
    int lo = 0, hi = batch;                           // request that owns this token
    while (lo < hi) { const int mid = lo + ((hi - lo) >> 1);
        if (cu_num_draft_tokens[mid] <= tok) lo = mid + 1; else hi = mid; }
    const int req = lo;
    const int64_t did = draft_token_ids[tok];
    const float* tp = target_probs + (long)tok * target_stride;
    const float* dp = no_draft_probs ? draft_probs : draft_probs + (long)tok * draft_stride;
    const float* iq = inv_q + (long)req * inv_q_stride;

    float best = -1.0f; int bi = vocab;
    for (int v = lane; v < vocab; v += 32) {
        float prob = tp[v];
        if (no_draft_probs) { if (v == did) prob = 0.0f; }
        else { const float d = prob - dp[v]; prob = d > 0.0f ? d : 0.0f; }
        const float val = prob * iq[v];
        if (val > best || (val == best && v < bi)) { best = val; bi = v; }
    }
    tms::warp_argmax(best, bi);
    if (lane == 0) out[tok] = bi;
}

// eagle_prepare_inputs_padded: per request index math for the padded draft
// layout. rejected = num_draft>0 ? num_draft+1-valid : 0; sample index = q_last
// - rejected; also writes num_rejected_tokens.
__global__ void eagle_prepare_inputs_padded(int* token_indices_to_sample,
        int* num_rejected_tokens, const int* cu_num_draft_tokens,
        const int* valid_sampled_tokens_count, const int* query_start_loc, int num_reqs) {
    const int req = blockIdx.x * blockDim.x + threadIdx.x;
    if (req >= num_reqs) return;
    const long start = req == 0 ? 0 : cu_num_draft_tokens[req - 1];
    const long nd = cu_num_draft_tokens[req] - start;
    const long valid = valid_sampled_tokens_count[req];
    const long rej = nd > 0 ? nd + 1 - valid : 0;
    const int q_last = query_start_loc[req + 1] - 1;
    token_indices_to_sample[req] = q_last - int(rej);
    num_rejected_tokens[req] = int(rej);
}

} // namespace tmeg
