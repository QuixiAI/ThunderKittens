# MetalForge → Ampere: kernel gap analysis

Precise diff of **MetalForge**'s Metal kernel set (`.reference/metal-forge/`, an
Apple-Silicon LLM-serving kernel layer) against our current **CUDA/SM86 port**
(`kernels/`). "Have" = functional equivalence on Ampere, not name identity.

Our port surface compared against: `kernels/quant/` (formats + qgemv/qgemm/
qflux/lm_head/quant_rt/qgemv_int), `kernels/serving/` (paged attn v1/v2, MLA,
rope_kv, attn_q, attn_varlen, beam, sampling, spec), `kernels/elementwise/`,
`kernels/moe/`, `kernels/lin_attn_tm/`, and the P1–P5 base (`mha_ampere`,
`int8_ampere`, `fp8_ampere`, bf16 GEMM, `mamba2`, `fftconv`, `based`,
`linear_attention`, `hedgehog`, `rotary`).

Status legend: ❌ missing (real port) · ⚠️ family present, mode missing · ✅ covered.

---

## ❌ Net-new families missing on Ampere

| # | MetalForge kernels | What it is | Why it's absent here |
|---|---|---|---|
| 1 | **✅ LANDED (Wave M3)** — `gdn_linear_attention*` | **Gated DeltaNet** linear attention | `kernels/lin_attn_tm/gdn_kernels.cuh`: per-token serial recurrence (S*=g; kv=<k,S>; delta=(v-kv)*beta; S+=k*delta; y=<q,S>), GQA + varlen + paged state, one warp per (req,hv,dv). fp64 replay exact; tk_cuda `gdn_linear_attention`. |
| 2 | **✅ LANDED (Wave M1)** — `moe_fp8_gemm_*`, `moe_fp4_gemm_*` (nvfp4, dual-fp4), `moe_wna16_gemm` (int4/int8) | **Quantized MoE grouped GEMM** | `kernels/moe_quant/tm_moe_quant_kernels.cuh` mma-fragment path: `moe_gemm_fp8` (rowwise scale), `moe_gemm_nvfp4` (dual-fp4, swizzled A-scale, per-expert alpha), `moe_gemm_wna16<BIT>` (uint32-interleaved int4/int8) + fused `silu_and_mul_quant` (static/per-block) + `per_token_group_quant_fp8` + `nvfp4_experts_quant` + `moe_route_scored` (sigmoid/softplus). 13/13 fp64/round-trip harness; tk_cuda bindings + `test_moe_quant.py`. |
| 3 | **✅ CORE LANDED (Wave M4)** — `rejection_greedy_sample`, `rejection_random_sample`, `sample_recovered_tokens`, `eagle_prepare_inputs_padded` | **EAGLE speculative decoding** | `kernels/serving/eagle_kernels.cuh`: rejection verify (greedy + externalized-uniform random), residual recovered-token argmax, prepare-inputs index math. Exact host replay. (`eagle_expand`/`copy_and_expand`/`step_slot_mapping` bookkeeping = follow-up.) |
| 4 | `convert_vertical_slash_indexes*`, `tau_tail_*` | **MInference sparse attention** — vertical/slash index builder + tail threshold | No sparse-pattern attention (we have dense + paged + MLA-sparse-by-explicit-index only) |
| 5 | `indexer_k_quant_and_cache`, `cp_gather_indexer_k_quant_cache` | **DeepSeek-V3.2 lightning-indexer** K-quant-and-cache build | We have MLA *sparse decode* (walks a given index list) but not the indexer that *produces* the list + its quantized K cache |
| 6 | `turboquant.metal` (`tq_encode`, `inverse_fwht_in_place`) | **TurboQuant** rotation-based cache-compression codec | Not ported (we have plain hadamard, not the TQ encode/decode codec) |
| 7 | `moe_lora_align`, `layout_lora` | **LoRA** apply + MoE-LoRA alignment | No LoRA kernels at all |
| 8 | **✅ LANDED (Wave M4)** — DRY, no-repeat-ngram, top-a, top-nσ, η/ε-cutoff, XTC, quadratic, skew | **Sampler tail-cutoff zoo** | `kernels/serving/logits_proc_kernels.cuh`: all 9 processors (warp-per-row masks + skew CDF block-scan + serial ngram/DRY). 9/9 exact / fp64. Deterministic — the draw stays on `rng_gumbel`. |
| 9 | `silu_and_mul_fp8_quant`, `silu_and_mul_per_block_{fp8,int8}_quant` | **Fused activation → quantized output** (SwiGLU that emits fp8/int8 in one pass) | Our GLU emits bf16/fp16/fp32; fp8 quant is only fused into the *norm* (add_norm), not the activation |

---

## ⚠️ Partial / variant gaps (family present, mode missing)

| Area | Have | Missing vs MetalForge |
|---|---|---|
| **int8 quant** | ✅ **M2** — `azp_int8_quant` (dyn min/max + zp, [-128,127]); `per_token_group_int8_quant`; M1 `per_token_group_quant_fp8` | (done) |
| **norm+quant epilogues** | ✅ **M2** — `rms_norm_quant<FP8,DYN,RESID>` (fp8/int8, static/dynamic, ±residual) + M1 fused `silu_and_mul` fp8/int8 per-block | per-block *norm* fp8 + `rms_norm_silu` remain (minor) |
| **MoE routing** | one `moe_route_topk` (softmax renorm) | sigmoid / softplus-sqrt / **grouped-with-group-limit** (DeepSeek), bias-dtype variants |
| **fp4 quantize side** | fp4 *dequant* in all GEMMs | `scaled_{mxfp4,nvfp4}_experts_quant` (the *encode* path for MoE fp4) |
| **merge_attn_states** | LSE merge (`paged_attention_reduce`) | **fp8-output** merge variant |
| **KV fp8 scales** | `kv_cache_scales` (absmax) | running `kv_scale_update` |
| **Mamba selective scan** | `mamba2` SSD + ✅ **M3** `selective_scan_fwd_varlen` (ragged Mamba-1 scan, paged state, softplus/D/z/n_groups) | **APC** (mid-sequence chunk checkpointing) remains |
| **QK-norm+RoPE** | `mla_q_norm_rope`, `rope_kv` w/ norm | **MiniMax** `minimax_qk_rms_norm` specific variant |
| **bitmask** | `apply_token_bitmask` (consume) | `packbits` (produce the bitmask) |
| **column permute** | folded into `qgemm_actorder` (in-kernel gather) | standalone `permute_cols_16bit` (materialized) |

---

## ✅ Fully covered (no gap)

Paged attention v1/v2 (+ GQA-staged, partitioned, cascade) · MLA decode
(bf16/fp8/sparse/partitioned) + insert · `reshape_and_cache` / `copy_blocks` /
`gather` · RMSNorm/LayerNorm (+fused add) · RoPE · SwiGLU/GeGLU ×6 · softmax ·
fp8/int8/fp4/q4 dequant + **the full 30-format GGUF k-quant/i-quant zoo**
(MetalForge has *fewer* formats here) · q4 rowwise GEMV/GEMM · hadamard FWHT ·
W8A8 / BitNet W2A8 (native `dp4a`) · core samplers + rep-penalty / bad-words /
bitmask · bf16/int8/fp8 dense GEMM · flash attention (dense) · MoE permute
pipeline (align / gather / finalize).

## 🎁 We have that MetalForge doesn't

MetalForge is **inference-only**, so it lacks everything on the training side we
ported: **attention backward** (FA2), **cross-entropy** (+ z-loss / softcap /
mw), **embedding backward** (atomic + sorted-segment), **GLU / GELU backward**,
**dropout**, **AdamW**, and the two **fused norm backwards**. We also carry
**more quant formats** (all GGUF k/i-quants) and **fftconv**.

---

## Suggested port priority (serving impact × effort)

1. **Quantized MoE grouped GEMM** (#2) — highest impact; MoE serving is
   memory-bound on expert weights, and we already have the format decoders
   (`load_wfrag<FMT>`, `dequant8<FMT>`) + the permute pipeline
   (`tm_moe_kernels.cuh`), so this is "wire the dequant into the grouped GEMM."
2. **Fused activation→fp8 quant** (#9) + **int8 / per-block norm epilogues**
   (⚠️) — small deltas on kernels we already own, real bandwidth wins.
3. **GDN linear attention** (#1) — one self-contained kernel; matters for
   Qwen3-Next-class models.
4. **Sampler zoo** (#8) — each is a cheap elementwise logits pass; batchable.
5. **EAGLE** (#3), **MInference sparse** (#4), **indexer** (#5) — larger,
   model-specific; do only if targeting those stacks.
6. **TurboQuant** (#6), **LoRA** (#7) — niche; defer.

---

*Generated 2026-07-05 from `.reference/metal-forge/csrc/metalforge/kernels/`
(29 native `.metal` files + vendored MLX Steel headers) vs `kernels/` on `main`.*
