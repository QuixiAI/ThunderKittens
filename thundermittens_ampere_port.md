# ThunderMittens → Ampere/3090 port

Mission: port ALL ThunderMittens kernel families (~44) and ALL quant formats
(30) to CUDA/SM86. No format is "unsupported": storage is bytes, dequant is
software placed in latency shadow, compute rides fp16/bf16 mma (or dp4a/IMMA,
which Ampere HAS and Metal didn't). References: Marlin dequant.h
(.reference/vllm/csrc/libtorch_stable/quantization/marlin/), ThunderMittens
include/ops/warp/register/tile/dequant.metal + kernels/tk/quant.py (golden
reference, pure numpy).

## Format inventory (30) — the foundation layer

Bit-arithmetic tranche (Q1): q8_0, q4_0, q4_1, q5_0, q5_1, kU4, kU4B8,
fp8_e4m3, e5m2, fp8_block, fp4_e2m1, mxfp8, mxfp4, nvfp4, mxfp6_e3m2,
mxfp6_e2m3, bitnet, hqq.
Table/codebook tranche (Q2): q2_K..q6_K (k-quants), iq4_nl, iq4_xs, iq2_xxs,
iq2_xs, iq3_xxs, iq1_s (E8-lattice grids + sign tables from
dequant_tables.metal / ggml-common.h).

CUDA home: `kernels/quant/quant_formats.cuh` — format structs
`{block_k, block_bytes, dequant(const uint8_t* base, int col) -> __half}`
mirroring dequant.metal verbatim; `dequant8<FMT>` span helper. Golden tests
against ~/ThunderMittens/ThunderMittens/kernels/tk/quant.py (numpy, runs
anywhere) — quantize random W, kernel-dequant on GPU, compare exactly.

## Kernel family inventory and status

ALREADY ON AMPERE (from the TK port; may need TM feature parity later):
attn fwd/bwd (mha_ampere), based, hedgehog, linear_attention, mamba2,
fftconv, layernorm, rotary, flux, gemm bf16/int8/fp8(+scaled).

TO PORT (waves; TM sources under ~/ThunderMittens/ThunderMittens/kernels/):
- W1 quant core: quant_formats.cuh + qgemv (all formats, warp-per-row,
  vs fp16 GEMV bar) + quant_rt (runtime quant of activations/KV).
- W2 quant matmul: qgemm (Marlin-style dequant-into-mma k-loop; block scales
  incl. mx/nv variants), qgemm_int + qgemv_int (W8A8/W4A8 — Ampere dp4a +
  IMMA make these BETTER than Metal's idot4 emulation), lm_head (fused
  quantized head + sampling), qflux.
- W3 elementwise/norm family: rms_norm, softmax, gelu, glu(+bwd), add_norm,
  add_rt, dropout, hadamard, cross_entropy (fused-linear CE), embedding
  (+bwd scatter-add).
- W4 attention/serving: kv_cache, rope_kv, paged_attn_v2, attn_varlen,
  attn_q (quantized KV attention over the format layer), mla (paged MLA
  decode), cascade/N-level prefix decode, attn_causal/multiwarp variants
  where they add features over mha_ampere.
- W5 decoding stack: sampling family (top-k/p/min-p/typical/categorical/
  argmax fused Gumbel, penalties, grammar bitmask, bad/stop words), beam
  search (advance + KV reorder), spec decode (linear + tree verify,
  spec_compact, kv meta update).
- W6 misc + training: moe (grouped schedule/gather), optim (fused AdamW),
  cmplx_matmul, gemm_staged/matmul_custom techniques folded into existing
  GEMMs, lin_attn_causal/decay deltas vs existing linear_attention.

Verification per kernel: port the TM correctness test (they live in
kernels/<fam>/correctness/) against torch/numpy on this box; perf bar =
fp16/bf16 baseline of the same op (quant kernels must beat the fp16
equivalent like TM did, adjusted for 3090's compute/bandwidth ratio).

## Status

- 2026-07-03 W4 STARTED, kv_cache core LANDED: kernels/serving/kv_cache.cu —
  paged layout (vLLM-compatible: (blocks, bs, HKV, D); block_table i32 with
  <0 unmapped; slot_mapping i64), kv_cache_zero/scatter/gather/clone/
  copy_blocks + fused v1 paged_attention decode (warp per (head,batch),
  online softmax, GQA + ALiBi + sliding window + block-sparse fused).
  20/20 tests PASS first run (D 64/128 x MHA/GQA x 4 mask variants, max
  6e-5 vs fp64 full-softmax ref; scatter->gather bit-exact round trip).
  paged_attn_v2.cu LANDED 8/8 first run: partition (+fp8 e4m3 per-KV-head
  scale variant) + LSE reduce (D 64/128 + D=512 instantiated for MLA) +
  cascade_prefix_partition; suffix partials compose via flat pointer offset
  (dtmp+Ppre*D, dml+Ppre - valid because shift < row stride, all rows shift
  identically); PS {16,48,512} x window {0,40} + fp8 + cascade all exact
  vs fp64 oracle. rope_kv.cu LANDED 8/8 first run: fused split-half RoPE + paged K/V
  insert (warp per (token,kv_head); optional fused RMSNorm with gemma (1+w)
  flag folded into one templated kernel; rope_q companion). D 64/128.
  attn_q.cu LANDED 8/8 (quantized-KV prefill attention: FMT::dequant inline
  in QK+AV loops, q8_0/q4_0 x D64/128 x causal/full).
  mla.cu LANDED 8/8 first run: mla_q_norm_rope (GPT-J interleaved RoPE +
  norm modes 0/1/2, D=192 V2/V3 + D=512 V4), mla_decode<512,64> bf16
  absorb path (576-wide dot, 512-wide accumulate, MQA headless cache),
  mla_kv_insert_fp8 + mla_decode_fp8 (V4: 448 NoPE e4m3 w/ per-64 UE8M0
  scales + 64 bf16 GPT-J rope, packed 576B+8B rows) - full insert->decode
  round trip vs fp64 oracle over the real cache bytes.
  W4 REMAINING: attn_varlen (worklist builder + head-major pad/gather),
  beam copy-pair/remap kernels, xcache reader, mla sparse + partitioned
  variants, mla bf16 insert.

- 2026-07-03 W2 STARTED, qgemm core LANDED: kernels/quant/qgemm.cu —
  torch-linear semantics Y(M,N)=X(M,K)@dequant(Wq)^T, raw m16n8k16 fp16 mma.
  ALL 29 formats PASS at M=64 (rel <= 0.021%, tolerance 2%): fragment path
  (load_wfrag<FMT> = generic Marlin zero-shuffle over FMT::dequant) for
  block_k<=64, and the TM-style dequant-to-fp16-then-GEMM route for
  superblock formats via an fp16_raw passthrough format (same kernel,
  dequant_to_fp16 prologue). Perf is v1-correctness only (0.6-2.3 TFLOP/s:
  1 warp/16x16 tile, global W reads, M=64) — perf pass = cp.async ring +
  wider warp tiles + reuse the 44-TFLOP GEMM for the dequant route.
  W2 MORE LANDED same day: qgemv_int.cu (w8a8 PASS at 528 GB/s via native
  __dp4a - 2 rows/warp uint4 loads; w2a8 BitNet PASS 144 GB/s, per-group
  scale once per 8-code span; integer oracle golden_int/ via
  gen_golden_int.py). quant_rt.cu: per-token + per-tensor fp8/int8
  quantizers ALL PASS (scale bit-exact = absmax/QMAX; codes correctly
  rounded within half a step; RNE encoders matching np.rint; per-tensor
  absmax via atomicMax on order-preserving float->uint mapping).
  qflux.cu LANDED: gelu_tanh(X@dequant(Wq)^T + bias) via the shared
  tm_qmm.cuh fragment loaders (extracted from qgemm) - 29/29 formats PASS.
  tm_rng.cuh LANDED with numpy bit-parity proven (the W5 contract).
  lm_head.cu LANDED (argmax + Gumbel-max categorical, partial/reduce shape,
  fp16 + ANY quantized format - tested q8_0/q4_0/mxfp4/nvfp4/mxfp8, 16/16
  tokens exact vs fp64 oracle with bit-identical Gumbel; goes beyond TM
  which only shipped q8_0/q4_0 here). warp_argmax helper in lm_head.cu.
  W2 COMPLETE (2026-07-03): torch extension LIVE at kernels/tm_cuda/
  (tk_cuda package: qgemv/qgemm/qflux_gelu over all 29 formats via X-macro
  dispatch, qgemv_w8a8/w2a8, quantize_per_token/per_tensor, lm_head_sample).
  End-to-end pytest 26/26: quantize with TM's quant.py -> torch CUDA tensors
  -> kernels -> numpy oracles. Kernels consolidated in
  kernels/quant/tm_kernels.cuh (single source for harnesses + extension).
  BUILD NOTE: torch cu130 vs system CUDA 12.9 - cpp_extension refuses;
  build the .so directly with nvcc (command in setup.py header comment).
  Deferred to W5/perf-pass: lm_head top-k/top-p (needs masked_topk),
  qgemm_actorder + fp8_block2d blockscale variants, qgemm perf shape.
- 2026-07-03 W1b LANDED: ALL 30 FORMATS NOW BIT-EXACT ON AMPERE.
  kernels/quant/{quant_tables.cuh (converted from dequant_tables.metal:
  iq2xxs/iq2xs/iq3xxs/iq1s grids + ksigns/kmask/kvalues, __constant__),
  quant_formats_tables.cuh (q2_K q3_K q4_K q5_K q6_K iq4_nl iq4_xs iq2_xxs
  iq2_xs iq3_xxs iq1_s)}. Every format dequant-EXACT (max diff 0) vs
  quant.py, gemv PASS. Coverage: 1.56 bits/weight (iq1_s) to 8.5 (q8_0).
  PERF DEBT (expected): i-quants 1.2-7 GB/s in gemv — divergent __constant__
  grid lookups serialize; fix in the perf pass = smem-resident tables +
  port TM's tk_dequant8 span specializations (q2_K/q5_K/q6_K/iq4 exist in
  dequant.metal lines 621-760). k-quants 28-165 GB/s.
- 2026-07-03 W1 LANDED: kernels/quant/{quant_formats.cuh, qgemv.cu,
  gen_golden.py}. All 18 tranche-1 formats DEQUANT-EXACT (bit-for-bit vs
  quant.py float32 reference, max diff 0) and qgemv PASS on RTX 3090 —
  including mxfp8, mxfp4, nvfp4, mxfp6 both variants, bitnet (2.5
  bits/weight). GEMV N=512 K=4096: q8_0 326 GB/s, fp8_e4m3 301, q5_0 196,
  q4_0 166; mx/nv formats currently 30-80 GB/s (per-element scale re-decode
  not yet hoisted; perf pass after coverage). e8m0 scale decode = one
  `__uint_as_float(e << 23)` — the byte IS a float exponent field.

## Design notes

- Formats are per-BYTE definitions from quant.py — the packed tensors are
  identical across Metal and CUDA. Model checkpoints quantized once work on
  both.
- qgemv: warp per output row; lane owns 8-contiguous-col span; block-major
  walk (see qgemv.metal header comment for why). half2 X loads; shuffle
  reduce.
- qgemm: Marlin placement — cp.async ring stays fp-agnostic (bytes),
  dequant in registers between smem fetch and mma_ABt fragment feed, scales
  fetched per k-block (e8m0 scale = exponent-field shift, dequant.h
  kFE8M0fnu shows the bf16 trick).
- Ampere unlocks Metal lacked: dp4a/IMMA int8 (use for W8A8/W4A8 instead of
  idot4 emulation), cp.async, 99KB smem, warp shuffles == simd ops.
- Where TM used constant-memory tables (i-quants): __constant__ or smem-
  resident tables (measure; 3090 constant cache is fine for warp-uniform
  indices, i-quant indices are NOT uniform -> put tables in smem).
