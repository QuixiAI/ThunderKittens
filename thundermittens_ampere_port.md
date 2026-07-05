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

- 2026-07-04 STEP 4 COMPLETE + BINDINGS GREEN: full pytest 75/75 stable
  (3 consecutive runs; test_step4.py 8 + elementwise 22 + w6 7 + serving 10
  + quant 26 + parametrized). New python API: lm_head_sample_topk/topp,
  qgemm_actorder/blockscale, mla_kv_insert, mla_decode_partition,
  mla_decode_fp8_partition, mla_decode_fp8_sparse(partition_size=),
  paged_attention_gqa_staged, attn_window. Test gotcha: qgemm_blockscale
  reference must replay the kernel's HALF code*scale multiply (random e4m3
  codes reach +-448 — fp32-ref rounding gap exceeds tolerance).

- 2026-07-04 STEP 5 CANONICAL SWEEP (idle 3090, external job cleared).
  perf/sweep_quant.sh over all 29 formats, N=512 K=4096. GEMV GB/s:
    q8_0 283, e5m2 288, mxfp8 254, fp8_e4m3 260, fp8_block 252,
    q6_K 212, mxfp6 201/200, q5_0 171, q4_1 160, q5_1 156, q4_0 156,
    q4_K 152, q5_K 143, nvfp4 141, mxfp4 136, kU4/kU4B8 137/135, hqq 145,
    fp4_e2m1 146, bitnet 91, q2_K 91, iq4_nl 71, q3_K 59, iq4_xs 59,
    iq3_xxs 48, iq2_xxs 36, iq2_xs 15, iq1_s 15.
  IMPORTANT CORRECTION: the "2.7x vs torch" from the contended run was an
  ARTIFACT — idle cuBLAS GEMV scales to ~714 GB/s (near the 936 peak), so
  the CANONICAL M=1 N=K=4096 advantage is ~1.4x for 4-bit (q4_0 1.48x,
  q4_K 1.42x, nvfp4 1.33x); mxfp8 0.93x, iq4_nl 0.16x (codebook-bound).
  Honest story: 4-bit quant GEMV beats fp16 torch at decode by moving ~4x
  less weight memory, but our GEMV isn't at peak bandwidth (dequant cost).
  ksplit at the decode shape (idle): q8_0 3.16->7.11, q4_0 2.67->7.08,
  nvfp4 2.00->6.82, mxfp4 2.33->7.00, q4_K 2.58->5.63, iq2_xxs 2.50->5.09
  TFLOP/s (~2.4x over single-pass). All bit-exact. README + this ledger
  corrected to the canonical numbers (the 2.7x line is gone).
- 2026-07-04 STEP 5 PREFILL ROUTE (closes the M>=64 gap the cheap way): the
  qgemm binding now, for M>=64, dequants W to a fp16 (N,K) buffer once
  (dequant_to_fp16<FMT>, all formats) and hands the GEMM to cuBLAS via
  torch::matmul — the whole-W materialization is amortized across the M rows.
  End-to-end (dequant included) at M=256 N=K=4096: q8_0 29.5, q4_0 29.6,
  nvfp4 28.5, mxfp4 29.2, q4_K 27.7 TFLOP/s (iq2_xxs 14.6 — codebook dequant
  bound) vs the old naive 7-9 = ~4x. cuBLAS-only is 40-62 TFLOP/s; the
  dequant pass is the remaining overhead. Decode (M<64) still uses GEMV /
  ksplit (no fp16 W blow-up). rel < 0.035% all formats; pytest 29/29 quant
  (+3 test_qgemm_prefill M=128), 78/78 total. This is the pragmatic answer
  to "cp.async ring" — cuBLAS IS the tuned ring; a hand-written fused
  dequant-in-pipeline ring could save the separate dequant pass (~40->60
  TFLOP/s) but is a large effort for a secondary (prefill) path. STEP 5 DONE.

- 2026-07-04 STEP 5 PERF (part 2): qgemm_ksplit (decode-shape occupancy fix).
  The warp-per-16x16-tile qgemm is only (N/16)*(M/16) warps — at M=64,N=512
  that's 128 warps, half the 82-SM chip idle with no latency hiding. ksplit
  slices the K axis across blockIdx.z (each warp reduces [k_beg,k_end),
  atomicAdd fp32 partials into a zeroed Y), aligned to the format block so a
  slice never splits a block. All 29 formats bit-exact (qgemm + ksplit PASS).
  Under CONTENTION (all 8 GPUs busy), decode shape M=64 N=512 K=4096:
  q8_0 2.2->8.0, q4_0 1.5->8.1, mxfp4 1.4->8.0, nvfp4 0.57->2.6,
  q4_K 0.66->2.0, iq2_xxs 1.4->2.8 TFLOP/s (3-6x). Binding auto-routes the
  plain qgemm through ksplit when the tile count < ~half the chip (832);
  FLUX keeps the single-pass grid (bias+gelu epilogue can't ride the
  partial-sum accumulate). Consolidated in tm_kernels.cuh (tmq::qgemm_ksplit
  + qgemm_pick_kchunk) + standalone qgemm.cu harness.

- 2026-07-04 STEP 5 PERF (part 1, correctness-gated): dequant8 SPAN
  SPECIALIZATIONS landed for 15 formats (q8_0 q4_0 mxfp8 mxfp4 nvfp4
  fp4_e2m1 mxfp6 both + q2_K q3_K q4_K q5_K q6_K + iq4_nl iq4_xs iq2_xxs
  iq2_xs iq3_xxs iq1_s): block scale / packed sub-scale / grid entry decoded
  ONCE per 8-span (i-quants: one 8-byte grid lookup serves the whole span —
  the divergent __constant__ traffic drops 8x). ALL 29 formats still
  dequant-EXACT + gemv PASS; qgemm/qflux/lm_head suites re-verified.
  Directional numbers UNDER CONTENTION (external job on all 8 GPUs):
  mxfp8 85 GB/s, mxfp4 154, nvfp4 160, mxfp6 228 (were 30-80);
  q2_K 103, q4_K 157, q6_K 168 (were 28-165); iq2_xxs 22, iq3_xxs 17,
  iq1_s/iq2_xs ~9 (were 1.2-7). CANONICAL numbers pending idle GPUs.
  REMAINING Step-5 (needs idle GPUs to measure): qgemm cp.async ring /
  44-TFLOP-GEMM dequant route, decode occupancy batching, i-quant smem
  tables (if constant-cache still binds after the 8x lookup cut),
  vs-torch table + final 8-GPU sweep + README perf refresh.

- 2026-07-04 STEP 4 (deferred items) KERNELS COMPLETE, all harnesses green:
  * mla bf16 kv_insert<128/256/512> (norm modes 0/2, neg-slot skip) +
    mla_decode_partition<512,64> + mla_decode_fp8_v<SPARSE,PART> (fp8 sparse
    gather-by-index, dense partition, sparse partition — all combine via the
    existing paged_attention_reduce<bf16,512>): mla.out now 14/14.
  * lm_head top-k (Family-B per-tile partials + Family-A global merge +
    Gumbel-max) and top-p (per-tile tempered lse -> TRUE full-vocab
    normalizer, 32-step bisection): fp16 + all quant formats;
    lm_head_topkp.out 3/3 exact vs fp64 oracle + bit-parity RNG replay.
  * qgemm_actorder (GPTQ act-order, in-kernel X gather by perm — fused into
    the fragment fill) + qgemm_blockscale (fp8_raw codes x (N/128,K/128)
    fp16 tile scales): qgemm_variants.out 2/2 vs fp64 over exactly-
    dequantized weights.
  * paged_attention_gqa_staged (flashinfer-style smem KV staging shared by
    the kv-head's query-head warps): BIT-FOR-BIT equal to v1 (same op
    order). attn_window (Mistral/Gemma sliding-window causal, warp/query
    row): both vs fp64. kv_cache.out now 28/28.
  Step-4 bindings added to tm_cuda_ext.cu (lm_head_sample_topk/topp,
  qgemm_actorder/blockscale) + tm_cuda_serving.cu (mla_kv_insert,
  mla_decode_partition, mla_decode_fp8_partition/sparse, gqa_staged,
  attn_window) + test_step4.py; rebuild in flight.

- 2026-07-04 W6 COMPLETE (coverage): kernels/moe/ (route_topk via
  masked_topk, histogram/scan/scatter permute, 32-pad schedule w/ -1
  sentinels, gather, grouped GEMMs square/rect/fused-swiglu [scalar-FMA
  correctness form; mma route = perf pass], finalize) — moe_test.out 8/8,
  END-TO-END MoE MLP 5.9e-7 vs dense fp64. kernels/lin_attn_tm/ (non-causal
  linear_attn, serial causal scan, 3-kernel chunk-PARALLEL lin_chunk_kv/
  scan/out [column-owner smem state, no barriers], cmplx_matmul) —
  linattn_test.out 4/4. Bindings tm_cuda_w6.cu (moe_route_topk, moe_align,
  moe_gather, moe_gemm(+swiglu), moe_finalize, linear_attn(causal,chunked),
  cmplx_matmul) + test_w6.py.

- 2026-07-04 W3 + W6 BINDINGS GREEN: full pytest 67/67 (elementwise 22 +
  w6 7 + serving 10 + quant 26 + adamw-tolerance fix: torch computes
  sqrt(v)/sqrt(bc2)+eps vs TM sqrt(v/bc2)+eps — fp32 op-order only).
  4-TU extension: tm_cuda_ext + serving + elementwise + w6.
  BUILD GOTCHA: nvcc needs -I../serving; #define inside a macro argument
  (DISPATCH_T(x, {#define HAD...})) does NOT expand — use templated
  launcher fns.

- 2026-07-04 W3 KERNELS LANDED (kernels/elementwise/): tm_elementwise_kernels.cuh
  + elementwise_test.cu — ALL 56 fp64-oracle checks PASS on first full run
  (2 harness bugs only: wrong rstd buffer fed to layernorm_bwd_dx, fp64-vs-fp32
  add tolerance). Coverage: rms_norm/layernorm fwd + BOTH backwards each
  (dx-only Liger factorization w/ host rstd|mean, fully-fused in-kernel stats +
  atomic fp32 dW/dB), add_norm rms+ln with fp8 e4m3 epilogues (static + dynamic
  absmax/448 per-row scale; codes BIT-EXACT vs host RNE encoder replay),
  softmax, gelu fwd/bwd, glu all 6 modes fwd/bwd (reglu/geglu/swiglu/
  swiglu_oai(alpha,limit)/geglu_erf A&S-approx-consistent/geglu_quick; bwds
  verified against fp64 central finite differences of independently transcribed
  forwards), dropout (mask-free counter-RNG, exact replay), cross_entropy
  online-LSE fwd/bwd + _mw 4-warp variants (ignore_index/label_smoothing/
  z_loss/softcap all exact vs fp64 analytic, every config x both schedules),
  embedding lookup + atomic AND sorted-segment backwards + multimodal
  build_src/merge, hadamard FWHT D64/128/256/512 (register+shfl_xor
  butterflies), adamw, add. Bindings TU tm_cuda_elementwise.cu (~30 fns,
  fp32/fp16/bf16 dispatch) + test_elementwise.py (torch autograd oracles)
  written; extension rebuild in flight.

- 2026-07-04 STEP 1 COMPLETE: serving/decode bindings live in tk_cuda.
  tm_cuda_serving.cu (~23 wrappers: kv scatter/gather/copy_blocks, paged v1
  w/ alibi/mask/window + v2 partition/reduce fused host-side, rope_kv/rope_q,
  attn_q (q8_0/q4_0/fp8), mla decode bf16+fp8 / insert / q_norm_rope, sample
  (all modes), penalties/bitmask/bad_words, beam_advance, spec verify
  linear+tree, build_dynamic_tree, spec_compact, kv_meta) + split
  serving/*_kernels.cuh headers (harnesses re-verified). Combined pytest
  36/36 GREEN (test_serving.py 10 + test_tk_cuda.py 26).

- 2026-07-04 W5 COMPLETE (kernels/serving/): sampling.cu 8/8 — argmax/
  categorical/top_k EXACT vs host (bit-identical RNG), top_p/min_p/
  typical_p bisection samplers agree with double oracles, penalties (vLLM
  order + min-length EOS mask) + grammar bitmask + bad-words elementwise-
  exact. spec_beam.cu 3/3 with ZERO mismatches: beam advance (single-pass
  LSE + per-lane top-2BM + Family-B merge; TRT-LLM 2-pass select), spec
  linear rejection sampling + spec_compact (chunked block scan) +
  kv_meta, dynamic-tree build + target-only tree rejection verify
  (lane-0 walk + cooperative terminal Gumbel). The FULL quantized
  inference pipeline is now live on Ampere end-to-end: 30-format weights
  -> qgemm/qgemv/lm_head -> paged + quantized-KV + MLA attention ->
  samplers/beam/spec decode. Remaining waves: W3 elementwise/training,
  W6 misc.

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
  beam_xcache.cu LANDED 3/3 (beam_build_copy_pairs + beam_remap_block_table
  vs CPU ref; kv_cache_scales absmax/240 exact; paged_attention_xcache
  reading vLLM's x-packed layout, exact vs fp64).
  attn_varlen.cu LANDED 3/3: block_exclusive_scan_i32 primitive (reused by
  W5 spec_compact + MoE), varlen_build_worklist (device cu_seqlens -> tile
  worklist + pad offsets, sentinel-filled), pad_gather/regather round trip
  bit-exact, attn_varlen_prefill (ragged + prefix context>=qlen + GQA over
  paged KV, worklist-driven 8-warps-per-tile) exact vs fp64.
  W4 COMPLETE (2026-07-04). Serving total: 66/66 tests. Small follow-ups
  deferred: mla sparse + partitioned variants, mla bf16 insert, the
  gqa_staged shared-memory decode variant (perf), tile/mma prefill shapes
  (perf pass).

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
