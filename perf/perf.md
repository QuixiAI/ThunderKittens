# ThunderKittens (Ampere Edition) Performance Handbook

This is the operating guide for baselining and optimizing every kernel under
`kernels/` on NVIDIA Ampere (SM 8.6, RTX 3090). It mirrors the discipline of the
ThunderMittens Metal handbook (`~/ThunderMittens/perf/perf.md`) but is CUDA- and
Ampere-specific and reflects the **current** state of the repo, not the
compile-only baseline it started from.

The goal is not to collect tricks. Run a disciplined loop: find references, form
a bottleneck hypothesis, measure a clean baseline, run controlled experiments,
keep only verified wins, and record enough detail that the next pass starts from
evidence instead of memory. The running notebook is
`perf/optimization_status.md` (baseline snapshots live in
`perf/baseline_status.md`).

## Principles

Optimization starts from correctness and measurement. A change is not a win until
it passes the kernel's correctness test (pytest and/or the standalone fp64/golden
harness), improves the target metric on realistic shapes, and does not regress a
supported edge shape or the numeric tolerance.

Attack a specific, named bottleneck:

- **Memory-bound** (most decode/elementwise/quant-GEMV): reduce bytes moved,
  coalesce global loads, exploit L2/constant-cache reuse, use narrower formats,
  kill extra global passes. Judge against the 936 GB/s roofline.
- **Compute-bound** (prefill GEMM, large-M): raise arithmetic intensity, feed the
  m16n8k16 tensor cores densely, cut scalar side work, fuse epilogues. Judge
  against ~71 TFLOP/s bf16 / ~284 TOP/s int8.
- **Latency-bound** (serial recurrences, tiny shapes): grow resident work, fix
  launch geometry, unroll serial loops, cut per-lane divergence.
- **Occupancy-bound** (decode kernels with 1 warp/block): more resident warps per
  SM — batch multiple rows/(head,batch) per block, split-K across the grid,
  shrink register/smem footprint. The chip is **82 SMs**; a grid of 128 warps
  leaves it half idle (this is exactly what `qgemm_ksplit` fixed).
- **Sync/smem-bound**: remove needless `__syncthreads`, cut smem traffic and bank
  conflicts, prefer warp-shuffle reductions when cross-warp sharing does not pay.

**The Ampere baseline assumption (mirror of TM's "don't blindly port H100"):**
this hardware is SM86. It has **no `tma::`/tensor-maps and no warpgroup MMA**
(both SM90+), and **no fp8/fp4/mxfp tensor units** (SM89/Blackwell). So:

- Async copy is `cp.async` (`cp.async.ca/cg`, `commit_group`/`wait_group`), not
  TMA. The `wgmma`/warpgroup path is emulated on SM80 (`warpgroup_sm80.cuh`).
- Tensor-core matmul is the `m16n8k16` fp16/bf16 `mma.sync` (see
  `tm_qmm.cuh::mma16816`), fp32 accumulate.
- Quantized compute rides that fp16 mma or native **`dp4a`/IMMA** int8 — the
  manifesto: *storage format is bits at rest; dequant is software in the
  latency shadow; compute rides fp16 mma / dp4a.* Never say "unsupported."
- Watch the register file (255 regs/thread hard cap) and 99KB usable smem/block.
  Spills are real on D=128 backward and the wide linear-attention kernels.
- Known substrate trap (found during the port): the 32-bit row-major
  shared→register fast path in `shared_to_register.cuh` computes addresses
  inconsistent with `st::idx()` on SM80; the SM80 family uses plain idx() scalar
  accesses. Don't "re-optimize" it back without re-verifying with a pattern fill.

## Hardware Reality On This Host

- **8× NVIDIA RTX 3090** (GA102, SM 8.6 / Ampere), 24 GB each, driver 580.65.06,
  CUDA toolkit 12.9, PyTorch (cu130) in `.venv`.
- No NVLink; the multi-GPU `parallel/` kernels (multimem) stay backlogged.

Rooflines for judging results on one 3090:

- **DRAM bandwidth: 936 GB/s** theoretical (memory-bound roofline).
- **BF16/FP16 tensor: ~71 TFLOP/s** dense (fp32 accum; sparsity unused).
- **FP32: ~35.6 TFLOP/s**; **INT8 dp4a/IMMA: ~284 TOP/s**.
- 82 SMs; 6 MB L2; 128 KB L1/smem per SM (**99 KB usable smem/block**);
  64 KB constant cache (warp-uniform indices only — i-quant grid indices are
  divergent, so smem-resident tables win there).

**IMPORTANT — this is no longer a compile-only host.** The original handbook said
"only `layernorm` and cuBLAS baselines execute." That is dead. The full
ThunderMittens port and the MetalForge parity waves are **SM86-native and
validated on these 3090s**: 30 quant formats bit-exact; qgemv/qgemm/qflux/
lm_head; the serving stack (paged v1/v2, MLA, rope_kv, attn_q, varlen, beam,
spec); the elementwise/norm/training family; MoE; linear attention; and the
MetalForge waves (quantized-MoE GEMMs, norm/activation quant, GDN, varlen Mamba
scan, the sampler zoo, EAGLE). `tk_cuda` pytest is green end to end. **Only the
SM90+/Blackwell-native kernels** (`mha_h100*`, `bf16_b300*`, TMA/wgmma demos)
remain build-only here; they need an H100/B200/B300 host.

Canonical idle-3090 numbers already on record (perf floor to beat, not ceiling):

- Weight-only quant **GEMV** (N=512, K=4096): q8_0 **283 GB/s**, e5m2 288,
  fp8 260, mxfp8 254, fp8_block 252, q6_K 212, mxfp6 201, q5_0 171, q4_1 160,
  q4_0/q4_K 156/152, nvfp4 141, mxfp4 136, kU4 137, hqq 145, bitnet/q2_K 91,
  iq4_nl 71, q3_K/iq4_xs 59, iq3_xxs 48, iq2_xxs 36, iq2_xs/iq1_s ~15.
- Quant **GEMM** decode shape (M=64,N=512,K=4096) via `qgemm_ksplit`: q8_0/q4_0
  ~7–8, nvfp4/mxfp4 ~7 TFLOP/s (~2.4× the single-pass tiled kernel).
- Quant **GEMM** prefill (M≥64) routes through dequant-to-fp16 + cuBLAS:
  **~28–30 TFLOP/s** end-to-end vs ~7–9 for the naive per-tile kernel.
- Dense **bf16 GEMM** 44.0 TFLOP/s @4096³ (80% cuBLAS); **mha_ampere** fwd 48.7
  TFLOP/s; **flux** 44.8/47.3; **int8** 50–51 TOP/s; **rotary** peak 763 GB/s;
  **W8A8** 528 GB/s (native dp4a).

Known perf **debts** (start here — the highest-leverage open work):

- **i-quant GEMV** lookup-bound at 15–71 GB/s: divergent `__constant__` grid
  gathers serialize. `dequant8` span specializations cut the re-lookups 8× (done)
  but the grids still want to be **smem-resident**.
- **MX/NV GEMV** at 136–254 GB/s vs q8_0's 283: the per-lane block-scale decode is
  hoisted per 8-span (done) but the format is denser; check for further coalescing.
- **Quant GEMM at M≥64** currently leans on cuBLAS (materializes fp16 W). A
  **fused cp.async-ring dequant-in-pipeline GEMM** would drop the separate dequant
  pass (~40→60 TFLOP/s) — the single largest remaining perf item, appropriately
  backlogged because decode (GEMV/ksplit) already wins.
- **Decode attention / GDN / selective_scan** run one warp per (head,batch) /
  (req,hv,dv) / (dim,batch) — low occupancy. Batch multiple rows per block, or
  measure `gqa_staged` / the partition axis vs v1.

## Repo Facts To Preserve

Kernel sources by family (all SM86-native unless noted):

- **Quant format layer**: `kernels/quant/quant_formats.cuh` (18 bit-arithmetic
  formats + encoders + swizzle), `quant_formats_tables.cuh` + `quant_tables.cuh`
  (GGUF k-/i-quants), `dequant8<FMT>` span helpers, `tm_qmm.cuh`
  (`load_wfrag`/`load_xfrag`/`mma16816`, `fp8_raw`).
- **Quant matmul/decode**: `qgemv.cu`, `qgemm.cu` (+`qgemm_actorder`/`blockscale`/
  `qgemm_ksplit`), `qflux.cu`, `qgemv_int.cu` (w8a8/w2a8, dp4a), `quant_rt.cu`
  (per-token/-tensor fp8/int8), `lm_head*.cu` (fused head + argmax/categorical/
  top-k/top-p), `tm_kernels.cuh` (consolidated).
- **Quantized MoE (MetalForge M1)**: `kernels/moe_quant/tm_moe_quant_kernels.cuh`
  — `moe_gemm_fp8` (rowwise), `moe_gemm_nvfp4` (dual-fp4, swizzled A-scale),
  `moe_gemm_wna16<BIT>` (int4/int8), fused `silu_and_mul_quant`,
  `per_token_group_quant_fp8`, `nvfp4_experts_quant`, `moe_route_scored`.
- **Norm/activation quant (M2)**: `kernels/elementwise/tm_norm_quant_kernels.cuh`
  — `rms_norm_quant<FP8,DYN,RESID>`, `azp_int8_quant`, `per_token_group_int8`.
- **Serving/decode**: `kernels/serving/` — `kv_cache`, `paged_attn_v2`
  (v1/partition/reduce/gqa_staged/cascade), `mla` (bf16/fp8/sparse/partitioned +
  insert), `rope_kv`, `attn_q`, `attn_varlen`, `attn_window`, `beam_xcache`,
  `sampling`, `spec_beam`, plus MetalForge `logits_proc_kernels.cuh` (sampler zoo)
  and `eagle_kernels.cuh`. Shared helpers in `tm_warp.cuh`, RNG in `tm_rng.cuh`.
- **Elementwise/norm/training**: `kernels/elementwise/tm_elementwise_kernels.cuh`
  (rms/layernorm fwd+bwd, add_norm+fp8, softmax, gelu, glu×6, dropout,
  cross_entropy, embedding, hadamard, adamw, add).
- **Linear/state-space**: `kernels/lin_attn_tm/` (linear_attn, causal, chunk-
  parallel, cmplx_matmul, **GDN** `gdn_kernels.cuh`), `kernels/mamba2/`
  (tile-SSD + **varlen selective scan** `selective_scan_kernels.cuh`),
  `kernels/based/`, `kernels/hedgehog/`.
- **Dense/attention base**: `kernels/attention/mha_ampere/`,
  `kernels/gemm/{int8_ampere,fp8_ampere}/`, bf16 GEMM, `kernels/fftconv/`,
  `kernels/rotary/`.
- **SM90+/Blackwell (build-only here)**: `attention/mha_h100*`, `bf16_b300*`,
  any `tma::`/wgmma demo. Registry marks these with a min-CC gate.

**Build & correctness.** The torch extension (cu130) mismatches system CUDA 12.9,
so `cpp_extension` refuses — build the `.so` directly with `nvcc` (command in
`kernels/tm_cuda/setup.py` header; the 5-TU line is in the session notes). Every
standalone kernel has a **self-contained fp64/oracle harness with the build
command in its header comment** (e.g. `kernels/quant/qgemm.cu`,
`kernels/moe_quant/moe_quant_test.cu`, `kernels/serving/logits_proc_test.cu`).
Golden data via `kernels/quant/gen_golden*.py` (imports `~/ThunderMittens`
`quant.py`). The end-to-end gate:

```bash
cd kernels/tm_cuda
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  test_tk_cuda.py test_serving.py test_step4.py test_elementwise.py \
  test_w6.py test_moe_quant.py test_norm_quant.py test_gdn.py \
  test_selective_scan.py test_m4.py -q
```

## Reference Search Protocol

For each kernel, find reference implementations under `.reference/` and record the
exact files in `perf/optimization_status.md`. Roots present: `vllm`,
`flashinfer`, `TensorRT-LLM`, `onnxruntime`, `.reference/vllm/.../marlin/`
(dequant tricks), plus `~/ThunderMittens` (Metal originals + `quant.py`) and
`.reference/metal-forge/` (the MetalForge serving kernels being ported). Re-clone
CUTLASS/CUB or the original ThunderKittens CUDA if a kernel needs them.

Bucket every reference idea into: **portable algorithm** (consider), **SM90+/
Blackwell mechanism** (translate only if SM86 has a real analogue — usually
cp.async + m16n8k16 mma + dp4a), or **benchmark/shape idea** (usually adopt).

```bash
rg -n "cp.async|mma.sync|dp4a|ldmatrix|__shfl|cvta" .reference kernels
rg -n "moe_wna16|marlin|dequant|per_token_group|nvfp4" .reference
```

## Measurement Harness

- **`perf/bench_kernels.py`** — the registry-driven harness. Declares each kernel
  dir's `ARCH`, min CC to execute, build mode (`standalone`/`pytorch`), runner.
  Build phase compiles for the declared arch (SM80 when arch-portable on Ampere)
  and parses ptxas registers/spills/smem; run phase executes only what the host
  can run. Schedules single-GPU benches one per device (`--gpus`); multi-GPU last.
  ```bash
  .venv/bin/python perf/bench_kernels.py --phase all --kernel all
  .venv/bin/python perf/bench_kernels.py --phase run  --kernel qgemv --gpus 0
  ```
- **`perf/sweep_quant.sh`** — canonical 29-format qgemv/qgemm GB/s + TFLOP/s
  table on one idle GPU (the source of the numbers above).
- **`perf/bench_vs_torch.py`** — quant GEMV/GEMM vs `torch.matmul` fp16 at matched
  shapes (relative speedup is contention-robust; absolutes need an idle GPU).
- **`perf/bench_reference.py`** — cuBLAS/SDPA/LayerNorm framework roofline sweep.
- Per-kernel standalone `.out` harnesses print their own timing + correctness;
  record through the harness so metadata lands in `results.jsonl`.

Writes `perf/results/YYYY-MM-DD/<run-id>/{run.json,results.jsonl,summary.md,
build/*.log,run/*.log}` (git-ignored — copy only summary snippets into the
notebook).

**Timing rules (CUDA):** CUDA-event timing around each call, `cudaDeviceSynchronize`
outside the timed region, warm up first (cuBLAS heuristics, autotuners). Do not
regenerate inputs in the timed region. Consumer GPUs throttle — record min/max;
if the spread exceeds ~10%, re-run on an idle box before trusting an A/B. The
external 8-GPU job on this host skews absolutes; the M=1 "2.7× vs torch" number
was a contention artifact that corrected to ~1.4× idle — **absolutes need an idle
GPU; ratios on a shared GPU are still fair.**

Derived metrics:

```text
GEMM FLOPs            = 2*M*N*K
attention fwd FLOPs   ~= 4*B*H*N^2*D        (halve for causal)
quant decode          = effective_GBps = packed_weight_bytes_read / time / 1e9
norm/pointwise bytes  = conservative required reads + writes (note cache reuse)
```

**Profiling:** `kernels/common.mk` provides `make ncu` (Nsight Compute, full
sections) and `make nsys` per kernel. Use them when a timing A/B can't explain a
result — check achieved occupancy, memory throughput %, warp-stall reasons,
`smsp__inst_executed` for dp4a/mma issue, and bank conflicts. Don't commit
profiler artifacts; record their path + a summary.

## Shape Strategy

Don't baseline only square toys. Per family, cover small/edge, tile-aligned,
tile-ragged, real-model, and stress shapes; log skips with a reason.

- **Dense GEMM**: 1024/2048/4096/8192/16384 squares + LLM rectangles
  (`K=4096,N=11008/14336`, small-M prefill/decode).
- **Quant GEMV/GEMM**: LLM projections `N=3840/13824/2560, K=2560/6912`; batch
  sweep `M∈{1,2,4,8,16,32,64,128}` (find the GEMV→ksplit→cuBLAS crossovers);
  all 29 formats.
- **MoE**: E∈{8,16,64}, top-k∈{2,8}, H/inter LLM sizes; padded-schedule tile
  counts vs 82 SMs.
- **Attention/decode**: `(B,H,N,D)` D∈{64,128}, N∈{512…4096+}, causal/non-causal;
  paged decode context 128…8192 with `partition_size` sweeps; GQA ratios.
- **Norm/pointwise/sampler**: rows ∈{4096,16384,65536}, hidden ∈{1024…8192},
  vocab ∈{32k,128k,256k} for lm_head/samplers.
- **Linear/SSM**: `(B,H,N,D)` with N long (the recurrence occupancy stress).

## Per-Kernel Optimization Loop

1. **Inventory** — entry points, dtypes, shape contract, tests, existing bench.
2. **References** — `.reference/`, marlin/vllm/cutlass, TM/MetalForge originals.
3. **Baseline** — correctness first (pytest + standalone harness), then the
   harness on the shape set vs the three baselines (framework, naive-decomposed,
   current-TK). Raw output under `perf/results/`, summary into the notebook.
4. **Classify** — bytes, FLOPs, achieved vs roofline, occupancy from ptxas
   regs/smem + `ncu`, variance.
5. **Experiments** — one factor at a time (below). Hypothesis before edit.
6. **Execute** — small, reversible; focused correctness test first, then the same
   shape matrix, then broader tests for a candidate.
7. **Decide** — keep only if it beats targets by ≥3% (low-risk) / ≥8–10%
   (complexity-adding), regresses nothing that matters, and has a
   counter-backed explanation.
8. **Record** — update `perf/optimization_status.md` (status, tables, decision
   log, rejected alternatives). Commit only when asked; sole author Eric
   Hartford, no AI attribution.

## Experiment Catalogue (Ampere)

**Launch geometry / occupancy.** Grid should saturate 82 SMs with enough resident
warps to hide latency. For 1-warp-per-row decode kernels, batch multiple rows per
block or split-K across `blockIdx.z` (see `qgemm_ksplit`, heuristic
`tiles < ~832` → split). Sweep threads/block (multiple of 32), warps/block,
rows/block. Watch tail effects when grid size isn't a multiple of SM count and
wasted work on partial edge tiles.

**cp.async pipelines.** For GEMM/attention, double/triple-buffer global→smem with
`cp.async.cg` + `commit_group`/`wait_group`; tune ring depth (2 vs 3 stages) vs
smem budget and occupancy. The int8 ring is the proven template. `gemm_staged`'s
lesson still holds: staging only wins if reuse beats the added smem traffic and
occupancy loss.

**mma tiling & fragments.** Sweep BM/BN/BK; feed `m16n8k16` fragments densely
(fewer partial-K tiles). For quant, prefer **dequant-direct-to-fragment**
(`load_wfrag<FMT>`) over a full-dequant global round-trip when weights are used
once; route superblock formats through the fp16 materialize + cuBLAS only at
prefill M.

**Dequant strategy (highest-leverage for quant).** Hoist the block scale / sub-
scale / grid entry to **once per 8-span** (`dequant8<FMT>` specializations — done
for 15 formats). Move i-quant `__constant__` grids to **smem** (divergent indices
defeat the constant cache). Branchless bit-tricks for fp8/fp4 (Marlin: sign |
mantissa-shift + one pow-2 mul). Predecode scales per K-block.

**Memory layout & vectorization.** Ensure adjacent lanes read adjacent addresses
on hot paths. Use `half2`/`float4`/128-bit (`ld.global.v4`) loads where aligned.
`ldmatrix` for register↔shared fragment loads. Compare row-major vs swizzled smem
to kill bank conflicts.

**Reductions & numerics.** `__shfl_xor_sync` warp reductions before smem
reductions; smem cross-warp fixup only when it pays. Keep fp32 accumulation for
softmax/norm/attention/long-K unless a measured lower-precision variant passes
tolerance. Integer/exact kernels: exactness is contract.

**Fusion.** Fuse bias/residual/scale/gelu/gate/norm epilogues that would
round-trip through global (flux/qflux, add_norm, silu_and_mul_quant). Fuse
dequant into matmul/attention when the dequantized value is used once. Split a
fused kernel if register pressure or occupancy loss dominates the saved traffic.

**Branch/scalar hoisting.** Template on format/D/causal/block-k so decisions leave
the inner loop; precompute base offsets and use increments; specialize D=64/128
and aligned K. A little scalar decode can erase the byte savings in qgemv/qgemm/
attn_q/int paths.

**Routing & crossovers.** Find the GEMV→ksplit→cuBLAS-prefill crossover by
sweeping M. Route tiny elementwise to a single fused kernel; batch decode rows.

## Per-Kernel Starting Hypotheses

Replace with measured facts as the round progresses.

- **`qgemv`** (top decode priority). Effective packed-weight GB/s should scale
  with bits/weight. i-quants (15–71 GB/s) are lookup-bound → **smem-resident
  grids**; also try 2 output rows/warp, `half2` X loads, wider vectorized packed
  loads. MX/NV (136–254) vs q8_0 (283): check block-scale coalescing.
- **`qgemm`/`qflux`**. Decode uses `qgemm_ksplit`; prefill uses cuBLAS-on-dequant.
  The open win: a **fused cp.async-ring dequant-in-pipeline GEMM** (dequant into
  fragments inside the ring) to beat the separate dequant pass at M≥64. Also:
  actorder gather cost, `fp8_block` 2D-scale path.
- **`moe_gemm_{fp8,nvfp4,wna16}` (M1)**. Currently one 256-thread block per 32×32
  tile via `mma16816`, no cp.async. Add the **cp.async ring + `_shared_b` 64×64
  staging**, per-expert scale hoist, and measure vs a dequant-to-fp16 + cuBLAS
  grouped path at prefill.
- **`qgemv_int`/`w8a8`** (528 GB/s). Native dp4a; tune output-rows/block,
  activation layout, split-K; the exact-int path only matters where exactness is
  required — else compare to dequant-to-half.
- **`mha_ampere`** (fwd 48.7 TFLOP/s; bwd D=128 spills). Sweep seq block size,
  D-specialize, cp.async double-buffer K/V, dS^T staging, logsumexp storage;
  attack the D=128 backward register spills.
- **`attn_q`** (quantized-KV prefill). Save bandwidth only if K/V dequant doesn't
  dominate — K-dequant-to-shared vs V-dequant-to-register; causal/non-causal
  specialize; format sweep.
- **paged decode (`paged_attention` v1/v2/`gqa_staged`, `mla_decode*`)**. KV-
  bandwidth + occupancy bound. Sweep `partition_size` per context; measure the
  partition-grid axis (v2) and `gqa_staged` smem KV reuse vs v1; batch multiple
  (head,batch) per block to raise occupancy; fp8-cache dequant-on-read cost vs
  bf16.
- **`gdn`/`selective_scan`/`lin_attn*`**. Serial recurrences, one warp per
  (req,hv,dv)/(dim,batch)/(b,h) — low occupancy at short seq. Try chunk-parallel
  decomposition (as `lin_chunk_*`), state layout in smem/registers, `exp2` vs
  `exp` where valid, larger blocks covering more (dv/dim) lanes.
- **norm/`softmax`/`gelu`/`rotary`/samplers/`logits_proc`**. Bandwidth/reduction
  bound. Vectorized contiguous loads, rows/block sweep, warp-only reductions for
  small hidden, fp32 accum, hidden/vocab specialization. lm_head + samplers walk
  the vocab once — coalesce and fuse the head+sample where possible.
- **MoE permute (`route`/`pad`/`gather`/`finalize`)**. atomic contention in
  histogram/scatter; vectorize gather; measure grouped-GEMM vs per-expert
  dispatch crossover at small E.
- **`cmplx_matmul`/`fftconv`**. Register pressure + intermediate traffic; complex
  tile layout, pointwise-complex-mul fusion, fewer inter-stage global writes.

## Decision Rules

**Candidate win** when it: passes the focused correctness test; improves median on
priority realistic shapes by ≥3% (low-risk) or ≥8–10% (complexity-adding);
regresses no required correctness shape or numeric tolerance; regresses no
secondary perf shape beyond the agreed tolerance unless the routing intentionally
narrows the target; and has a bytes/FLOPs/counter-backed explanation.

**Reject/defer** when: the win is inside noise; it appears only on toy shapes; it
adds substantial complexity without a durable real-shape win; it depends on an
SM90+/Blackwell feature this host lacks; or it breaks a numeric contract
(exact-int kernels, RNG bit-parity samplers).

## Recording Format

Each kernel section in `perf/optimization_status.md`:

- Status: not started / baselining / experimenting / candidate / landed / deferred.
- Current best impl + current public route (e.g. "qgemm M≥64 → cuBLAS-on-dequant").
- References inspected (exact files).
- Correctness command + last result.
- Baseline table (framework / naive / current-TK on the shape set).
- Experiment table (one factor per row, before/after, keep/reject + why).
- Decision log + open questions.

## Final Verification Before Landing A Win

```bash
# focused correctness (standalone harness for the touched kernel)
cd kernels/<family> && nvcc <kernel>_test.cu -std=c++17 -O2 \
  -gencode arch=compute_86,code=sm_86 -o <kernel>_test.out -I../quant -I../serving
CUDA_VISIBLE_DEVICES=0 ./<kernel>_test.out
# format sweep if quant
CUDA_VISIBLE_DEVICES=0 bash perf/sweep_quant.sh
# full end-to-end regression (must not drop below the current passing count)
cd kernels/tm_cuda && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q
# golden dequant-exactness for any quant_formats.cuh change
cd kernels/quant && for f in golden/*; do ./qgemv.out "$f"; done | grep -c EXACT
```

For `include/` substrate changes, also run the broader unit tests and the
`perf/tools/sm80_*_smoke.cu` primitive smokes (wgmma/pipeline/lcsf/transpose).

## External References

- NVIDIA **CUDA C++ Programming Guide** & **Best Practices Guide** — occupancy,
  coalescing, cp.async, warp primitives, `__launch_bounds__`.
- **PTX ISA** — `mma.sync.m16n8k16`, `cp.async`, `dp4a`, `ldmatrix`, `cvta`.
- NVIDIA **Ampere GA10x Tuning Guide** — SM86 smem/register budgets, tensor/int
  throughput, async-copy.
- **Marlin** (`.reference/vllm/.../marlin/`) and **CUTLASS** — quant dequant
  tricks, mma fragment layouts, ring pipelines, grouped/MoE GEMM.
- **Nsight Compute / Systems** docs — occupancy, memory-throughput, stall-reason,
  and roofline sections.
- The two companions in this repo: `~/ThunderMittens/perf/perf.md` (the Metal
  handbook this mirrors) and `thundermittens_ampere_port.md` /
  `metalforge_gap_analysis.md` (what exists and what's measured).
