# ThunderKittens Performance Handbook

This document is the operating guide for baselining and optimizing the kernels
under `kernels/`. It follows the same discipline as the ThunderMittens Metal
handbook (`~/ThunderMittens/perf/perf.md`): find references, form a bottleneck
hypothesis, measure a clean baseline, run controlled experiments, keep only
verified wins, and record enough detail that the next pass starts from evidence
instead of memory.

The running notebook for the effort is `perf/baseline_status.md`.

## Hardware Reality On This Host

Current numbers were collected on:

- 8x NVIDIA GeForce RTX 3090 (GA102, SM 8.6 / Ampere), 24 GB each
- No NVLink fabric assumptions verified; multi-GPU kernels untested here
- Driver 580.65.06, CUDA toolkit 12.9, PyTorch 2.12.1+cu130

This matters because almost every kernel in this repo is written for Hopper or
Blackwell: they use TMA (`tma::`, tensor maps) and warpgroup MMA, which do not
exist before SM 9.0, and the multi-GPU kernels additionally use `multimem`.
**On this host, only arch-portable kernels (currently `layernorm` and the
cuBLAS baseline programs) can execute.** Everything else gets a compile-only
baseline: build health plus ptxas register/spill/smem statistics for its
declared architecture. Full runtime baselines require an H100 (SM90), B200
(SM100), or B300 (SM103) host; the harness will pick up the runnable set
automatically when pointed at one.

RTX 3090 reference rooflines for judging results measured here:

- DRAM bandwidth: 936 GB/s theoretical
- BF16/FP16 tensor-core FLOPs: ~71 TFLOP/s dense (142 with sparsity, not used)
- FP32: ~35.6 TFLOP/s

## Harness

The shared harness is `perf/bench_kernels.py`. Run it from the repo root:

```bash
.venv/bin/python perf/bench_kernels.py --phase all --kernel all
.venv/bin/python perf/bench_kernels.py --phase build --kernel gemm/bf16_h100
.venv/bin/python perf/bench_kernels.py --phase run --kernel layernorm
```

It owns a registry of every kernel directory: declared `ARCH`, minimum compute
capability to execute, build configuration (`standalone`/`pytorch`), and an
optional runner. The build phase compiles for the declared arch (or SM80 when
the source is arch-portable and the host is Ampere) and parses ptxas output;
the run phase executes only what the host can run and records structured
measurements. Single-GPU benchmarks are scheduled in parallel, one per device
(`--gpus` selects which; each kernel gets a dedicated GPU via
`CUDA_VISIBLE_DEVICES`, and the record notes the device). Multi-GPU kernels
run afterwards, serially, owning the whole box. Note the caveat: parallel
runs share host CPU, PCIe, and the power/thermal envelope — for
tight A/B comparisons at large sizes, prefer `--gpus <one-id>` on an idle
box.

Each run writes:

```text
perf/results/YYYY-MM-DD/<run-id>/run.json        # env + invocation metadata
perf/results/YYYY-MM-DD/<run-id>/results.jsonl   # one record per kernel/phase
perf/results/YYYY-MM-DD/<run-id>/build/*.log     # full nvcc/ptxas logs
perf/results/YYYY-MM-DD/<run-id>/run/*.log       # full run logs
perf/results/YYYY-MM-DD/<run-id>/summary.md      # human table
```

`perf/results/` is git-ignored. Copy only summary snippets into
`perf/baseline_status.md`.

Companion scripts:

- `perf/bench_reference.py` — PyTorch framework reference sweep (matmul, SDPA,
  layernorm) on the host GPU. This is the "framework baseline" leg below and
  the device roofline context.
- `perf/tk_bench_layernorm.py` — clean timing for the TK fused layernorm,
  invoked by the harness with `kernels/layernorm` as cwd.

The per-kernel `benchmark.py` scripts that ship beside kernels predate this
harness; several time one call per sync or regenerate inputs inside the timed
region. Prefer the harness for anything that will be recorded.

## Measurement Requirements

Every recorded result must include: git commit, kernel name and entry point,
device, driver, toolkit and framework versions, shape/dtype/flags, warmup and
iteration counts, median plus min/max (or percentiles), correctness tolerance
and observed error, derived throughput (TFLOP/s, GB/s, or tokens/s), and the
raw output location. `bench_kernels.py` records these in JSONL schema v1.

Timing rules:

- CUDA-event timing around each call; synchronize only outside the timed
  region. Warm up before timing (includes cuBLAS heuristics, autotuners, and
  JIT paths).
- Do not regenerate inputs inside the timed region.
- Watch clocks: consumer GPUs throttle. Record variance; if the min/max spread
  exceeds ~10%, re-run with the box idle before trusting a comparison.
- For standalone kernels, keep the shipped self-benchmark output but record it
  through the harness so metadata lands in `results.jsonl`.

Benchmark against three baselines wherever possible:

- Framework baseline: cuBLAS/cuBLASLt, `torch.matmul`,
  `torch.nn.functional.scaled_dot_product_attention`, `nn.LayerNorm`, Triton
  reference kernels shipped in `*/baselines/`.
- Naive decomposed baseline: for fused kernels, materialize the intermediate
  and call framework primitives.
- Current TK baseline: the kernel as it stands before an experiment.

Derived metrics:

```text
GEMM FLOPs            = 2*M*N*K
attention fwd FLOPs   ~= 4*B*H*N^2*D   (halve for causal)
norm/pointwise bytes  = conservative required reads + writes
```

## Shape Strategy

Do not baseline only square toy shapes. Per kernel family:

- GEMM: 1024/2048/4096/8192/16384 squares (the standalone kernels sweep these
  already) plus rectangular LLM shapes (`K=4096, N=11008/14336`, small-M
  prefill/decode).
- Attention: `(B,H,N,D)` with `D in {64,128}`, `N in {512..4096+}`, causal and
  non-causal.
- Norm/pointwise: `b=16`, `n in {1024..16384}`, `d_model` as supported
  (layernorm kernel is hard-coded to `d=1024`, seq divisible by 16, bf16).
- Record skipped shapes with the reason (unsupported contract, OOM, hardware).

## Per-Kernel Loop

1. Inventory: entry points, dtypes, shape contract, tests, existing benchmarks.
2. References: original papers/repos, `kernels/*/baselines/`, upstream TK
   history.
3. Baseline: correctness first, then the harness on the agreed shape set; raw
   output under `perf/results/`, summary into `perf/baseline_status.md`.
4. Classify the bottleneck: bytes, FLOPs, achieved vs roofline, occupancy
   (`ptxas` registers/smem from the build records; `ncu`/`nsys` targets exist
   in `kernels/common.mk`).
5. Experiments: one factor at a time — tile shape, launch geometry, stage
   count, barrier placement, layout/swizzle, fusion, routing.
6. Decide: keep a win only if it beats the target shapes by >=3% (low-risk) or
   >=8-10% (complexity-adding), regresses nothing that matters, and has an
   explanation backed by counters or a clean A/B.
7. Record: update `perf/baseline_status.md` (status, tables, decision log,
   open questions). Commit only when asked, with a normal descriptive message.

## Profiling

`kernels/common.mk` provides `make ncu` (Nsight Compute, full section set) and
`make nsys` (Nsight Systems trace) targets per kernel. Use them when a timing
A/B cannot explain a result. Do not commit profiler artifacts; record their
path and a summary.
