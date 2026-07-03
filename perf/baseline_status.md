# ThunderKittens Kernel Baseline Status

Running notebook for the performance-baseline effort. Method and harness are
described in `perf/perf.md`. Raw results live under `perf/results/` (git-ignored).

## Environment (all numbers below)

- Host: 8x NVIDIA GeForce RTX 3090 (SM 8.6 / Ampere, 24 GB), driver 580.65.06
- CUDA toolkit 12.9 (`/usr/local/cuda`), PyTorch 2.12.1+cu130, Python 3.12.3
- Repo: commit `02e9acbd` + perf harness + two small fixes (see Decision Log)
- Date: 2026-07-03

## The Hardware Gap (read first)

Every kernel in this repo except `layernorm` and the cuBLAS baseline programs
uses TMA and/or WGMMA/tcgen05, which require SM 9.0+ (H100) or SM 10.x (B200/
B300). The multi-GPU `parallel/` kernels additionally use `multimem`. **None of
these can execute on SM 8.6 hardware.** The runtime baseline on this host is
therefore: `layernorm` (rebuilt for SM80) plus cuBLAS reference GEMMs plus
PyTorch framework references. Every other kernel has a compile-only baseline:
build health and ptxas register/spill statistics for its declared arch.
Completing runtime baselines requires an H100/B200/B300 host; the harness
(`perf/bench_kernels.py`) will pick up the runnable set automatically there —
the standalone GEMM/attention kernels self-benchmark and their output is
captured/parsed by the run phase.

## Build Baseline — 42/42 kernels compile cleanly

Run: `perf/results/2026-07-03/baseline-build` (+ `lcf-rebuild`). Highlights,
worst kernel per directory (`regs` = max registers across entry points,
`spill` = total spill bytes reported by ptxas):

| kernel | arch | regs | spill B | note |
|---|---|---|---|---|
| attention/mha_h100 | SM90 | 252 | 1464 | spills: backward pass candidates |
| attention/mha_h100_lcf | SM90 | 254 | 0 | fixed missing `<chrono>` include |
| attention/bf16_b300_mha_causal | SM103 | 128 | 640 | |
| attention/bf16_b300_mha_noncausal | SM103 | 128 | 96 | |
| based | SM90 | 255 | 0 | at register ceiling |
| fftconv | SM90 | 168 | 760 | |
| flux | SM90 | 232 | 1488 | largest spill count |
| hedgehog | SM90 | 252 | 0 | |
| layernorm | SM80 | 96 | 0 | built SM80 to run on this host |
| linear_attention | SM90 | 255 | 0 | at register ceiling |
| mamba2 | SM90 | 168 | 0 | |
| rotary | SM90 | 168 | 144 | |
| gemm/bf16_h100, fp8_h100, fp8_h100_scaled | SM90 | 168 | 0 | |
| gemm/int8_h100 | SM90 | 168 | 32 | |
| gemm/bf16_b200, fp8_b200 | SM100 | 255 | 0 | |
| gemm/int8_b200 | SM100 | 255 | 1752 | largest GEMM spill |
| gemm/mxfp8_b200 | SM100 | 186 | 0 | |
| gemm/nvfp4_b200 | SM100 | 174 | 0 | |
| gemm/educational_{h100,b200} | SM90/100 | 32/30 | 0 | level 01 only |
| gemm/baselines/* (cuBLAS/Lt) | SM80/SM100 | ~30 | 0 | bf16+int8 rebuilt SM80 |
| parallel/* (13 kernels) | SM90/SM100 | 8–255 | ≤16 | all compile |

Full per-kernel table: `perf/results/2026-07-03/baseline-build/results.jsonl`
and `summary.md` there.

## Runtime Baseline — layernorm (the one executable TK kernel)

`fused_layernorm` = dropout + residual-add + LayerNorm, bf16, `d_model=1024`
(hard-coded), seq divisible by 16. Built `ARCH=SM80`, run on one RTX 3090.
Benchmark: `perf/tk_bench_layernorm.py` (CUDA events, 10 warmup, 50 iters x 3
repeats, median). Archived: `perf/results/2026-07-03/baseline-run/`.

Correctness (dropout_p=0 deterministic path): max abs err out = 0.052,
resid = 0.016 vs fp32 reference — passes bf16 tolerance 0.15.

Shapes `b=16, d=1024`, dropout_p=0.1; GB/s assumes 8 B/element moved:

| n | TK ms | TK GB/s | torch eager ms | triton fused ms | TK vs torch | TK vs triton |
|---|---|---|---|---|---|---|
| 1024 | 0.258 | 520 | 0.814 | 0.256 | 3.2x | 0.99x |
| 2048 | 0.482 | 557 | 1.972 | 0.510 | 4.1x | 1.06x |
| 4096 | 1.183 | 454 | 4.001 | 1.166 | 3.4x | 0.99x |
| 8192 | 2.387 | 450 | 8.332 | 2.391 | 3.5x | 1.00x |
| 16384 | 4.766 | 451 | 16.653 | 4.605 | 3.5x | 0.97x |

Reading: TK fused layernorm is ~3.2–4.1x faster than the eager PyTorch
composite and statistically tied with the FlashAttention Triton fused kernel
(±3%). At ~450–557 GB/s it reaches 48–60% of the 3090's 936 GB/s DRAM peak —
plausible for a dropout+curand+two-output kernel, and a reasonable
optimization target to revisit on real target hardware rather than here.

Note: in `baseline-run-parallel` (three GEMM sweeps running on neighboring
GPUs simultaneously) the same kernel read 381–418 GB/s — a ~10–15% haircut
from shared power/host contention. The isolated `layernorm-run` numbers above
are canonical; the relative ordering (TK ≈ triton >> torch eager) is identical
in both runs.

## Reference GEMM Baseline — cuBLAS on RTX 3090

`gemm/baselines/{bf16_cublas,bf16_cublas_lt,int8_cublas_lt}` rebuilt
`ARCH=SM80`, 500 warmup / 100 iters each. Canonical numbers below are from
`baseline-run-parallel` (each benchmark alone on an idle GPU):

| M=N=K | bf16 cuBLAS TFLOP/s | bf16 cuBLASLt TFLOP/s | int8 cuBLASLt TOP/s |
|---|---|---|---|
| 1024 | 46.5 | 46.8 | 102.1 |
| 2048 | 49.2 | 49.7 | 88.5 |
| 4096 | 54.8 | 54.7 | 87.6 |
| 8192 | 57.5 | 57.4 | 125.8 |
| 16384 | 56.4 | 56.3 | 122.3 |

Peak observed: bf16 ~57.5 TFLOP/s (~81% of the 3090's ~71 TFLOP/s dense bf16
peak); int8 ~126 TOP/s (~89% of the ~142 TOP/s dense int8 peak). cuBLAS and
cuBLASLt are equivalent for bf16 on this device.

Measurement-hygiene lesson (kept because it will bite again): the first,
serial run executed all four benchmarks back-to-back on GPU 0 and produced
wild swings — bf16 cuBLAS at 16384^3 read 56.0 then 32.3 TFLOP/s across runs,
and cuBLASLt read 15–30 TFLOP/s at small sizes — pure thermal/clock and
contention artifacts, not library behavior. Judge large-size numbers only
from an idle, cool GPU, and re-measure anything surprising before believing
it.

## Framework Reference — PyTorch on RTX 3090

`perf/bench_reference.py`, archived at
`perf/results/reference/rtx3090_reference.json`.

- bf16 `torch.matmul`: 45–60 TFLOP/s on square 1024–16384; LLM projection
  shapes (`M=2048, N=11008/14336, K=4096`) 55–56 TFLOP/s; small-M decode
  (`M=16`) drops to ~9 TFLOP/s (memory-bound).
- bf16 SDPA (flash backend): 49–55 TFLOP/s non-causal at N>=1024 (D=64 and
  128); causal 33–48 TFLOP/s.
- bf16 `nn.LayerNorm` (norm only, 4 B/element): 705–785 GB/s at n<=2048,
  ~318–338 GB/s at n>=4096.

These are the numbers TK kernels must beat (or match with better fusion) on
this device class.

## Per-Kernel Status Table

Status legend: `build-only` = compiles for declared arch, cannot execute here;
`baselined` = runtime numbers recorded on this host.

| kernel | status | runtime blocker |
|---|---|---|
| layernorm | **baselined** | — |
| gemm/baselines/bf16_cublas | **baselined** | — |
| gemm/baselines/bf16_cublas_lt | **baselined** | — |
| gemm/baselines/int8_cublas_lt | **baselined** (int8 IMMA works on Ampere) | — |
| attention/* | build-only | TMA/WGMMA (SM90) or SM103 |
| based, hedgehog, linear_attention, mamba2 | build-only | TMA/WGMMA (SM90) |
| fftconv, flux, rotary | build-only | TMA/WGMMA (SM90) |
| gemm/*_h100 | build-only | TMA/WGMMA (SM90) |
| gemm/*_b200 | build-only | tcgen05/TMA (SM100) |
| gemm/baselines/{fp8,mxfp8,nvfp4}_cublas_lt | build-only | fp8/fp4 hw (SM89/SM100+) |
| parallel/* (13) | build-only | multimem + TMA (SM90+), NVLink fabric |

## Decision Log

- 2026-07-03: Added `perf/` harness (`bench_kernels.py`, `tk_bench_layernorm.py`,
  `bench_reference.py`, `perf.md`, this file). `perf/results/` git-ignored.
- 2026-07-03: Fixed `include/pyutils/torchutils.cuh` to guard PGL /
  `TKParallelTensor` machinery behind `KITTENS_SM90/SM10X/SM120` (it broke every
  SM80 PyTorch-extension build) and to include `<torch/csrc/utils/pybind.h>`
  directly (was inherited transitively via `parallel_tensor.cuh`). SM90+ builds
  verified unaffected (all pytorch-config kernels rebuilt cleanly after the
  change).
- 2026-07-03: Fixed missing `#include <chrono>` in
  `kernels/attention/mha_h100_lcf/mha_h100_lcf.cu` (pre-existing build failure).
- 2026-07-03: Rebuilding arch-portable kernels with `ARCH=SM80` for execution on
  Ampere is done automatically by the harness (`sm80_ok` registry flag).

## Ampere (SM86) Port Status — 2026-07-03

The SM90+ kernels are being ported to run on this box's 3090s (plan:
`~/.claude/plans/all-the-kernels-that-serene-storm.md`). Library groundwork
(KITTENS_SM86 target, cp.async producer primitives, WGMMA emulation in
`include/ops/group/mma/warpgroup_sm80.cuh`) is landed and validated; SM90
compilation output is unchanged (PTX-verified). Validation tools live in
`perf/tools/` (sm80_*_smoke.cu).

| kernel | SM86 status | numbers (3090; * = under external GPU contention) |
|---|---|---|
| rotary (same-source) | CORRECT | IDLE GPUs: peak 763 GB/s (d64 n4096), 480–600 GB/s other shapes |
| attention/mha_ampere fwd (new dir; full mha_h100 parity) | CORRECT: causal+GQA+L, D=64/128 (o err <=1.1e-3, L err <=6e-4 vs fp32 ref) | IDLE GPUs: 48.7 TFLOP/s D=64 non-causal harness |
| attention/mha_ampere bwd (NEW kernel, no Ampere ancestor) | CORRECT: dQ/dK/dV <=7.5e-4 max vs torch fp32 autograd, all configs (D 64/128 x causal x GQA) | untuned; D=128 spills (1 wg/block) |
| gemm/bf16_h100 (same-source, M2N2 stages3) | CORRECT | IDLE GPUs: 44.0 TFLOP/s @4096³ = 80% cuBLAS (54.9) |
| flux/flux_gate (same-source, 128x128x64) | CORRECT (err count 0) | IDLE GPUs: 44.8 TFLOP/s |
| flux/flux_gelu (same-source) | builds; run pending | — |
| gemm/int8_ampere (new dir) | EXACT (err 0) | IDLE, tuned: 50.4 @4096³ / 51.5 @8192³ TOP/s (per-size configs: staged-writeback Nb=64 small, direct-register-writeback Nb=128 large; was 41/36) vs cuBLASLt 85.6/123 |
| mamba2 (same-source, NWG=1 on Ampere) | CORRECT | 0.39% mean rel err vs fp32 ssd_minimal |
| fftconv_non_pc (same-source + chrono fix) | PASSED (0/16.7M violations) | 15.1 ms* full B=4 H=1024 N=4096 |
| fftconv_pc | A100-only (112KB scratch > 99KB) | untested — no A100 on this box |
| flux/flux_gelu (same-source) | CORRECT (err count 0) | IDLE GPUs: 47.3 TFLOP/s |
| based/lin_attn_ampere (resurrected 50ee1f0a 4090 kernel) | CORRECT: randn max 0.164 / avg 0.0149 = exactly the bf16-state precision floor (validated vs exact-precision emulation) | 17.2 TFLOP/s* B16 H16 N1024 |
| linear_attention_ampere (new copy: single-buffered, no full-k tile, merged o store, 96.5KB) | CORRECT first run: rel_diff 0.127%, max 0.0078 | 209 us* B1 H8 N1024 D=F=128 |
| hedgehog_ampere (new copy: 1-slot k/v with cross-block carry, aliased kv/scratch staging, 97.5KB) | CORRECT first run (randn): O rel 0.81%, KV state rel 1.0%, K state rel 0.18% | 517 us* B2 H2 N1024 |
| gemm/fp8_ampere (new dir: fp8e4m3 STORAGE, Marlin bit-trick dequant to fp16 in registers, fp16 mma) | EXACT (max err 0 vs fp32 ref) | IDLE: 23.1 TFLOP/s @8192³ (fast dequant = 5x over __nv_cvt) |
| gemm/fp8_ampere scaled (per-row/col scales, port of fp8_h100_scaled) | PASS (max err 1.5e-5) | IDLE: 23.7 TFLOP/s @8192³ |
| BAR1 P2P driver (.reference/open-gpu-kernel-modules-580.65.06, tinygrad 9e39420bc4cb adapted to 580: 4 files) | all 5 .ko BUILT for 6.8.0-110-generic | awaiting user install: BIOS ReBAR (BAR1 now 256MiB) + sudo; see README-BAR1-P2P.md |
| unit tests ARCH=SM86 | 2467/2468 | 1 pre-existing warp complex-mma fail |

Upstream bugs found & fixed during the port (affect SM90 correctness too
where noted):
1. `kernels/rotary/rotary.cu`: sin/cos loaded with an element-typed
   `kittens::coord` (reads rows w..w+15 instead of 16w..) — wrong on all
   arches; the shipped test_correctness.py's large "tk diff" was this.
2. `include/ops/thread/util/util.cuh`: `move<int8_4>/<uint8_4>` (ldmatrix
   wrappers) were inside the fp8 SM90+ guard; int8 is an SM80 feature.
3. `include/ops/group/memory/tile/shared_to_register.cuh`: the 32-bit
   row-major register<->shared fast path (hand-rolled swizzle+blit
   addressing) writes addresses inconsistent with `st::idx()` (proven with a
   pattern-fill test on SM86: element (0,28) landed at (1,0)). SM80 family
   now uses canonical idx()-based accesses; SM90 keeps the fast path.
   NOTE: worth auditing on H100 too — unit tests don't cover int
   shared<->register.
4. Missing `#include <chrono>` in mha_h100_lcf.cu, the 4090 harness, and
   flux_gate.cu (RUN_MAIN paths).
5. based test harness (50ee1f0a `lin_attn_4090_harness.impl`, now
   `harness_ampere.impl`): head-replication used modulus `ATTN_B*ATTN_H`
   (=256) instead of elements-per-head, so the device saw v rows 0-3 tiled
   everywhere; and the check indexed `o_ref[i]` out of bounds for heads > 0
   (o_ref holds one head). The old "based is broken" impressions likely trace
   here — the kernel matches an exact-precision emulation to 7.8e-3 max.
6. based test generator (`generate_tests_ampere.py`, from the old repo):
   `pytorch_test(q, k, v, TESTNAME)` passes TESTNAME positionally into
   `add_scale`, forcing the scaled reference (and making every per-term test
   file identical to all-terms). The kernel computes the UNSCALED convention;
   fixed by calling with `add_scale=False, TESTNAME=TESTNAME`.

## Run Index

| run | contents |
|---|---|
| `perf/results/2026-07-03/baseline-build/` | build phase, all 42 kernels, ptxas stats |
| `perf/results/2026-07-03/lcf-rebuild/` | mha_h100_lcf rebuild after chrono fix |
| `perf/results/2026-07-03/baseline-run/` | serial run phase (superseded; thermal artifacts) |
| `perf/results/2026-07-03/baseline-run-parallel/` | **canonical run**: one benchmark per idle GPU (4 ok / 38 skip) |
| `perf/results/2026-07-03/int8-rerun/` | int8_cublas_lt re-record with TOP/s parsing |
| `perf/results/2026-07-03/layernorm-run/` | first archived layernorm run |
| `perf/results/reference/rtx3090_reference.json` | torch matmul/SDPA/layernorm sweep |

## Open Questions

- Which target device should runtime baselines be recorded on for the
  SM90/SM100 kernels — is an H100/B200 host available to this project?
- The educational GEMM levels 02–08 are compile-checked at level 01 only;
  sweep them if they become optimization targets.
- `parallel/` kernels additionally need a multi-GPU fabric with multicast
  support; 8x 3090 (PCIe/pairwise NVLink) cannot validate them even if the
  arch gap were closed.
