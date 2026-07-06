# QuixiCore CUDA Optimization Status

This is the running notebook for CUDA kernel optimization. Raw output belongs
under `perf/results/`; stable conclusions belong here. Historical and baseline
snapshots live in `perf/baseline_status.md`.

## Entry Template

Use this structure for every kernel family or optimization pass:

```text
## YYYY-MM-DD: <kernel or pass name>

Status: not started | baselining | experimenting | candidate | landed | deferred.
Current implementation:
Current public route:
References inspected:
Correctness:
Baseline:
Experiments:
Decision:
Open questions:
Raw results:
```

Record enough context to reproduce the run: GPU, driver, CUDA toolkit, PyTorch
or library versions, command, git commit or working-tree label, dtype, shape,
quant format, warmups, iterations, median, variance, correctness tolerance, and
observed error.

## 2026-07-06: Shared Performance Documentation Alignment

Status: landed documentation scaffold.

Added the shared optimization notebook expected by `perf/perf.md`. The CUDA
handbook already contained the full measurement and experiment discipline; this
file is the tracked place for ongoing kernel-specific decisions, rejected
experiments, and final performance tables.

Current baseline material remains in `perf/baseline_status.md`:

- RTX 3090 / SM86 environment and roofline notes.
- Build/run baseline index.
- Framework reference numbers.
- Ampere port status and known performance debts.
- Existing result paths under `perf/results/`.

## Current High-Leverage Backlog

Status: baselined, experiments pending or partially complete.

| Area | Current finding | Next experiment |
|---|---|---|
| i-quant GEMV | lookup-bound from divergent table access | Move hot grids/tables into shared memory and remeasure format sweep |
| Quant GEMM prefill | dequant-to-fp16 plus cuBLAS is current route | Prototype fused dequant-in-pipeline GEMM and compare by `M` |
| MX/NV GEMV | below q8_0 effective bandwidth | Audit block-scale load coalescing and metadata placement |
| Attention backward D=128 | correctness landed, register pressure visible | Profile spills/occupancy and sweep split geometry |
| Decode attention/MLA/GDN/selective scan | low occupancy in one-row/one-state mappings | Batch more rows/states per block or add partition/chunk parallelism |
| Quant MoE GEMM | 32-row M-blocking landed | Add cp.async staging and compare grouped-library route at prefill sizes |

## Open Questions

- Which CUDA host should be treated as the canonical baseline for SM90/SM100
  kernels that cannot execute on the RTX 3090 box?
- Which results should be promoted from historical notes into compact
  per-kernel tables here?
- Should `perf/bench_kernels.py` grow a single normalized JSON schema for all
  standalone CUDA harnesses?
