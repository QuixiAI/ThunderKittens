# Agent Instructions

This is the QuixiCore CUDA backend. Kernel work must be correctness-first,
measurement-driven, and recorded in the performance notebook.

## Read First

- User-facing overview: `README.md`.
- Repository layout: `docs/repository-structure.md`.
- Performance operating guide: `perf/perf.md`.
- Optimization notebook: `perf/optimization_status.md`.
- Baseline index: `perf/baseline_status.md`.
- Kernel metadata: `.quixicore/kernels.yaml` and
  `.quixicore/quant-formats.yaml`.

## Performance Optimization Requirement

Before committing any kernel implementation, kernel routing change, benchmark
change, or performance claim, the agent must complete at least one focused
performance optimization run on an affected kernel.

A valid run includes:

- The kernel, public route, dtype/format, and shape set.
- Correctness for the touched path.
- Baseline/current timing and candidate timing when testing a variant.
- GPU model, driver, CUDA toolkit, library/framework versions, command line,
  warmups, iterations, median, and variance or min/max.
- A keep/reject decision in `perf/optimization_status.md`.

If a suitable CUDA GPU/runtime is unavailable, do not commit a kernel
optimization or speedup claim. Stop and report the blocker, or restrict the
commit to docs/scaffolding with no performance claim.

Pure documentation and metadata-only commits may skip the kernel perf run, but
they must not claim a performance improvement.

## How To Optimize

- Start from `perf/perf.md`; form a bottleneck hypothesis before editing.
- Change one meaningful factor at a time: tile shape, launch geometry, memory
  layout, fusion, `cp.async`/shared-memory staging, dequant strategy, routing
  threshold, or specialization.
- Compare against cuBLAS/cuBLASLt, PyTorch, naive decompositions, and current
  CUDA kernels where relevant.
- Keep only wins that pass correctness, improve realistic priority shapes, and
  do not regress supported edge shapes or tolerances.
- Store raw output under `perf/results/`; copy durable conclusions into
  `perf/optimization_status.md`. Do not commit large Nsight traces.

## Useful Commands

```bash
.venv/bin/python perf/bench_kernels.py --phase all --kernel <kernel>
.venv/bin/python perf/bench_kernels.py --phase run --kernel <kernel> --gpus 0
CUDA_VISIBLE_DEVICES=0 bash perf/sweep_quant.sh
cd kernels/tm_cuda && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q
```

Use Nsight Compute/Systems when a timing A/B does not explain a result. Record
the trace path and conclusion, not the full trace.

## Engineering Hygiene

- Check `git status` before editing. Do not revert user changes.
- Keep backend-local optimizations behind the public QuixiCore contract.
- Update metadata, tests, docs, and bindings when changing public behavior.
- Do not import reference implementation code unless licensing and provenance
  have been reviewed.
- Keep commits scoped and descriptive.
