# Performance

CUDA performance work is tracked in `perf/`.

- `perf/perf.md` is the performance handbook and result summary.
- `perf/baseline_status.md` records baseline snapshots.
- `perf/bench_kernels.py` and sibling harnesses run local sweeps.
- Raw results should stay under ignored result directories.

Benchmark reports should include GPU, driver, CUDA version, command line, input
shape, dtype, quant format, and relevant kernel variant.
