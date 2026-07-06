# Benchmarking

QuixiCore CUDA benchmarks should use native CUDA, framework, and library timing
and profiling tools.

The operating guide is [`perf.md`](perf.md). The running status notebook is
[`optimization_status.md`](optimization_status.md), with baseline snapshots in
[`baseline_status.md`](baseline_status.md).

Use `scripts/bench` as the common entrypoint when possible. Existing CUDA
benchmark harnesses remain under this directory.

Record results with enough context to reproduce them: GPU, driver, CUDA version,
command line, kernel variant, shape, dtype, quant format, warmups, measured
iterations, correctness tolerance, observed error, and git commit.

Raw benchmark output should be written under `perf/results/`, which is ignored by
git. Summaries that matter for future work should be copied into a tracked
status document.
