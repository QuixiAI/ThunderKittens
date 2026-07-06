# Performance

Use `scripts/bench` as the common entrypoint when possible. Existing CUDA
benchmark harnesses remain under this directory.

Record results with enough context to reproduce them: GPU, driver, CUDA version,
command line, kernel variant, shape, dtype, quant format, and git commit.
