# Tests

Use `scripts/test` as the common entrypoint when possible.

Current CUDA tests include primitive unit tests under `tests/` and PyTorch
extension tests under `kernels/tm_cuda/`. New contract tests should mirror the
kernel taxonomy under `tests/correctness/<family>/<operation>/`.
