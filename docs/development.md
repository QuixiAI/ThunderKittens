# Development

QuixiCore CUDA uses CUDA-native source, per-kernel Makefiles, and PyTorch
extension builds while tracking the shared QuixiCore contract.

Use the common scripts first:

```bash
scripts/configure
scripts/build help
scripts/test help
scripts/bench --help
```

Legacy kernel directories may keep their local Makefiles during migration.
New contract work should follow `docs/repository-structure.md` and update
`.quixicore/kernels.yaml`.
