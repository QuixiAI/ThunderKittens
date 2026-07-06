# Kernel Roadmap

The CUDA backend already contains broad kernel coverage in legacy directories.
The roadmap is to make that coverage explicit and comparable across QuixiCore
backends.

Priorities:

1. Move new work into `kernels/<family>/<operation>/`.
2. Expand `.quixicore/kernels.yaml` from family-level status to operation-level
   status.
3. Bring CUDA and Metal operation coverage into parity.
4. Keep collectives capability-gated because they depend on multi-GPU runtime
   availability.
