# Repository Structure

QuixiCore-CUDA should share the same contract-facing structure as the other
QuixiCore backends while keeping CUDA-specific build, tuning, and architecture
choices below the operation boundary.

The rule is: public taxonomy is common; implementation details are native CUDA.

## Target Layout

```text
QuixiCore-CUDA/
  .quixicore/
    backend.yaml
    kernels.yaml
    quant-formats.yaml

  docs/
    repository-structure.md
    development.md
    kernel-roadmap.md
    performance.md
    backend-notes.md

  include/quixicore/cuda/
    backend.hpp
    runtime.hpp
    ops.hpp

  src/
    runtime/
    dispatch/
    errors/

  kernels/
    common/
    norms/
    activations/
    attention/
    linear_attention/
    ssm/
    matmul/
    quantization/
    moe/
    sampling/
    serving/
    optimizers/
    collectives/
    utils/

  bindings/
    c/
    python/
    pytorch/

  tests/
    correctness/
    integration/
    smoke/
    testdata/

  perf/
    harness/
    configs/
    results/
    baselines/

  examples/
  scripts/
  tools/
  assets/
```

CUDA-specific additions that may remain at the top level:

```text
Makefile
kernels/common.mk
Doxyfile
demos/
prototype/
```

`demos/` is for larger runnable examples. `prototype/` is temporary or
exploratory code; contract kernels should graduate into `kernels/`.

## Manifests

`.quixicore/backend.yaml` identifies this repository as the CUDA backend and
declares supported architectures and contract compatibility.

`.quixicore/kernels.yaml` should be the machine-readable parity source for
implemented operations:

```yaml
operations:
  paged_attention:
    family: attention
    status: implemented
    path: kernels/attention/paged_attention
    bindings:
      python: bindings/python/paged_attention.cpp
      pytorch: bindings/pytorch/paged_attention.cpp
    tests:
      correctness: tests/correctness/attention/paged_attention
    benchmarks:
      default: perf/configs/attention_paged.yaml
    variants:
      - name: cuda_sm80
        status: supported
      - name: cuda_sm90
        status: optimized
```

`.quixicore/quant-formats.yaml` should list supported quant formats, packing
layouts, and any CUDA-only hardware layout variants.

## Kernel Families

The top-level directories under `kernels/` are semantic families, not build
buckets:

- `norms/`: RMSNorm, LayerNorm, add-norm, norm-to-quant, QK norm.
- `activations/`: GELU, GLU, SiLU/SwiGLU helpers, standalone softmax.
- `attention/`: flash attention, causal/non-causal/varlen attention, backward,
  paged attention, MLA, rotary, quantized-KV attention, state merging.
- `linear_attention/`: Based, Hedgehog, linear attention, causal/decay linear
  attention, GDN, complex linear attention primitives.
- `ssm/`: Mamba, SSD, selective scan, FFT convolution.
- `matmul/`: dense GEMM, staged GEMM, complex matmul, Flux, architecture-tuned
  GEMM.
- `quantization/`: act quant, runtime quant, qgemm, qgemv, quantized LM head,
  fp8/int8/fp4 packing, TurboQuant.
- `moe/`: routing, expert alignment, gather/scatter, grouped GEMM, quantized
  MoE GEMM, LoRA alignment, finalize.
- `sampling/`: sampling, logit transforms, penalties, rejection sampling, beam
  search, speculative decode and EAGLE helpers.
- `serving/`: KV cache mutation, block/page tables, indexers, MInference, cache
  copy/gather helpers.
- `optimizers/`: AdamW and other training optimizer kernels.
- `collectives/`: NCCL-style collectives and fused collective kernels.
- `utils/`: bit packing, column permutation, Hadamard/FWHT, small reusable
  user-visible utilities.

## Operation Layout

Use one directory per contract operation. The filesystem rule is semantic
first, target variant second: do not create branch-only, architecture-first, or
marketing-name-first kernel trees.

```text
kernels/<family>/<operation>/
  README.md
  common/
    include/
    src/
  variants/
    cuda_sm80/
      include/
      src/
      tests/
      bench/
    cuda_sm90/
      include/
      src/
      tests/
      bench/
    cuda_sm100/
      include/
      src/
      tests/
      bench/
  tests/
  bench/
```

`common/` is for target-independent operation code only. Anything that depends
on a CUDA architecture, PTX instruction selection, Tensor Core shape, memory
hierarchy assumption, scheduler strategy, or launch geometry belongs under a
variant directory.

For very small operations, direct source files under the operation directory are
acceptable only when they are genuinely target-independent. As soon as an
operation has more than one architecture-specific implementation, move all
target-specific source into `variants/`.

Architecture names should describe the CUDA target, not the marketing feature
alone. Prefer `cuda_sm80`, `cuda_sm90`, `cuda_sm100`, or a more specific
operation variant under those directories.

Branches are not the architecture boundary. A backend repository should keep
all supported architecture variants in `main` side by side, with build and test
scripts selecting variants by requested or detected target.

## Target-Specific Internals

Public backend headers remain under `include/quixicore/cuda/`. CUDA primitive
libraries or low-level implementation headers that differ by architecture should
use explicit internal target directories, for example:

```text
include/internal/cuda/
  common/
  sm80/
  sm90/
  sm100/
```

Operation variants may include these internal headers, but contract-facing
headers should not expose target-specific implementation layouts.

## Tests And Benchmarks

Correctness and performance assets should mirror the kernel taxonomy:

```text
tests/correctness/<family>/<operation>/
perf/configs/<family>_<operation>.yaml
perf/baselines/<family>/<operation>/
```

The native build can still use Make, CMake, or direct extension builds, but
common developer entrypoints should exist:

```text
scripts/configure
scripts/build
scripts/test
scripts/bench
scripts/coverage-report
scripts/clean
```

For CUDA these scripts should wrap the existing Make/CUDA extension workflow.

## Current Migration Map

The current CUDA tree predates the common layout. New work should target the
canonical structure; existing code can move incrementally.

| Current area | Target area |
| --- | --- |
| `kernels/elementwise` | `kernels/norms/`, `kernels/activations/`, `kernels/optimizers/`, `kernels/utils/` |
| `kernels/quant` | `kernels/quantization/` |
| `kernels/serving` | `kernels/attention/`, `kernels/sampling/`, `kernels/serving/`, `kernels/quantization/` |
| `kernels/tm_cuda` | `bindings/` plus family-specific kernel directories |
| `kernels/layernorm` | `kernels/norms/layernorm/` |
| `kernels/rotary` | `kernels/attention/rotary/` |
| `kernels/attention` | `kernels/attention/` |
| `kernels/lin_attn_tm` | `kernels/linear_attention/` |
| `kernels/linear_attention` | `kernels/linear_attention/` |
| `kernels/based` | `kernels/linear_attention/based/` |
| `kernels/hedgehog` | `kernels/linear_attention/hedgehog/` |
| `kernels/mamba2` | `kernels/ssm/mamba2/` |
| `kernels/fftconv` | `kernels/ssm/fftconv/` |
| `kernels/flux` | `kernels/matmul/flux/` |
| `kernels/gemm` | `kernels/matmul/` |
| `kernels/moe` | `kernels/moe/` |
| `kernels/moe_quant` | `kernels/moe/` and `kernels/quantization/` as appropriate |
| `kernels/parallel` | `kernels/collectives/` |

Move files in behavior-preserving steps. Rename APIs only when synchronizing
the CUDA, Metal, ROCm, XPU, and Gaudi bindings deliberately.

## Rules For New Work

- Add new kernels under the semantic family directory, not under legacy buckets.
- Keep binding code in `bindings/`.
- Keep CUDA tuning, PTX, cooperative groups, CUTLASS/TK-style implementation
  details, and architecture specialization under operation variants.
- Keep all supported CUDA architecture variants in the repo; do not depend on
  one branch per architecture.
- Build, test, and benchmark entrypoints must filter variants by the selected
  CUDA target and must not try to compile unrelated architecture variants.
- Use `collectives/` for multi-GPU CUDA/NCCL extensions; mark them capability
  gated in `.quixicore/kernels.yaml`.
- If an operation has no meaningful CUDA implementation, mark it unsupported in
  metadata rather than adding a stub kernel.
