# QuixiCore CUDA

QuixiCore CUDA is the NVIDIA CUDA implementation of the QuixiCore kernel library.

It is a standalone native implementation for NVIDIA GPUs starting from Ampere, covering both consumer and datacenter hardware. It shares no implementation code with the other QuixiCore backends.

It implements the contract defined by [QuixiAI/QuixiCore](https://github.com/QuixiAI/QuixiCore): the same operation names, quant formats, correctness expectations, benchmark methodology, and public library identity as the other QuixiCore backends.

**Native implementations. Shared contract. No shared code.**

## QuixiCore Standard Files

- Contract metadata: [`.quixicore/backend.yaml`](.quixicore/backend.yaml)
- Kernel coverage manifest: [`.quixicore/kernels.yaml`](.quixicore/kernels.yaml)
- Quant format manifest: [`.quixicore/quant-formats.yaml`](.quixicore/quant-formats.yaml)
- Repository structure: [`docs/repository-structure.md`](docs/repository-structure.md)
- Contribution workflow: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

Common developer entrypoints:

```bash
scripts/configure
scripts/build
scripts/test
scripts/bench
scripts/coverage-report
scripts/clean
```

These scripts keep the QuixiCore workflow consistent while wrapping CUDA-native
Makefiles, CUDA extension builds, and benchmark tools.

<div align="center" >
    <img src="assets/thunderkittens.png" height=350 alt="QuixiCore CUDA logo" style="margin-bottom:px"/><br/>
    <em>Native NVIDIA CUDA kernels for Ampere and newer GPUs.</em><br/><br/>
</div>

## Origin And Focus

This repository is the [QuixiAI](https://github.com/QuixiAI) CUDA backend for QuixiCore. It was renamed from QuixiAI's ThunderKittens Ampere fork and remains based on
[ThunderKittens](https://github.com/HazyResearch/ThunderKittens). Upstream's README says:

> *"We no longer actively support Ampere GPUs."*

This backend is the counterexample. **Every SM90+ kernel in this repository runs on an RTX 3090** —
attention forward *and backward*, GEMMs, the linear-attention family, Mamba-2, FFT convolutions —
plus a quantization stack ported from [QuixiCore Metal](https://github.com/QuixiAI/QuixiCore-Metal)
(the Apple Metal port of TK) covering **30 weight formats, every one bit-exact**, on silicon that
has hardware support for almost none of them.

The premise, learned from [Marlin](https://github.com/IST-DASLab/marlin) and proven three
platforms deep (CUDA H100 → Apple Metal → Ampere):

> **"Native support" is a statement about which instructions exist — not about which computations
> are possible.** A quant format is bits at rest. Dequantization is software placed in the latency
> shadow the memory system already forces on you. Compute rides whatever the hardware multiplies
> fast. The feature list is a menu of fast paths, not a boundary of the possible.

All numbers below are measured on **RTX 3090** (SM 8.6, 24 GB, ~936 GB/s), driver 580.65.06,
CUDA 12.9. Method and full logs: [`perf/baseline_status.md`](perf/baseline_status.md).

## The SM90 → SM86 port

Kernels written for H100 machinery — TMA, WGMMA, warp-specialized producer/consumer templates,
`setmaxnreg` — running on hardware with none of it. cp.async rings replace TMA, a warp-level
emulation layer replaces WGMMA (`include/ops/group/mma/warpgroup_sm80.cuh`), and the SM90/SM100
compilation output is byte-identical (PTX-verified) — this fork is a strict superset of upstream.

| kernel | correctness | RTX 3090 performance |
|---|---|--:|
| **attention fwd** (causal, GQA, L; full `mha_h100` parity) | ≤1.1e-3 vs fp32 ref | **48.7 TFLOP/s** (D=64) |
| **attention bwd** (FA2, new kernel — no Ampere ancestor existed) | ≤7.5e-4 vs torch autograd | all configs D=64/128 × causal × GQA |
| **bf16 GEMM** (same-source, arch branch) | matches cuBLAS | **44.0 TFLOP/s** @4096³ = 80% of cuBLAS |
| **int8 GEMM** | bit-exact | 51.5 TOP/s @8192³ |
| **fp8 GEMM** (+ scaled variant) — *on a chip with no fp8 anything* | **exact** (max err 0) | 23.1 TFLOP/s |
| **flux** gate / gelu | err count 0 | 44.8 / 47.3 TFLOP/s |
| **rotary** | ≤0.015 | up to 763 GB/s |
| **mamba2** | 0.39% rel vs `ssd_minimal` | — |
| **based / linear-attention / hedgehog** | at the bf16-state precision floor (proven vs an exact-precision emulation) | 17.2 TFLOP/s (based) |
| **fftconv, layernorm** | pass | layernorm 3.2–4.1× torch eager, ties Triton |

Three **upstream bugs** were found and fixed along the way (they affect SM90 too): an
element-typed `kittens::coord` in rotary's sin/cos loads, `ldmatrix` int8 wrappers gated behind
the fp8 SM90 guard, and a 32-bit register↔shared fast path inconsistent with `st::idx()`.

## The quant stack: 30 formats, zero excuses

Ported from QuixiCore Metal's format layer (`kernels/quant/`), packed **byte-identically** — one
quantized checkpoint serves Metal and CUDA. Every format's GPU dequant is **bit-exact (max diff
0)** against the reference quantizer, from 1.56 to 8.5 bits per weight:

| family | formats |
|---|---|
| GGUF legacy | q8_0, q4_0, q4_1, q5_0, q5_1 |
| GGUF k-quants | q2_K, q3_K, q4_K, q5_K, q6_K |
| GGUF i-quants (E8 lattice / codebook) | iq1_s **(1.56 bpw)**, iq2_xxs, iq2_xs, iq3_xxs, iq4_nl, iq4_xs |
| Marlin / AWQ / GPTQ / HQQ | kU4, kU4B8, hqq |
| fp8 | fp8_e4m3, e5m2, fp8_block |
| **OCP MX / NVIDIA** — *"Blackwell-only," allegedly* | **mxfp8, mxfp4, mxfp6_e3m2, mxfp6_e2m3, nvfp4**, fp4_e2m1 |
| ternary | bitnet **(2.5 bpw)** |

Every format runs through the whole kernel family — dequant, GEMV, GEMM (Marlin zero-shuffle
fragment path: format → mma fragment in registers, no shared staging; a `qgemm_ksplit` variant
slices K across the grid for decode-shape occupancy — **~2.4× over the single-pass kernel** at
`M=64, N=512, K=4096`, e.g. q8_0 3.16 → 7.11 TFLOP/s), fused GEMM+GELU, GPTQ act-order and
block-scale GEMMs, and a fused LM-head sampler (argmax / categorical / top-k / top-p). At the
decode-critical **M=1 GEMV** — where quant serving actually lives — the quantized path is
**~1.4× faster than `torch.matmul` fp16** for ≥4-bit formats (q4_0 1.48×, q4_K 1.42×; it moves
~4× less weight memory). Weight-only GEMV bandwidth on idle 3090 (N=512, K=4096): q8_0 **283 GB/s**,
q6_K 212, q5_0 171, q4_0 156, mxfp8 254, nvfp4 141; the E8-lattice i-quants trail at 15–71 GB/s
(divergent codebook gathers — the span-decode pass cut the re-lookups 8×, but they stay
lookup-bound). At **prefill (M ≥ 64)** the GEMM dequants the weight to fp16 once and hands the
matmul to cuBLAS — **~28–30 TFLOP/s** end-to-end (q8_0/q4_0/nvfp4 at M=256, dequant included),
vs ~7–9 for the naive per-tile kernel, so a batched quantized forward runs at cuBLAS-class speed.
Activation-quantized paths (W8A8 at **528 GB/s**, BitNet W2A8) use native
`dp4a` — hardware Apple never had, so this port *exceeds* the original. The dequant recipes are
Marlin's: an e8m0 block scale is one `__uint_as_float(e << 23)`; fp4→fp16 is a shift and one
power-of-two multiply.

Callable from Python (`kernels/tm_cuda/`, package `tk_cuda`, **117/117** end-to-end tests):

```python
import tk_cuda
y   = tk_cuda.qgemm(x_fp16, Wq, "nvfp4")          # any of 29 formats, same packed bytes as Metal
tok = tk_cuda.lm_head_sample(h, Wq, "mxfp4",       # fused head + sampling, no (T,V) logits
                             temperature=0.8, seed=99, mode="categorical")
```

Sampling is driven by a counter-based RNG with **proven bit-parity** to its numpy reproduction —
the fused GPU draw equals the reference draw exactly, so stochastic kernels have deterministic
oracles.

The rest of the QuixiCore Metal serving stack is **ported and live** — the serving path (paged attention
v1/v2, quantized + fp8 KV cache, MLA decode incl. bf16/fp8/sparse/partitioned, RoPE insert,
sliding-window and GQA-staged attention, varlen prefill), the decode stack (full sampler family,
beam search, speculative + tree verification), the elementwise/norm/training family (RMSNorm,
LayerNorm, add-norm with fp8 epilogues, GELU, GLU ×6, cross-entropy, embedding, dropout, hadamard,
AdamW), and MoE (routing → padded grouped GEMMs → finalize), linear attention, and complex matmul.
Every kernel is validated against an fp64 or torch-autograd oracle. Roadmap and status in
[`thundermittens_ampere_port.md`](thundermittens_ampere_port.md).

## The serving layer: MetalForge → CUDA

A second, exhaustive port brings **[MetalForge](https://github.com/AlpinDale)**'s Apple-Silicon
LLM-serving kernel set to Ampere. MetalForge is [**AlpinDale**](https://github.com/AlpinDale)'s
([@AlpinDale](https://x.com/AlpinDale)) Metal serving layer; its "one simdgroup per row +
`simd_*` reduction" idiom maps 1:1 onto our warp-per-row + `warp_*` reductions, so the algorithms
port faithfully. Every family below is validated against an fp64 / round-trip / host-replay oracle
(gap analysis in [`metalforge_gap_analysis.md`](metalforge_gap_analysis.md)):

| MetalForge kernel family | what it is | CUDA/SM86 port |
|---|---|---|
| **Quantized MoE grouped GEMM** | fp8 rowwise · nvfp4 dual-fp4 (swizzled A-scale, per-expert α) · WNA16 int4/int8 | `kernels/moe_quant/` — mma-fragment path + fused `silu_and_mul` quant + experts-quant + scored routing (sigmoid/softplus/grouped-topk) |
| **Norm / activation quant epilogues** | RMSNorm→fp8/int8 (static/dynamic/±residual), AZP int8, per-token-group quant | `kernels/elementwise/tm_norm_quant_kernels.cuh` |
| **GDN gated DeltaNet** | per-token serial linear-attention recurrence, GQA + varlen + paged state | `kernels/lin_attn_tm/gdn_kernels.cuh` |
| **Mamba selective scan** | ragged Mamba-1 scan + **APC** mid-sequence chunk checkpointing | `kernels/mamba2/selective_scan_kernels.cuh` |
| **Sampler tail-cutoff zoo** | top-nσ · top-a · η/ε-cutoff · XTC · quadratic · skew · no-repeat-ngram · DRY | `kernels/serving/logits_proc_kernels.cuh` |
| **EAGLE speculative decode** | rejection verify (greedy/random) + recovered-token argmax + full padded-draft bookkeeping | `kernels/serving/eagle_kernels.cuh` |
| **Sparse attention** | 2-way LSE merge (`merge_attn_states`, + fp8 output) · tau/temperature gate · MInference vertical/slash builder | `kernels/serving/sparse_serving_kernels.cuh` |
| **DeepSeek lightning-indexer** | fp8 K-quant-and-cache (per-block UE8M0 scale) + the `cp_gather` inverse | `kernels/serving/sparse_serving_kernels.cuh` |
| **TurboQuant codec** | random-sign FWHT rotation (self-inverse) + `tq_encode` K-uniform / V-centroid cache codec | `kernels/quant/turboquant.cuh` |
| **LoRA / act-order** | per-LoRA-sharded MoE align + GPTQ column permute | `kernels/quant/turboquant.cuh` |

The quantized MoE GEMMs go straight to the tensor-core mma-fragment path (not a scalar
correctness pass) — the same Marlin dequant discipline the weight-only GEMMs use, now feeding the
per-expert padded schedule. `dp4a`/IMMA-backed int paths again *exceed* the Metal original on
hardware Apple never had.

Throughput on an **idle 3090** (`perf/bench_metalforge.py`, full table in
[`perf/perf.md`](perf/perf.md)): quantized MoE grouped GEMM (E=8, N=K=4096, 2048 rows) **10.5
TFLOP/s** fp8 / 7.0 int4 / 5.7 int8; **GDN decode 345k tok/s** (128 seqs, GQA, D=128, ~360 GB/s
over recurrent state); RMSNorm→fp8 **304 GB/s**; `merge_attn_states` **415 GB/s**; SwiGLU→fp8
**374 GB/s**; the TurboQuant FWHT rotation **815 GB/s** (~87% of peak). An optimization pass took
the MoE GEMMs **1.4–1.6×** (32-row M-blocking reuses each dequantized weight fragment across the
tile) and `rms_norm_quant` **1.7×** (multi-warp block-per-row + folded amax); the serial
selective-scan (21 GB/s on a single long sequence, occupancy-bound) and further MoE tiling remain
the headroom.

## Quick start

```bash
# any per-kernel build (Makefiles sit next to sources):
cd kernels/gemm/bf16_h100 && make ARCH=SM86     # SM90 and SM100 still work, unchanged

# quant stack, standalone golden tests (needs the Metal format reference checkout):
cd kernels/quant
python gen_golden.py golden                      # quantize with TM's quant.py (numpy)
nvcc qgemv.cu -std=c++20 -O2 -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 -o qgemv.out
./qgemv.out golden/nvfp4                         # dequant: EXACT; gemv: PASS

# torch extension (see kernels/tm_cuda/setup.py header if your torch/toolkit CUDA versions differ):
cd kernels/tm_cuda && python setup.py build_ext --inplace
python -m pytest test_tk_cuda.py -v
```

Requirements: CUDA 12.3+, C++20, an Ampere GPU (`-DKITTENS_SM86` for consumer cards, `SM80` for
A100) — or any SM90/SM100 GPU, exactly as upstream.

## What ThunderKittens is

Everything upstream says still holds here: TK is an embedded CUDA framework built around
register and shared-memory **tiles** (`rt`, `st`, ≥16×16), warp and warpgroup ops over them, and
templates that overlap compute with I/O. Kernels read like the algorithm, not like the hardware
manual. See the [upstream README](https://github.com/HazyResearch/ThunderKittens#readme) and
[Hamza Elshafie's deep-dive](https://hamzaelshafie.bearblog.dev/dissecting-thunderkittens-anatomy-of-a-compact-dsl-for-high-performance-ai-kernels/)
for the framework tour and the H100/B200 story.

Learn more from the Hazy Research blog posts:

* [GPUs Go Brrr, May 2024](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
* [Easier, Better, Faster, Cuter, Oct. 2024](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2)
* [ThunderKittens: Bringing fp8 to theaters near you, Nov 2024](https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8)
* [ThunderMLA: FlashMLA, Faster and Fused-er! Mar 2025](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla)
* [ThunderKittens Now on Blackwells! Mar 2025](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell)
* [ThunderKittens 2.0: Even Faster Kernels for Your GPUs, Feb 2026](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)

Backend lineage:

* [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) for NVIDIA (upstream)
* [HipKittens](https://github.com/HazyResearch/HipKittens) for AMD
* [QuixiCore Metal](https://github.com/QuixiAI/QuixiCore-Metal) for Apple Silicon
* **QuixiCore CUDA** for NVIDIA Ampere+ hardware

Papers: [Single GPU](https://arxiv.org/abs/2410.20399) ·
[Multiple GPUs](https://arxiv.org/abs/2511.13940)

## Credits

- [HazyResearch](https://hazyresearch.stanford.edu/) — ThunderKittens itself. This backend tracks
  upstream and adds Ampere; it takes nothing away.
- [QuixiCore Metal](https://github.com/QuixiAI/QuixiCore-Metal) — the Apple Metal backend whose
  kernels and format layer this backend brings to CUDA, and the proof that "none of this was
  supposed to work" is a solvable condition.
- **[AlpinDale](https://github.com/AlpinDale)** ([@AlpinDale](https://x.com/AlpinDale)) — author of
  **MetalForge**, the Apple-Silicon LLM-serving kernel layer that inspired the entire serving-layer
  port above: quantized MoE GEMMs, the sampler zoo, EAGLE, GDN, Mamba selective scan, MInference /
  lightning-indexer sparse attention, and the TurboQuant cache codec. Every one of those families
  is a direct CUDA transcription of AlpinDale's Metal kernels.
- [Marlin](https://github.com/IST-DASLab/marlin) (IST-DASLab / Neural Magic) — the dequant bit
  tricks and the placement discipline that make quantized compute free on "unsupported" hardware.

## License

MIT. See [`LICENSE`](LICENSE).
