# ThunderKittens — Ampere Edition

<div align="center" >
    <img src="assets/thunderkittens.png" height=350 alt="ThunderKittens logo" style="margin-bottom:px"/><br/>
    <em>Tile primitives for speedy kernels — now running where they were never supposed to.</em><br/><br/>
</div>

This is the [QuixiAI](https://github.com/QuixiAI) fork of
[ThunderKittens](https://github.com/HazyResearch/ThunderKittens). Upstream's README says:

> *"We no longer actively support Ampere GPUs."*

This fork is the counterexample. **Every SM90+ kernel in this repository runs on an RTX 3090** —
attention forward *and backward*, GEMMs, the linear-attention family, Mamba-2, FFT convolutions —
plus a quantization stack ported from [ThunderMittens](https://github.com/QuixiAI/ThunderMittens)
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

Ported from ThunderMittens' format layer (`kernels/quant/`), packed **byte-identically** — one
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

Callable from Python (`kernels/tm_cuda/`, package `tk_cuda`, **75/75** end-to-end tests):

```python
import tk_cuda
y   = tk_cuda.qgemm(x_fp16, Wq, "nvfp4")          # any of 29 formats, same packed bytes as Metal
tok = tk_cuda.lm_head_sample(h, Wq, "mxfp4",       # fused head + sampling, no (T,V) logits
                             temperature=0.8, seed=99, mode="categorical")
```

Sampling is driven by a counter-based RNG with **proven bit-parity** to its numpy reproduction —
the fused GPU draw equals the reference draw exactly, so stochastic kernels have deterministic
oracles.

The rest of the ThunderMittens stack is **ported and live** — the serving path (paged attention
v1/v2, quantized + fp8 KV cache, MLA decode incl. bf16/fp8/sparse/partitioned, RoPE insert,
sliding-window and GQA-staged attention, varlen prefill), the decode stack (full sampler family,
beam search, speculative + tree verification), the elementwise/norm/training family (RMSNorm,
LayerNorm, add-norm with fp8 epilogues, GELU, GLU ×6, cross-entropy, embedding, dropout, hadamard,
AdamW), and MoE (routing → padded grouped GEMMs → finalize), linear attention, and complex matmul.
Every kernel is validated against an fp64 or torch-autograd oracle. Roadmap and status in
[`thundermittens_ampere_port.md`](thundermittens_ampere_port.md).

## Quick start

```bash
# any per-kernel build (Makefiles sit next to sources):
cd kernels/gemm/bf16_h100 && make ARCH=SM86     # SM90 and SM100 still work, unchanged

# quant stack, standalone golden tests (needs a ~/ThunderMittens checkout for the reference):
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

The Kittens Cinematic Universe:

* [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) for NVIDIA (upstream)
* [HipKittens](https://github.com/HazyResearch/HipKittens) for AMD
* [ThunderMittens](https://github.com/QuixiAI/ThunderMittens) for Apple Silicon
* **This fork** for the NVIDIA hardware people actually own

Papers: [Single GPU](https://arxiv.org/abs/2410.20399) ·
[Multiple GPUs](https://arxiv.org/abs/2511.13940)

## Credits

- [HazyResearch](https://hazyresearch.stanford.edu/) — ThunderKittens itself. This fork tracks
  upstream and adds Ampere; it takes nothing away.
- [ThunderMittens](https://github.com/QuixiAI/ThunderMittens) — the Apple Metal port whose
  kernels and format layer this fork brings to CUDA, and the proof that "none of this was
  supposed to work" is a solvable condition.
- [Marlin](https://github.com/IST-DASLab/marlin) (IST-DASLab / Neural Magic) — the dequant bit
  tricks and the placement discipline that make quantized compute free on "unsupported" hardware.

## License

MIT, same as upstream.
