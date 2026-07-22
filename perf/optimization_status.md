# ThunderKittens (Ampere) Optimization Status

Running notebook for the optimize-and-verify loop described in `perf/perf.md`
(baseline snapshots live in `perf/baseline_status.md`). Each entry records a
kernel's correctness oracle + tolerance and a focused A/B on this host, ending
in a keep/reject decision. Raw run logs under `perf/results/` (git-ignored);
only summary snippets are copied here.

## Environment (all numbers below)

- Host: NVIDIA GeForce RTX 3090 (SM 8.6 / Ampere, 24 GB), driver 580.65.06,
  CUDA toolkit 12.9 (`/usr/local/cuda`). All runs pinned to `CUDA_VISIBLE_DEVICES=1`
  (GPU 0 hosts a foreign job — never touched).
- Timing: CUDA-event around each launch, `cudaDeviceSynchronize` outside the
  timed region, 10 warmups + 50 timed iters, mean reported. Consumer GPU, so
  absolutes are host-load sensitive; ratios on the same idle GPU are fair.

---

## Wave 8 — int8 kernels ported from embeddinggemma.c

Two opt-in CUDA paths from `embeddinggemma.c` (`src/engine_cuda.cu`, flags
`EI_CUDA_W4A8_GEMM` / `EI_CUDA_INT8_ATTN`) ported as **shape-named** specialized
variants into the matching families. Named by shape/format, never by model.
Reference source: the standalone device code was lifted verbatim where possible;
the correctness oracles and A/B baselines are new.

### quant/qgemm_w4a8 — Q4_0 weight x int8 activation, per-32-block scales, IMMA (m16n8k32)

New file: `kernels/quant/qgemm_w4a8.cu` (harness + `namespace w4a8`). Complements
the weight-only `qgemm<q4_0>` (fp16 activation, m16n8k16): here the activation is
also quantized (int8, per-32-block amax/127), so the K=32 MMA tile is exactly one
Q4/Q8 block and the score rides the int8 tensor cores (`mma.m16n8k32.s8`,
~284 TOP/s roofline vs ~71 TFLOP/s fp16). cp.async 4-stage double-buffer, SoA
nibble+fp16-scale repack, per-block int32->fp32 dequant-accumulate, -8 Q4 zero
point folded into the signed weight. This is the **only** Q4bit x A8bit IMMA path
in the repo — `gemm/int8_ampere` is dense A8xW8, `qgemv_int` is W8A8/W2A8 GEMV;
no duplication.

- **Correctness** — fp64 oracle recomputes the exact integer block sums scaled by
  the fp16 `(as*ws)` products (the values the kernel multiplies), tolerance the
  repo's `rel = sum|got-ref|/sum|ref| < 0.02`. Reuses the checked-in
  `golden/q4_0` weight (N=512, K=4096); activations generated + int8-quantized in
  the harness. The `quantize_soa` support kernel is checked bit-for-bit vs host.

  | shape (M,N,K) | rel | max abs | result |
  |---|---|---|---|
  | 64, 512, 4096 (golden/q4_0) | 0.0000% | 2.86e-06 | PASS |
  | 256, 512, 4096 | 0.0000% | 2.86e-06 | PASS |
  | 512, 512, 4096 | 0.0000% | 2.86e-06 | PASS |
  | 512, 4096, 4096 (synth) | 0.0000% | 4.77e-06 | PASS |

- **Perf A/B (RTX 3090, GPU 1).** Direct vs the repo's existing Q4_0 GEMM kernels
  at the canonical decode shape (M=64, N=512, K=4096):

  | kernel | TFLOP/s (TOP/s) | ms |
  |---|---|---|
  | `qgemm<q4_0>` (fp16 act, single-pass) | 2.65 | 0.101 |
  | `qgemm_ksplit<q4_0>` (fp16 act, K-split) | 7.10 | 0.038 |
  | **qgemm_w4a8** (int8 act, IMMA) | 3.36 | 0.080 |

  Synthetic sweep vs the strongest baseline — dequant-to-fp16 + cuBLAS fp16 GEMM
  on **pre-materialized** fp16 weights (ignores the dequant pass and uses 4x the
  weight memory), same shape:

  | shape (M,N,K) | w4a8 TOP/s | cuBLAS-fp16 TFLOP/s | w4a8/cuBLAS |
  |---|---|---|---|
  | 64, 768, 4096   | 5.0  | 16.4 | 0.31x |
  | 64, 4096, 4096  | 16.8 | 40.3 | 0.42x |
  | 512, 768, 4096  | 28.2 | 35.1 | 0.80x |
  | 512, 2304, 4096 | 55.0 | 59.2 | 0.93x |
  | 512, 4096, 4096 | 50.3 | 53.2 | 0.95x |
  | **2048, 4096, 4096** | **61.8** | 57.3 | **1.08x** |

- **Decision: KEEP** as an opt-in specialized variant for compute-bound wide-N /
  large-M projections (M >= ~512). It beats the existing single-pass `qgemm<q4_0>`
  by 1.27x at decode and, at M=2048, matches/beats cuBLAS-on-fp16 (1.08x) while
  the weight stays 4-bit resident — 4x less weight bandwidth/footprint, no fp16
  materialization, no separate dequant pass, and the activation quant fuses in.
  **Not** the default decode path: at small M/N the grid is only
  `ceil(N/BN) x ceil(M/BM)` CTAs (8 at N=512) so it is occupancy-bound and loses
  to `qgemm_ksplit`. Follow-up (backlogged): a K-split over `blockIdx.z` with an
  fp32 atomic combine, mirroring `qgemm_ksplit`, to fill the 82 SMs at decode.

### serving/attn_q_int8 — int8 Q.K^T flash score stage, head-dim 256, GQA (mode 1)

New file: `kernels/serving/attn_q_int8.cu`. Distinct from `attn_q` (which
quantizes K/V *storage* and computes in fp16): this quantizes Q and K to int8
with a per-row (amax/127) scale, runs S = Q.K^T on the int8 tensor cores
(cuBLAS `CUDA_R_8I -> CUDA_R_32I`, as the source does), and dequantizes the int32
partials **inside** the softmax by `qscale[q]*kscale[k]` (no extra query x key
pass). P.V stays fp16. The two hand-written ported kernels are `quantize_rows_i8`
(per-row int8 quant, D=256) and `attention_softmax_i8` (fused dequant + row
softmax). The source's int8 P.V "mode 2" is **intentionally not ported** — the
source found it breaks embedding accuracy.

- **Correctness** — fp64 oracle recomputes `softmax(qs*ks * Qi8.Ki8^T) . Vf` from
  the same int8 inputs; kernel output must match `rel < 0.02`. A second number
  reports the int8-vs-full-fp16 attention-output error (the accuracy the int8 QK
  path costs — the justification for mode 1 over mode 2).

  | T | vs fp64 oracle (rel / max) | int8 vs fp16 output | gate |
  |---|---|---|---|
  | 256  | 0.0255% / 7.8e-04 | 1.96% | PASS |
  | 512  | 0.0247% / 7.6e-04 | 1.88% | PASS |
  | 2048 | 0.0253% / 8.7e-04 | 2.11% | PASS |

- **Perf A/B (RTX 3090, GPU 1), GQA 3:1, head-dim 256:**

  | T | QK stage int8 | QK stage fp16 | QK ratio | full attn ratio |
  |---|---|---|---|---|
  | 256  | 4.1 TOP/s  | 5.3 TFLOP/s  | 0.77x | 0.89x |
  | 512  | 16.0 TOP/s | 20.2 TFLOP/s | 0.79x | 0.93x |
  | 2048 | 58.1 TOP/s | 37.9 TFLOP/s | 1.53x | 1.14x |

- **Decision: KEEP** as an opt-in variant gated to long sequences: the int8 QK^T
  wins from T >= ~1024 (1.53x on the score GEMM at T=2048) and the full attention
  from T >= ~2048 (1.14x). At short T the per-row quantize overhead is not
  amortized and PV/softmax dominate, so it is ~neutral-to-slower — matching the
  source, which gates int8 attention behind a min-tokens threshold (~80-192). The
  ~2% output error vs fp16 is acceptable for embedding pooling (validated
  end-to-end in the source); int8 P.V (mode 2) rejected upstream, not ported.
