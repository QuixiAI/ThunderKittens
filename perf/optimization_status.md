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

---

## Wave 9 — embedding pooling head ported from embeddinggemma.c

fp32 sentence-embedding pooling head from `embeddinggemma.c` (`src/engine_cuda.cu`
`pool_kernel`), ported as a **shape-named** specialized variant. Named by the
D-vector shape, never by model.

### serving/pool_mean_rms_l2 — masked mean-pool -> per-token RMSNorm -> L2 (D in {256,512,768,1024})

New file: `kernels/serving/pool_mean_rms_l2.cu`. One warp owns a whole sequence's
D-vector (lane `l` owns dims l, l+32, ..., D/32 registers per lane). For each
valid token in the sequence's `[offsets[s], offsets[s+1])` range it RMS-normalizes
the token row with the learned weight (`inv = rsqrt(sum_d row[d]^2 / D + eps)`,
accumulate `row[d]*w[d]*inv`), then after the token loop mean-pools by
`1/(stop-start)` and L2-normalizes the pooled vector (unit-scale guard on the
all-zero case). Weights are applied as plain `w` — the +1 Gemma fold is baked into
the exported norm weights offline, so both reference kernels use `x*scale*w`, no
`(1+w)` here. Templated on D and dispatched at runtime; D must be a multiple of 32.
Only the plain mean->RMS->L2 head is ported; the source's fused-singleton-pool
epilogue is intentionally **not** ported (fails parity on GEMM shapes).

- **Correctness** — fp64 oracle recomputes the CPU reference `ei_mean_pool_rms_l2`
  (`src/kernels.c`) in double precision from the same inputs; the fp32 kernel must
  match `rel = sum|got-ref|/sum|ref| < 0.02` (pure fp32, so it lands far tighter)
  with cosine ~1. Swept over one packed batch with n_tokens in {1,4,7,37,128,512}
  (covers the singleton and long-sequence cases).

  | D | rel | max abs | cosine | gate |
  |---|---|---|---|---|
  | 256  | 0.000015% | 1.0e-07 | 1.00000000 | PASS |
  | 512  | 0.000015% | 9.5e-08 | 1.00000000 | PASS |
  | 768  | 0.000014% | 8.2e-08 | 1.00000000 | PASS |
  | 1024 | 0.000015% | 8.2e-08 | 1.00000000 | PASS |

- **Perf A/B (RTX 3090, GPU 1)** — fused single-pass kernel vs the naive composed
  baseline (stage 1: RMS-normalize every token row into a `[total_tokens][D]` temp
  in global memory; stage 2: mean + L2 over the temp). The fused version keeps the
  normalized rows in registers, so it avoids the full temp-matrix round-trip.

  | shape (B seqs x T tok) | D | fused ms | naive ms | speedup |
  |---|---|---|---|---|
  | 1024 x 64  | 256  | 0.0837 | 0.2701 | 3.23x |
  | 1024 x 64  | 512  | 0.1831 | 0.5702 | 3.12x |
  | 1024 x 64  | 768  | 0.3720 | 0.9924 | 2.67x |
  | 1024 x 64  | 1024 | 0.4510 | 1.2466 | 2.76x |
  | 256 x 256  | 768  | 0.6716 | 1.1093 | 1.65x |
  | 4096 x 16  | 768  | 0.4745 | 0.9521 | 2.01x |

  Fused reaches ~800 GB/s at D=256 (bandwidth-bound, as expected for a pooling
  reduction); the naive baseline pays 2-3x for materializing the normalized token
  matrix. Speedup shrinks toward 1.3-1.7x only for very long per-sequence token
  counts (T=256), where the stage-1 temp write is better amortized, but the fused
  path still wins in every cell.

- **Decision: KEEP** as the default (and only) pooling-head path — it is a new op
  (no byte-identical-off constraint), strictly faster than composing the separate
  RMS/mean/L2 ops, exact vs the fp64 oracle, and matches the checked reference
  math bit-for-bit in structure. No env flag: there is no slower variant to gate.

---

## Wave 10 — fused attention-projection kernels ported from embeddinggemma.c

Two fused CUDA kernels ported as **shape-named** specialized variants, named by
their tensor shapes (head-dim 256, GQA head-group counts, symmetric window),
never by model.

### serving/qk_norm_rope — fused per-head QK-RMSNorm + NeoX RoPE + packed-QKV split + f16 cast (head-dim 256, GQA)

New file: `kernels/serving/qk_norm_rope.cu`. Ported from embeddinggemma.c
(`src/engine_cuda.cu` `qkv_norm_rope_f16_kernel` / `qk_norm_rope_kernel`;
portable fp64 reference is the CPU `ei_qk_norm_rope_qk_inplace` in
`src/kernels.c`). One warp owns one `(token, head)` task over the packed
projection output `combined = [Q: N_HEAD*D][K: N_HEAD_KV*D][V: N_HEAD_KV*D]` (f32).
For each Q/K head it RMS-normalizes the D=256 head row with the learned per-dim
norm weight, folds the query logit scale into the reciprocal
(`inv = rsqrt(ss/D + eps) * scale`, `scale = 0.0625` for Q and `1.0` for K — the
source applies 0.0625 inside `inv`, i.e. to `x0,x1` before the rotation, identical
to applying it after because RoPE is a rotation), applies NeoX split-half rotary
from a precomputed `(cos,sin)` table indexed by the token position, and casts to
f16 into the split Q/K outputs. Every V head is a plain f32->f16 copy (no norm,
no rope), exactly the source's V task. Templated on `(N_HEAD, N_HEAD_KV, D)`.

- **Correctness** — fp64 oracle recomputes the reference norm->rope->cast math in
  double precision from the same inputs (scale folded into `inv`, matching the
  kernel; `cos/sin` in double). Swept over sequence lengths `T in {1,7,37,128,512,
  2048}`, both rope bases (swa `1e4` / full `1e6`), and three GQA head-group counts.
  f16 output, so the gate is `rel = sum|got-ref|/sum|ref| < 0.02`, `cosine > 0.999`.

  | GQA (N_HEAD:N_HEAD_KV) | rel | max abs | cosine | gate |
  |---|---|---|---|---|
  | 3:1 | 0.01762% | 1.95e-03 | 0.99999998 | PASS |
  | 4:2 | 0.01762% | 1.95e-03 | 0.99999998 | PASS |
  | 8:2 | 0.01761% | 1.95e-03 | 0.99999998 | PASS |

  (rel is at the f16-output floor; the 3:1 row is the EmbeddingGemma reference shape.)

- **Perf A/B (RTX 3090, GPU 1)** — fused single-pass kernel vs the naive composed
  baseline (stage 1: split + RMSNorm each head into an f32 `[tokens][*]` temp in
  global memory + cast V; stage 2: rope + f16-cast Q/K from the temp). The fused
  version keeps the normalized head row in registers, avoiding the temp round-trip.

  | GQA | T | fused ms | naive ms | speedup |
  |---|---|---|---|---|
  | 3:1 | 2048 | 0.0248 | 0.0489 | 1.97x |
  | 4:2 | 2048 | 0.0371 | 0.0715 | 1.93x |
  | 8:2 | 2048 | 0.0536 | 0.1113 | 2.08x |

- **Decision: KEEP** as the default (and only) QK-norm+rope path — a new op (no
  byte-identical-off constraint), exact vs the fp64 oracle at the f16-output floor,
  and ~2x faster than composing separate norm/rope passes because it never
  materializes the f32 intermediate. No env flag: there is no slower variant to gate.
