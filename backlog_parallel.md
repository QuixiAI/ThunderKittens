# Backlog: BAR1 P2P + parallel/ kernels on the 3090 box

Parked on 2026-07-03. Everything up to here is committed (`1b609d22`).
State details also live in `perf/baseline_status.md` ("Ampere (SM86) Port
Status" section). This file is the resume-point for the multi-GPU track.

## 1. Install the patched driver (user action, ~15 min + reboot)

The BAR1-P2P-patched 580.65.06 kernel modules are **already built** for
kernel 6.8.0-110-generic in `.reference/open-gpu-kernel-modules-580.65.06/`
(see `README-BAR1-P2P.md` there for the patch details — tinygrad 9e39420bc4cb
adapted to 580, only 4 files because 580 mainlined most of it).

- BIOS: **nothing to do** — verified Above-4G decoding is already on (GPU
  BARs at 0x401xxxxxxxx) and all 8 GPUs report ReBAR to 32GB
  (`/sys/bus/pci/devices/*/resource1_resize` = 0xffc0). The patched
  `nv-pci.c` performs the resize at driver probe.
- Steps: stop all GPU work → `bash install.sh` in that directory (sudo) →
  reboot.
- Verify:
  1. `nvidia-smi -q -i 0 | grep -A3 'BAR1 Memory'` → expect ~32GB
     (if still 256MiB: `dmesg | grep -i 'bar\|rebar'`; fallback is adding
     `pci=realloc` to the kernel cmdline — still no BIOS visit).
  2. `dmesg | grep -i 'p2p\|static bar1'`
  3. `python -c "import torch; print(torch.cuda.can_device_access_peer(0,1))"`
  4. cuda-samples `p2pBandwidthLatencyTest` if available. Expect
     PCIe-limited numbers (no NVLink here); the win is that P2P works at all.
- Rollback: reinstall the distro driver package (stock 580.65.06 modules).

## 2. Phase 6.5 — port parallel/ pgl-P2P kernels

Precondition: step 1 verified. Scope from the approved plan
(`~/.claude/plans/all-the-kernels-that-serene-storm.md`):

- Port order (each reuses compute halves already ported in P2/P4):
  `all_gather` → `all_to_all` → `gemm_rs` / `gemm_ar` → `ag_gemm` →
  `ulysses_attn` → `ring_attn` → `moe_dispatch_gemm` (h100 variants are the
  sources).
- Approach: the collective side is pgl peer pointers WITHOUT multicast —
  the `TKParallelTensor` multicast=False path. `include/pyutils/
  torchutils.cuh` currently gates ALL PGL machinery behind SM90+; that gate
  needs loosening for the non-multicast subset (peer pointer exchange via
  cudaIpc / can_device_access_peer, no mc handles).
- OUT of scope (multimem is a Hopper+ instruction): `all_reduce`,
  `reduce_scatter`, `all_reduce_educational`. Optional later rewrite as P2P
  ring using the same pgl plumbing.
- Correctness-first; expect PCIe-bandwidth-limited perf (~few GB/s per
  pair over BAR1).

## 3. Smaller open items (grab-bag, any order)

- **P4c (optional)**: mha_h100_lcf same-source Ampere demo — lcf producer
  hooks like bf16_h100_gemm.cu, D64 kv 128x64 2-stage ≤96KB. Task #14.
- **int8 GEMM deeper tuning**: now 50-52 TOP/s large sizes (42% of
  cuBLASLt's 123). Ideas: coalesced direct-store variant (stage each warp's
  tile through a small reused smem buffer), DEPTH=6 with Nb=128, ldmatrix
  B-fragment reuse across k-chunks. 1024-size regressed 61→25 in the final
  binary (run-order/thermal? re-measure isolated before chasing).
- **fp8 GEMM perf**: 23 TFLOP/s vs bf16's 44 — the register dequant is in
  the inner loop. Marlin's answer: convert fp8→fp16 tiles into smem once,
  then hardware ldmatrix. Worth it only if fp8 storage matters for a real
  workload.
- **fftconv_pc**: A100-only config (112KB scratch > 99KB) — untestable on
  this box, port sketch in the plan.
- **H100 audit (upstream)**: the 32-bit register↔shared fast path bug fixed
  for SM80 in `include/ops/group/memory/tile/shared_to_register.cuh` may
  affect SM90 too — unit tests don't cover int shared↔register. Worth an
  upstream issue/PR along with the rotary coord fix and the based
  harness/generator fixes (all committed here, see commit message).
- **Upstreaming**: consider a PR to HazyResearch with the bug fixes
  (independent of the Ampere port itself).

## Environment gotchas for whoever resumes (probably future me)

- State-changing git (add/commit/checkout/fetch) is permission-denied for
  the assistant in this session setup; `git config`/`remote set-url` and
  read-only git work. User runs commits via `! git ...`.
- No `Co-Authored-By: Claude` in commits — Eric Hartford is sole author.
- nvcc lives at `/usr/local/cuda/bin/nvcc` (not on PATH). Standalone builds
  need `-std=c++20 --extended-lambda --expt-relaxed-constexpr -DKITTENS_SM86
  -gencode arch=compute_86,code=sm_86`.
- Python: `/home/quixi/ThunderKittens/.venv/bin/python` (no pip — use
  `~/.local/bin/uv pip install --python .venv/bin/python`).
- Background shell tasks: always `cd` explicitly (cwd drifts).
