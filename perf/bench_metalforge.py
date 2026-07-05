"""MetalForge serving-layer kernels — throughput on an idle RTX 3090 (SM86).

Headline numbers for the CUDA port of AlpinDale's MetalForge kernels: quantized
MoE grouped GEMMs (TFLOP/s), GDN linear-attention decode, Mamba selective scan,
RMSNorm->quant / SwiGLU->quant epilogues, 2-way LSE merge, and the TurboQuant
FWHT rotation (GB/s, bandwidth-bound). Run on a *dedicated* idle GPU:

    CUDA_VISIBLE_DEVICES=<gpu> python perf/bench_metalforge.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "kernels", "tm_cuda"))
import tk_cuda  # noqa: F401,E402
from tk_cuda import _C  # noqa: E402

DEV = "cuda"
torch.manual_seed(0)
np.random.seed(0)


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def row(name, ms, metric):
    print(f"  {name:<34} {ms*1e3:8.3f} ms   {metric}")


print("=" * 74)
print("MetalForge serving kernels — idle RTX 3090 (SM86)")
print("=" * 74)

# ---------------- Quantized MoE grouped GEMM (TFLOP/s) ----------------
print("\n[1] Quantized MoE grouped GEMM   E=8 experts, N=K=4096, rows=2048")
E, N, K = 8, 4096, 4096
TILES = 2048 // 32
rows = TILES * 32
eot = np.array([t % E for t in range(TILES)], np.int32)
A = torch.from_numpy((np.random.randn(rows, K) * 0.3).astype(np.float32)).half().cuda()
d_eot = torch.from_numpy(eot).cuda()
gflop = 2.0 * rows * N * K

# fp8 rowwise
Bcode = torch.from_numpy(np.random.randint(0, 255, (E, N, K), np.uint8)).cuda().contiguous()
Bsc = torch.from_numpy(np.random.uniform(0.02, 0.2, (E, N)).astype(np.float32)).cuda()
t = _time(lambda: _C.moe_gemm_fp8(A, Bcode, Bsc, d_eot, N, K))
row("moe_gemm_fp8", t, f"{gflop/t/1e12:6.2f} TFLOP/s")

# wna16 int4 / int8
for bit in (4, 8):
    PACK, groups, packed_k = 32 // bit, K // 64, K // (32 // bit)
    qw = torch.from_numpy(np.random.randint(0, 2**31, (E, N, packed_k), np.int64).astype(np.uint32)).cuda().contiguous()
    sc = torch.from_numpy(np.random.uniform(0.01, 0.08, (E, N, groups)).astype(np.float32)).cuda().half()
    if bit == 4:
        qz = torch.from_numpy(np.random.randint(0, 255, (E, (N + 1) // 2, groups), np.uint8)).cuda().contiguous()
    else:
        qz = torch.from_numpy(np.random.randint(0, 255, (E, N, groups), np.uint8)).cuda().contiguous()
    t = _time(lambda: _C.moe_gemm_wna16(A, qw, sc, qz, d_eot, N, K, 64, bit))
    row(f"moe_gemm_wna16 int{bit}", t, f"{gflop/t/1e12:6.2f} TFLOP/s")

# ---------------- GDN linear-attention decode (tok/s + state GB/s) ----------------
print("\n[2] GDN linear-attention decode   R=128 seqs, Hk=2 Hv=8 (GQA), Dk=Dv=128")
R, Hk, Hv, Dk, Dv = 128, 2, 8, 128, 128
cu = np.arange(R + 1, dtype=np.int32)            # 1 token per request (decode step)
T = R
q = torch.from_numpy((np.random.rand(T, Hk * Dk).astype(np.float32) - 0.5)).cuda()
k = torch.from_numpy((np.random.rand(T, Hk * Dk).astype(np.float32) - 0.5)).cuda()
v = torch.from_numpy((np.random.rand(T, Hv * Dv).astype(np.float32) - 0.5)).cuda()
g = torch.from_numpy((np.random.rand(T, Hv).astype(np.float32) * 0.14 + 0.85)).cuda()
be = torch.from_numpy((np.random.rand(T, Hv).astype(np.float32) * 0.8 + 0.1)).cuda()
state = torch.zeros(R * Hv * Dv * Dk, dtype=torch.float32, device=DEV)
d_cu = torch.from_numpy(cu).cuda()
slot = torch.from_numpy(np.arange(R, dtype=np.int32)).cuda()
t = _time(lambda: _C.gdn_linear_attention(q, k, v, g, be, state, d_cu, slot, Hk, Hv, Dk, Dv))
state_gb = R * Hv * Dv * Dk * 4 * 2 / 1e9       # read + write the recurrent state
row("gdn_linear_attention", t, f"{T/t/1e3:7.1f}k tok/s   {state_gb/t:5.0f} GB/s state")

# ---------------- Mamba selective scan (GB/s) ----------------
print("\n[3] Mamba selective_scan_varlen   dim=2048, dstate=16, T=2048")
dim, dstate, ng, TT = 2048, 16, 1, 2048
u = torch.from_numpy((np.random.rand(dim, TT).astype(np.float32) - 0.5)).cuda()
delta = torch.from_numpy((np.random.rand(dim, TT).astype(np.float32))).cuda()
A_ = torch.from_numpy((-np.random.rand(dim, dstate).astype(np.float32) - 0.1)).cuda()
Bt = torch.from_numpy((np.random.rand(ng * dstate, TT).astype(np.float32) - 0.5)).cuda()
Ct = torch.from_numpy((np.random.rand(ng * dstate, TT).astype(np.float32) - 0.5)).cuda()
Dd = torch.from_numpy((np.random.rand(dim).astype(np.float32))).cuda()
dbias = torch.from_numpy((np.random.rand(dim).astype(np.float32))).cuda()
z = torch.from_numpy((np.random.rand(dim, TT).astype(np.float32) - 0.5)).cuda()
qsl = torch.from_numpy(np.array([0, TT], np.int32)).cuda()
cache = torch.from_numpy(np.zeros(1, np.int32)).cuda()
hinit = torch.from_numpy(np.zeros(1, np.uint8)).cuda()
sstate = torch.zeros(dim * dstate, dtype=torch.float32, device=DEV)
t = _time(lambda: _C.selective_scan_varlen(u, delta, A_, Bt, Ct, Dd, dbias, z, qsl, cache, hinit,
                                           sstate, dstate, ng, True, -1))
ss_gb = (4 * dim * TT) * 4 / 1e9                 # u+delta+z+out streamed
row("selective_scan_varlen", t, f"{ss_gb/t:5.0f} GB/s")

# ---------------- RMSNorm -> fp8 quant (GB/s) ----------------
print("\n[4] Fused quant epilogues   tokens=8192, hidden=4096")
Tn, H = 8192, 4096
x = torch.from_numpy((np.random.randn(Tn, H) * 0.5).astype(np.float32)).half().cuda()
w = torch.from_numpy((np.random.randn(H) * 0.1 + 1).astype(np.float32)).half().cuda()
t = _time(lambda: _C.rms_norm_quant(x, w, True, True, 1e-5))
row("rms_norm_quant (fp8 dyn)", t, f"{Tn*H*2/t/1e9:5.0f} GB/s in")

# SwiGLU -> fp8 quant
inp = torch.from_numpy((np.random.randn(Tn, 2 * H) * 0.5).astype(np.float32)).half().cuda()
t = _time(lambda: _C.silu_and_mul_quant(inp, True, 0, 4.0))
row("silu_and_mul_quant (fp8)", t, f"{Tn*2*H*2/t/1e9:5.0f} GB/s in")

# ---------------- 2-way LSE merge (GB/s) ----------------
print("\n[5] Sparse-serving primitives")
Tt, Hh, Dd2, PT = 8192, 8, 128, 6000
po = torch.from_numpy((np.random.rand(Tt, Hh, Dd2) - 0.5).astype(np.float32)).half().cuda()
so = torch.from_numpy((np.random.rand(Tt, Hh, Dd2) - 0.5).astype(np.float32)).half().cuda()
pl = torch.from_numpy((np.random.rand(Tt, Hh) * 5 - 2).astype(np.float32)).cuda()
sl = torch.from_numpy((np.random.rand(Tt, Hh) * 5 - 2).astype(np.float32)).cuda()
t = _time(lambda: _C.merge_attn_states(po, pl, so, sl, PT))
merge_gb = Tt * Hh * Dd2 * 2 * 3 / 1e9           # read prefix+suffix, write out (fp16)
row("merge_attn_states", t, f"{merge_gb/t:5.0f} GB/s")

# lightning-indexer fp8 K-quant-and-cache
ni_tok, hd = 8192, 128
kk = torch.from_numpy((np.random.randn(ni_tok, hd) * 0.5).astype(np.float32)).half().cuda()
sm = torch.from_numpy(np.arange(ni_tok, dtype=np.int64)).cuda()
nblk = (ni_tok + 63) // 64
t = _time(lambda: _C.indexer_k_quant(kk, sm, nblk, hd, hd, 64, False))
row("indexer_k_quant_and_cache", t, f"{ni_tok*hd*2/t/1e9:5.0f} GB/s in")

# ---------------- TurboQuant FWHT rotation (GB/s) ----------------
print("\n[6] TurboQuant FWHT rotation   rows=65536, D=128")
Rr, D = 65536, 128
xr = torch.from_numpy((np.random.randn(Rr, D)).astype(np.float32)).cuda()
sign = torch.from_numpy(np.where(np.random.rand(D) > 0.5, 1.0, -1.0).astype(np.float32)).cuda()
t = _time(lambda: _C.fwht_rotate(xr, sign, False))
row("fwht_rotate (fwd)", t, f"{Rr*D*4*2/t/1e9:5.0f} GB/s")

print("\n" + "=" * 74)
