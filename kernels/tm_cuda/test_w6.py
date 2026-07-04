"""W6 (MoE + linear attention + cmplx_matmul) vs torch oracles.

Run: CUDA_VISIBLE_DEVICES=6 python -m pytest test_w6.py -q
"""
import pytest
import torch
import torch.nn.functional as F

from tk_cuda import _C

torch.manual_seed(7)
DEV = "cuda"


def t32(*shape, lo=-1.0, hi=1.0):
    return (torch.rand(*shape, device=DEV, dtype=torch.float32) * (hi - lo) + lo).contiguous()


# ------------------------------------------------------------------- MoE
def _route_ref(logits, K):
    # stable top-k with smaller-id ties + renormalized softmax
    vals, ids = torch.sort(logits, dim=-1, descending=True, stable=True)
    ids, vals = ids[:, :K], vals[:, :K]
    w = torch.softmax(vals, dim=-1)
    return ids.int(), w


def test_moe_route():
    T, E, K = 128, 16, 4
    logits = t32(T, E, lo=-3, hi=3)
    ids, wts = _C.moe_route_topk(logits, K)
    rids, rw = _route_ref(logits, K)
    torch.testing.assert_close(ids, rids, rtol=0, atol=0)
    torch.testing.assert_close(wts, rw, rtol=1e-5, atol=1e-6)


def test_moe_end_to_end():
    T, E, K, H, INTER = 200, 11, 4, 64, 96
    x = t32(T, H)
    logits = t32(T, E, lo=-3, hi=3)
    w1 = t32(E, H, 2 * INTER, lo=-0.3, hi=0.3)
    w2 = t32(E, INTER, H, lo=-0.3, hi=0.3)

    ids, wts = _C.moe_route_topk(logits, K)
    offsets, sri, inv, off_pad, eot, gix, inv_pad = _C.moe_align(ids, E)
    perm = _C.moe_gather(x, gix)
    inter = _C.moe_gemm_swiglu(perm, w1, eot)
    down = _C.moe_gemm(inter, w2, eot)
    out = _C.moe_finalize(down, inv_pad, wts)

    # dense torch reference
    rids, rw = _route_ref(logits, K)
    ref = torch.zeros(T, H, device=DEV)
    for e in range(E):
        gate_up = x @ w1[e]
        h = F.silu(gate_up[:, :INTER]) * gate_up[:, INTER:]
        y = h @ w2[e]
        mask = (rids == e).float() * rw          # (T, K)
        ref += mask.sum(-1, keepdim=True) * y
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


# ------------------------------------------------- linear attention (D=64)
def _lin_ref(q, k, v, causal):
    s = torch.einsum("bhnd,bhmd->bhnm", q, k)
    if causal:
        n = s.size(-1)
        s = s.tril()  # inclusive
    return torch.einsum("bhnm,bhmd->bhnd", s, v)


@pytest.mark.parametrize("causal,chunked", [(False, False), (True, False), (True, True)])
def test_linear_attn(causal, chunked):
    B, H, N, D = 2, 3, 256, 64
    q, k, v = (t32(B, H, N, D, lo=-0.5, hi=0.5) for _ in range(3))
    o = _C.linear_attn(q, k, v, causal, chunked)
    ref = _lin_ref(q, k, v, causal)
    torch.testing.assert_close(o, ref, rtol=1e-4, atol=1e-4)


def test_linear_attn_bf16():
    B, H, N, D = 1, 2, 128, 64
    q, k, v = (t32(B, H, N, D, lo=-0.5, hi=0.5).bfloat16() for _ in range(3))
    o = _C.linear_attn(q, k, v, True, True).float()
    ref = _lin_ref(q.float(), k.float(), v.float(), True)
    torch.testing.assert_close(o, ref, rtol=5e-2, atol=5e-1)


# ------------------------------------------------------------ cmplx_matmul
def test_cmplx_matmul():
    N, K, M = 96, 48, 64
    A, B = t32(2, N, K), t32(2, K, M)
    D = _C.cmplx_matmul(A, B)
    ac = torch.complex(A[0], A[1])
    bc = torch.complex(B[0], B[1])
    dc = ac @ bc
    torch.testing.assert_close(D[0], dc.real, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(D[1], dc.imag, rtol=1e-4, atol=1e-4)
