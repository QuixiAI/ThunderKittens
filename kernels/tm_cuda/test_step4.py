"""Step-4 deferred items vs torch/numpy oracles: bf16 MLA insert, partitioned
+ sparse MLA decodes, gqa_staged (bit-equal to v1), attn_window, lm_head
top-k/top-p, qgemm actorder/blockscale.

Run: CUDA_VISIBLE_DEVICES=6 python -m pytest test_step4.py -q
"""
import math

import numpy as np
import pytest
import torch

from tk_cuda import _C

torch.manual_seed(11)
DEV = "cuda"


def _rng_gumbel_np(seed, a, b):
    M = np.uint64(0xFFFFFFFF)
    x = (np.uint64(seed) * np.uint64(0x9E3779B9) + np.uint64(a) * np.uint64(0x85EBCA77)
         + np.uint64(b) * np.uint64(0xC2B2AE3D)) & M
    x ^= x >> np.uint64(16); x = (x * np.uint64(0x7FEB352D)) & M
    x ^= x >> np.uint64(15); x = (x * np.uint64(0x846CA68B)) & M
    x ^= x >> np.uint64(16)
    u = np.maximum((x >> np.uint64(8)).astype(np.float64) / 16777216.0, 1e-20)
    # kernel computes -log(-log(u)) in fp32
    return -np.log(-np.log(u.astype(np.float32)).astype(np.float32)).astype(np.float32)


# ------------------------------------------------------------- MLA family
def _mla_setup():
    B, H, BS, MAXB, LAT, ROPE = 2, 4, 16, 8, 512, 64
    ctx = [49, 90]
    nb = B * MAXB + 1
    bt = torch.full((B, MAXB), -1, device=DEV, dtype=torch.int32)
    nxt = 1
    for b in range(B):
        for c in range((ctx[b] + BS - 1) // BS):
            bt[b, c] = nxt
            nxt += 1
    return B, H, BS, MAXB, LAT, ROPE, ctx, nb, bt


def test_mla_kv_insert_and_partition():
    B, H, BS, MAXB, LAT, ROPE, ctx, nb, bt = _mla_setup()
    QK = LAT + ROPE
    T = sum(ctx)
    kv_c = (torch.rand(T, LAT, device=DEV) - 0.5).bfloat16()
    k_pe = (torch.rand(T, ROPE, device=DEV) - 0.5).bfloat16()
    P = 64
    theta = torch.arange(P, device=DEV)[:, None] / (
        10000.0 ** (torch.arange(ROPE // 2, device=DEV)[None, :] / (ROPE // 2)))
    cos, sin = theta.cos().bfloat16().contiguous(), theta.sin().bfloat16().contiguous()
    positions = torch.zeros(T, device=DEV, dtype=torch.int32)
    slots = torch.empty(T, device=DEV, dtype=torch.int64)
    i = 0
    for b in range(B):
        for t in range(ctx[b]):
            positions[i] = t % P
            slots[i] = int(bt[b, t // BS]) * BS + t % BS
            i += 1
    cache = torch.zeros(nb, BS, QK, device=DEV).bfloat16()
    _C.mla_kv_insert(kv_c, k_pe, cos, sin, positions, slots, cache, BS)

    # reference rows
    cf, pf = cache.float().reshape(-1, QK), positions.long()
    e, o = k_pe.float()[:, 0::2], k_pe.float()[:, 1::2]
    cth, sth = cos.float()[pf], sin.float()[pf]
    ref_rope = torch.stack([e * cth - o * sth, e * sth + o * cth], -1).reshape(T, ROPE)
    got = cf[slots]
    torch.testing.assert_close(got[:, :LAT], kv_c.float(), rtol=0, atol=1e-2)
    torch.testing.assert_close(got[:, LAT:], ref_rope, rtol=0, atol=2e-2)

    # partitioned decode == serial decode (same cache)
    q = (torch.rand(B, H, QK, device=DEV) - 0.5).bfloat16()
    ctx_t = torch.tensor(ctx, device=DEV, dtype=torch.int32)
    scale = 1.0 / math.sqrt(QK)
    o1 = _C.mla_decode(q, cache, bt, ctx_t, BS, scale)
    o2 = _C.mla_decode_partition(q, cache, bt, ctx_t, BS, scale, 32, max(ctx))
    torch.testing.assert_close(o1.float(), o2.float(), rtol=1e-2, atol=1e-2)


def test_mla_fp8_sparse_and_partition():
    B, H, BS, MAXB, LAT, ROPE, ctx, nb, bt = _mla_setup()
    T = sum(ctx)
    kv = (torch.rand(T, LAT, device=DEV) - 0.5).bfloat16()
    theta = torch.arange(64, device=DEV)[:, None] / (
        10000.0 ** (torch.arange(32, device=DEV)[None, :] / 32.0))
    cos, sin = theta.cos().bfloat16().contiguous(), theta.sin().bfloat16().contiguous()
    positions = torch.zeros(T, device=DEV, dtype=torch.int32)
    slots = torch.empty(T, device=DEV, dtype=torch.int64)
    i = 0
    for b in range(B):
        for t in range(ctx[b]):
            positions[i] = t % 64
            slots[i] = int(bt[b, t // BS]) * BS + t % BS
            i += 1
    data = torch.zeros(nb, BS, 576, device=DEV, dtype=torch.uint8)
    scl = torch.zeros(nb, BS, 8, device=DEV, dtype=torch.uint8)
    _C.mla_kv_insert_fp8(kv, cos, sin, positions, slots, data, scl, BS)

    q = (torch.rand(B, H, LAT, device=DEV) - 0.5).bfloat16()
    ctx_t = torch.tensor(ctx, device=DEV, dtype=torch.int32)
    scale = 1.0 / math.sqrt(LAT)

    # dense partition == dense serial
    o1 = _C.mla_decode_fp8(q, data, scl, bt, ctx_t, BS, scale)
    o2 = _C.mla_decode_fp8_partition(q, data, scl, bt, ctx_t, BS, scale, 32, max(ctx))
    torch.testing.assert_close(o1.float(), o2.float(), rtol=1e-2, atol=1e-2)

    # sparse over ALL tokens == dense; partitioned sparse == sparse
    max_topk = max(ctx)
    idx = torch.full((B, max_topk), -1, device=DEV, dtype=torch.int32)
    tlen = torch.tensor(ctx, device=DEV, dtype=torch.int32)
    for b in range(B):
        idx[b, :ctx[b]] = torch.arange(ctx[b], device=DEV, dtype=torch.int32)
    o3 = _C.mla_decode_fp8_sparse(q, data, scl, bt, idx, tlen, BS, scale)
    torch.testing.assert_close(o1.float(), o3.float(), rtol=1e-2, atol=1e-2)
    o4 = _C.mla_decode_fp8_sparse(q, data, scl, bt, idx, tlen, BS, scale, partition_size=16)
    torch.testing.assert_close(o3.float(), o4.float(), rtol=1e-2, atol=1e-2)


# ------------------------------------------------- gqa_staged / attn_window
def test_gqa_staged_bit_equal():
    B, H, HKV, D, BS, MAXB = 2, 8, 2, 128, 16, 8
    ctx = [37, 100]
    nb = B * MAXB + 1
    bt = torch.full((B, MAXB), -1, device=DEV, dtype=torch.int32)
    nxt = 1
    for b in range(B):
        for c in range((ctx[b] + BS - 1) // BS):
            bt[b, c] = nxt
            nxt += 1
    kc = (torch.rand(nb, BS, HKV, D, device=DEV) - 0.5).half()
    vc = (torch.rand(nb, BS, HKV, D, device=DEV) - 0.5).half()
    q = (torch.rand(B, H, D, device=DEV) - 0.5).half()
    ctx_t = torch.tensor(ctx, device=DEV, dtype=torch.int32)
    scale = 1.0 / math.sqrt(D)
    o1 = _C.paged_attention(q, kc, vc, bt, ctx_t, BS, scale, HKV)
    o2 = _C.paged_attention_gqa_staged(q, kc, vc, bt, ctx_t, BS, scale, HKV)
    assert torch.equal(o1, o2)          # identical op order -> bit-for-bit


@pytest.mark.parametrize("window", [0, 24])
def test_attn_window(window):
    B, H, N, D = 2, 3, 64, 64
    q, k, v = ((torch.rand(B, H, N, D, device=DEV) - 0.5).half() for _ in range(3))
    scale = 1.0 / math.sqrt(D)
    o = _C.attn_window(q, k, v, scale, window)
    s = torch.einsum("bhnd,bhmd->bhnm", q.float(), k.float()) * scale
    mask = torch.ones(N, N, device=DEV).tril()
    if window > 0:
        mask *= torch.ones(N, N, device=DEV).triu(-(window - 1))
    s = s.masked_fill(mask == 0, float("-inf"))
    ref = torch.einsum("bhnm,bhmd->bhnd", torch.softmax(s, -1), v.float())
    torch.testing.assert_close(o.float(), ref, rtol=2e-2, atol=2e-3)


# --------------------------------------------------- lm_head top-k / top-p
def test_lm_head_topk():
    T, V, K, k = 16, 4096, 128, 16
    h = (torch.randn(T, K, device=DEV) * 0.3).half()
    W = (torch.randn(V, K, device=DEV) * 0.3).half()
    temp, seed = 0.8, 77
    got = _C.lm_head_sample_topk(h, W, "fp16", k, temp, seed)
    logits = (h.float() @ W.float().T).double()
    vals, ids = torch.sort(logits, dim=-1, descending=True, stable=True)
    ref = torch.empty(T, dtype=torch.int32)
    for t in range(T):
        cand = ids[t, :k].cpu().numpy().astype(np.uint64)
        g = _rng_gumbel_np(seed, np.uint64(t), cand)
        pert = vals[t, :k].cpu().numpy() / temp + g
        ref[t] = int(cand[int(np.argmax(pert))])
    torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=0)


def test_lm_head_topp_valid():
    # top-p: check the pick is a plausible nucleus member with high perturbed logit
    T, V, K = 16, 4096, 128
    h = (torch.randn(T, K, device=DEV) * 0.3).half()
    W = (torch.randn(V, K, device=DEV) * 0.3).half()
    temp, seed, p = 0.8, 5, 0.9
    got = _C.lm_head_sample_topp(h, W, "fp16", p, 32, temp, seed).cpu()
    logits = (h.float() @ W.float().T) / temp
    probs = torch.softmax(logits, -1)
    sp, si = torch.sort(probs, -1, descending=True)
    for t in range(T):
        csum = torch.cumsum(sp[t], 0)
        ncut = int((csum < p).sum()) + 1
        nucleus = set(si[t, :ncut + 8].tolist())   # small slack for the fp32 bisection edge
        assert int(got[t]) in nucleus


def test_qgemm_actorder_blockscale():
    import os
    import sys
    sys.path.insert(0, os.path.expanduser("~/ThunderMittens/ThunderMittens/kernels"))
    from tk.quant import QUANT_FORMATS

    M, N, K = 32, 64, 128
    X = (torch.randn(M, K, device=DEV) * 0.3).half()
    W = (torch.randn(N, K) * 0.3).numpy().astype(np.float32)
    packed, dequant = QUANT_FORMATS["q4_0"][0], QUANT_FORMATS["q4_0"][1]
    Wq = torch.from_numpy(packed(W)).reshape(N, -1).contiguous().cuda()
    Wd = torch.from_numpy(dequant(packed(W))).float().cuda()   # exact dequantized values
    perm = torch.randperm(K, device=DEV).int().contiguous()
    Y = _C.qgemm_actorder(X, Wq, perm, "q4_0")
    ref = X.float()[:, perm.long()] @ Wd.T
    torch.testing.assert_close(Y, ref, rtol=5e-3, atol=5e-3)

    # blockscale: random e4m3 codes + tile scales, reference via numpy e4m3 LUT
    lut = np.zeros(256, dtype=np.float32)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, mm = (c >> 3) & 0xF, c & 7
        lut[c] = s * (mm * 2.0**-9 if e == 0 else (1 + mm / 8.0) * 2.0 ** (e - 7))
    codes = torch.randint(0, 255, (N, K), device=DEV, dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] &= 0xFE                       # no NaNs
    sc2d = (torch.rand(N // 128 + 1, K // 128, device=DEV)[:max(N // 128, 1)] * 0.1 + 0.02).half()
    sc2d = sc2d.contiguous()
    Y2 = _C.qgemm_blockscale(X, codes.contiguous(), sc2d)
    Wd2 = torch.from_numpy(lut[codes.cpu().numpy()]).cuda()
    scale_full = sc2d.float().repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
    # the kernel multiplies code*scale in HALF (large e4m3 values -> visible rounding)
    Wd2s = (Wd2.half() * scale_full.half()).float()
    ref2 = X.float() @ Wd2s.T
    torch.testing.assert_close(Y2, ref2, rtol=1e-2, atol=1e-2)
