"""End-to-end tests for the tk_cuda serving/decode bindings, numpy oracles
(adapting ThunderMittens' correctness suites). RNG-dependent tests reproduce
the murmur3 counter RNG exactly. Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest test_serving.py -q
"""
import os
import sys
import warnings

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.expanduser("~/ThunderMittens/ThunderMittens/kernels"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tk_cuda  # noqa: E402
from tk_cuda import _C  # noqa: E402

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(17)


def rng_uniform(seed, a, b):
    x = np.uint32(seed * 0x9E3779B9 + a * 0x85EBCA77 + b * 0xC2B2AE3D)
    x ^= x >> np.uint32(16); x = np.uint32(x * np.uint32(0x7FEB352D))
    x ^= x >> np.uint32(15); x = np.uint32(x * np.uint32(0x846CA68B))
    x ^= x >> np.uint32(16)
    return np.float32(x >> np.uint32(8)) * np.float32(1 / 16777216)


def rng_gumbel(seed, a, b):
    u = max(float(rng_uniform(np.uint32(seed), np.uint32(a), np.uint32(b))), 1e-20)
    return np.float32(-np.log(-np.log(np.float32(u))))


# ---- shared paged-cache fixture ----
B, H, HKV, D, BS, MAXB = 2, 8, 2, 64, 16, 8
CTX = [37, 100]


def make_cache():
    bt = -np.ones((B, MAXB), np.int32)
    nb = 1
    for b in range(B):
        for c in range((CTX[b] + BS - 1) // BS):
            bt[b, c] = nb
            nb += 1
    K = (RNG.standard_normal((nb + 1, BS, HKV, D)) * 0.4).astype(np.float16)
    V = (RNG.standard_normal((nb + 1, BS, HKV, D)) * 0.4).astype(np.float16)
    return bt, K, V


def np_paged_attention(q, K, V, bt, ctx, scale, window=0, alibi=None, mask=None):
    out = np.zeros_like(q, np.float64)
    for b in range(B):
        for h in range(H):
            kvh = h // (H // HKV)
            t0 = max(0, ctx[b] - window) if window > 0 else 0
            sc, vals = [], []
            for t in range(t0, ctx[b]):
                blk = bt[b, t // BS]
                if blk < 0 or (mask is not None and mask[b, t // BS] == 0):
                    continue
                s = float(q[b, h].astype(np.float64) @ K[blk, t % BS, kvh].astype(np.float64)) * scale
                if alibi is not None:
                    s += alibi[h] * (t - ctx[b] + 1)
                sc.append(s)
                vals.append(V[blk, t % BS, kvh].astype(np.float64))
            sc = np.array(sc)
            w = np.exp(sc - sc.max())
            out[b, h] = (w[:, None] * np.stack(vals)).sum(0) / w.sum()
    return out


def test_paged_attention_v1_and_v2():
    bt, K, V = make_cache()
    q = RNG.standard_normal((B, H, D)).astype(np.float16)
    scale = 1.0 / np.sqrt(D)
    tq = torch.from_numpy(q).cuda()
    tK, tV = torch.from_numpy(K).cuda(), torch.from_numpy(V).cuda()
    tbt = torch.from_numpy(bt).cuda()
    tctx = torch.tensor(CTX, dtype=torch.int32).cuda()
    ref = np_paged_attention(q, K, V, bt, CTX, scale)
    o1 = _C.paged_attention(tq, tK, tV, tbt, tctx, BS, scale, HKV).float().cpu().numpy()
    assert np.abs(o1 - ref).max() < 5e-3
    o2 = _C.paged_attention_v2(tq, tK, tV, tbt, tctx, BS, scale, HKV, 32, max(CTX)).float().cpu().numpy()
    assert np.abs(o2 - ref).max() < 5e-3
    # window + alibi on v1
    al = (-0.05 * (np.arange(H) + 1)).astype(np.float32)
    refw = np_paged_attention(q, K, V, bt, CTX, scale, window=24, alibi=al)
    ow = _C.paged_attention(tq, tK, tV, tbt, tctx, BS, scale, HKV,
                            alibi=torch.from_numpy(al).cuda(), window=24).float().cpu().numpy()
    assert np.abs(ow - refw).max() < 5e-3


def test_kv_scatter_gather_roundtrip():
    bt, K, V = make_cache()
    T = sum(CTX)
    nk = RNG.standard_normal((T, HKV, D)).astype(np.float16)
    nv = RNG.standard_normal((T, HKV, D)).astype(np.float16)
    slots, cu = [], [0]
    for b in range(B):
        for t in range(CTX[b]):
            slots.append(int(bt[b, t // BS]) * BS + t % BS)
        cu.append(cu[-1] + CTX[b])
    kc = torch.zeros(K.shape, dtype=torch.float16).cuda().contiguous()
    vc = torch.zeros(V.shape, dtype=torch.float16).cuda().contiguous()
    _C.kv_scatter(torch.from_numpy(nk).cuda(), torch.from_numpy(nv).cuda(),
                  torch.tensor(slots, dtype=torch.int64).cuda(), kc, vc, BS)
    gk, gv = _C.kv_gather(kc, vc, torch.from_numpy(bt).cuda(),
                          torch.tensor(cu, dtype=torch.int32).cuda(), T, HKV, D, BS)
    np.testing.assert_array_equal(gk.cpu().numpy(), nk)
    np.testing.assert_array_equal(gv.cpu().numpy(), nv)


def test_rope_kv_and_q():
    T, PMAX = 20, 64
    M = T * HKV
    k = RNG.standard_normal((M, D)).astype(np.float16)
    v = RNG.standard_normal((M, D)).astype(np.float16)
    pos = ((np.arange(T) * 7) % PMAX).astype(np.int32)
    th = np.arange(PMAX)[:, None] / (10000.0 ** (2 * np.arange(D // 2) / D))
    cos, sin = np.cos(th).astype(np.float16), np.sin(th).astype(np.float16)
    nb = (T + BS - 1) // BS + 1
    slots = np.arange(T, dtype=np.int64) + BS
    kc = torch.zeros((nb + 2, BS, HKV, D), dtype=torch.float16).cuda()
    vc = torch.zeros_like(kc)
    _C.rope_kv_insert(torch.from_numpy(k).cuda(), torch.from_numpy(v).cuda(),
                      torch.from_numpy(cos).cuda(), torch.from_numpy(sin).cuda(),
                      torch.from_numpy(pos).cuda(), torch.from_numpy(slots).cuda(),
                      kc, vc, HKV, BS)
    kcn = kc.cpu().numpy().reshape(-1, HKV, D)
    for t in range(T):
        for h in range(HKV):
            row = k[t * HKV + h].astype(np.float64)
            c, s = cos[pos[t]].astype(np.float64), sin[pos[t]].astype(np.float64)
            k1, k2 = row[:D // 2], row[D // 2:]
            ref = np.concatenate([k1 * c - k2 * s, k2 * c + k1 * s])
            got = kcn[slots[t], h].astype(np.float64)
            assert np.abs(got - ref).max() < 4e-3


def test_attn_q_quantized_kv():
    from tk.quant import quantize_kv, dequantize_kv
    N = 64
    q = RNG.standard_normal((B, H, N, D)).astype(np.float16)
    Kf = (RNG.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    Vf = (RNG.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    Kq = quantize_kv(Kf, "q8_0")
    Vq = quantize_kv(Vf, "q8_0")
    Kd = dequantize_kv(Kq).astype(np.float64)
    Vd = dequantize_kv(Vq).astype(np.float64)
    scale = 1.0 / np.sqrt(D)
    out = _C.attn_q(torch.from_numpy(q.reshape(-1)).cuda().view(torch.half),
                    torch.from_numpy(Kq.reshape(B, H, N, -1).copy()).cuda().reshape(-1),
                    torch.from_numpy(Vq.reshape(B, H, N, -1).copy()).cuda().reshape(-1),
                    "q8_0", True, scale, B, H, N, D)
    got = out.float().cpu().numpy().reshape(B, H, N, D)
    for b in range(B):
        for h in range(H):
            for qi in range(0, N, 17):
                sc = (q[b, h, qi].astype(np.float64) @ Kd[b, h, :qi + 1].T) * scale
                w = np.exp(sc - sc.max())
                ref = (w[:, None] * Vd[b, h, :qi + 1]).sum(0) / w.sum()
                assert np.abs(got[b, h, qi] - ref).max() < 5e-3


def test_mla_decode():
    LAT, ROPE = 512, 64
    QK = LAT + ROPE
    bt = -np.ones((B, MAXB), np.int32)
    nb = 1
    for b in range(B):
        for c in range((CTX[b] + BS - 1) // BS):
            bt[b, c] = nb
            nb += 1
    cache = (RNG.standard_normal((nb + 1, BS, QK)) * 0.2).astype(np.float32)
    q = (RNG.standard_normal((B, H, QK)) * 0.3).astype(np.float32)
    scale = 1.0 / np.sqrt(QK)
    out = _C.mla_decode(torch.from_numpy(q).cuda().bfloat16(),
                        torch.from_numpy(cache).cuda().bfloat16(),
                        torch.from_numpy(bt).cuda(), torch.tensor(CTX, dtype=torch.int32).cuda(),
                        BS, scale).float().cpu().numpy()
    qb = torch.from_numpy(q).bfloat16().float().numpy().astype(np.float64)
    cb = torch.from_numpy(cache).bfloat16().float().numpy().astype(np.float64)
    for b in range(B):
        for h in range(0, H, 3):
            sc, vals = [], []
            for t in range(CTX[b]):
                blk = bt[b, t // BS]
                if blk < 0:
                    continue
                sc.append(float(qb[b, h] @ cb[blk, t % BS]) * scale)
                vals.append(cb[blk, t % BS, :LAT])
            sc = np.array(sc)
            w = np.exp(sc - sc.max())
            ref = (w[:, None] * np.stack(vals)).sum(0) / w.sum()
            assert np.abs(out[b, h] - ref).max() < 8e-3


@pytest.mark.parametrize("mode", ["argmax", "categorical", "top_k"])
def test_samplers_exact(mode):
    T, V = 12, 500
    seed, temp = 321, 0.8
    logits = (RNG.standard_normal((T, V)) * 3).astype(np.float32)
    got = _C.sample(torch.from_numpy(logits).cuda(), mode, seed, temp, 0.9, 24).cpu().numpy()
    inv = np.float32(1.0 / temp)
    for t in range(T):
        if mode == "argmax":
            ref = int(logits[t].argmax())
        elif mode == "categorical":
            pert = [np.float32(logits[t, v] * inv) + rng_gumbel(seed, t, v) for v in range(V)]
            ref = int(np.argmax(pert))
        else:  # top_k: K masked-argmax rounds then gumbel among chosen
            chosen = []
            work = logits[t].copy()
            for _ in range(24):
                bi = int(work.argmax())
                chosen.append(bi)
                work[bi] = -np.inf
            best, ref = -np.inf, chosen[0]
            for c in chosen:
                p = np.float32(logits[t, c] * inv) + rng_gumbel(seed, t, c)
                if p > best or (p == best and c < ref):
                    best, ref = p, c
        assert got[t] == ref


def test_penalties_and_masks():
    T, V, L = 6, 200, 10
    logits = (RNG.standard_normal((T, V)) * 2).astype(np.float32)
    prev = ((np.arange(T * L) * 13) % V).astype(np.int32).reshape(T, L)
    parent = np.arange(T, dtype=np.int32)
    bias = (RNG.standard_normal(V) * 0.1).astype(np.float32)
    out = _C.apply_penalties(torch.from_numpy(logits).cuda(), torch.from_numpy(prev).cuda(),
                             torch.from_numpy(parent).cuda(), 0.8, 1.2, 0.4, 0.15,
                             torch.from_numpy(bias).cuda(), 7, 5, 2).cpu().numpy()
    inv = np.float32(1 / 0.8)
    for t in range(T):
        cnt = np.bincount(prev[t], minlength=V)
        for v in range(0, V, 7):
            e = np.float32(logits[t, v] * inv)
            if cnt[v] > 0:
                e = np.float32(e * 1.2) if e < 0 else np.float32(e / 1.2)
                e = np.float32(e - 0.4 - 0.15 * cnt[v])
            e = np.float32(e + bias[v])
            if v == 7:
                e = np.float32(-3.4028234663852886e38)
            assert out[t, v] == e
    # bitmask
    words = (V + 31) // 32
    bm = RNG.integers(0, 2**31, (T, words)).astype(np.int32)
    mout = _C.apply_token_bitmask(torch.from_numpy(logits).cuda(), torch.from_numpy(bm).cuda()).cpu().numpy()
    for t in range(T):
        for v in range(0, V, 11):
            allow = (int(bm[t, v >> 5]) >> (v & 31)) & 1
            assert mout[t, v] == (logits[t, v] if allow else np.float32(-3.4028234663852886e38))


def test_beam_and_spec():
    # beam advance
    BM, V = 4, 250
    logits = (RNG.standard_normal((B * BM, V)) * 2).astype(np.float32)
    cum = RNG.standard_normal(B * BM).astype(np.float32)
    nt, par, nc = _C.beam_advance(torch.from_numpy(logits).cuda(), torch.from_numpy(cum).cuda(), BM)
    nt, par, nc = nt.cpu().numpy(), par.cpu().numpy(), nc.cpu().numpy()
    for b in range(B):
        cands = []
        for i in range(BM):
            row = b * BM + i
            lse = np.log(np.exp(logits[row].astype(np.float64) - logits[row].max()).sum()) + logits[row].max()
            order = np.argsort(-logits[row], kind="stable")[: 2 * BM]
            for tok in order:
                cands.append((float(cum[row]) + float(logits[row, tok]) - lse, i, int(tok)))
        cands.sort(key=lambda x: -x[0])
        for k in range(BM):
            assert nt[b, k] == cands[k][2] and par[b, k] == cands[k][1]
            assert abs(nc[b, k] - cands[k][0]) < 1e-3
    # spec linear + compact + meta
    S, V2 = 4, 100
    draft = ((np.arange(B * S) * 37) % V2).astype(np.int32).reshape(B, S)
    dp = np.abs(RNG.standard_normal((B, S, V2))).astype(np.float32) + 0.01
    tp = np.abs(RNG.standard_normal((B, S + 1, V2))).astype(np.float32) + 0.01
    dp /= dp.sum(-1, keepdims=True)
    tp /= tp.sum(-1, keepdims=True)
    bonus = ((np.arange(B) * 91) % V2).astype(np.int32)
    u = RNG.random((B, S)).astype(np.float32)
    seed = 777
    out, cnt = _C.spec_verify_linear(torch.from_numpy(draft).cuda(), torch.from_numpy(dp).cuda(),
                                     torch.from_numpy(tp).cuda(), torch.from_numpy(bonus).cuda(),
                                     torch.from_numpy(u).cuda(), seed)
    out, cnt = out.cpu().numpy(), cnt.cpu().numpy()
    for b in range(B):
        rej = S
        for i in range(S):
            dt = draft[b, i]
            if u[b, i] * dp[b, i, dt] <= tp[b, i, dt]:
                assert out[b, i] == dt
                continue
            r = np.maximum(0.0, tp[b, i] - dp[b, i])
            pert = [((np.float32(np.log(r[v])) if r[v] > 0 else np.float32(-3.4028234663852886e38))
                     + rng_gumbel(seed, b * S + i, v)) for v in range(V2)]
            assert out[b, i] == int(np.argmax(pert))
            rej = i
            break
        assert cnt[b] == rej
    seq = (10 + 7 * np.arange(B)).astype(np.int32)
    pt, pp, cu = _C.spec_compact(torch.from_numpy(out).cuda().int(), torch.from_numpy(cnt).cuda().int(),
                                 torch.from_numpy(seq).cuda())
    nsl = _C.spec_update_kv_meta(torch.from_numpy(seq).cuda(), torch.from_numpy(cnt).cuda().int())
    pt, cu, nsl = pt.cpu().numpy(), cu.cpu().numpy(), nsl.cpu().numpy()
    run = 0
    for b in range(B):
        assert cu[b] == run
        for j in range(cnt[b] + 1):
            assert pt[run + j] == out[b, j]
        run += cnt[b] + 1
        assert nsl[b] == seq[b] + cnt[b] + 1
    assert cu[B] == run
