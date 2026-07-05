"""MF-M4 logits-processor zoo + EAGLE spec decode vs numpy oracles.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest test_m4.py -q
"""
import numpy as np
import torch

from tk_cuda import _C

np.random.seed(31)
DEV = "cuda"


def _mk(R=6, V=4000):
    return (np.random.rand(R, V) * 16 - 8).astype(np.float32)


def _isninf(a):
    return a < -1e30


def test_top_nsigma():
    lg = _mk()
    ns = np.random.uniform(0.5, 2.5, 6).astype(np.float32)
    o = _C.top_nsigma_mask(torch.from_numpy(lg).cuda(), torch.from_numpy(ns).cuda()).cpu().numpy()
    for r in range(6):
        var = max((lg[r].astype(np.float64) ** 2).sum() / len(lg[r]) * len(lg[r])
                  - lg[r].astype(np.float64).sum() ** 2 / len(lg[r]), 0)
        var = max((np.sum(lg[r].astype(np.float64) ** 2) - lg[r].astype(np.float64).sum() ** 2 / lg[r].size)
                  / (lg[r].size - 1), 0.0)
        thr = lg[r].max() - ns[r] * np.sqrt(var)
        assert np.array_equal(_isninf(o[r]), lg[r] < thr)


def test_top_a_epsilon_eta():
    lg = _mk()
    for name, fn, param in [
        ("top_a", _C.top_a_mask, np.random.uniform(0.05, 0.3, 6)),
        ("epsilon", _C.epsilon_cutoff_mask, np.random.uniform(1e-4, 5e-3, 6)),
    ]:
        p = param.astype(np.float32)
        o = fn(torch.from_numpy(lg).cuda(), torch.from_numpy(p).cuda()).cpu().numpy()
        for r in range(6):
            mx = lg[r].max()
            Z = np.exp(lg[r] - mx).sum()
            prob = np.exp(lg[r] - mx) / Z
            if name == "top_a":
                thr = (1 / Z) ** 2 * p[r]
                mask = prob < thr
            else:
                mask = (lg[r] < mx) & (prob < p[r])
            assert np.array_equal(_isninf(o[r]), mask), name


def test_xtc():
    lg = _mk()
    th = np.random.uniform(0.05, 0.2, 6).astype(np.float32)
    ax = np.ones(6, np.int32)
    ax[0] = 0
    o = _C.xtc_mask(torch.from_numpy(lg).cuda(), torch.from_numpy(th).cuda(),
                    torch.from_numpy(ax).cuda()).cpu().numpy()
    for r in range(6):
        if not ax[r]:
            assert np.array_equal(o[r], lg[r])
            continue
        mx = lg[r].max()
        Z = np.exp(lg[r] - mx).sum()
        prob = np.exp(lg[r] - mx) / Z
        elig = prob >= th[r]
        cnt = int(elig.sum())
        kp = prob[elig].min() if cnt else 1e30
        remove = (cnt > 1) & (prob >= th[r]) & (prob > kp)
        assert np.array_equal(_isninf(o[r]), remove)


def test_quadratic_skew():
    lg = _mk()
    fa = np.random.uniform(0.1, 0.9, 6).astype(np.float32)
    cu = np.random.uniform(0.5, 2.0, 6).astype(np.float32)
    o = _C.quadratic_transform(torch.from_numpy(lg).cuda(), torch.from_numpy(fa).cuda(),
                               torch.from_numpy(cu).cuda()).cpu().numpy()
    for r in range(6):
        mx = lg[r].max()
        k = fa[r] * (3 - cu[r]) * 0.5
        s = fa[r] * (cu[r] - 1) * 0.5
        diff = lg[r] - mx
        diff = diff - diff * diff * (s * diff - k)
        ref = np.where(np.isfinite(diff), lg[r] - diff, lg[r])
        np.testing.assert_allclose(o[r], ref, rtol=1e-4, atol=1e-4)
    # skew on probs
    probs = np.exp(lg - lg.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    sk = np.random.uniform(-1, 1, 6).astype(np.float32)
    o = _C.skew_transform(torch.from_numpy(probs.astype(np.float32)).cuda(),
                          torch.from_numpy(sk).cuda()).cpu().numpy()
    for r in range(6):
        ex = np.exp(sk[r])
        cdf = np.cumsum(probs[r].astype(np.float64))
        ref = cdf ** ex - np.concatenate([[0.0], cdf[:-1] ** ex])
        np.testing.assert_allclose(o[r], ref, rtol=1e-3, atol=1e-4)


def test_eagle_rejection_and_recovered():
    B, V = 5, 500
    nd = np.random.randint(1, 5, B)
    cu = np.cumsum(nd).astype(np.int32)
    TD = int(cu[-1])
    draft = np.random.randint(0, V, TD).astype(np.int64)
    targ = np.random.randint(0, V, TD).astype(np.int64)
    targ[::3] = draft[::3]
    bonus = np.random.randint(0, V, B).astype(np.int64)
    OUT = 8
    out = _C.rejection_greedy_sample(torch.from_numpy(targ).cuda(), torch.from_numpy(draft).cuda(),
        torch.from_numpy(bonus).cuda(), torch.from_numpy(cu).cuda(), B, OUT).cpu().numpy()
    for r in range(B):
        s = 0 if r == 0 else cu[r - 1]
        ref = np.full(OUT, -1, np.int64)
        rej = False
        for p in range(nd[r]):
            ref[p] = targ[s + p]
            if draft[s + p] != targ[s + p]:
                rej = True
                break
        if not rej:
            ref[nd[r]] = bonus[r]
        np.testing.assert_array_equal(out[r, :nd[r] + 1], ref[:nd[r] + 1])

    # recovered tokens (residual argmax)
    tp = np.random.rand(TD, V).astype(np.float32)
    dp = np.random.rand(TD, V).astype(np.float32)
    invq = (np.random.rand(B, V) + 0.1).astype(np.float32)
    rec = _C.sample_recovered_tokens(torch.from_numpy(tp).cuda(), torch.from_numpy(dp).cuda(),
        torch.from_numpy(draft).cuda(), torch.from_numpy(invq).cuda(),
        torch.from_numpy(cu).cuda(), B).cpu().numpy()
    for tok in range(TD):
        r = int(np.searchsorted(cu, tok, "right"))
        resid = np.maximum(tp[tok] - dp[tok], 0) * invq[r]
        assert rec[tok] == int(resid.argmax())
