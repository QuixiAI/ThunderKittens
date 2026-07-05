"""MF-M3 GDN gated-deltanet vs numpy recurrence replay.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest test_gdn.py -q
"""
import numpy as np
import torch

from tk_cuda import _C

torch.manual_seed(11)
np.random.seed(11)
DEV = "cuda"


def _gdn_ref(q, k, v, g, be, cu, Hk, Hv, Dk, Dv):
    T = q.shape[0]
    y = np.zeros((T, Hv, Dv), np.float64)
    R = len(cu) - 1
    state_final = np.zeros((R, Hv, Dv, Dk), np.float64)
    for r in range(R):
        s0, s1 = cu[r], cu[r + 1]
        for hv in range(Hv):
            hk = hv // (Hv // Hk)
            S = np.zeros((Dv, Dk), np.float64)
            for t in range(s0, s1):
                kk = k[t, hk].astype(np.float64)
                qq = q[t, hk].astype(np.float64)
                S *= g[t, hv]
                kv = S @ kk                       # (Dv,)
                delta = (v[t, hv].astype(np.float64) - kv) * be[t, hv]  # (Dv,)
                S += np.outer(delta, kk)
                y[t, hv] = S @ qq
            state_final[r, hv] = S
    return y, state_final


def test_gdn():
    R, Hk, Hv, Dk, Dv = 3, 2, 4, 64, 64
    lens = [5, 12, 8]
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    T = int(cu[-1])
    q = (np.random.rand(T, Hk, Dk) * 0.8 - 0.4).astype(np.float32)
    k = (np.random.rand(T, Hk, Dk) * 0.8 - 0.4).astype(np.float32)
    v = (np.random.rand(T, Hv, Dv) * 0.8 - 0.4).astype(np.float32)
    g = (np.random.rand(T, Hv) * 0.14 + 0.85).astype(np.float32)
    be = (np.random.rand(T, Hv) * 0.8 + 0.1).astype(np.float32)
    slot = np.arange(R, dtype=np.int32)
    state = np.zeros((R, Hv, Dv, Dk), np.float32)

    y = _C.gdn_linear_attention(
        torch.from_numpy(q.reshape(T, Hk * Dk)).cuda(),
        torch.from_numpy(k.reshape(T, Hk * Dk)).cuda(),
        torch.from_numpy(v.reshape(T, Hv * Dv)).cuda(),
        torch.from_numpy(g).cuda(), torch.from_numpy(be).cuda(),
        torch.from_numpy(state.reshape(-1)).cuda(),
        torch.from_numpy(cu).cuda(), torch.from_numpy(slot).cuda(),
        Hk, Hv, Dk, Dv).cpu().numpy()

    yref, _ = _gdn_ref(q, k, v, g, be, cu, Hk, Hv, Dk, Dv)
    rel = np.abs(y.reshape(T, Hv, Dv) - yref).sum() / max(np.abs(yref).sum(), 1e-9)
    assert rel < 1e-4, rel
