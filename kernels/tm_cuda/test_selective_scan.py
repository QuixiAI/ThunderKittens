"""MF-M3 varlen Mamba-1 selective scan vs numpy recurrence replay.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest test_selective_scan.py -q
"""
import numpy as np
import torch

from tk_cuda import _C

np.random.seed(21)
DEV = "cuda"


def test_selective_scan_varlen():
    B, dim, dstate, ng = 3, 16, 16, 4
    lens = [6, 10, 4]
    qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    TT = int(qsl[-1])
    u = (np.random.rand(dim, TT) - 0.5).astype(np.float32)
    delta = (np.random.rand(dim, TT) * 2 - 1).astype(np.float32)
    A = (-np.random.rand(dim, dstate) * 0.9 - 0.1).astype(np.float32)
    Bt = (np.random.rand(ng, dstate, TT) - 0.5).astype(np.float32)
    Ct = (np.random.rand(ng, dstate, TT) - 0.5).astype(np.float32)
    D = (np.random.rand(dim) - 0.5).astype(np.float32)
    dbias = (np.random.rand(dim) * 0.4 - 0.2).astype(np.float32)
    z = (np.random.rand(dim, TT) * 2 - 1).astype(np.float32)
    cache = np.arange(B, dtype=np.int32)
    hinit = np.zeros(B, np.uint8)
    state = np.zeros((B, dim, dstate), np.float32)

    out = _C.selective_scan_varlen(
        torch.from_numpy(u).cuda(), torch.from_numpy(delta).cuda(),
        torch.from_numpy(A).cuda(), torch.from_numpy(Bt.reshape(ng * dstate, TT)).cuda(),
        torch.from_numpy(Ct.reshape(ng * dstate, TT)).cuda(),
        torch.from_numpy(D).cuda(), torch.from_numpy(dbias).cuda(),
        torch.from_numpy(z).cuda(), torch.from_numpy(qsl).cuda(),
        torch.from_numpy(cache).cuda(), torch.from_numpy(hinit).cuda(),
        torch.from_numpy(state.reshape(-1)).cuda(),
        dstate, ng, True, -1).cpu().numpy()

    # numpy replay
    ref = np.zeros((dim, TT), np.float64)
    for b in range(B):
        for d in range(dim):
            grp = d // (dim // ng)
            S = np.zeros(dstate, np.float64)
            for t in range(qsl[b], qsl[b + 1]):
                uv = u[d, t]
                dv = delta[d, t] + dbias[d]
                dv = np.log1p(np.exp(dv)) if dv <= 20 else dv
                S = np.exp(dv * A[d]) * S + Bt[grp, :, t] * dv * uv
                s = D[d] * uv + (S * Ct[grp, :, t]).sum()
                zv = z[d, t]
                ref[d, t] = s * zv / (1 + np.exp(-zv))
    rel = np.abs(out - ref).sum() / max(np.abs(ref).sum(), 1e-9)
    assert rel < 1e-4, rel
