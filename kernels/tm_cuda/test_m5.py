"""MF-M5 sparse attention + serving extras vs numpy oracles.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest test_m5.py -q
"""
import numpy as np
import torch

from tk_cuda import _C

np.random.seed(51)
DEV = "cuda"


def _lut():
    lut = np.zeros(256, np.float64)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 0xF, c & 7
        lut[c] = s * (m * 2.0**-9 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
    return lut


LUT = _lut()


def test_merge_attn_states():
    T, H, D, PT = 48, 4, 64, 40
    po = (np.random.rand(T, H, D) * 2 - 1).astype(np.float32)
    so = (np.random.rand(T, H, D) * 2 - 1).astype(np.float32)
    pl = (np.random.rand(T, H) * 5 - 2).astype(np.float32)
    sl = (np.random.rand(T, H) * 5 - 2).astype(np.float32)
    out, olse = _C.merge_attn_states(torch.from_numpy(po).half().cuda(),
                                     torch.from_numpy(pl).cuda(),
                                     torch.from_numpy(so).half().cuda(),
                                     torch.from_numpy(sl).cuda(), PT)
    out, olse = out.float().cpu().numpy(), olse.cpu().numpy()
    for t in range(T):
        for h in range(H):
            if t >= PT:
                ref, rlse = so[t, h], sl[t, h]
            else:
                m = max(pl[t, h], sl[t, h])
                pe, se = np.exp(pl[t, h] - m), np.exp(sl[t, h] - m)
                z = pe + se
                ref = po[t, h] * (pe / z) + so[t, h] * (se / z)
                rlse = np.log(z) + m
            np.testing.assert_allclose(out[t, h], ref, rtol=2e-2, atol=2e-2)
            assert abs(olse[t, h] - rlse) < 1e-3


def test_tau_tail():
    T, H, D, PMAX = 20, 4, 16, 64
    q_dim = H * D
    qkv = (np.random.rand(T, 3 * q_dim) * 2 - 1).astype(np.float32)
    lin = (np.random.rand(T, 2 * H) * 2 - 1).astype(np.float32)
    tab = (np.random.rand(PMAX, H) * 2 - 1).astype(np.float32)
    pos = np.arange(T, dtype=np.int64) % PMAX
    qkv0 = qkv.copy()
    qt = torch.from_numpy(qkv).half().cuda()
    _C.tau_tail(qt, torch.from_numpy(lin).half().cuda(), torch.from_numpy(tab).half().cuda(),
                torch.from_numpy(pos).cuda(), H, D)
    out = qt.float().cpu().numpy()
    for t in range(T):
        for h in range(H):
            tq = np.tanh(lin[t, h])
            tv = np.tanh(lin[t, H + h])
            tp = tab[pos[t], h]
            for d in range(D):
                qi = h * D + d
                vi = 2 * q_dim + h * D + d
                np.testing.assert_allclose(out[t, qi], qkv0[t, qi] * (tq + tp), rtol=3e-2, atol=3e-2)
                np.testing.assert_allclose(out[t, vi], qkv0[t, vi] * (tv + tp), rtol=3e-2, atol=3e-2)
                # K slice untouched (bit-exact fp16)
                ki = q_dim + h * D + d
                assert out[t, ki] == np.float32(np.float16(qkv0[t, ki]))


def test_indexer_k_quant():
    NT, HD, QB, CBS = 30, 128, 64, 16
    NB = NT // CBS + 2
    nqb = HD // QB
    k = (np.random.rand(NT, HD) * 4 - 2).astype(np.float32)
    slot = np.arange(NT, dtype=np.int64)
    cache = _C.indexer_k_quant(torch.from_numpy(k).half().cuda(),
                               torch.from_numpy(slot).cuda(), NB, HD, QB, CBS).cpu().numpy().reshape(-1)
    cache_stride = HD + nqb * 4               # per-block: CBS*HD data, then CBS*nqb scales
    gs = rs = 0.0
    for t in range(NT):
        block, off = int(slot[t] // CBS), int(slot[t] % CBS)
        base = block * CBS * cache_stride     # block base; data at +off*HD, scales at +CBS*HD
        for qb in range(nqb):
            data = cache[base + off * HD + qb * QB: base + off * HD + (qb + 1) * QB]
            so = base + CBS * HD + (off * HD + qb * QB) * 4 // QB
            sc = np.frombuffer(cache[so:so + 4].tobytes(), np.float32)[0]
            deq = LUT[data] * sc
            ref = k[t, qb * QB:(qb + 1) * QB].astype(np.float16).astype(np.float64)
            gs += np.abs(deq - ref).sum()
            rs += np.abs(ref).sum()
    assert gs / max(rs, 1e-9) < 6e-2


def test_convert_vertical_slash():
    B, H, num_rows, nnz_v, nnz_s = 2, 2, 4, 8, 6
    qsl = np.array([200, 240], np.int32)
    ksl = qsl.copy()
    vert = np.zeros((B, H, nnz_v), np.int32)
    slash = np.zeros((B, H, nnz_s), np.int32)
    for b in range(B):
        for h in range(H):
            vert[b, h] = np.sort(np.random.randint(0, 300, nnz_v))
            slash[b, h] = np.sort(np.random.randint(0, 300, nnz_s))[::-1]
    bc, bo, cc, ci = _C.convert_vertical_slash_indexes(
        torch.from_numpy(qsl).cuda(), torch.from_numpy(ksl).cuda(),
        torch.from_numpy(vert.reshape(B * H, nnz_v)).cuda(),
        torch.from_numpy(slash.reshape(B * H, nnz_s)).cuda(), H, num_rows, 64, 64, True)
    bc = bc.cpu().numpy()
    # sanity: block counts are non-negative and within nnz_s; column counts within nnz_v
    assert (bc >= 0).all() and (bc <= nnz_s).all()
    cc = cc.cpu().numpy()
    assert (cc >= 0).all() and (cc <= nnz_v).all()
    # (exact per-element parity is covered by the standalone C++ host-replay harness)
