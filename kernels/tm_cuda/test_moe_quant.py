"""MF-M1 quantized MoE vs numpy oracles: fp8/wna16 grouped GEMM, scored routing,
fused activation-quant, per-token-group quant.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest test_moe_quant.py -q
"""
import numpy as np
import pytest
import torch

from tk_cuda import _C

torch.manual_seed(3)
np.random.seed(3)
DEV = "cuda"


def _e4m3_lut():
    lut = np.zeros(256, np.float64)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 0xF, c & 7
        lut[c] = s * (m * 2.0**-9 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
    return lut


LUT = _e4m3_lut()


def _e4m3_encode(x):
    # nearest-code encode via the LUT (host reference; matches the CUDA RNE encoder
    # closely enough for round-trip checks)
    x = np.asarray(x, np.float32)
    codes = np.abs(x[..., None] - LUT[None, :]).argmin(-1).astype(np.uint8)
    return codes


def test_moe_gemm_fp8():
    E, TILES, N, K = 4, 6, 128, 256
    rows = TILES * 32
    eot = np.array([(-1 if t == TILES - 1 else t % E) for t in range(TILES)], np.int32)
    A = (np.random.randn(rows, K) * 0.5).astype(np.float32)
    Wf = (np.random.randn(E, N, K) * 0.1).astype(np.float32)
    Bcode = _e4m3_encode(Wf)                         # (E,N,K) uint8
    Bsc = np.random.uniform(0.02, 0.2, (E, N)).astype(np.float32)

    Y = _C.moe_gemm_fp8(torch.from_numpy(A).half().cuda(),
                        torch.from_numpy(Bcode).cuda().contiguous(),
                        torch.from_numpy(Bsc).cuda(),
                        torch.from_numpy(eot).cuda(), N, K).cpu().numpy()
    Wdeq = LUT[Bcode] * Bsc[:, :, None]              # (E,N,K)
    rel_max = 0.0
    for r in range(rows):
        e = eot[r // 32]
        if e < 0:
            continue
        ref = A[r].astype(np.float64) @ Wdeq[e].T
        rel = np.abs(Y[r] - ref).sum() / max(np.abs(ref).sum(), 1e-9)
        rel_max = max(rel_max, rel)
    assert rel_max < 5e-3, rel_max


@pytest.mark.parametrize("bit", [4, 8])
def test_moe_gemm_wna16(bit):
    E, TILES, N, K, GS = 3, 5, 128, 256, 64
    rows = TILES * 32
    PACK, groups, packed_k = 32 // bit, K // GS, K // (32 // bit)
    QMAX = (1 << bit) - 1
    eot = np.array([(-1 if t == TILES - 1 else t % E) for t in range(TILES)], np.int32)
    A = (np.random.randn(rows, K) * 0.5).astype(np.float32)
    order4 = [0, 2, 4, 6, 1, 3, 5, 7]
    order8 = [0, 2, 1, 3]
    order = order4 if bit == 4 else order8
    q = np.random.randint(0, QMAX + 1, (E, N, K)).astype(np.int64)
    zp = np.random.randint(0, QMAX + 1, (E, N, groups)).astype(np.int64)
    sc = np.random.uniform(0.01, 0.08, (E, N, groups)).astype(np.float32)
    qw = np.zeros((E, N, packed_k), np.uint32)
    for k in range(K):
        local = order[k % PACK]
        qw[:, :, k // PACK] |= (q[:, :, k].astype(np.uint32) << (local * bit))
    if bit == 4:
        qz = np.zeros((E, (N + 1) // 2, groups), np.uint8)
        for n in range(N):
            qz[:, n // 2, :] |= (zp[:, n, :].astype(np.uint8) << ((n & 1) * 4))
    else:
        qz = zp.astype(np.uint8)

    Y = _C.moe_gemm_wna16(torch.from_numpy(A).half().cuda(),
                          torch.from_numpy(qw).cuda().contiguous(),
                          torch.from_numpy(sc).cuda().half(),
                          torch.from_numpy(qz).cuda().contiguous(),
                          torch.from_numpy(eot).cuda(), N, K, GS, bit).cpu().numpy()
    scf = sc.astype(np.float64)
    Wdeq = (q - zp[:, :, np.repeat(np.arange(groups), GS)]) * scf[:, :, np.repeat(np.arange(groups), GS)]
    rel_max = 0.0
    for r in range(rows):
        e = eot[r // 32]
        if e < 0:
            continue
        ref = A[r].astype(np.float64) @ Wdeq[e].T
        rel = np.abs(Y[r] - ref).sum() / max(np.abs(ref).sum(), 1e-9)
        rel_max = max(rel_max, rel)
    assert rel_max < 8e-3, rel_max


@pytest.mark.parametrize("mode", [0, 1])
def test_moe_route_scored(mode):
    T, E, K = 64, 32, 6
    logits = (np.random.randn(T, E) * 2).astype(np.float32)
    ids, w = _C.moe_route_scored(torch.from_numpy(logits).half().cuda(), K, mode, True, 2.5)
    ids, w = ids.cpu().numpy(), w.cpu().numpy()
    lh = logits.astype(np.float32)

    def score(x):
        if mode == 0:
            return 1 / (1 + np.exp(-x))
        y = np.where(x > 20, x, np.log1p(np.exp(x)))
        return np.sqrt(y)

    sc = score(torch.from_numpy(lh).half().float().numpy())   # score the fp16-rounded logits
    for t in range(T):
        order = sorted(range(E), key=lambda e: (-sc[t, e], e))[:K]
        np.testing.assert_array_equal(ids[t], order)
        denom = sum(sc[t, e] for e in order)
        for k, e in enumerate(order):
            assert abs(w[t, k] - sc[t, e] * 2.5 / denom) < 2e-3


def test_silu_and_mul_quant():
    T, H = 40, 256
    inp = (np.random.randn(T, 2 * H) * 2).astype(np.float32)
    g, u = inp[:, :H], inp[:, H:]
    ref = (g / (1 + np.exp(-g))) * u
    # static
    codes, sc = _C.silu_and_mul_quant(torch.from_numpy(inp).half().cuda(), True, 0, 4.0)
    deq = LUT[codes.cpu().numpy()] * 4.0
    rel = np.abs(deq - ref).sum() / np.abs(ref).sum()
    assert rel < 6e-2, rel
    # per-block
    codes, scv = _C.silu_and_mul_quant(torch.from_numpy(inp).half().cuda(), True, 128, 0.0)
    scv = scv.cpu().numpy()
    deq = LUT[codes.cpu().numpy()] * np.repeat(scv, 128, axis=1)
    rel = np.abs(deq - ref).sum() / np.abs(ref).sum()
    assert rel < 4e-2, rel


def test_per_token_group_quant_fp8():
    T, H, GS = 32, 512, 128
    inp = (np.random.randn(T, H) * 2).astype(np.float32)
    codes, sc = _C.per_token_group_quant_fp8(torch.from_numpy(inp).half().cuda(), GS, False, 1e-6)
    deq = LUT[codes.cpu().numpy()] * np.repeat(sc.cpu().numpy(), GS, axis=1)
    rel = np.abs(deq - inp).sum() / np.abs(inp).sum()
    assert rel < 5e-2, rel
