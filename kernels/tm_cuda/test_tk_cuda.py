"""End-to-end tests for the tk_cuda torch extension: quantize with
ThunderMittens' quant.py (the single source of truth), run the CUDA kernels,
compare against numpy oracles. Run: .venv/bin/python -m pytest test_tk_cuda.py -v
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.expanduser("~/ThunderMittens/ThunderMittens/kernels"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tk.quant import QUANT_FORMATS, quantize_w8a8, quantize_act_int8, quantize_bitnet  # noqa: E402
import tk_cuda  # noqa: E402

torch.cuda.init()
RNG = np.random.default_rng(3)
N, K, M = 256, 2048, 32
W = (RNG.standard_normal((N, K)) * 0.05).astype(np.float32)
FMTS = ["q8_0", "q4_K", "mxfp4", "nvfp4", "mxfp8", "iq2_xxs", "bitnet"]


def _pack(fmt):
    packed = QUANT_FORMATS[fmt][0](W)
    wdeq = QUANT_FORMATS[fmt][1](packed).astype(np.float32)
    Wq = torch.from_numpy(packed.reshape(N, -1).copy()).cuda()
    return Wq, wdeq


@pytest.mark.parametrize("fmt", FMTS)
def test_qgemv(fmt):
    Wq, wdeq = _pack(fmt)
    x = RNG.standard_normal(K).astype(np.float16)
    ref = wdeq @ x.astype(np.float32)
    d = tk_cuda.qgemv(Wq, torch.from_numpy(x).cuda(), fmt).float().cpu().numpy()
    rel = np.abs(d - ref).sum() / max(np.abs(ref).sum(), 1e-30)
    assert rel < 5e-3, rel


@pytest.mark.parametrize("fmt", FMTS)
def test_qgemm(fmt):
    Wq, wdeq = _pack(fmt)
    X = RNG.standard_normal((M, K)).astype(np.float16)
    ref = X.astype(np.float32) @ wdeq.T
    y = tk_cuda.qgemm(torch.from_numpy(X).cuda(), Wq, fmt).cpu().numpy()
    rel = np.abs(y - ref).sum() / max(np.abs(ref).sum(), 1e-30)
    assert rel < 2e-2, rel


@pytest.mark.parametrize("fmt", ["q4_0", "nvfp4"])
def test_qflux_gelu(fmt):
    Wq, wdeq = _pack(fmt)
    X = RNG.standard_normal((M, K)).astype(np.float16)
    b = RNG.standard_normal(N).astype(np.float32)
    z = X.astype(np.float32) @ wdeq.T + b[None, :]
    ref = 0.5 * z * (1 + np.tanh(0.7978845608028654 * (z + 0.044715 * z**3)))
    y = tk_cuda.qflux_gelu(torch.from_numpy(X).cuda(), Wq, torch.from_numpy(b).cuda(), fmt).cpu().numpy()
    rel = np.abs(y - ref).sum() / max(np.abs(ref).sum(), 1e-30)
    assert rel < 3e-2, rel


def test_w8a8():
    Wq, w_scale = quantize_w8a8(W)
    x = (RNG.standard_normal((K, 1)) * 0.5).astype(np.float32)
    _, Xq, a_scale = quantize_act_int8(x)
    ref = (Wq.astype(np.int32) @ Xq.astype(np.int32)).reshape(-1).astype(np.float32)
    ref = ref * w_scale.astype(np.float16).astype(np.float32) * np.float16(a_scale[0, 0]).astype(np.float32)
    d = tk_cuda.qgemv_w8a8(
        torch.from_numpy(Wq).cuda(), torch.from_numpy(Xq.reshape(-1)).cuda(),
        torch.from_numpy(w_scale.astype(np.float16)).cuda(),
        torch.from_numpy(np.array([a_scale[0, 0]], np.float16)).cuda()).float().cpu().numpy()
    rel = np.abs(d - ref).sum() / max(np.abs(ref).sum(), 1e-30)
    assert rel < 2e-2, rel


def test_w2a8():
    packed = quantize_bitnet(W)
    x = (RNG.standard_normal((K, 1)) * 0.5).astype(np.float32)
    _, Xq, a_scale = quantize_act_int8(x)
    wdeq = QUANT_FORMATS["bitnet"][1](packed).astype(np.float32)
    ref = (wdeq @ Xq.astype(np.float32)).reshape(-1) * np.float16(a_scale[0, 0]).astype(np.float32)
    d = tk_cuda.qgemv_w2a8(
        torch.from_numpy(packed.reshape(N, -1).copy()).cuda(),
        torch.from_numpy(Xq.reshape(-1)).cuda(),
        torch.from_numpy(np.array([a_scale[0, 0]], np.float16)).cuda(), K).float().cpu().numpy()
    rel = np.abs(d - ref).sum() / max(np.abs(ref).sum(), 1e-30)
    assert rel < 2e-2, rel


@pytest.mark.parametrize("kind,qmax", [("int8", 127.0), ("fp8", 448.0)])
def test_quantize_per_token(kind, qmax):
    A = (RNG.standard_normal((33, 257)) * 3).astype(np.float32)
    codes, scale = tk_cuda.quantize_per_token(torch.from_numpy(A).cuda(), kind)
    s = scale.cpu().numpy()
    np.testing.assert_array_equal(s, np.abs(A).max(axis=1) / qmax)  # scale bit-exact


@pytest.mark.parametrize("fmt", ["fp16", "q8_0", "nvfp4"])
@pytest.mark.parametrize("mode", ["argmax", "categorical"])
def test_lm_head_sample(fmt, mode):
    T = 8
    h = RNG.standard_normal((T, K)).astype(np.float16)
    if fmt == "fp16":
        Wt = torch.from_numpy(W.astype(np.float16)).cuda()
        wdeq = W.astype(np.float16).astype(np.float32)
    else:
        Wt, wdeq = _pack(fmt)
    out = tk_cuda.lm_head_sample(torch.from_numpy(h).cuda(), Wt, fmt,
                                 temperature=0.8, seed=99, mode=mode).cpu().numpy()
    # oracle with the identical RNG
    def rng_uniform(seed, a, b):
        x = np.uint32(seed * 0x9E3779B9 + a * 0x85EBCA77 + b * 0xC2B2AE3D)
        x ^= x >> np.uint32(16); x = np.uint32(x * np.uint32(0x7FEB352D))
        x ^= x >> np.uint32(15); x = np.uint32(x * np.uint32(0x846CA68B))
        x ^= x >> np.uint32(16)
        return np.float32(x >> np.uint32(8)) * np.float32(1 / 16777216)
    import warnings
    warnings.filterwarnings("ignore")
    logits = (h.astype(np.float32) @ wdeq.T) * (1 / 0.8)
    if mode == "categorical":
        g = np.array([[-np.log(-np.log(max(float(rng_uniform(np.uint32(99), np.uint32(t), np.uint32(v))), 1e-20)))
                       for v in range(N)] for t in range(T)], np.float32)
        logits = logits + g
    ref = logits.argmax(axis=1)
    # tie tolerance: accept if chosen logit within 1e-2 of the max
    ok = sum(1 for t in range(T)
             if out[t] == ref[t] or logits[t, out[t]] >= logits[t].max() - 1e-2)
    assert ok == T
