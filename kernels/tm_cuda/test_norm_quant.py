"""MF-M2 norm/AZP/group quant vs torch oracles (round-trip at quant precision).

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest test_norm_quant.py -q
"""
import numpy as np
import torch

from tk_cuda import _C

torch.manual_seed(4)
DEV = "cuda"


def _lut():
    lut = np.zeros(256, np.float64)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 0xF, c & 7
        lut[c] = s * (m * 2.0**-9 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
    return lut


LUT = _lut()


def _rms(x, w, eps=1e-5):
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * r * w


def test_rms_norm_quant_fp8_int8():
    M, D = 40, 768
    x = (torch.randn(M, D, device=DEV) * 1.5).half()
    w = (torch.randn(D, device=DEV)).half()
    ref = _rms(x.float(), w.float())
    # fp8 dynamic
    codes, scale = _C.rms_norm_quant(x, w, True, True, 1e-5)
    deq = torch.from_numpy(LUT[codes.cpu().numpy()]).cuda() * scale[:, None]
    rel = (deq - ref).abs().sum() / ref.abs().sum()
    assert rel < 6e-2, rel
    # int8 dynamic
    codes, scale = _C.rms_norm_quant(x, w, False, True, 1e-5)
    deq = torch.from_numpy(codes.cpu().numpy().view('int8')).float().cuda() * scale[:, None]
    rel = (deq - ref).abs().sum() / ref.abs().sum()
    assert rel < 2.5e-2, rel


def test_rms_norm_add_quant():
    M, D = 32, 512
    x = (torch.randn(M, D, device=DEV) * 1.5).half()
    r = (torch.randn(M, D, device=DEV) * 1.5).half()
    w = torch.randn(D, device=DEV).half()
    ref = _rms((x + r).float(), w.float())
    codes, scale, res_out = _C.rms_norm_quant(x, w, False, True, 1e-5, 1.0, r)
    deq = torch.from_numpy(codes.cpu().numpy().view('int8')).float().cuda() * scale[:, None]
    assert (deq - ref).abs().sum() / ref.abs().sum() < 2.5e-2
    torch.testing.assert_close(res_out.float(), (x + r).float(), rtol=0, atol=2e-2)


def test_azp_int8():
    M, D = 40, 768
    x = (torch.randn(M, D, device=DEV) * 2).half()
    codes, scale, azp = _C.azp_int8_quant(x, True)
    deq = (torch.from_numpy(codes.cpu().numpy().view('int8')).float().cuda() - azp[:, None].float()) * scale[:, None]
    rel = (deq - x.float()).abs().sum() / x.float().abs().sum()
    assert rel < 1.2e-2, rel
    # AZP maps row min->-128, max->127 approximately
    for r in range(M):
        assert codes[r].min() <= -120 and codes[r].max() >= 120


def test_per_token_group_int8():
    T, H, GS = 32, 512, 128
    x = (torch.randn(T, H, device=DEV) * 2).half()
    codes, scales = _C.per_token_group_int8_quant(x, GS)
    deq = torch.from_numpy(codes.cpu().numpy().view('int8')).float().cuda() * scales.repeat_interleave(GS, 1)
    rel = (deq - x.float()).abs().sum() / x.float().abs().sum()
    assert rel < 2.5e-2, rel
