"""W3 elementwise/norm family vs torch oracles (autograd for every backward).

Run: CUDA_VISIBLE_DEVICES=6 python -m pytest test_elementwise.py -q
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tk_cuda import _C

torch.manual_seed(1234)
DEV = "cuda"


def t32(*shape, lo=-2.0, hi=2.0):
    return (torch.rand(*shape, device=DEV, dtype=torch.float32) * (hi - lo) + lo).contiguous()


# ---------------------------------------------------------------- norms
def _rms_ref(x, w, eps):
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * r * w


def test_rms_norm_fwd_bwd():
    M, D, eps = 33, 768, 1e-5
    x = t32(M, D).requires_grad_(True)
    w = t32(D).requires_grad_(True)
    dy = t32(M, D)
    y = _rms_ref(x, w, eps)
    torch.testing.assert_close(_C.rms_norm(x.detach(), w.detach(), eps), y,
                               rtol=1e-5, atol=1e-5)
    (y * dy).sum().backward()
    rstd = torch.rsqrt(x.detach().pow(2).mean(-1) + eps).float().contiguous()
    dx = _C.rms_norm_bwd_dx(x.detach(), w.detach(), dy, rstd)
    torch.testing.assert_close(dx, x.grad, rtol=2e-4, atol=2e-5)
    dx2, dw = _C.rms_norm_bwd_fused(x.detach(), w.detach(), dy, eps)
    torch.testing.assert_close(dx2, x.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(dw, w.grad, rtol=2e-4, atol=2e-4)


def test_layernorm_fwd_bwd():
    M, D, eps = 29, 512, 1e-5
    x = t32(M, D).requires_grad_(True)
    w, b = t32(D).requires_grad_(True), t32(D).requires_grad_(True)
    dy = t32(M, D)
    y = F.layer_norm(x, (D,), w, b, eps)
    torch.testing.assert_close(_C.layernorm(x.detach(), w.detach(), b.detach(), eps), y,
                               rtol=1e-5, atol=1e-5)
    (y * dy).sum().backward()
    xd = x.detach()
    mean = xd.mean(-1).contiguous()
    rstd = torch.rsqrt(xd.var(-1, unbiased=False) + eps).contiguous()
    dx = _C.layernorm_bwd_dx(xd, w.detach(), dy, mean, rstd)
    torch.testing.assert_close(dx, x.grad, rtol=2e-4, atol=2e-5)
    dx2, dw, db = _C.layernorm_bwd_fused(xd, w.detach(), dy, eps)
    torch.testing.assert_close(dx2, x.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(dw, w.grad, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(db, b.grad, rtol=2e-4, atol=2e-4)


def test_norms_bf16():
    M, D, eps = 16, 1024, 1e-5
    x = t32(M, D).bfloat16()
    w = t32(D).bfloat16()
    ref = _rms_ref(x.float(), w.float(), eps)
    got = _C.rms_norm(x, w, eps).float()
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


# ------------------------------------------------------------ add_norm
def _e4m3_lut():
    lut = np.zeros(256, dtype=np.float64)
    for code in range(256):
        s = -1.0 if code & 0x80 else 1.0
        e, m = (code >> 3) & 0xF, code & 7
        lut[code] = s * (m * 2.0**-9 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
    return lut


def test_add_norm():
    M, D, eps = 17, 512, 1e-5
    x, res, w, b = t32(M, D), t32(M, D), t32(D), t32(D)
    s = x + res
    o_rms = _rms_ref(s, w, eps)
    o, ro = _C.rms_norm_add(x, res, w, eps)
    torch.testing.assert_close(ro, s, rtol=0, atol=0)
    torch.testing.assert_close(o, o_rms, rtol=1e-5, atol=1e-5)
    o, ro = _C.layernorm_add(x, res, w, b, eps)
    torch.testing.assert_close(o, F.layer_norm(s, (D,), w, b, eps), rtol=1e-5, atol=1e-5)

    lut = _e4m3_lut()
    inv_scale = 3.0
    codes, ro = _C.rms_norm_add_fp8(x, res, w, eps, inv_scale)
    deq = torch.from_numpy(lut[codes.cpu().numpy()]).float().cuda() / inv_scale
    torch.testing.assert_close(deq, o_rms, rtol=0.07, atol=0.07)

    codes, ro, scale = _C.rms_norm_add_fp8_dyn(x, res, w, eps)
    amax = o_rms.abs().amax(-1)
    torch.testing.assert_close(scale, amax / 448.0, rtol=1e-5, atol=1e-8)
    deq = torch.from_numpy(lut[codes.cpu().numpy()]).float().cuda() * scale[:, None]
    torch.testing.assert_close(deq, o_rms, rtol=0.07, atol=0.02)

    o_ln = F.layer_norm(s, (D,), w, b, eps)
    codes, ro = _C.layernorm_add_fp8(x, res, w, b, eps, inv_scale)
    deq = torch.from_numpy(lut[codes.cpu().numpy()]).float().cuda() / inv_scale
    torch.testing.assert_close(deq, o_ln, rtol=0.07, atol=0.07)
    codes, ro, scale = _C.layernorm_add_fp8_dyn(x, res, w, b, eps)
    deq = torch.from_numpy(lut[codes.cpu().numpy()]).float().cuda() * scale[:, None]
    torch.testing.assert_close(deq, o_ln, rtol=0.07, atol=0.02)


# ------------------------------------------------- softmax / activations
def test_softmax():
    x = t32(21, 1000, lo=-4, hi=4)
    torch.testing.assert_close(_C.softmax(x), torch.softmax(x, -1), rtol=1e-5, atol=1e-6)


def test_gelu():
    x = t32(4099, lo=-4, hi=4).requires_grad_(True)
    dy = t32(4099)
    y = F.gelu(x, approximate="tanh")
    torch.testing.assert_close(_C.gelu(x.detach()), y, rtol=1e-5, atol=1e-6)
    (y * dy).sum().backward()
    torch.testing.assert_close(_C.gelu_bwd(x.detach(), dy), x.grad, rtol=2e-4, atol=1e-5)


GLU_MODES = {"reglu": 0, "geglu": 1, "swiglu": 2, "swiglu_oai": 3,
             "geglu_erf": 4, "geglu_quick": 5}


def _glu_ref(mode, a, b, alpha=1.702, limit=7.0):
    if mode == "reglu":
        return F.relu(a) * b
    if mode == "geglu":
        return F.gelu(a, approximate="tanh") * b
    if mode == "swiglu":
        return F.silu(a) * b
    if mode == "swiglu_oai":
        x0 = torch.clamp(a, max=limit)
        x1 = torch.clamp(b, -limit, limit)
        return x0 * torch.sigmoid(alpha * x0) * (1 + x1)
    if mode == "geglu_erf":
        return 0.5 * a * (1 + torch.erf(a / math.sqrt(2))) * b
    return a * torch.sigmoid(1.702 * a) * b


@pytest.mark.parametrize("mode", list(GLU_MODES))
def test_glu(mode):
    n = 2051
    a = t32(n, lo=-3, hi=3).requires_grad_(True)
    b = t32(n, lo=-3, hi=3).requires_grad_(True)
    dc = t32(n)
    y = _glu_ref(mode, a, b)
    # geglu_erf forward uses the A&S rational approximation (max abs err ~1.5e-7 in erf)
    torch.testing.assert_close(_C.glu(a.detach(), b.detach(), GLU_MODES[mode]), y,
                               rtol=1e-4, atol=2e-6)
    (y * dc).sum().backward()
    da, db = _C.glu_bwd(a.detach(), b.detach(), dc, GLU_MODES[mode])
    torch.testing.assert_close(da, a.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(db, b.grad, rtol=2e-4, atol=2e-5)


def _rng_uniform_np(seed, a, b=0):
    M = np.uint64(0xFFFFFFFF)
    x = (np.uint64(seed) * np.uint64(0x9E3779B9) + np.uint64(a) * np.uint64(0x85EBCA77)
         + np.uint64(b) * np.uint64(0xC2B2AE3D)) & M
    x ^= x >> np.uint64(16); x = (x * np.uint64(0x7FEB352D)) & M
    x ^= x >> np.uint64(15); x = (x * np.uint64(0x846CA68B)) & M
    x ^= x >> np.uint64(16)
    return (x >> np.uint64(8)).astype(np.float64) / 16777216.0


def test_dropout():
    n, seed, p = 10007, 77, 0.3
    x = t32(n)
    u = _rng_uniform_np(seed, np.arange(n, dtype=np.uint64))
    keep = torch.from_numpy(u >= p).cuda()
    ref = torch.where(keep, x / (1 - p), torch.zeros_like(x))
    torch.testing.assert_close(_C.dropout(x, seed, p), ref, rtol=0, atol=0)
    torch.testing.assert_close(_C.dropout_bwd(x, seed, p), ref, rtol=0, atol=0)


# ------------------------------------------------------- cross entropy
@pytest.mark.parametrize("mw", [0, 1])
def test_cross_entropy_plain(mw):
    Tn, V, ii = 12, 5000, -100
    logits = t32(Tn, V, lo=-6, hi=6).requires_grad_(True)
    tgt = torch.randint(0, V, (Tn,), device=DEV, dtype=torch.int32)
    tgt[3] = ii
    tgt[7] = ii
    for smooth in (0.0, 0.1):
        loss_ref = F.cross_entropy(logits, tgt.long(), ignore_index=ii,
                                   label_smoothing=smooth, reduction="none")
        loss, lse = _C.cross_entropy(logits.detach(), tgt, ii, smooth, 0.0, 0.0, mw)
        torch.testing.assert_close(loss, loss_ref, rtol=1e-4, atol=1e-5)
        if logits.grad is not None:
            logits.grad = None
        go = t32(Tn, lo=0.1, hi=1.0)
        (loss_ref * go).sum().backward()
        grad = _C.cross_entropy_bwd(logits.detach(), tgt, lse, go, ii, smooth, 0.0, 0.0, mw)
        torch.testing.assert_close(grad, logits.grad, rtol=1e-4, atol=1e-6)


def test_cross_entropy_zloss_softcap():
    Tn, V, ii, zl, cap = 8, 4096, -100, 1e-4, 30.0
    logits = t32(Tn, V, lo=-40, hi=40).requires_grad_(True)
    tgt = torch.randint(0, V, (Tn,), device=DEV, dtype=torch.int32)
    capped = cap * torch.tanh(logits / cap)
    lse_ref = torch.logsumexp(capped, -1)
    loss_ref = (lse_ref - capped.gather(1, tgt.long()[:, None])[:, 0]) + zl * lse_ref**2
    loss, lse = _C.cross_entropy(logits.detach(), tgt, ii, 0.0, zl, cap)
    torch.testing.assert_close(loss, loss_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-5, atol=1e-5)
    go = t32(Tn, lo=0.1, hi=1.0)
    (loss_ref * go).sum().backward()
    grad = _C.cross_entropy_bwd(logits.detach(), tgt, lse, go, ii, 0.0, zl, cap)
    torch.testing.assert_close(grad, logits.grad, rtol=1e-4, atol=1e-6)


# ----------------------------------------------------------- embedding
def test_embedding():
    n_tok, vocab, D, scale = 97, 50, 129, 1.5
    table = t32(vocab, D).requires_grad_(True)
    pos = t32(n_tok, D)
    ids = torch.randint(0, vocab, (n_tok,), device=DEV, dtype=torch.int32)
    ids[5] = -1
    ids[9] = vocab + 3
    ids[:20] = 7  # duplication for the atomic path
    valid = (ids >= 0) & (ids < vocab)
    emb = F.embedding(torch.where(valid, ids, 0).long(), table) * scale
    ref = torch.where(valid[:, None], emb, torch.zeros_like(emb)) + pos
    out = _C.embedding(ids, table.detach(), pos, scale)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)

    dy = t32(n_tok, D)
    (emb * torch.where(valid[:, None], dy, torch.zeros_like(dy))).sum().backward()
    dt = _C.embedding_bwd(ids, dy, vocab, scale)
    torch.testing.assert_close(dt, table.grad, rtol=1e-4, atol=1e-5)
    perm = torch.argsort(ids, stable=True).int().contiguous()
    dt2 = _C.embedding_bwd_sorted(ids[perm.long()].contiguous(), perm, dy, vocab, scale)
    torch.testing.assert_close(dt2, table.grad, rtol=1e-4, atol=1e-5)


def test_multimodal():
    n_tok, n_modal, D = 64, 30, 96
    so = torch.tensor([5, 30, 50], device=DEV, dtype=torch.int32)
    sl = torch.tensor([10, 4, 9], device=DEV, dtype=torch.int32)
    ms = torch.tensor([0, 12, 20], device=DEV, dtype=torch.int32)
    src = _C.build_multimodal_src(so, sl, ms, n_tok)
    src_ref = torch.full((n_tok,), -1, device=DEV, dtype=torch.int32)
    for k in range(3):
        for o in range(sl[k]):
            src_ref[so[k] + o] = ms[k] + o
    torch.testing.assert_close(src, src_ref, rtol=0, atol=0)
    text, modal = t32(n_tok, D), t32(n_modal, D)
    out = _C.merge_multimodal(text, modal, src)
    ref = torch.where(src[:, None] >= 0, modal[src.long().clamp(min=0)], text)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


# ----------------------------------------------- hadamard / adamw / add
def _fwht(x):
    d = x.shape[-1]
    h = 1
    y = x.clone()
    while h < d:
        y = y.reshape(-1, d // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack([a + b, a - b], 2)
        h *= 2
    return y.reshape(x.shape)


@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_hadamard(D):
    rows = 37
    x = t32(rows, D)
    scale = 1.0 / math.sqrt(D)
    torch.testing.assert_close(_C.hadamard(x, scale), _fwht(x) * scale,
                               rtol=1e-5, atol=1e-5)


def test_adamw():
    n, lr, wd, step = 5003, 1e-3, 0.01, 7
    p = t32(n).requires_grad_(True)
    g = t32(n)
    opt = torch.optim.AdamW([p], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    m = t32(n, lo=0, hi=1)
    v = t32(n, lo=0, hi=1)
    opt.state[p] = {"step": torch.tensor(float(step - 1)), "exp_avg": m.clone(),
                    "exp_avg_sq": v.clone()}
    p.grad = g.clone()
    p_before = p.detach().clone()
    opt.step()
    po, mo, vo = _C.adamw(p_before, g, m, v, lr, 0.9, 0.999, 1e-8, wd, step)
    # torch computes sqrt(v)/sqrt(bc2)+eps, TM sqrt(v/bc2)+eps — fp32 op-order
    # noise only (the exact TM formula is validated by the fp64 C++ harness)
    torch.testing.assert_close(po, p.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(mo, opt.state[p]["exp_avg"], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(vo, opt.state[p]["exp_avg_sq"], rtol=1e-4, atol=1e-6)


def test_add():
    x, y = t32(5003), t32(5003)
    torch.testing.assert_close(_C.add(x, y), x + y, rtol=0, atol=0)
