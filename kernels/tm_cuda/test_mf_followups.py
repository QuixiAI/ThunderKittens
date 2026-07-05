"""MetalForge follow-up kernels vs numpy/torch oracles.

Covers: EAGLE bookkeeping (prepare_next_token / step_slot_mapping / expand /
copy_and_expand), lightning-indexer cp_gather, fp8-output merge_attn_states,
TurboQuant tq_encode round-trip, and the Mamba selective_scan APC variant.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest test_mf_followups.py -q
"""
import numpy as np
import pytest
import torch

from tk_cuda import _C

np.random.seed(71)
DEV = "cuda"


def _e4m3_table():
    v = np.zeros(256, np.float32)
    for c in range(256):
        s = -1.0 if (c >> 7) & 1 else 1.0
        e = (c >> 3) & 0xF
        m = c & 0x7
        if e == 0:
            v[c] = s * (m / 8.0) * 2.0 ** (-6)
        elif e == 0xF and m == 0x7:
            v[c] = np.nan
        else:
            v[c] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return v


E4M3 = _e4m3_table()


# ----------------------- EAGLE bookkeeping -----------------------
def test_eagle_prepare_next_token():
    R, nsp, vocab = 4, 3, 100
    sampled = np.array([[5, 7, -1], [-1, -1, -1], [9, 200, 4], [1, -1, -1]], np.int64)
    discard = np.array([0, 0, 1, 0], np.uint8)
    backup = np.array([11, 22, 33, 44], np.int64)
    nt, vc = _C.eagle_prepare_next_token(
        torch.from_numpy(sampled).cuda(), torch.from_numpy(discard).cuda(),
        torch.from_numpy(backup).cuda(), vocab)
    nt, vc = nt.cpu().numpy(), vc.cpu().numpy()
    for r in range(R):
        valid = [t for t in sampled[r] if t != -1 and t < vocab]
        if discard[r]:
            assert nt[r] == backup[r] and vc[r] == 0
        else:
            assert nt[r] == (valid[-1] if valid else backup[r])
            assert vc[r] == len(valid)


def test_eagle_step_slot_mapping():
    input_bs, bs = 5, 4
    block_size, max_len, pad, nbpr = 8, 64, -1, 4
    pos = np.array([3, 7, 15, 63, 0], np.int64)
    bt = np.arange(input_bs * nbpr, dtype=np.int32).reshape(input_bs, nbpr) + 10
    seq = np.array([4, 8, 16, 64, 1], np.int32)
    cp, sm = _C.eagle_step_slot_mapping(
        torch.from_numpy(seq).cuda(), torch.from_numpy(pos).cuda(),
        torch.from_numpy(bt).cuda(), block_size, max_len, pad, bs, input_bs, nbpr)
    cp, sm = cp.cpu().numpy(), sm.cpu().numpy()
    for r in range(input_bs):
        if r >= bs:
            assert sm[r] == pad
            continue
        np_ = pos[r] + 1
        ex = np_ >= max_len
        cl = 0 if ex else np_
        assert cp[r] == cl
        bn = min(cl // block_size, nbpr - 1)
        slot = int(bt[r, bn]) * block_size + (cl % block_size)
        assert sm[r] == (pad if ex else slot)


def test_eagle_expand_int64():
    cu = np.array([2, 2, 5, 7], np.int64)
    inp = np.array([9, -1, 3, -1], np.int64)
    rf, rt = -1, 77
    out = _C.eagle_expand_int64(torch.from_numpy(inp).cuda(), torch.from_numpy(cu).cuda(),
                                int(cu[-1]), rf, rt).cpu().numpy()
    exp = np.zeros(cu[-1], np.int64)
    for r in range(len(inp)):
        s = 0 if r == 0 else cu[r - 1]
        exp[s:cu[r]] = rt if inp[r] == rf else inp[r]
    np.testing.assert_array_equal(out, exp)


def test_copy_and_expand_eagle():
    R = 2
    pad, para, npad = -9, -8, 2
    qsl = np.array([0, 3, 7], np.int32)
    qel = np.array([2, 5], np.int32)
    total_in = 7
    ttok = (100 + np.arange(7)).astype(np.int64)
    tpos = np.arange(7).astype(np.int64)
    ntok = np.array([55, 66], np.int64)
    out_len = 64
    oi, op, rej, msk, ni, hm = _C.copy_and_expand_eagle(
        torch.from_numpy(ttok).cuda(), torch.from_numpy(tpos).cuda(), torch.from_numpy(ntok).cuda(),
        torch.from_numpy(qsl).cuda(), torch.from_numpy(qel).cuda(), pad, para, npad, 0, out_len)
    oi, op, rej, msk = (x.cpu().numpy() for x in (oi, op, rej, msk))
    for req in range(R):
        qs, nqs, qe = qsl[req], qsl[req + 1], qel[req]
        num_valid = qe - qs + 1
        out_start = qs + req * npad
        num_rej = nqs - qe - 1
        total_out = num_valid + npad + num_rej
        start_pos, bonus = tpos[qs], ntok[req]
        for j in range(total_out):
            oidx = out_start + j
            iv, ib = j < num_valid, j == num_valid
            ip = num_valid < j < num_valid + npad
            ir = j >= num_valid + npad
            in_idx = min(qs + j, total_in - 1)
            tid = pad
            if iv: tid = ttok[in_idx]
            elif ib: tid = bonus
            elif ip: tid = para
            assert oi[oidx] == tid
            assert op[oidx] == (0 if ir else start_pos + j)
            assert rej[oidx] == (1 if ir else 0)
            assert msk[oidx] == (1 if ip else 0)


# ----------------------- lightning-indexer cp_gather -----------------------
def test_cp_gather_indexer():
    head_dim, qbs, cbs, num_tokens = 128, 128, 4, 6
    block_table = np.array([[7, 3]], np.int32)  # (batch=1, num_blocks=2)
    num_blocks = block_table.shape[1]
    K = np.random.randn(num_tokens, head_dim).astype(np.float16)
    slot = np.array([block_table[0, t // cbs] * cbs + (t % cbs) for t in range(num_tokens)], np.int64)
    cache = _C.indexer_k_quant(torch.from_numpy(K).cuda(), torch.from_numpy(slot).cuda(),
                               8, head_dim, qbs, cbs, False)          # (num_blocks_phys, cbs, stride)
    cu = np.array([0, num_tokens], np.int32)
    dst_k, dst_scale = _C.cp_gather_indexer(
        cache, torch.from_numpy(block_table).cuda(), torch.from_numpy(cu).cuda(),
        head_dim, cbs, num_tokens, qbs)
    dst_k = dst_k.cpu().numpy()
    dst_scale = dst_scale.cpu().numpy()
    cache = cache.cpu().numpy()
    stride = cache.shape[2]
    cflat = cache.reshape(cache.shape[0], cbs * stride)
    for t in range(num_tokens):
        s = slot[t]
        blk, off = s // cbs, s % cbs
        data = cflat[blk, off * head_dim:off * head_dim + head_dim]
        np.testing.assert_array_equal(dst_k[t], data)
        soff = cbs * head_dim + (off * head_dim) * 4 // qbs
        sc = cflat[blk, soff:soff + 4].view(np.float32)[0]
        assert dst_scale[t, 0] == sc


# ----------------------- fp8-output merge_attn_states -----------------------
def test_merge_attn_states_fp8():
    T, H, D, prefix = 6, 3, 16, 4
    out_scale = 0.5
    po = np.random.randn(T, H, D).astype(np.float16)
    so = np.random.randn(T, H, D).astype(np.float16)
    pl = np.random.uniform(-1, 2, (T, H)).astype(np.float32)
    sl = np.random.uniform(-1, 2, (T, H)).astype(np.float32)
    out, _ = _C.merge_attn_states_fp8(
        torch.from_numpy(po).cuda(), torch.from_numpy(pl).cuda(),
        torch.from_numpy(so).cuda(), torch.from_numpy(sl).cuda(), prefix, out_scale)
    dec = E4M3[out.cpu().numpy().astype(np.int32)]
    si = 1.0 / out_scale
    ref = np.zeros((T, H, D), np.float32)
    for t in range(T):
        for h in range(H):
            if t >= prefix:
                ref[t, h] = so[t, h].astype(np.float32)
            else:
                m = max(pl[t, h], sl[t, h])
                pe, se = np.exp(pl[t, h] - m), np.exp(sl[t, h] - m)
                z = pe + se
                ref[t, h] = po[t, h].astype(np.float32) * (pe / z) + so[t, h].astype(np.float32) * (se / z)
    np.testing.assert_allclose(dec, ref * si, rtol=0.08, atol=0.06)


# ----------------------- TurboQuant tq_encode round-trip -----------------------
def _unpack(buf, base, idx, bits):
    bp = idx * bits
    by, bo = bp >> 3, bp & 7
    raw = int(buf[base + by])
    if bo + bits > 8:
        raw |= int(buf[base + by + 1]) << 8
    return (raw >> bo) & ((1 << bits) - 1)


@pytest.mark.parametrize("HS,k_bits,k_signed,v_bits,k_tol,v_tol", [
    (64, 8, 1, 8, 8e-3, 6e-2),
    (128, 8, 1, 8, 8e-3, 6e-2),
    (128, 4, 0, 3, 1.2e-1, 3e-1),
    (256, 8, 1, 4, 8e-3, 2e-1),
])
def test_tq_encode(HS, k_bits, k_signed, v_bits, k_tol, v_tol):
    NT, NKV, block_size = 5, 2, 8
    K = np.random.randn(NT, NKV, HS).astype(np.float32)
    V = np.random.randn(NT, NKV, HS).astype(np.float32)
    sign = np.where(np.random.rand(HS) > 0.5, 1.0, -1.0).astype(np.float32)
    nc = 1 << v_bits
    cent = np.linspace(-2.5, 2.5, nc).astype(np.float32)
    slot = np.arange(NT, dtype=np.int64)
    kc, vc, ks, vs, kz = _C.tq_encode(
        torch.from_numpy(K).cuda(), torch.from_numpy(V).cuda(), torch.from_numpy(slot).cuda(),
        torch.from_numpy(cent).cuda(), torch.from_numpy(sign).cuda(),
        block_size, block_size, k_bits, k_signed, v_bits)
    sg = HS // 32
    kp, vp = (HS * k_bits + 7) // 8, (HS * v_bits + 7) // 8
    kc, vc = kc.cpu().numpy(), vc.cpu().numpy()
    ks, vs, kz = (x.cpu().float().numpy() for x in (ks, vs, kz))
    kge = krs = vge = vrs = 0.0
    for tk in range(NT):
        for h in range(NKV):
            s = slot[tk]
            sb, kb, vb = (s * NKV + h) * sg, (s * NKV + h) * kp, (s * NKV + h) * vp
            for d in range(HS):
                g = d // 32
                idx = _unpack(kc, kb, d, k_bits)
                if k_signed and idx >= (1 << (k_bits - 1)):
                    idx -= (1 << k_bits)
                rec = (idx + kz[sb + g]) * ks[sb + g]
                kge += abs(rec - K[tk, h, d]); krs += abs(K[tk, h, d])
            rot = np.zeros(HS, np.float64)
            for d in range(HS):
                g = d // 32
                idx = _unpack(vc, vb, d, v_bits)
                rot[d] = cent[idx] * vs[sb + g]
            hstep = 1
            while hstep < HS:
                rot = rot.reshape(HS // (2 * hstep), 2, hstep)
                a, b = rot[:, 0, :], rot[:, 1, :]
                rot = np.stack([a + b, a - b], 1).reshape(HS)
                hstep *= 2
            vdec = sign * rot / np.sqrt(HS)
            vge += np.abs(vdec - V[tk, h]).sum(); vrs += np.abs(V[tk, h]).sum()
    assert kge / max(krs, 1e-30) <= k_tol
    assert vge / max(vrs, 1e-30) <= v_tol


# ----------------------- Mamba selective_scan APC -----------------------
def _softplus(x):
    return np.where(x <= 20.0, np.log1p(np.exp(np.minimum(x, 20.0))), x)


def test_selective_scan_apc():
    dim, dstate, n_groups, block_size = 4, 8, 1, 4
    seqlen = 10
    U = np.random.randn(dim, seqlen).astype(np.float32)
    Dl = np.random.uniform(-0.5, 0.5, (dim, seqlen)).astype(np.float32)
    A = np.random.uniform(-1, -0.1, (dim, dstate)).astype(np.float32)
    B = np.random.randn(n_groups * dstate, seqlen).astype(np.float32)
    C = np.random.randn(n_groups * dstate, seqlen).astype(np.float32)
    Dd = np.full(dim, 0.5, np.float32)
    dbias = np.full(dim, 0.1, np.float32)
    Z = np.random.randn(dim, seqlen).astype(np.float32)
    n_blocks = 3
    cache_indices = np.array([[5, 2, 9]], np.int32)
    max_slot = 10
    qsl = np.array([0, seqlen], np.int32)
    state = np.zeros((max_slot, dim, dstate), np.float32)
    bfirst = np.array([0], np.int32)
    blast = np.array([n_blocks - 1], np.int32)
    initidx = np.array([0], np.int32)
    hinit = np.zeros(1, np.uint8)
    dummy = np.zeros(1, np.int32)

    st = torch.from_numpy(state).cuda()
    out = _C.selective_scan_apc(
        torch.from_numpy(U).cuda(), torch.from_numpy(Dl).cuda(), torch.from_numpy(A).cuda(),
        torch.from_numpy(B).cuda(), torch.from_numpy(C).cuda(), torch.from_numpy(Dd).cuda(),
        torch.from_numpy(dbias).cuda(), torch.from_numpy(Z).cuda(), torch.from_numpy(qsl).cuda(),
        torch.from_numpy(cache_indices).cuda(), torch.from_numpy(hinit).cuda(), st,
        torch.from_numpy(bfirst).cuda(), torch.from_numpy(blast).cuda(),
        torch.from_numpy(initidx).cuda(), torch.from_numpy(dummy).cuda(), torch.from_numpy(dummy).cuda(),
        dstate, n_groups, 1, 1, 1, 1, -1, block_size, 0)
    out = out.cpu().numpy()
    st = st.cpu().numpy()

    ref = np.zeros((dim, seqlen), np.float64)
    for d in range(dim):
        S = np.zeros(dstate)
        tp = 0
        chunk = 0
        while tp < seqlen:
            ct = min(block_size, seqlen - tp)
            for off in range(ct):
                t = tp + off
                u = U[d, t]
                dv = _softplus(Dl[d, t] + dbias[d])
                s = Dd[d] * u
                S = np.exp(dv * A[d]) * S + B[:, t] * dv * u
                s += (S * C[:, t]).sum()
                zv = Z[d, t]
                s *= zv / (1.0 + np.exp(-zv))
                ref[d, t] = s
            sbi = blast[0] if chunk == n_blocks - 1 else (tp + ct - 1) // block_size
            slot = cache_indices[0, sbi]
            np.testing.assert_allclose(st[slot, d], S, rtol=1e-4, atol=1e-4)
            tp += ct
            chunk += 1
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
