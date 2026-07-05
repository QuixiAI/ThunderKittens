"""MF-M6 (TurboQuant FWHT rotation, permute_cols, moe_lora_align) vs numpy.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest test_m6.py -q
"""
import numpy as np
import pytest
import torch

from tk_cuda import _C

np.random.seed(61)
DEV = "cuda"


@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_fwht_rotate(D):
    R = 20
    x = (np.random.rand(R, D) * 4 - 2).astype(np.float32)
    sign = np.where(np.random.rand(D) > 0.5, 1.0, -1.0).astype(np.float32)
    xt = torch.from_numpy(x).cuda()
    st = torch.from_numpy(sign).cuda()
    fwd = _C.fwht_rotate(xt, st, False)
    inv = _C.fwht_rotate(fwd, st, True)
    # self-inverse
    np.testing.assert_allclose(inv.cpu().numpy(), x, rtol=1e-4, atol=1e-4)
    # forward parity vs host FWHT-with-signs
    ref = (x * sign).astype(np.float64)
    h = 1
    while h < D:
        ref = ref.reshape(R, D // (2 * h), 2, h)
        a, b = ref[:, :, 0, :], ref[:, :, 1, :]
        ref = np.stack([a + b, a - b], 2).reshape(R, D)
        h *= 2
    ref /= np.sqrt(D)
    np.testing.assert_allclose(fwd.cpu().numpy(), ref, rtol=1e-4, atol=1e-4)


def test_permute_cols():
    R, C = 40, 256
    x = (np.random.rand(R, C) * 2 - 1).astype(np.float16)
    perm = np.random.permutation(C).astype(np.int32)
    out = _C.permute_cols(torch.from_numpy(x).cuda(), torch.from_numpy(perm).cuda()).cpu().numpy()
    np.testing.assert_array_equal(out, x[:, perm])


def test_moe_lora_align():
    max_loras, E, topk, block, T = 3, 8, 2, 4, 24
    assignments = T * topk
    scap, ecap = 64, 32
    topk_ids = np.random.randint(0, E, assignments).astype(np.int32)
    tlm = np.random.randint(0, max_loras, T).astype(np.int32)
    lora_ids = np.arange(max_loras, dtype=np.int32)
    aen = np.ones(max_loras, np.uint8)
    aen[1] = 0
    sorted_, expert, ntp = _C.moe_lora_align(
        torch.from_numpy(topk_ids).cuda(), torch.from_numpy(tlm).cuda(),
        torch.from_numpy(lora_ids).cuda(), torch.from_numpy(aen).cuda(),
        E, topk, block, scap, ecap)
    sorted_ = sorted_.cpu().numpy().reshape(max_loras, scap)
    ntp = ntp.cpu().numpy()
    for L in range(max_loras):
        if not aen[L]:
            assert ntp[L] == 0
            continue
        cnt = np.zeros(E, int)
        for i in range(assignments):
            if tlm[i // topk] != L:
                continue
            e = topk_ids[i]
            cnt[e] += 1
        pad = sum(((c + block - 1) // block) * block for c in cnt)
        assert ntp[L] == pad
        # each expert segment holds exactly its routing rows
        off = 0
        for e in range(E):
            seg = sorted_[L, off:off + cnt[e]]
            for v in seg:
                if v < assignments:
                    assert topk_ids[v] == e
            off += ((cnt[e] + block - 1) // block) * block
