#!/usr/bin/env python
"""Golden data for the integer GEMV paths (W8A8, W2A8/BitNet), from TM's quant.py.
Oracle = TM contract: integer sums exactly, scales as the only float ops.
Usage: python gen_golden_int.py [outdir]
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/ThunderMittens/ThunderMittens/kernels"))
from tk.quant import quantize_w8a8, quantize_act_int8, quantize_bitnet  # noqa: E402

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "golden_int"
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(7)
    N, K = 512, 4096
    W = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    X = (rng.standard_normal((K, 1)) * 0.5).astype(np.float32)

    # W8A8
    Wq, w_scale = quantize_w8a8(W)                       # int8 (N,K), f32 (N,)
    _, Xq, a_scale = quantize_act_int8(X)                # int8 (K,1), f32 (1,1)
    d_ref = (Wq.astype(np.int32) @ Xq.astype(np.int32).reshape(K, 1)).astype(np.float64)
    d_ref = (d_ref[:, 0] * w_scale.astype(np.float16).astype(np.float64)
             * np.float16(a_scale[0, 0]).astype(np.float64)).astype(np.float32)
    Wq.tofile(f"{outdir}/Wq8.bin")
    Xq.tofile(f"{outdir}/Xq8.bin")
    w_scale.astype(np.float16).tofile(f"{outdir}/w_scale.bin")
    np.array([a_scale[0, 0]], np.float16).tofile(f"{outdir}/a_scale.bin")
    d_ref.tofile(f"{outdir}/D_w8a8_ref.bin")

    # W2A8 (bitnet weights x the same int8 activations)
    packed = quantize_bitnet(W)                          # (N, K/32, 10) uint8
    scales = packed[:, :, 0:2].reshape(N, -1).view(np.float16).astype(np.float32)  # (N, K/32)
    qs = packed[:, :, 2:10].astype(np.uint32)            # (N, K/32, 8)
    codes = np.stack([(qs >> (2 * j)) & 3 for j in range(4)], axis=-1).reshape(N, K // 32, 32)
    tern = codes.astype(np.int32) - 1
    xg = Xq.reshape(K // 32, 32).astype(np.int32)
    isums = np.einsum('ngk,gk->ng', tern, xg).astype(np.float32)   # int32-exact per group
    d2 = (isums * scales).sum(axis=1) * np.float16(a_scale[0, 0]).astype(np.float32)
    packed.tofile(f"{outdir}/Wq2.bin")
    d2.astype(np.float32).tofile(f"{outdir}/D_w2a8_ref.bin")

    with open(f"{outdir}/meta.txt", "w") as f:
        f.write(f"{N} {K}\n")
    print(f"wrote {outdir}: N={N} K={K}")

if __name__ == "__main__":
    main()
