#!/usr/bin/env python
"""Generate golden test data for the CUDA quant format layer using ThunderMittens'
quant.py as the single source of truth (byte-identical packing across Metal/CUDA).

Usage: python gen_golden.py [outdir] [fmt fmt ...]   (default: all tranche-1 formats)
Writes per format: <outdir>/<fmt>/{Wq.bin, W_deq.bin, X.bin, D_ref.bin, meta.txt}
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/ThunderMittens/ThunderMittens/kernels"))
from tk.quant import QUANT_FORMATS  # noqa: E402

TRANCHE1 = ["q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "kU4B8", "kU4", "hqq",
            "fp8_e4m3", "e5m2", "fp8_block", "fp4_e2m1", "mxfp8", "mxfp4",
            "nvfp4", "mxfp6_e3m2", "mxfp6_e2m3", "bitnet"]
TRANCHE2 = ["q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
            "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s"]

def main():
    args = sys.argv[1:]
    outdir = args[0] if args else "golden"
    fmts = args[1:] if len(args) > 1 else TRANCHE1 + TRANCHE2
    rng = np.random.default_rng(42)
    N, K = 512, 4096
    W = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    X = rng.standard_normal(K).astype(np.float16)

    for fmt in fmts:
        q, dq = QUANT_FORMATS[fmt]
        packed = q(W)                       # (N, nb, block_bytes) uint8
        wdeq = dq(packed).astype(np.float32)  # (N, K)
        dref = (wdeq @ X.astype(np.float32)).astype(np.float32)
        M = 64
        X2 = rng.standard_normal((M, K)).astype(np.float16)
        yref = (X2.astype(np.float32) @ wdeq.T).astype(np.float32)
        bias = rng.standard_normal(N).astype(np.float32)
        z = yref + bias[None, :]
        yflux = (0.5 * z * (1.0 + np.tanh(0.7978845608028654 * (z + 0.044715 * z**3)))).astype(np.float32)
        d = os.path.join(outdir, fmt)
        os.makedirs(d, exist_ok=True)
        packed.astype(np.uint8).tofile(os.path.join(d, "Wq.bin"))
        wdeq.tofile(os.path.join(d, "W_deq.bin"))
        X.tofile(os.path.join(d, "X.bin"))
        dref.tofile(os.path.join(d, "D_ref.bin"))
        X2.tofile(os.path.join(d, "X2.bin"))
        yref.tofile(os.path.join(d, "Y_ref.bin"))
        bias.tofile(os.path.join(d, "bias.bin"))
        yflux.tofile(os.path.join(d, "Yflux_ref.bin"))
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write(f"{fmt} {N} {K}\n")
        bits = packed.size * 8 / (N * K)
        print(f"{fmt:12s} packed {packed.shape} = {bits:.2f} bits/weight")

if __name__ == "__main__":
    main()
