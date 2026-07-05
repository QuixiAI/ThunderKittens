"""Step-5 vs-torch table: tk_cuda quant GEMV/GEMM vs torch fp16 matmul at
matched shapes. The quant kernels read less weight memory (the whole point);
the bar is beating fp16 torch.matmul for >=4-bit formats at decode shapes.

Speedup ratios are meaningful even under GPU contention (both kernels share
the same busy device); absolute GB/s and TFLOP/s are only canonical on an
idle GPU. Uses TM's quant.py packers so weights are byte-identical.

Run: CUDA_VISIBLE_DEVICES=<gpu> python perf/bench_vs_torch.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/ThunderMittens/ThunderMittens/kernels"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "kernels", "tm_cuda"))
from tk.quant import QUANT_FORMATS  # noqa: E402

import tk_cuda  # noqa: E402
from tk_cuda import _C  # noqa: E402

DEV = "cuda"
FORMATS = ["q8_0", "q4_0", "q4_K", "q6_K", "mxfp4", "nvfp4", "mxfp8", "iq4_nl"]


def _time(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench(N, K, M):
    W = (np.random.randn(N, K).astype(np.float32) * 0.1)
    x = torch.randn(M, K, device=DEV, dtype=torch.float16)
    Wt = torch.from_numpy(W).to(DEV).half()

    # torch fp16 baseline (M,K)@(K,N)
    t_torch = _time(lambda: x @ Wt.T)
    gflop = 2.0 * M * N * K
    print(f"\nN={N} K={K} M={M}   torch.matmul fp16: {t_torch*1e3:.3f} ms "
          f"{gflop/t_torch/1e12:.2f} TFLOP/s")
    print(f"  {'format':<10} {'ms':>8} {'TFLOP/s':>9} {'vs torch':>9}")
    for fmt in FORMATS:
        pack = QUANT_FORMATS[fmt][0]
        Wq = torch.from_numpy(pack(W)).reshape(N, -1).contiguous().to(DEV)
        if M == 1:
            xv = x[0].contiguous()
            fn = lambda: tk_cuda.qgemv(Wq, xv, fmt)  # noqa: E731
        else:
            fn = lambda: tk_cuda.qgemm(x, Wq, fmt)   # noqa: E731
        try:
            t = _time(fn)
        except Exception as e:  # noqa: BLE001
            print(f"  {fmt:<10} ERROR {e}")
            continue
        print(f"  {fmt:<10} {t*1e3:8.3f} {gflop/t/1e12:9.2f} {t_torch/t:8.2f}x")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    # M=1 pure GEMV (the decode-critical path, bandwidth-bound);
    # narrow-N=512 decode where ksplit fires; wide N=K=4096 FFN at M=64/256.
    bench(4096, 4096, 1)
    bench(512, 4096, 64)     # attention out-proj / small head: ksplit shape
    for M in (64, 256):
        bench(4096, 4096, M)
