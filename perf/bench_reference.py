#!/usr/bin/env python3
"""Framework reference sweep for the host GPU.

Records PyTorch throughput for the core ops the TK kernels target — dense
matmul, scaled-dot-product attention, and layernorm — so kernel numbers can be
judged against what the deployed frameworks achieve on the same device. This is
the "framework baseline" leg of the three-baselines rule in perf/perf.md, and
on hosts that cannot execute the Hopper/Blackwell kernels it doubles as the
device roofline context.

Usage:  .venv/bin/python perf/bench_reference.py [--out results.json]
"""

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess

import torch

WARMUP = 10
ITERS = 30
REPEATS = 3


def time_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    medians = []
    for _ in range(REPEATS):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        medians.append(statistics.median(
            s.elapsed_time(e) for s, e in zip(starts, ends)))
    return statistics.median(medians)


def bench_matmul(results):
    shapes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
              (8192, 8192, 8192), (16384, 16384, 16384),
              # rectangular LLM projection shapes
              (16, 4096, 4096), (128, 4096, 4096), (2048, 4096, 4096),
              (2048, 11008, 4096), (2048, 4096, 11008), (2048, 14336, 4096)]
    for dtype in (torch.bfloat16, torch.float16):
        for M, N, K in shapes:
            try:
                a = torch.randn(M, K, dtype=dtype, device="cuda")
                b = torch.randn(K, N, dtype=dtype, device="cuda")
                ms = time_fn(lambda: a @ b)
                tflops = 2 * M * N * K / (ms / 1e3) / 1e12
                results.append(dict(op="matmul", dtype=str(dtype).split(".")[-1],
                                    shape=f"M={M},N={N},K={K}",
                                    median_ms=round(ms, 4), tflops=round(tflops, 2)))
            except torch.cuda.OutOfMemoryError:
                results.append(dict(op="matmul", dtype=str(dtype).split(".")[-1],
                                    shape=f"M={M},N={N},K={K}", status="oom"))
            torch.cuda.empty_cache()


def bench_sdpa(results):
    B = 16
    for D in (64, 128):
        H = 2048 // D  # keep model width constant at 2048
        for N in (512, 1024, 2048, 4096):
            for causal in (False, True):
                q, k, v = (torch.randn(B, H, N, D, dtype=torch.bfloat16,
                                       device="cuda") for _ in range(3))
                fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=causal)
                ms = time_fn(fn)
                flops = 4 * B * H * N * N * D
                if causal:
                    flops //= 2
                results.append(dict(op="sdpa", dtype="bfloat16",
                                    shape=f"B={B},H={H},N={N},D={D}",
                                    causal=causal, median_ms=round(ms, 4),
                                    tflops=round(flops / (ms / 1e3) / 1e12, 2)))
                torch.cuda.empty_cache()


def bench_layernorm(results):
    b = 16
    d = 1024
    for n in (1024, 2048, 4096, 8192, 16384):
        x = torch.randn(b, n, d, dtype=torch.bfloat16, device="cuda")
        norm = torch.nn.LayerNorm(d).cuda().to(torch.bfloat16)
        with torch.no_grad():
            ms = time_fn(lambda: norm(x))
        gb = 4 * b * n * d / 1e9  # read 2B + write 2B per element
        results.append(dict(op="layernorm", dtype="bfloat16",
                            shape=f"b={b},n={n},d={d}", median_ms=round(ms, 4),
                            gbps=round(gb / (ms / 1e3), 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    results = []
    bench_matmul(results)
    bench_sdpa(results)
    bench_layernorm(results)

    meta = dict(
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0),
        compute_cap=".".join(map(str, torch.cuda.get_device_capability(0))),
        torch=torch.__version__,
        python=platform.python_version(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True).strip(),
        warmup=WARMUP, iters=ITERS, repeats=REPEATS,
    )
    payload = dict(meta=meta, results=results)

    for r in results:
        print(r)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
