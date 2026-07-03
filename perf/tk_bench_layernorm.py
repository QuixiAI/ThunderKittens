#!/usr/bin/env python3
"""Clean baseline timing for the TK fused layernorm kernel (dropout+residual+LN).

Must be run with kernels/layernorm as the working directory (so the built _C
extension and the triton baseline import). Compares:

  tk       : _C.fused_layernorm (the ThunderKittens kernel)
  torch    : eager dropout + residual-add + nn.LayerNorm composite
  triton   : FlashAttention-style fused layer_norm_fn (baselines/layer_norm_triton)

Prints a human table; with --json, the last stdout line is a JSON payload
consumed by perf/bench_kernels.py.
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.getcwd())  # the built _C extension lives in the kernel dir

import torch
import torch.nn as nn

D_MODEL = 1024  # hard-coded in layernorm.cu
BATCH = 16
SEQLENS = [1024, 2048, 4096, 8192, 16384]
WARMUP = 10
ITERS = 50
REPEATS = 3


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def make_inputs(b, n):
    x = torch.randn(b, n, D_MODEL, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(b, n, D_MODEL, dtype=torch.bfloat16, device="cuda")
    norm = nn.LayerNorm(D_MODEL).cuda()
    w = norm.weight.detach().to(torch.bfloat16)
    bias = norm.bias.detach().to(torch.bfloat16)
    return x, residual, w, bias, norm


def time_fn(fn, warmup=WARMUP, iters=ITERS, repeats=REPEATS):
    """Median per-call ms via CUDA events; batches iters between syncs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    medians = []
    for _ in range(repeats):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        medians.append(statistics.median(times))
    return dict(
        median_ms=statistics.median(medians),
        min_ms=min(medians),
        max_ms=max(medians),
    )


def gbps(b, n, ms):
    # x read (2B) + residual read (2B) + out write (2B) + out_resid write (2B)
    bytes_moved = 8 * b * n * D_MODEL
    return bytes_moved / (ms / 1e3) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    from _C import fused_layernorm  # built extension in cwd

    try:
        from baselines.layer_norm_triton import layer_norm_fn
    except Exception as e:
        layer_norm_fn = None
        log(f"triton baseline unavailable: {e}")

    torch.manual_seed(0)
    results = []

    # ---- correctness at p=0 (deterministic path) ----
    x, residual, w, bias, norm = make_inputs(4, 1024)
    with torch.no_grad():
        out_tk, resid_tk = fused_layernorm(x, residual, w, bias, 0.0)
        ref_resid = (x.float() + residual.float())
        ref_out = torch.nn.functional.layer_norm(
            ref_resid, (D_MODEL,), w.float(), bias.float(), norm.eps)
    err_out = (out_tk.float() - ref_out).abs().max().item()
    err_resid = (resid_tk.float() - ref_resid).abs().max().item()
    correctness = dict(max_abs_err_out=err_out, max_abs_err_resid=err_resid,
                       tol=0.15)
    log(f"correctness (p=0): max_abs_err out={err_out:.4f} resid={err_resid:.4f}")

    p = args.dropout
    for n in SEQLENS:
        x, residual, w, bias, norm = make_inputs(BATCH, n)
        row = dict(shape=f"b={BATCH},n={n},d={D_MODEL}", dropout_p=p)

        with torch.no_grad():
            t = time_fn(lambda: fused_layernorm(x, residual, w, bias, p))
        row["tk"] = dict(**t, gbps=gbps(BATCH, n, t["median_ms"]))

        dropout = nn.Dropout(p)
        def torch_composite():
            with torch.no_grad():
                dropped = dropout(x)
                resid = residual + dropped
                return norm(resid.to(norm.weight.dtype)), resid.float()
        t = time_fn(torch_composite)
        row["torch_eager"] = dict(**t, gbps=gbps(BATCH, n, t["median_ms"]))

        if layer_norm_fn is not None:
            rowscale = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
            def triton_fused():
                with torch.no_grad():
                    return layer_norm_fn(
                        x, norm.weight, norm.bias, residual=residual,
                        eps=norm.eps, dropout_p=p, rowscale=rowscale,
                        prenorm=True, residual_in_fp32=True, is_rms_norm=False)
            t = time_fn(triton_fused)
            row["triton_fused"] = dict(**t, gbps=gbps(BATCH, n, t["median_ms"]))

        results.append(row)
        msg = f"n={n:6d}  tk {row['tk']['median_ms']:.3f} ms ({row['tk']['gbps']:.0f} GB/s)"
        msg += f"  torch {row['torch_eager']['median_ms']:.3f} ms"
        if "triton_fused" in row:
            msg += f"  triton {row['triton_fused']['median_ms']:.3f} ms"
        log(msg)

    payload = dict(kernel="layernorm", entry="fused_layernorm",
                   correctness=correctness, batch=BATCH, d_model=D_MODEL,
                   warmup=WARMUP, iters=ITERS, repeats=REPEATS, rows=results)
    if args.json:
        print(json.dumps(payload))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
