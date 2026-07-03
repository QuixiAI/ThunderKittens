#!/usr/bin/env python3
"""ThunderKittens kernel baseline harness.

Builds every kernel in kernels/ for its declared architecture (build-health +
ptxas register/spill baseline) and runs the ones the host GPU can actually
execute, recording structured results.

Usage (from repo root, with .venv active or via .venv/bin/python):

    python perf/bench_kernels.py --phase build --kernel all
    python perf/bench_kernels.py --phase run   --kernel all
    python perf/bench_kernels.py --phase all   --kernel layernorm,gemm/baselines/bf16_cublas

Results land in perf/results/YYYY-MM-DD/<run-id>/:
    run.json        environment + invocation metadata
    results.jsonl   one record per kernel per phase (schema_version 1)
    build/<k>.log   full nvcc/ptxas output
    run/<k>.log     full run stdout/stderr
    summary.md      human-readable table
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import platform
import queue
import re
import subprocess
import sys
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS = os.path.join(REPO_ROOT, "kernels")
CUDA_BIN = "/usr/local/cuda/bin"
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Kernel registry
#
# arch      : ARCH the Makefile declares (build target for the build phase)
# min_cc    : minimum device compute capability required to *execute*
# sm80_ok   : source has no Hopper-only features; can be rebuilt with
#             ARCH=SM80 and executed on an Ampere host
# config    : standalone | python | pytorch  (pytorch needs torch in env)
# make_vars : extra make variables
# out       : binary name for standalone kernels (for run phase)
# ---------------------------------------------------------------------------
REGISTRY = [
    # attention
    dict(name="attention/mha_h100", arch="SM90", min_cc=90, config="pytorch"),
    dict(name="attention/mha_h100_lcf", arch="SM90", min_cc=90, config="standalone"),
    dict(name="attention/bf16_b300_mha_causal", arch="SM103", min_cc=103, config="pytorch"),
    dict(name="attention/bf16_b300_mha_noncausal", arch="SM103", min_cc=103, config="pytorch"),
    # linear attention / SSM family
    dict(name="based", arch="SM90", min_cc=90, config="pytorch"),
    dict(name="hedgehog", arch="SM90", min_cc=90, config="pytorch"),
    dict(name="linear_attention", arch="SM90", min_cc=90, config="standalone"),
    dict(name="mamba2", arch="SM90", min_cc=90, config="pytorch"),
    # Ampere ports (see perf/baseline_status.md "Ampere (SM86) Port Status");
    # standalone .cu with build commands in their headers, no Makefile targets yet
    dict(name="attention/mha_ampere", arch="SM86", min_cc=80, sm80_ok=True,
         config="standalone", out="mha_ampere.out",
         notes="resurrected 4090 kernel + full mha_h100 parity incl. backward"),
    dict(name="gemm/int8_ampere", arch="SM86", min_cc=80, sm80_ok=True,
         config="standalone", out="int8_ampere_gemm.out"),
    dict(name="gemm/fp8_ampere", arch="SM86", min_cc=80, sm80_ok=True,
         config="standalone", out="fp8_ampere_gemm.out",
         notes="fp8 storage + Marlin-style register dequant; + _scaled variant"),
    # convolutions / fused
    dict(name="fftconv", arch="SM90", min_cc=90, config="pytorch"),
    dict(name="flux", arch="SM90", min_cc=90, config="pytorch"),
    # pointwise / norm
    dict(name="layernorm", arch="SM90", min_cc=80, sm80_ok=True, config="pytorch",
         runner="layernorm"),
    dict(name="rotary", arch="SM90", min_cc=90, config="pytorch"),
    # dense GEMM (TK)
    dict(name="gemm/bf16_h100", arch="SM90", min_cc=90, config="standalone",
         out="bf16_h100_gemm.out"),
    dict(name="gemm/fp8_h100", arch="SM90", min_cc=90, config="standalone",
         out="fp8_h100_gemm.out"),
    dict(name="gemm/fp8_h100_scaled", arch="SM90", min_cc=90, config="standalone",
         out="fp8_h100_gemm_scaled.out"),
    dict(name="gemm/int8_h100", arch="SM90", min_cc=90, config="standalone",
         out="int8_h100_gemm.out"),
    dict(name="gemm/bf16_b200", arch="SM100", min_cc=100, config="standalone",
         out="bf16_b200_gemm.out"),
    dict(name="gemm/fp8_b200", arch="SM100", min_cc=100, config="standalone",
         out="fp8_b200_gemm.out"),
    dict(name="gemm/int8_b200", arch="SM100", min_cc=100, config="standalone",
         out="int8_b200_gemm.out"),
    dict(name="gemm/mxfp8_b200", arch="SM100", min_cc=100, config="standalone",
         out="mxfp8_b200_gemm.out"),
    dict(name="gemm/nvfp4_b200", arch="SM100", min_cc=100, config="standalone",
         out="nvfp4_b200_gemm.out"),
    dict(name="gemm/educational_h100", arch="SM90", min_cc=90, config="standalone",
         make_vars={"LEVEL": "01"}, out="level_01.out",
         notes="level 01 only; levels 02-08 exist"),
    dict(name="gemm/educational_b200", arch="SM100", min_cc=100, config="standalone",
         make_vars={"LEVEL": "01"}, out="level_01.out",
         notes="level 01 only; levels 02-08 exist"),
    # GEMM baselines (cuBLAS / cuBLASLt) -- pure library calls, arch-portable
    dict(name="gemm/baselines/bf16_cublas", arch="SM100", min_cc=80, sm80_ok=True,
         config="standalone", out="bf16_cublas_gemm.out", runner="stdout"),
    dict(name="gemm/baselines/bf16_cublas_lt", arch="SM100", min_cc=80, sm80_ok=True,
         config="standalone", out="bf16_cublas_lt_gemm.out", runner="stdout"),
    dict(name="gemm/baselines/int8_cublas_lt", arch="SM100", min_cc=80, sm80_ok=True,
         config="standalone", out="int8_cublas_lt_gemm.out", runner="stdout",
         notes="int8 IMMA supported on Ampere; run attempt on SM86 is best-effort"),
    dict(name="gemm/baselines/fp8_cublas_lt", arch="SM100", min_cc=89,
         config="standalone", out="fp8_cublas_lt_gemm.out",
         notes="fp8 needs SM89+ hardware"),
    dict(name="gemm/baselines/mxfp8_cublas_lt", arch="SM100", min_cc=100,
         config="standalone", out="mxfp8_cublas_lt_gemm.out"),
    dict(name="gemm/baselines/nvfp4_cublas_lt", arch="SM100", min_cc=100,
         config="standalone", out="nvfp4_cublas_lt_gemm.out"),
    # multi-GPU kernels (all use SM90+ multimem/TMA; need NVLink/NVSwitch fabric)
    dict(name="parallel/all_gather", arch="SM90", min_cc=90, config="pytorch", multi_gpu=True),
    dict(name="parallel/all_reduce", arch="SM90", min_cc=90, config="pytorch", multi_gpu=True),
    dict(name="parallel/all_reduce_educational", arch="SM90", min_cc=90, config="standalone",
         out="all_reduce_educational.out", multi_gpu=True),
    dict(name="parallel/all_to_all", arch="SM90", min_cc=90, config="pytorch", multi_gpu=True),
    dict(name="parallel/reduce_scatter", arch="SM90", min_cc=90, config="pytorch", multi_gpu=True),
    dict(name="parallel/ag_gemm", arch="SM100", min_cc=100, config="pytorch", multi_gpu=True),
    dict(name="parallel/ag_gemm_fp8", arch="SM100", min_cc=100, config="pytorch", multi_gpu=True),
    dict(name="parallel/gemm_ar", arch="SM90", min_cc=90, config="pytorch", multi_gpu=True),
    dict(name="parallel/gemm_rs", arch="SM100", min_cc=100, config="pytorch", multi_gpu=True),
    dict(name="parallel/gemm_rs_fp8", arch="SM100", min_cc=100, config="pytorch", multi_gpu=True),
    dict(name="parallel/moe_dispatch_gemm", arch="SM90", min_cc=90, config="pytorch", multi_gpu=True),
    dict(name="parallel/ring_attn", arch="SM90", min_cc=90, config="pytorch", multi_gpu=True),
    dict(name="parallel/ulysses_attn", arch="SM90", min_cc=90, config="pytorch", multi_gpu=True),
]


def host_compute_cap():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
        text=True).strip().splitlines()
    cc, name = out[0].split(", ", 1)
    major, minor = cc.split(".")
    return int(major) * 10 + int(minor), name.strip(), len(out)


def build_env():
    env = os.environ.copy()
    venv_bin = os.path.join(REPO_ROOT, ".venv", "bin")
    env["PATH"] = f"{venv_bin}:{CUDA_BIN}:" + env.get("PATH", "")
    env["VIRTUAL_ENV"] = os.path.join(REPO_ROOT, ".venv")
    return env


def pytorch_make_env(env):
    """Precompute torch include/lib paths once so make doesn't shell out per kernel."""
    py = os.path.join(REPO_ROOT, ".venv", "bin", "python")
    code = ("from torch.utils.cpp_extension import include_paths, library_paths;"
            "print(' '.join('-I'+p for p in include_paths()));"
            "print(' '.join('-L'+p for p in library_paths()))")
    inc, lib = subprocess.check_output([py, "-c", code], text=True).strip().splitlines()
    env = env.copy()
    env["PYTORCH_INCLUDES"] = inc
    env["PYTORCH_LIBDIR"] = lib
    return env


PTXAS_REG_RE = re.compile(r"Used (\d+) registers")
PTXAS_SPILL_RE = re.compile(r"(\d+) bytes spill")
PTXAS_SMEM_RE = re.compile(r"(\d+) bytes smem")
TFLOPS_RE = re.compile(r"([\d.]+)\s*(TFLOP|TOP)/?s", re.IGNORECASE)
PROBLEM_RE = re.compile(r"Problem size:\s*(.+)")


def parse_ptxas(log):
    regs = [int(m) for m in PTXAS_REG_RE.findall(log)]
    spills = [int(m) for m in PTXAS_SPILL_RE.findall(log)]
    smem = [int(m) for m in PTXAS_SMEM_RE.findall(log)]
    return dict(
        num_kernels=len(regs),
        max_registers=max(regs) if regs else None,
        max_smem_bytes=max(smem) if smem else None,
        total_spill_bytes=sum(spills) if spills else 0,
    )


def effective_arch(entry, host_cc):
    """Arch used for the build. If the host can't run the declared arch but the
    source is arch-portable, build for SM80 so we can execute it."""
    if entry.get("sm80_ok") and host_cc < 90:
        return "SM80"
    return entry["arch"]


def do_build(entry, host_cc, env, outdir, timeout):
    name = entry["name"]
    kdir = os.path.join(KERNELS, name)
    arch = effective_arch(entry, host_cc)
    cmd = ["make", "-B", f"ARCH={arch}"]
    for k, v in entry.get("make_vars", {}).items():
        cmd.append(f"{k}={v}")
    t0 = datetime.datetime.now()
    try:
        proc = subprocess.run(cmd, cwd=kdir, env=env, capture_output=True,
                              text=True, timeout=timeout)
        status = "ok" if proc.returncode == 0 else "build_error"
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        status = "build_timeout"
        log = (e.stdout or "") + (e.stderr or "")
    dt = (datetime.datetime.now() - t0).total_seconds()
    logpath = os.path.join(outdir, "build", name.replace("/", "__") + ".log")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    with open(logpath, "w") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n# cwd: {kdir}\n\n" + log)
    rec = dict(schema_version=SCHEMA_VERSION, phase="build", kernel=name,
               arch=arch, declared_arch=entry["arch"], config=entry["config"],
               status=status, build_seconds=round(dt, 1), log=os.path.relpath(logpath, outdir))
    rec.update(parse_ptxas(log))
    if entry.get("notes"):
        rec["notes"] = entry["notes"]
    return rec


def run_stdout_binary(entry, kdir, env, outdir, timeout):
    binary = os.path.join(kdir, entry["out"])
    if not os.path.exists(binary):
        return dict(status="skip", skip_reason="binary not built")
    proc = subprocess.run([binary], cwd=kdir, env=env, capture_output=True,
                          text=True, timeout=timeout)
    log = proc.stdout + proc.stderr
    logpath = os.path.join(outdir, "run", entry["name"].replace("/", "__") + ".log")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    with open(logpath, "w") as f:
        f.write(log)
    if proc.returncode != 0:
        return dict(status="run_error", log=os.path.relpath(logpath, outdir))
    # pair "Problem size" lines with the TFLOP/s that follows each
    measurements = []
    problem = None
    for line in log.splitlines():
        m = PROBLEM_RE.search(line)
        if m:
            problem = m.group(1).strip()
        m = TFLOPS_RE.search(line)
        if m and problem:
            unit = "tops" if m.group(2).upper() == "TOP" else "tflops"
            measurements.append({"shape": problem, unit: float(m.group(1))})
            problem = None
    return dict(status="ok", measurements=measurements,
                log=os.path.relpath(logpath, outdir))


def run_layernorm(entry, kdir, env, outdir, timeout):
    script = os.path.join(REPO_ROOT, "perf", "tk_bench_layernorm.py")
    py = os.path.join(REPO_ROOT, ".venv", "bin", "python")
    proc = subprocess.run([py, script, "--json"], cwd=kdir, env=env,
                          capture_output=True, text=True, timeout=timeout)
    log = proc.stdout + proc.stderr
    logpath = os.path.join(outdir, "run", entry["name"].replace("/", "__") + ".log")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    with open(logpath, "w") as f:
        f.write(log)
    if proc.returncode != 0:
        return dict(status="run_error", log=os.path.relpath(logpath, outdir))
    # last line of stdout is a JSON blob
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return dict(status="ok", measurements=payload, log=os.path.relpath(logpath, outdir))


def skip_reason(entry, host_cc, n_gpus):
    """None if the host can execute this kernel, else the reason it can't."""
    if host_cc < entry["min_cc"]:
        return (f"requires compute capability {entry['min_cc']/10:.1f}+"
                f" (host is {host_cc/10:.1f})")
    if entry.get("multi_gpu") and n_gpus < 8:
        return f"multi-GPU kernel needs 8 GPUs (host has {n_gpus})"
    if entry.get("runner") is None:
        return "no automated runner wired for this kernel"
    return None


def do_run(entry, env, outdir, timeout, gpu=None):
    """Execute one kernel's benchmark. `gpu` pins single-GPU runs to a device."""
    name = entry["name"]
    kdir = os.path.join(KERNELS, name)
    rec = dict(schema_version=SCHEMA_VERSION, phase="run", kernel=name,
               config=entry["config"])
    if gpu is not None:
        env = env.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        rec["gpu"] = gpu
    runner = entry.get("runner")
    if runner == "stdout":
        rec.update(run_stdout_binary(entry, kdir, env, outdir, timeout))
    elif runner == "layernorm":
        rec.update(run_layernorm(entry, kdir, env, outdir, timeout))
    return rec


def env_metadata(host_cc, gpu_name, n_gpus):
    def cmd(c):
        try:
            return subprocess.check_output(c, shell=True, text=True).strip()
        except Exception:
            return None
    py = os.path.join(REPO_ROOT, ".venv", "bin", "python")
    torch_ver = cmd(f"{py} -c 'import torch; print(torch.__version__)'")
    return dict(
        git_commit=cmd(f"git -C {REPO_ROOT} rev-parse --short HEAD"),
        git_dirty=bool(cmd(f"git -C {REPO_ROOT} status --porcelain")),
        gpu=gpu_name, n_gpus=n_gpus, compute_cap=host_cc / 10,
        driver=cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        cuda_toolkit=cmd(f"{CUDA_BIN}/nvcc --version | grep release"),
        torch=torch_ver,
        python=platform.python_version(),
        platform=platform.platform(),
        hostname=platform.node(),
    )


def write_summary(outdir, meta, records):
    lines = ["# ThunderKittens kernel baseline run", ""]
    lines.append(f"- date: {meta['date']}  commit: {meta['env']['git_commit']}"
                 f"{' (dirty)' if meta['env']['git_dirty'] else ''}")
    lines.append(f"- device: {meta['env']['n_gpus']}x {meta['env']['gpu']}"
                 f" (CC {meta['env']['compute_cap']}), driver {meta['env']['driver']}")
    lines.append(f"- toolkit: {meta['env']['cuda_toolkit']}; torch {meta['env']['torch']}")
    lines.append("")
    builds = [r for r in records if r["phase"] == "build"]
    runs = {r["kernel"]: r for r in records if r["phase"] == "run"}
    if builds:
        lines.append("| kernel | arch | build | regs | smem | spills | runtime | result |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for b in sorted(builds, key=lambda r: r["kernel"]):
            r = runs.get(b["kernel"], {})
            rstat = r.get("status", "-")
            if rstat == "skip":
                rstat = f"skip: {r.get('skip_reason', '')}"
            result = ""
            meas = r.get("measurements")
            if isinstance(meas, list) and meas:
                unit = "tflops" if "tflops" in meas[0] else "tops"
                best = max(m[unit] for m in meas)
                result = f"peak {best:.1f} {'TFLOP/s' if unit == 'tflops' else 'TOP/s'}"
            elif isinstance(meas, dict):
                result = "see results.jsonl"
            lines.append(
                f"| {b['kernel']} | {b['arch']} | {b['status']} ({b['build_seconds']}s) "
                f"| {b.get('max_registers') or ''} | {b.get('max_smem_bytes') or ''} "
                f"| {b.get('total_spill_bytes') or 0} | {rstat} | {result} |")
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["build", "run", "all"], default="all")
    ap.add_argument("--kernel", default="all",
                    help="comma-separated kernel names (registry names) or 'all'")
    ap.add_argument("--jobs", type=int, default=4, help="parallel builds")
    ap.add_argument("--build-timeout", type=int, default=2400)
    ap.add_argument("--run-timeout", type=int, default=1800)
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids for parallel single-GPU runs"
                         " (default: all)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    host_cc, gpu_name, n_gpus = host_compute_cap()
    entries = REGISTRY if args.kernel == "all" else [
        e for e in REGISTRY if e["name"] in set(args.kernel.split(","))]
    if not entries:
        sys.exit(f"no kernels matched {args.kernel!r}")

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    run_id = args.run_id or (datetime.datetime.now().strftime("%H%M%S") + "-" +
                             uuid.uuid4().hex[:6])
    outdir = os.path.join(REPO_ROOT, "perf", "results", date, run_id)
    os.makedirs(outdir, exist_ok=True)

    env = build_env()
    if any(e["config"] == "pytorch" for e in entries):
        env = pytorch_make_env(env)

    meta = dict(date=date, run_id=run_id, argv=sys.argv[1:],
                env=env_metadata(host_cc, gpu_name, n_gpus))
    with open(os.path.join(outdir, "run.json"), "w") as f:
        json.dump(meta, f, indent=2)

    records = []
    results_path = os.path.join(outdir, "results.jsonl")

    def emit(rec):
        records.append(rec)
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        status = rec.get("status")
        extra = rec.get("skip_reason", "")
        print(f"[{rec['phase']:5s}] {rec['kernel']:45s} {status} {extra}", flush=True)

    if args.phase in ("build", "all"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(do_build, e, host_cc, env, outdir, args.build_timeout): e
                    for e in entries}
            for fut in concurrent.futures.as_completed(futs):
                emit(fut.result())

    if args.phase in ("run", "all"):
        single, multi = [], []
        for e in entries:
            reason = skip_reason(e, host_cc, n_gpus)
            if reason:
                emit(dict(schema_version=SCHEMA_VERSION, phase="run",
                          kernel=e["name"], config=e["config"], status="skip",
                          skip_reason=reason))
            elif e.get("multi_gpu"):
                multi.append(e)
            else:
                single.append(e)

        # single-GPU benchmarks: one per device, in parallel
        gpu_ids = args.gpus.split(",") if args.gpus else [str(i) for i in range(n_gpus)]
        gpu_slots = queue.Queue()
        for g in gpu_ids:
            gpu_slots.put(g)

        def run_on_free_gpu(e):
            g = gpu_slots.get()
            try:
                return do_run(e, env, outdir, args.run_timeout, gpu=g)
            finally:
                gpu_slots.put(g)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
            for fut in concurrent.futures.as_completed(
                    [ex.submit(run_on_free_gpu, e) for e in single]):
                emit(fut.result())

        # multi-GPU kernels own the whole box, serially
        for e in multi:
            emit(do_run(e, env, outdir, args.run_timeout))

    write_summary(outdir, meta, records)
    print(f"\nresults: {outdir}")


if __name__ == "__main__":
    main()
