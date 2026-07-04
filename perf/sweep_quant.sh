#!/usr/bin/env bash
# Canonical quant-kernel perf sweep (Step 5). Run on an IDLE GPU (external
# jobs skew every number). Emits a markdown table of qgemv GB/s + qgemm
# TFLOP/s per format from the golden harnesses.
#   usage: CUDA_VISIBLE_DEVICES=<idle-gpu> ./sweep_quant.sh [out.md]
set -u
cd "$(dirname "$0")/../kernels/quant"
OUT="${1:-/tmp/quant_sweep.md}"

echo "| format | qgemv GB/s | qgemm TFLOP/s |" > "$OUT"
echo "|---|---|---|" >> "$OUT"
for f in $(ls golden/); do
    g=$(./qgemv.out "golden/$f" 2>/dev/null | grep gemv | grep -o '[0-9.]* GB/s' | head -1)
    m=$(./qgemm.out "golden/$f" 2>/dev/null | grep qgemm | grep -o '[0-9.]* TFLOP/s' | head -1)
    echo "| $f | ${g:-n/a} | ${m:-n/a} |" >> "$OUT"
    echo "$f: gemv ${g:-n/a}, gemm ${m:-n/a}"
done
echo "wrote $OUT"
