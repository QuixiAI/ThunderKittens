#!/usr/bin/env bash
# Direct nvcc build of the tk_cuda _C extension (torch cu130 vs system CUDA 12.9
# mismatch makes cpp_extension refuse to build; build the .so by hand).
set -e
cd "$(dirname "$0")"
TORCH=$(../../.venv/bin/python -c "import torch,os;print(os.path.dirname(torch.__file__))")
PYINC=$(../../.venv/bin/python -c "import sysconfig;print(sysconfig.get_path('include'))")
/usr/local/cuda/bin/nvcc \
  tm_cuda_ext.cu tm_cuda_serving.cu tm_cuda_elementwise.cu tm_cuda_w6.cu \
  tm_cuda_moe_quant.cu tm_cuda_m4.cu tm_cuda_m5.cu tm_cuda_m6.cu tm_cuda_mf_followups.cu \
  -shared -Xcompiler -fPIC -std=c++20 -O2 \
  -DTORCH_EXTENSION_NAME=_C -DTORCH_API_INCLUDE_EXTENSION_H \
  -I../quant -I../serving -I../elementwise -I../moe -I../lin_attn_tm \
  -I../moe_quant -I../mamba2 \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PYINC" \
  -gencode arch=compute_86,code=sm_86 \
  -L"$TORCH/lib" -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -ltorch_python \
  -o tk_cuda/_C.cpython-312-x86_64-linux-gnu.so
echo "BUILDEXIT done"
