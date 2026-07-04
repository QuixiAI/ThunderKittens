# NOTE: on this box torch (cu130) mismatches the system CUDA toolkit (12.9) and
# torch's cpp_extension refuses to build. Workaround: build the .so directly -
#   TORCH=.venv/lib/python3.12/site-packages/torch
#   /usr/local/cuda/bin/nvcc tm_cuda_ext.cu -shared -Xcompiler -fPIC -std=c++20 -O2 \
#     -DTORCH_EXTENSION_NAME=_C -DTORCH_API_INCLUDE_EXTENSION_H -I../quant \
#     -I$TORCH/include -I$TORCH/include/torch/csrc/api/include -I<python-include> \
#     -gencode arch=compute_86,code=sm_86 -L$TORCH/lib -ltorch -ltorch_cuda \
#     -ltorch_cpu -lc10 -lc10_cuda -ltorch_python \
#     -o tk_cuda/_C.cpython-312-x86_64-linux-gnu.so
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

QUANT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quant")

setup(
    name="tk_cuda",
    version="0.1",
    packages=["tk_cuda"],
    ext_modules=[
        CUDAExtension(
            name="tk_cuda._C",
            sources=["tm_cuda_ext.cu"],
            include_dirs=[QUANT_DIR],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2", "-std=c++20",
                         "-gencode", "arch=compute_86,code=sm_86"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
