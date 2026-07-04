"""tk_cuda: ThunderMittens kernels on CUDA/SM86 (RTX 3090+).

API mirrors ThunderMittens' `tk` package where ported. Weights quantize with
TM's quant.py (packed tensors are byte-identical across Metal and CUDA):

    import sys, os; sys.path.insert(0, os.path.expanduser("~/ThunderMittens/ThunderMittens/kernels"))
    from tk.quant import QUANT_FORMATS
    packed = QUANT_FORMATS["nvfp4"][0](W)          # numpy (N, K/bk, bytes) uint8
    Wq = torch.from_numpy(packed).reshape(N, -1).cuda()
    y = tk_cuda.qgemm(x_fp16, Wq, "nvfp4")
"""
from . import _C

QUANT_FORMATS = [
    "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "kU4B8", "kU4", "hqq",
    "fp8_e4m3", "e5m2", "fp8_block", "fp4_e2m1", "mxfp8", "mxfp4", "nvfp4",
    "mxfp6_e3m2", "mxfp6_e2m3", "bitnet",
    "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
    "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s",
]


def _flat(Wq):
    return Wq.reshape(Wq.shape[0], -1).contiguous() if Wq.dim() == 3 else Wq


def qgemv(Wq, x, format):
    """D(N) = dequant(Wq) @ x. Wq (N, K/bk, bytes) or (N, bytes*) uint8; x (K,) fp16."""
    return _C.qgemv(_flat(Wq), x.contiguous(), format)


def qgemm(X, Wq, format):
    """Y(M,N) fp32 = X(M,K) fp16 @ dequant(Wq(N,K))^T."""
    return _C.qgemm(X.contiguous(), _flat(Wq), format)


def qflux_gelu(X, Wq, bias, format):
    """gelu_tanh(X @ dequant(Wq)^T + bias). bias (N,) fp32."""
    return _C.qflux_gelu(X.contiguous(), _flat(Wq), bias.contiguous(), format)


def qgemv_w8a8(Wq, Xq, w_scale, a_scale):
    """int8 W (N,K) x int8 X (K,) -> fp16 (N,). Scales fp16: per-channel (N,), per-token (1,)."""
    return _C.qgemv_w8a8(Wq, Xq, w_scale, a_scale)


def qgemv_w2a8(Wq, Xq, a_scale, K):
    """BitNet ternary W (N, K/32, 10) x int8 X (K,) -> fp16 (N,)."""
    return _C.qgemv_w2a8(_flat(Wq), Xq, a_scale, K)


def quantize_per_token(A, kind="fp8"):
    """(codes u8 (T,D), scale f32 (T,)): symmetric per-row absmax/QMAX quantization."""
    return _C.quantize_per_token(A.contiguous(), kind)


def quantize_per_tensor(A, kind="fp8"):
    return _C.quantize_per_tensor(A.contiguous(), kind)


def lm_head_sample(h, W, format="fp16", temperature=1.0, seed=0, mode="argmax"):
    """Fused LM head + sampling without materializing (T,V) logits.
    mode: 'argmax' | 'categorical' (Gumbel-max, reproducible from seed)."""
    return _C.lm_head_sample(h.contiguous(), _flat(W), format, temperature, seed,
                             mode == "categorical")
