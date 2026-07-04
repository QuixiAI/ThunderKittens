/**
 * @file
 * @brief CUDA (SM80+) port of ThunderMittens' quant format layer.
 *
 * The premise: a quant format is a contract about bits at rest, not about compute.
 * Each format is a struct exposing `block_k` (weights per block), `block_bytes`, and
 * `dequant(const uint8_t* base, int col) -> float` for the weight at column `col` of
 * the block starting at `base`. Packed tensors are produced by
 * ~/ThunderMittens/ThunderMittens/kernels/tk/quant.py (numpy; byte-identical across
 * Metal and CUDA - one checkpoint serves both). Layouts mirror llama.cpp GGUF
 * (ggml-common.h), Marlin (dequant.h), and the OCP MX / NVFP4 specs.
 *
 * Decoders compute in fp32 (fp16 scale widened once; code values are exact small
 * floats), so dequant here matches quant.py's float32 `dequantize_*` bit-exactly for
 * every format whose reference multiplies fp32(scale) * fp32(code) - which is all of
 * them in this tranche.
 *
 * Tranche 1 (bit arithmetic): q8_0 q4_0 q4_1 q5_0 q5_1 kU4B8 kU4 hqq fp8_e4m3 e5m2
 * fp8_block fp8_raw fp4_e2m1 mxfp8 mxfp4 nvfp4 mxfp6_e3m2 mxfp6_e2m3 bitnet.
 * Tranche 2 (tables/codebooks; separate header): q2_K..q6_K, iq4_nl/xs, iq2_xxs/xs,
 * iq3_xxs, iq1_s.
 */
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace tmq {

// ---- float-code decoders (bit tricks; fp32 out) -------------------------------------------------
// Normal codes shift exponent/mantissa into the fp16 field positions (an exact half bit pattern),
// then one power-of-two multiply fixes the bias gap. Subnormals take an explicit small-integer
// path. Same recipes as Marlin dequant.h / ThunderMittens tk_*_decode.

// fp8 e4m3 (1-4-3, bias 7): normal = half_bits(code<<7) * 2^(15-7); subnormal = m * 2^-9.
__device__ __forceinline__ float e4m3_decode(uint8_t v) {
    const uint16_t h = uint16_t(v & 0x7F) << 7;
    const float mag = (v & 0x78) ? __half2float(__ushort_as_half(h)) * 256.0f
                                 : float(v & 0x7) * 0.001953125f;
    return (v & 0x80) ? -mag : mag;
}
// fp4 e2m1 (1-2-1, bias 1): normal = half_bits(code<<9) * 2^(15-1); subnormals are 0 / 0.5.
__device__ __forceinline__ float e2m1_decode(unsigned nib) {
    const uint16_t h = uint16_t(nib & 0x7) << 9;
    const float mag = (nib & 0x6) ? __half2float(__ushort_as_half(h)) * 16384.0f
                                  : ((nib & 1) ? 0.5f : 0.0f);
    return (nib & 0x8) ? -mag : mag;
}
// fp8 e5m2 (1-5-2, bias 15): e5m2 IS truncated fp16 - a pure bitcast.
__device__ __forceinline__ float e5m2_decode(uint8_t v) {
    return __half2float(__ushort_as_half(uint16_t(uint16_t(v) << 8)));
}
// fp6 e3m2 (1-3-2, bias 3): normal = half_bits(code<<8) * 2^(15-3); subnormal = m * 2^-4.
__device__ __forceinline__ float e3m2_decode(unsigned c) {
    const uint16_t h = uint16_t(c & 0x1F) << 8;
    const float mag = (c & 0x1C) ? __half2float(__ushort_as_half(h)) * 4096.0f
                                 : float(c & 0x3) * 0.0625f;
    return (c & 0x20) ? -mag : mag;
}
// fp6 e2m3 (1-2-3, bias 1): normal = half_bits(code<<7) * 2^(15-1); subnormal = m * 0.125.
__device__ __forceinline__ float e2m3_decode(unsigned c) {
    const uint16_t h = uint16_t(c & 0x1F) << 7;
    const float mag = (c & 0x18) ? __half2float(__ushort_as_half(h)) * 16384.0f
                                 : float(c & 0x7) * 0.125f;
    return (c & 0x20) ? -mag : mag;
}
// e8m0 (MX power-of-two block scale): the byte IS a float exponent field. Two ALU ops, exact
// (e=0 -> 2^-127 lands subnormal ~= 0; encoders never emit it for nonzero blocks).
__device__ __forceinline__ float e8m0_decode(uint8_t e) {
    return __uint_as_float(uint32_t(e) << 23);
}
__device__ __forceinline__ float half_at(const uint8_t* p, int idx = 0) {
    return __half2float(reinterpret_cast<const __half*>(p)[idx]);
}

// ---- integer dot primitive: on Ampere this is NATIVE dp4a (Metal needed a software loop) --------
__device__ __forceinline__ int idot4(unsigned a, unsigned b, int acc = 0) {
    return __dp4a(int(a), int(b), acc);   // signed int8 x4 dot, matching TM's idot4 semantics
}

// ================================ formats ========================================================

// ---- q8_0 : { half d; int8 qs[32]; } - 34 bytes, 32/block. value = d * q ----
struct q8_0 {
    static constexpr int block_k = 32, block_bytes = 34;
    __device__ static float dequant(const uint8_t* base, int col) {
        return half_at(base) * float(reinterpret_cast<const int8_t*>(base + 2)[col]);
    }
    // integer path (W8A8): raw code + per-group scale, kept separate.
    __device__ static int   code(const uint8_t* base, int col) { return reinterpret_cast<const int8_t*>(base + 2)[col]; }
    __device__ static float gscale(const uint8_t* base)        { return half_at(base); }
};

// ---- q4_0 : { half d; uint8 qs[16]; } - 18 bytes, 32/block. value = d * (nibble - 8).
//   ggml nibble packing: col<16 -> qs[col]&0xF ; col>=16 -> qs[col-16]>>4. ----
struct q4_0 {
    static constexpr int block_k = 32, block_bytes = 18;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 2;
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return half_at(base) * float(nib - 8);
    }
};

// ---- q4_1 : { half d; half m; uint8 qs[16]; } - 20 bytes, 32/block. value = d*nibble + m. ----
struct q4_1 {
    static constexpr int block_k = 32, block_bytes = 20;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 4;
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return half_at(base, 0) * float(nib) + half_at(base, 1);
    }
};

// ---- q5_0 : { half d; uint8 qh[4]; uint8 qs[16]; } - 22 bytes. value = d*(q-16),
//   q = nibble | (5th bit = bit `col` of the qh uint32). ----
struct q5_0 {
    static constexpr int block_k = 32, block_bytes = 22;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint32_t qh = uint32_t(base[2]) | (uint32_t(base[3]) << 8) |
                            (uint32_t(base[4]) << 16) | (uint32_t(base[5]) << 24);
        const uint8_t* qs = base + 6;
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        const int q = nib | (((qh >> col) & 1) << 4);
        return half_at(base) * float(q - 16);
    }
};

// ---- q5_1 : { half d; half m; uint8 qh[4]; uint8 qs[16]; } - 24 bytes. value = d*q + m. ----
struct q5_1 {
    static constexpr int block_k = 32, block_bytes = 24;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint32_t qh = uint32_t(base[4]) | (uint32_t(base[5]) << 8) |
                            (uint32_t(base[6]) << 16) | (uint32_t(base[7]) << 24);
        const uint8_t* qs = base + 8;
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        const int q = nib | (((qh >> col) & 1) << 4);
        return half_at(base, 0) * float(q) + half_at(base, 1);
    }
};

// ---- kU4B8 : GPTQ/Marlin grouped int4, group=128. { half scale; uint8 qs[64]; } - 66 bytes.
//   value = scale * (nibble - 8). ----
struct kU4B8 {
    static constexpr int block_k = 128, block_bytes = 66;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 2;
        const int nib = (col < 64) ? (qs[col] & 0x0F) : (qs[col - 64] >> 4);
        return half_at(base) * float(nib - 8);
    }
};

// ---- kU4 : AWQ grouped int4, group=128, per-group zero-point. { half scale; half zp;
//   uint8 qs[64]; } - 68 bytes. value = scale * (nibble - zp). ----
struct kU4 {
    static constexpr int block_k = 128, block_bytes = 68;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 4;
        const int nib = (col < 64) ? (qs[col] & 0x0F) : (qs[col - 64] >> 4);
        return half_at(base, 0) * (float(nib) - half_at(base, 1));
    }
};

// ---- hqq : HQQ int4 + per-group zero-point, group 64. { half scale; half zp; uint8 qs[32]; }
//   - 36 bytes. value = scale * (nibble - zp). ----
struct hqq {
    static constexpr int block_k = 64, block_bytes = 36;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 4;
        const int nib = (col < 32) ? (qs[col] & 0x0F) : (qs[col - 32] >> 4);
        return half_at(base, 0) * (float(nib) - half_at(base, 1));
    }
};

// ---- fp8_e4m3 : per-group (32) half-scaled fp8. { half scale; uint8 qs[32]; } - 34 bytes. ----
struct fp8_e4m3 {
    static constexpr int block_k = 32, block_bytes = 34;
    __device__ static float dequant(const uint8_t* base, int col) {
        return half_at(base) * e4m3_decode(base[2 + col]);
    }
};

// ---- e5m2 : per-group (32) half-scaled fp8 e5m2. { half scale; uint8 qs[32]; } - 34 bytes. ----
struct e5m2 {
    static constexpr int block_k = 32, block_bytes = 34;
    __device__ static float dequant(const uint8_t* base, int col) {
        return half_at(base) * e5m2_decode(base[2 + col]);
    }
};

// ---- fp8_block : 128x128 block-scaled fp8 e4m3 (compressed-tensors), tile scale replicated
//   into each row's k-block. { half scale; uint8 qs[128]; } - 130 bytes. ----
struct fp8_block {
    static constexpr int block_k = 128, block_bytes = 130;
    __device__ static float dequant(const uint8_t* base, int col) {
        return half_at(base) * e4m3_decode(base[2 + col]);
    }
};

// ---- fp8_raw : codes-only fp8 e4m3 (scale lives in a separate buffer), 128/block. ----
struct fp8_raw {
    static constexpr int block_k = 128, block_bytes = 128;
    __device__ static float dequant(const uint8_t* base, int col) {
        return e4m3_decode(base[col]);
    }
};

// ---- fp4_e2m1 : per-group (32) half-scaled fp4 (nibbles, q4_0-style packing). 18 bytes. ----
struct fp4_e2m1 {
    static constexpr int block_k = 32, block_bytes = 18;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 2;
        const unsigned nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return half_at(base) * e2m1_decode(nib);
    }
};

// ---- mxfp8 : OCP MX - 32-block, e8m0 power-of-two scale + fp8 e4m3. { uint8 e8m0;
//   uint8 qs[32]; } - 33 bytes. value = 2^(e8m0-127) * e4m3(q). ----
struct mxfp8 {
    static constexpr int block_k = 32, block_bytes = 33;
    __device__ static float dequant(const uint8_t* base, int col) {
        return e8m0_decode(base[0]) * e4m3_decode(base[1 + col]);
    }
};

// ---- nvfp4 : 16-block, fp8 e4m3 block scale + fp4 e2m1 codes. { uint8 e4m3_scale;
//   uint8 qs[8]; } - 9 bytes. value = e4m3(scale) * e2m1(nib). ----
struct nvfp4 {
    static constexpr int block_k = 16, block_bytes = 9;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 1;
        const unsigned nib = (col < 8) ? (qs[col] & 0x0F) : (qs[col - 8] >> 4);
        return e4m3_decode(base[0]) * e2m1_decode(nib);
    }
};

// ---- mxfp4 : OCP MX - 32-block, e8m0 scale + fp4 e2m1 codes. { uint8 e8m0; uint8 qs[16]; }
//   - 17 bytes. value = 2^(e8m0-127) * e2m1(nib). ----
struct mxfp4 {
    static constexpr int block_k = 32, block_bytes = 17;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 1;
        const unsigned nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return e8m0_decode(base[0]) * e2m1_decode(nib);
    }
};

// ---- mxfp6 (e3m2 / e2m3) : OCP MX 6-bit. { uint8 e8m0; uint8 codes[24]; } - 25 bytes, 32/block.
//   4 six-bit codes per 3 bytes (little-endian 24-bit groups). ----
template<bool E3M2>
struct mxfp6 {
    static constexpr int block_k = 32, block_bytes = 25;
    __device__ static float dequant(const uint8_t* base, int col) {
        const int g = col >> 2, within = col & 3;
        const uint8_t* p = base + 1 + 3 * g;
        const unsigned val = unsigned(p[0]) | (unsigned(p[1]) << 8) | (unsigned(p[2]) << 16);
        const unsigned c = (val >> (6 * within)) & 0x3F;
        return e8m0_decode(base[0]) * (E3M2 ? e3m2_decode(c) : e2m3_decode(c));
    }
};
using mxfp6_e3m2 = mxfp6<true>;
using mxfp6_e2m3 = mxfp6<false>;

// ---- bitnet : BitNet b1.58 ternary, group 32, per-group absmean scale. 2-bit codes 4/byte
//   (code {0,1,2} -> value = scale*(code-1)). { half scale; uint8 qs[8]; } - 10 bytes.
//   (Unlike Metal, Ampere ALSO has the true integer path: dp4a on the codes; see code()/gscale().) ----
struct bitnet {
    static constexpr int block_k = 32, block_bytes = 10;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 2;
        const unsigned code = (qs[col >> 2] >> ((col & 3) * 2)) & 0x3;
        return half_at(base) * float(int(code) - 1);
    }
    __device__ static int code(const uint8_t* base, int col) {
        const uint8_t* qs = base + 2;
        return int((qs[col >> 2] >> ((col & 3) * 2)) & 0x3) - 1;
    }
    __device__ static float gscale(const uint8_t* base) { return half_at(base); }
};

// ---- span helper: dequant 8 contiguous columns (the qgemv per-lane unit) ----
template<typename FMT>
__device__ __forceinline__ void dequant8(const uint8_t* base, int col0, float w[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) w[i] = FMT::dequant(base, col0 + i);
}

}  // namespace tmq
