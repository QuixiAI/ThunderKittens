/**
 * @file
 * @brief Tranche 2 of the CUDA quant format layer: GGUF k-quants and i-quants
 * (table/codebook formats). Same contract as quant_formats.cuh:
 * `{block_k, block_bytes, dequant(base, col) -> float}`, decode in fp32,
 * bit-exact vs quant.py's dequantize_*. Layouts mirror ggml-common.h;
 * decoders mirror ThunderMittens dequant.metal (which mirrors ggml-metal).
 *
 * Formats: q2_K q3_K q4_K q5_K q6_K (256-superblock k-quants with packed
 * sub-scales) and iq4_nl iq4_xs iq2_xxs iq2_xs iq3_xxs iq1_s (non-linear
 * codebook / E8-lattice grid formats). Tables in quant_tables.cuh
 * (__constant__; indices are data-dependent so hot kernels may want smem).
 */
#pragma once
#include "quant_formats.cuh"
#include "quant_tables.cuh"

namespace tmq {

// ---- q2_K : { u8 scales[16]; u8 qs[64]; half d; half dmin; } - 84 bytes, 256/block.
//   16 sub-blocks of 16; scales byte = 4-bit dl-scale | 4-bit min. value = d*sc*q - dmin*m. ----
struct q2_K {
    static constexpr int block_k = 256, block_bytes = 84;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* scales = base;
        const uint8_t* qs = base + 16;
        const float d = half_at(base + 80), dmin = half_at(base + 82);
        const int chunk = col >> 7, pos = col & 127, sidx = pos >> 5, sub = (pos >> 4) & 1, l = pos & 15;
        const int is = chunk * 8 + sidx * 2 + sub;
        const int q = (qs[chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3;
        return d * float(scales[is] & 0xF) * float(q) - dmin * float(scales[is] >> 4);
    }
};

// ---- q3_K : { u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d; } - 110 bytes, 256/block.
//   low 2 bits in qs, high bit in hmask; 16 6-bit signed scales 4-way packed. value = d*(sc-32)*q3. ----
struct q3_K {
    static constexpr int block_k = 256, block_bytes = 110;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* hmask = base;
        const uint8_t* qs = base + 32;
        const uint8_t* sca = base + 96;
        const float d = half_at(base + 108);
        const int chunk = col >> 7, pos = col & 127, sidx = pos >> 5, sub = (pos >> 4) & 1, l = pos & 15;
        const int is = chunk * 8 + sidx * 2 + sub;
        const int low2 = (qs[chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3;
        const int hb = (hmask[sub * 16 + l] & (1 << (chunk * 4 + sidx))) ? 1 : 0;
        const int q3v = (low2 | (hb << 2)) - 4;
        const int w = is >> 2, b = is & 3;
        int s;
        if      (w == 0) s = (sca[b] & 0xF)            | ((sca[8 + b] & 3) << 4);
        else if (w == 1) s = (sca[4 + b] & 0xF)        | (((sca[8 + b] >> 2) & 3) << 4);
        else if (w == 2) s = ((sca[b] >> 4) & 0xF)     | (((sca[8 + b] >> 4) & 3) << 4);
        else             s = ((sca[4 + b] >> 4) & 0xF) | (((sca[8 + b] >> 6) & 3) << 4);
        return d * float(s - 32) * float(q3v);
    }
};

// ---- q4_K : { half d; half dmin; u8 scales[12]; u8 qs[128]; } - 144 bytes, 256/block.
//   8 sub-blocks of 32; 6-bit scale+min via get_scale_min_k4. value = (d*sc)*nib - (dmin*m). ----
struct q4_K {
    static constexpr int block_k = 256, block_bytes = 144;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base, 0), dmin = half_at(base, 1);
        const uint8_t* scales = base + 4;
        const uint8_t* qs     = base + 16;
        const int chunk = col / 64, pos = col % 64;
        int sub, nib;
        if (pos < 32) { sub = chunk * 2;     nib = qs[chunk * 32 + pos]        & 0x0F; }
        else          { sub = chunk * 2 + 1; nib = qs[chunk * 32 + (pos - 32)] >> 4;   }
        int sc, m;
        if (sub < 4) { sc = scales[sub] & 63; m = scales[sub + 4] & 63; }
        else {
            sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
            m  = (scales[sub + 4] >> 4)   | ((scales[sub]     >> 6) << 4);
        }
        return d * float(sc) * float(nib) - dmin * float(m);
    }
};

// ---- q5_K : { half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; } - 176 bytes.
//   8 sub-blocks of 32; scale/min as q4_K; 5-bit q = nibble | (qh bit)<<4. ----
struct q5_K {
    static constexpr int block_k = 256, block_bytes = 176;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base, 0), dmin = half_at(base, 1);
        const uint8_t* sca = base + 4;
        const uint8_t* qh  = base + 16;
        const uint8_t* qs  = base + 48;
        const int chunk = col >> 6, pos = col & 63, sub = pos >> 5, l = pos & 31;
        const int is = 2 * chunk + sub;
        const int nib = sub ? (qs[chunk * 32 + l] >> 4) : (qs[chunk * 32 + l] & 0xF);
        const int hb = (qh[l] & (1 << (2 * chunk + sub))) ? 1 : 0;
        const int q = nib + hb * 16;
        int sc, mn;
        if (is < 4) { sc = sca[is] & 63; mn = sca[is + 4] & 63; }
        else {
            sc = (sca[is + 4] & 0x0F) | ((sca[is - 4] >> 6) << 4);
            mn = (sca[is + 4] >> 4)   | ((sca[is]     >> 6) << 4);
        }
        return d * float(sc) * float(q) - dmin * float(mn);
    }
};

// ---- q6_K : { u8 ql[128]; u8 qh[64]; int8 scales[16]; half d; } - 210 bytes, 256/block.
//   16 sub-blocks of 16; 6-bit q = (4 low in ql | 2 high in qh) - 32; int8 scales. ----
struct q6_K {
    static constexpr int block_k = 256, block_bytes = 210;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* ql = base;
        const uint8_t* qh = base + 128;
        const int8_t* sca = reinterpret_cast<const int8_t*>(base + 192);
        const float d = half_at(base + 208);
        const int chunk = col >> 7, pos = col & 127, group = pos >> 5, l = pos & 31;
        const int ql_byte = ql[chunk * 64 + l + 32 * (group & 1)];
        const int nib = (group & 2) ? (ql_byte >> 4) : (ql_byte & 0xF);
        const int hbits = (qh[chunk * 32 + l] >> (2 * group)) & 3;
        const int q = (nib | (hbits << 4)) - 32;
        const int sc_idx = chunk * 8 + (l >> 4) + group * 2;
        return d * float(int(sca[sc_idx])) * float(q);
    }
};

// ---- iq4_nl : { half d; u8 qs[16]; } - 18 bytes, 32/block. value = d * kvalues_iq4nl[nib]
//   (q4_0-style nibble layout). ----
struct iq4_nl {
    static constexpr int block_k = 32, block_bytes = 18;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 2;
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return half_at(base) * float(kvalues_iq4nl[nib]);
    }
};

// ---- iq4_xs : 256-superblock IQ4_NL. { half d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }
//   - 136 bytes. 8 sub-blocks of 32; 6-bit sub-scale ls - 32. value = d*ls * codebook[nib]. ----
struct iq4_xs {
    static constexpr int block_k = 256, block_bytes = 136;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base);
        const uint16_t scales_h = *reinterpret_cast<const uint16_t*>(base + 2);
        const uint8_t* scales_l = base + 4;
        const uint8_t* qs = base + 8;
        const int ib = col >> 5, local = col & 31;
        const int sl = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
        const int sh = (scales_h >> (2 * ib)) & 0x3;
        const int ls = (sl | (sh << 4)) - 32;
        const float dl = d * float(ls);
        const int nib = (local < 16) ? (qs[16 * ib + local] & 0x0F)
                                     : (qs[16 * ib + (local - 16)] >> 4);
        return dl * float(kvalues_iq4nl[nib]);
    }
};

// ---- iq2_xxs : E8-lattice 2.0625 bpw. { half d; u16 qs[32]; } - 66 bytes, 256/block.
//   Per 32: 4 u16 grid indices + 4 u16 signs/scale; grid entry = 8 packed u8 magnitudes. ----
struct iq2_xxs {
    static constexpr int block_k = 256, block_bytes = 66;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base);
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(base + 2);
        const int ib32 = col >> 5, p = col & 31, sub = p >> 3, elem = p & 7;
        const uint16_t* q2 = qs + 4 * ib32;
        const uint32_t aux_g = uint32_t(q2[0]) | (uint32_t(q2[1]) << 16);
        const uint32_t aux_s = uint32_t(q2[2]) | (uint32_t(q2[3]) << 16);
        const uint32_t g = (aux_g >> (8 * sub)) & 0xff;
        const uint32_t gv = uint32_t((iq2xxs_grid[g] >> (8 * elem)) & 0xffULL);
        const uint8_t signs = ksigns_iq2xs[(aux_s >> (7 * sub)) & 127];
        const float dl = d * (0.5f + float((aux_s >> 28) & 0xf)) * 0.25f;
        const float sgn = (signs & kmask_iq2xs[elem]) ? -1.0f : 1.0f;
        return dl * float(gv) * sgn;
    }
};

// ---- iq2_xs : E8-lattice 2.3125 bpw. { half d; u16 qs[32]; u8 scales[8]; } - 74 bytes.
//   u16 = 9-bit grid index | 7-bit signs; 4-bit per-half scale. ----
struct iq2_xs {
    static constexpr int block_k = 256, block_bytes = 74;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base);
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(base + 2);
        const uint8_t* scales = base + 66;
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, sub2 = (p & 15) >> 3, elem = p & 7;
        const uint16_t idx16 = qs[4 * ib32 + 2 * il + sub2];
        const uint32_t g = idx16 & 511;
        const uint8_t signs = ksigns_iq2xs[idx16 >> 9];
        const int sc = (scales[ib32] >> (4 * il)) & 0xF;
        const float dl = d * (0.5f + float(sc)) * 0.25f;
        const uint32_t gv = uint32_t((iq2xs_grid[g] >> (8 * elem)) & 0xffULL);
        const float sgn = (signs & kmask_iq2xs[elem]) ? -1.0f : 1.0f;
        return dl * float(gv) * sgn;
    }
};

// ---- iq3_xxs : E8-lattice 3.0625 bpw. { half d; u8 qs[96]; } - 98 bytes. First 64 bytes =
//   8-bit grid indices (8 per 32); next 32 = u16 signs/scale. Grid entry = u32 of 4 magnitudes. ----
struct iq3_xxs {
    static constexpr int block_k = 256, block_bytes = 98;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base);
        const uint8_t* qs = base + 2;
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, w = p & 15, r = w >> 2, i = w & 3;
        const uint8_t* q3 = qs + 8 * ib32;
        const uint16_t* gas = reinterpret_cast<const uint16_t*>(qs + 64) + 2 * ib32;
        const uint32_t aux32 = uint32_t(gas[0]) | (uint32_t(gas[1]) << 16);
        const uint32_t gv = (iq3xxs_grid[q3[4 * il + r]] >> (8 * i)) & 0xff;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> (14 * il + 7 * (r >> 1))) & 127];
        const float dl = d * (0.5f + float(aux32 >> 28)) * 0.5f;
        const float sgn = (signs & kmask_iq2xs[i + 4 * (r & 1)]) ? -1.0f : 1.0f;
        return dl * float(gv) * sgn;
    }
};

// ---- iq1_s : 1.5625 bpw. { half d; u8 qs[32]; u16 qh[8]; } - 50 bytes. Grid index = qs byte |
//   high bits from qh; 3-bit scale + sign bit give dl and the ml offset (value = dl*nib + ml). ----
struct iq1_s {
    static constexpr int block_k = 256, block_bytes = 50;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base);
        const uint8_t* qs = base + 2;
        const uint16_t* qh = reinterpret_cast<const uint16_t*>(base + 34);
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, w = p & 15;
        const int which = w >> 2, i = w & 3;
        const uint8_t* qsp = qs + 4 * ib32 + 2 * il;
        const uint16_t qhv = qh[ib32];
        const float dl = d * float(2 * ((qhv >> 12) & 7) + 1);
        const float ml = dl * ((qhv & 0x8000) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA));
        const uint32_t h = uint32_t(qhv >> (6 * il));
        const uint32_t gi = (which >> 1) == 0 ? (qsp[0] | ((h << 8) & 0x700))
                                              : (qsp[1] | ((h << 5) & 0x700));
        const uint32_t b = (iq1s_grid_gpu[gi] >> (8 * i)) & 0xff;
        const uint32_t nib = (which & 1) ? (b >> 4) : (b & 0xF);
        return dl * float(nib) + ml;
    }
};

}  // namespace tmq
