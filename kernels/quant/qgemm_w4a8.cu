// W4A8 quantized GEMM: Q4_0 weight x int8 activation on the Ampere int8 tensor
// cores (IMMA mma.m16n8k32.s8), CUDA/SM86. Shape-named specialized variant of
// the quant-family GEMM: it keeps native Q4_0 nibbles + per-32-block fp16 scales
// and quantizes the activation to int8 with a per-32-block scale, so the K=32
// MMA tile is exactly one Q4/Q8 block. cp.async double/triple-buffers the SoA
// nibble+scale tiles; per K-block the int32 IMMA partial is dequantized by
// wscale*xscale into an fp32 accumulator. The -8 Q4 zero point is folded into
// the signed int8 weight (no separate sum term).
//
// This complements the weight-only `qgemm<q4_0>` (fp16 activation, m16n8k16):
// at compute-bound M the int8 tensor cores (~284 TOP/s) beat the fp16 path
// (~71 TFLOP/s) once the activation is also quantized. Ported from
// embeddinggemma.c src/engine_cuda.cu (EI_CUDA_W4A8_GEMM), named by shape.
//
//   wq: uint8 [N][bpr][16] nibbles     ws: half [N][bpr]
//   aq: int8  [M][bpr][32]             as: half [M][bpr]     bpr = K/32
//   Y(M,N) = X(M,K) @ dequant(Wq(N,K))^T, fp32 accumulate.
//
// Oracle: fp64 recomputation of the exact integer block sums scaled by the fp16
// (as*ws) products -- the same values the kernel multiplies, so the tolerance
// covers only fp32-vs-fp64 accumulation (mirrors qgemv_int's "scales are the
// only float ops" contract). Baseline: the repo's documented Q4_0 prefill path,
// dequant-to-fp16 + cuBLAS fp16 GEMM, same shape.
//
// Build:
//   /usr/local/cuda/bin/nvcc qgemm_w4a8.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -lcublas -o qgemm_w4a8.out
// Run (reuses the checked-in q4_0 weight golden):
//   CUDA_VISIBLE_DEVICES=1 ./qgemm_w4a8.out golden/q4_0 [M]
#include "quant_formats.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <random>

using namespace tmq;

// =============================== W4A8 kernel ===============================
// (device code ported verbatim from embeddinggemma.c src/engine_cuda.cu)
namespace w4a8 {

__device__ __forceinline__ void imma_m16n8k32(int acc[4], const uint32_t a[4],
                                              const uint32_t b[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void)acc; (void)a; (void)b;
#endif
}

// Expand 4 packed Q4_0 nibbles (low or high) to signed int8, -8 folded in.
__device__ __forceinline__ uint32_t pack_q4(const uint8_t *qs, int c, bool high) {
    uint32_t p = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int v = (high ? (qs[c + i] >> 4) : (qs[c + i] & 0x0f)) - 8;
        p |= static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(v)))
             << (i * 8);
    }
    return p;
}

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
#if __CUDA_ARCH__ >= 800
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
#endif
}
__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}
template <int N> __device__ __forceinline__ void cp_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES>
__global__ __launch_bounds__(WARPS_M *WARPS_N * 32)
void gemm_kernel(const uint8_t *__restrict__ wq, const __half *__restrict__ ws,
                 const int8_t *__restrict__ aq, const __half *__restrict__ as,
                 float *__restrict__ out, int M, int N, int K, int out_stride) {
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    constexpr int MMA_M = WM / 16, MMA_N = WN / 8;
    constexpr int KW = BK * 32, KWW = BK * 16, NT = WARPS_M * WARPS_N * 32;
    extern __shared__ char smem_raw[];
    int8_t (*sA)[BM * KW] = reinterpret_cast<int8_t (*)[BM * KW]>(smem_raw);
    uint8_t (*sB)[BN * KWW] = reinterpret_cast<uint8_t (*)[BN * KWW]>(
        smem_raw + STAGES * BM * KW);
    __half (*sAs)[BM * BK] = reinterpret_cast<__half (*)[BM * BK]>(
        smem_raw + STAGES * BM * KW + STAGES * BN * KWW);
    __half (*sBs)[BN * BK] = reinterpret_cast<__half (*)[BN * BK]>(
        smem_raw + STAGES * BM * KW + STAGES * BN * KWW + STAGES * BM * BK * 2);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int group = lane >> 2, tig = lane & 3;
    const int warpM = warp / WARPS_N, warpN = warp % WARPS_N;
    const int m_base = blockIdx.y * BM, n_base = blockIdx.x * BN;
    const int bpr = K / 32, nchunk = bpr / BK;

    float acc[MMA_M][MMA_N][4];
#pragma unroll
    for (int i = 0; i < MMA_M; i++)
#pragma unroll
        for (int j = 0; j < MMA_N; j++)
#pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0.0f;

    auto load_chunk = [&](int kc, int buf) {
        for (int u = tid; u < BM * BK * 2; u += NT) {
            int m = u / (BK * 2), rem = u % (BK * 2), bb = rem >> 1, half = rem & 1;
            int gm = m_base + m;
            int8_t *d = &sA[buf][m * KW + bb * 32 + half * 16];
            if (gm < M) cp_async16(d, aq + (static_cast<size_t>(gm) * bpr + kc * BK + bb) * 32 + half * 16);
            else *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
        }
        for (int u = tid; u < BN * BK; u += NT) {
            int n = u / BK, bb = u % BK, gn = n_base + n;
            uint8_t *d = &sB[buf][n * KWW + bb * 16];
            if (gn < N) cp_async16(d, wq + (static_cast<size_t>(gn) * bpr + kc * BK + bb) * 16);
            else *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
        }
        for (int u = tid; u < BM * BK; u += NT) {
            int m = u / BK, bb = u % BK, gm = m_base + m;
            sAs[buf][u] = gm < M ? as[static_cast<size_t>(gm) * bpr + kc * BK + bb] : __float2half(0.f);
        }
        for (int u = tid; u < BN * BK; u += NT) {
            int n = u / BK, bb = u % BK, gn = n_base + n;
            sBs[buf][u] = gn < N ? ws[static_cast<size_t>(gn) * bpr + kc * BK + bb] : __float2half(0.f);
        }
        cp_commit();
    };

    int prefetch = 0;
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) { if (s < nchunk) load_chunk(s, s); prefetch++; }
    for (int kc = 0; kc < nchunk; kc++) {
        int buf = kc % STAGES;
        if (prefetch < nchunk) { load_chunk(prefetch, prefetch % STAGES); prefetch++; }
        cp_wait<STAGES - 1>();
        __syncthreads();
#pragma unroll
        for (int bb = 0; bb < BK; bb++) {
            uint32_t af[MMA_M][4]; uint32_t bf[MMA_N][2];
            float axs[MMA_M][2], bxs[MMA_N][2];
#pragma unroll
            for (int mi = 0; mi < MMA_M; mi++) {
                int r0 = warpM * WM + mi * 16 + group, r1 = r0 + 8;
                const int8_t *p0 = &sA[buf][r0 * KW + bb * 32], *p1 = &sA[buf][r1 * KW + bb * 32];
                af[mi][0] = *reinterpret_cast<const uint32_t *>(p0 + tig * 4);
                af[mi][2] = *reinterpret_cast<const uint32_t *>(p0 + 16 + tig * 4);
                af[mi][1] = *reinterpret_cast<const uint32_t *>(p1 + tig * 4);
                af[mi][3] = *reinterpret_cast<const uint32_t *>(p1 + 16 + tig * 4);
                axs[mi][0] = __half2float(sAs[buf][r0 * BK + bb]);
                axs[mi][1] = __half2float(sAs[buf][r1 * BK + bb]);
            }
#pragma unroll
            for (int ni = 0; ni < MMA_N; ni++) {
                int nrow = warpN * WN + ni * 8 + group;
                const uint8_t *qs = &sB[buf][nrow * KWW + bb * 16];
                bf[ni][0] = pack_q4(qs, tig * 4, false); bf[ni][1] = pack_q4(qs, tig * 4, true);
                int n0 = warpN * WN + ni * 8 + tig * 2;
                bxs[ni][0] = __half2float(sBs[buf][n0 * BK + bb]);
                bxs[ni][1] = __half2float(sBs[buf][(n0 + 1) * BK + bb]);
            }
#pragma unroll
            for (int mi = 0; mi < MMA_M; mi++)
#pragma unroll
                for (int ni = 0; ni < MMA_N; ni++) {
                    int c[4] = {0, 0, 0, 0}; imma_m16n8k32(c, af[mi], bf[ni]);
                    acc[mi][ni][0] = fmaf(axs[mi][0] * bxs[ni][0], static_cast<float>(c[0]), acc[mi][ni][0]);
                    acc[mi][ni][1] = fmaf(axs[mi][0] * bxs[ni][1], static_cast<float>(c[1]), acc[mi][ni][1]);
                    acc[mi][ni][2] = fmaf(axs[mi][1] * bxs[ni][0], static_cast<float>(c[2]), acc[mi][ni][2]);
                    acc[mi][ni][3] = fmaf(axs[mi][1] * bxs[ni][1], static_cast<float>(c[3]), acc[mi][ni][3]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < MMA_M; mi++)
#pragma unroll
        for (int ni = 0; ni < MMA_N; ni++) {
            int m0 = m_base + warpM * WM + mi * 16 + group, m1 = m0 + 8;
            int n0 = n_base + warpN * WN + ni * 8 + tig * 2, n1 = n0 + 1;
            if (m0 < M) { size_t base = static_cast<size_t>(m0) * out_stride;
                if (n0 < N) out[base + n0] = acc[mi][ni][0]; if (n1 < N) out[base + n1] = acc[mi][ni][1]; }
            if (m1 < M) { size_t base = static_cast<size_t>(m1) * out_stride;
                if (n0 < N) out[base + n0] = acc[mi][ni][2]; if (n1 < N) out[base + n1] = acc[mi][ni][3]; }
        }
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES>
static inline int smem_bytes() {
    return STAGES * BM * BK * 32 + STAGES * BN * BK * 16 +
           STAGES * BM * BK * 2 + STAGES * BN * BK * 2;
}

// Quantize an fp16 [M][K] activation to SoA int8 blocks + fp16 per-block scale
// (symmetric amax/127). One warp per 32-element block. The -8 Q4 offset is
// folded into the weight nibbles by the GEMM, so no zero-point/sum here.
__global__ void quantize_soa_kernel(const __half *input, int8_t *aq, __half *as,
                                    uint32_t n_tokens, uint32_t n_cols) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t blocks_per_row = n_cols / 32;
    const size_t task = static_cast<size_t>(blockIdx.x) * (blockDim.x / 32) + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * blocks_per_row;
    if (task >= task_count) return;
    const uint32_t row = static_cast<uint32_t>(task / blocks_per_row);
    const uint32_t block_index = static_cast<uint32_t>(task - static_cast<size_t>(row) * blocks_per_row);
    const float value = __half2float(input[static_cast<size_t>(row) * n_cols + block_index * 32 + lane]);
    float amax = fabsf(value);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, off));
    amax = __shfl_sync(0xffffffffu, amax, 0);
    const float scale = amax / 127.0f;
    int quant = scale == 0.0f ? 0 : __float2int_rn(value / scale);
    quant = max(-128, min(127, quant));
    aq[task * 32 + lane] = static_cast<int8_t>(quant);
    if (lane == 0) as[task] = __float2half(scale);
}

// Repack native Q4_0 blocks {half d; uint8 qs[16]} into the SoA nibble + fp16
// scale arrays the GEMM expects.
struct block_q4_0 { __half d; uint8_t qs[16]; };
__global__ void repack_q4_soa_kernel(const block_q4_0 *w, uint8_t *wq, __half *ws,
                                     size_t n_blocks) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_blocks) return;
    const block_q4_0 blk = w[i];
#pragma unroll
    for (int k = 0; k < 16; k++) wq[i * 16 + k] = blk.qs[k];
    ws[i] = blk.d;
}

// Host-side launcher: config picked by output width N (wide N -> 64x128/8 warps,
// narrow N -> 64x64/4 warps); both BK=2 / 4-stage cp.async.
static void launch(const uint8_t *wq, const __half *ws, const int8_t *aq,
                   const __half *as, float *out, int M, int N, int K, int out_stride) {
    if (N > 768) {
        constexpr int BM = 64, BN = 128, BK = 2, WM = 1, WN = 8, ST = 4;
        const int smem = smem_bytes<BM, BN, BK, WM, WN, ST>();
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_kernel<BM, BN, BK, WM, WN, ST><<<grid, WM * WN * 32, smem>>>(
            wq, ws, aq, as, out, M, N, K, out_stride);
    } else {
        constexpr int BM = 64, BN = 64, BK = 2, WM = 1, WN = 4, ST = 4;
        const int smem = smem_bytes<BM, BN, BK, WM, WN, ST>();
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_kernel<BM, BN, BK, WM, WN, ST><<<grid, WM * WN * 32, smem>>>(
            wq, ws, aq, as, out, M, N, K, out_stride);
    }
}

}  // namespace w4a8

// =============================== test harness ==============================
static std::vector<char> read_file(const std::string &p) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> b(n);
    if (fread(b.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p.c_str()); exit(2); }
    fclose(f); return b;
}

// Host int8 per-32-block symmetric quantize (matches quantize_soa_kernel).
static void host_quant_soa(const std::vector<float> &X, int M, int K,
                           std::vector<int8_t> &aq, std::vector<__half> &as) {
    const int bpr = K / 32;
    aq.assign(size_t(M) * bpr * 32, 0);
    as.assign(size_t(M) * bpr, __float2half(0.f));
    for (int m = 0; m < M; m++)
        for (int b = 0; b < bpr; b++) {
            float amax = 0.f;
            for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(X[size_t(m) * K + b * 32 + i]));
            const float scale = amax / 127.0f;
            const size_t task = size_t(m) * bpr + b;
            as[task] = __float2half(scale);
            for (int i = 0; i < 32; i++) {
                int q = scale == 0.f ? 0 : int(std::rint(X[size_t(m) * K + b * 32 + i] / scale));
                q = std::max(-128, std::min(127, q));
                aq[task * 32 + i] = int8_t(q);
            }
        }
}

// Host q4_0 pack of a 32-element block -> 18 bytes {half d; uint8 qs[16]}.
static void host_pack_q4_0(const float *x, uint8_t *out) {
    float amax = 0.f;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 7.0f, inv = d ? 1.0f / d : 0.0f;
    __half dh = __float2half(d); std::memcpy(out, &dh, 2);
    uint8_t nib[32];
    for (int i = 0; i < 32; i++) { int v = int(std::rint(x[i] * inv)) + 8; nib[i] = uint8_t(std::max(0, std::min(15, v))); }
    for (int i = 0; i < 16; i++) out[2 + i] = nib[i] | (nib[i + 16] << 4);
}

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "golden/q4_0";
    const bool synth = dir == "synth";
    int N, K, M; char fmt[64] = "q4_0";
    std::vector<char> Wq_raw;
    if (synth) {
        N = argc > 2 ? atoi(argv[2]) : 4096;
        K = argc > 3 ? atoi(argv[3]) : 4096;
        M = argc > 4 ? atoi(argv[4]) : 64;
        const int bpr0 = K / 32;
        std::mt19937 wr(99); std::normal_distribution<float> wnd(0.f, 0.05f);
        std::vector<float> Wf(size_t(N) * K); for (auto &x : Wf) x = wnd(wr);
        Wq_raw.resize(size_t(N) * bpr0 * 18);
        for (int n = 0; n < N; n++)
            for (int b = 0; b < bpr0; b++)
                host_pack_q4_0(&Wf[size_t(n) * K + b * 32],
                               reinterpret_cast<uint8_t *>(Wq_raw.data()) + (size_t(n) * bpr0 + b) * 18);
    } else {
        M = argc > 2 ? atoi(argv[2]) : 64;
        FILE *mf = fopen((dir + "/meta.txt").c_str(), "r");
        if (!mf || fscanf(mf, "%63s %d %d", fmt, &N, &K) != 3) { fprintf(stderr, "bad meta\n"); return 2; }
        fclose(mf);
        if (std::string(fmt) != "q4_0") { fprintf(stderr, "expected q4_0 golden, got %s\n", fmt); return 2; }
        Wq_raw = read_file(dir + "/Wq.bin");
        if (Wq_raw.size() != size_t(N) * (K / 32) * 18) { fprintf(stderr, "Wq size mismatch\n"); return 2; }
    }
    const int bpr = K / 32;
    printf("== qgemm_w4a8 (Q4_0 W x int8 A, IMMA m16n8k32)  M=%d N=%d K=%d  [%s]\n",
           M, N, K, synth ? "synth" : dir.c_str());
    std::vector<uint8_t> wq(size_t(N) * bpr * 16);
    std::vector<__half> ws(size_t(N) * bpr);
    std::vector<float> Wdeq(size_t(N) * K);     // for the fp16 cuBLAS baseline
    for (int n = 0; n < N; n++)
        for (int b = 0; b < bpr; b++) {
            const uint8_t *blk = reinterpret_cast<const uint8_t *>(Wq_raw.data()) + (size_t(n) * bpr + b) * 18;
            __half d; std::memcpy(&d, blk, 2);
            ws[size_t(n) * bpr + b] = d;
            for (int k = 0; k < 16; k++) wq[(size_t(n) * bpr + b) * 16 + k] = blk[2 + k];
            const float df = __half2float(d);
            for (int c = 0; c < 32; c++) {
                int nib = (c < 16) ? (blk[2 + c] & 0x0F) : (blk[2 + c - 16] >> 4);
                Wdeq[size_t(n) * K + b * 32 + c] = df * float(nib - 8);
            }
        }

    // ---- activation: random X(M,K), int8 per-block quantize on host ----
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<float> Xf(size_t(M) * K);
    for (auto &x : Xf) x = nd(rng);
    std::vector<int8_t> aq; std::vector<__half> as;
    host_quant_soa(Xf, M, K, aq, as);

    // ---- fp64 oracle: exact integer block sums scaled by fp16 (as*ws) ----
    std::vector<float> ref(size_t(M) * N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int b = 0; b < bpr; b++) {
                const uint8_t *blk = reinterpret_cast<const uint8_t *>(Wq_raw.data()) + (size_t(n) * bpr + b) * 18;
                long isum = 0;
                for (int i = 0; i < 32; i++) {
                    int nib = (i < 16) ? (blk[2 + i] & 0x0F) : (blk[2 + i - 16] >> 4);
                    isum += long(aq[(size_t(m) * bpr + b) * 32 + i]) * long(nib - 8);
                }
                const double sa = __half2float(as[size_t(m) * bpr + b]);
                const double sw = __half2float(ws[size_t(n) * bpr + b]);
                acc += sa * sw * double(isum);
            }
            ref[size_t(m) * N + n] = float(acc);
        }

    // ---- device buffers ----
    uint8_t *dwq; __half *dws, *das; int8_t *daq; float *dY;
    cudaMalloc(&dwq, wq.size()); cudaMalloc(&dws, ws.size() * 2);
    cudaMalloc(&daq, aq.size()); cudaMalloc(&das, as.size() * 2);
    cudaMalloc(&dY, sizeof(float) * size_t(M) * N);
    cudaMemcpy(dwq, wq.data(), wq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dws, ws.data(), ws.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(daq, aq.data(), aq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(das, as.data(), as.size() * 2, cudaMemcpyHostToDevice);

    // ---- optional: verify the device quantize support kernel matches host ----
    {
        std::vector<__half> Xh(size_t(M) * K);
        for (size_t i = 0; i < Xh.size(); i++) Xh[i] = __float2half(Xf[i]);
        __half *dXh; int8_t *daq2; __half *das2;
        cudaMalloc(&dXh, Xh.size() * 2); cudaMalloc(&daq2, aq.size()); cudaMalloc(&das2, as.size() * 2);
        cudaMemcpy(dXh, Xh.data(), Xh.size() * 2, cudaMemcpyHostToDevice);
        size_t tasks = size_t(M) * bpr;
        w4a8::quantize_soa_kernel<<<(tasks + 7) / 8, 256>>>(dXh, daq2, das2, M, K);
        cudaDeviceSynchronize();
        std::vector<int8_t> aq2(aq.size()); cudaMemcpy(aq2.data(), daq2, aq.size(), cudaMemcpyDeviceToHost);
        long mism = 0;
        // host quant used fp32 X; device used fp16 X -> allow +-1 code differences
        for (size_t i = 0; i < aq.size(); i++) if (std::abs(int(aq2[i]) - int(aq[i])) > 1) mism++;
        printf("   quantize_soa support kernel: %ld/%zu codes differ by >1 (fp16-vs-fp32 rounding)\n", mism, aq.size());
        cudaFree(dXh); cudaFree(daq2); cudaFree(das2);
    }

    // ---- candidate: W4A8 IMMA GEMM ----
    w4a8::launch(dwq, dws, daq, das, dY, M, N, K, N);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERROR\n"); return 1; }
    const int iters = 50, warm = 10;
    for (int i = 0; i < warm; i++) w4a8::launch(dwq, dws, daq, das, dY, M, N, K, N);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) w4a8::launch(dwq, dws, daq, das, dY, M, N, K, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= iters;

    std::vector<float> got(size_t(M) * N);
    cudaMemcpy(got.data(), dY, sizeof(float) * got.size(), cudaMemcpyDeviceToHost);
    double gsum = 0, rsum = 0, gmax = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = std::abs(double(got[i]) - double(ref[i]));
        gmax = std::max(gmax, d); gsum += d; rsum += std::abs(double(ref[i]));
    }
    double rel = gsum / std::max(rsum, 1e-30);
    double top = 2.0 * M * N * K / 1e12;
    printf("qgemm_w4a8: rel %.4f%% max %.4g | %.4f ms  %.2f TOP/s  (%s)\n",
           100 * rel, gmax, ms, top / (ms / 1e3), rel < 0.02 ? "PASS" : "FAIL");
    int rc = rel < 0.02 ? 0 : 1;

    // ---- baseline: repo's Q4_0 prefill path = dequant-to-fp16 + cuBLAS fp16 ----
    {
        std::vector<__half> Wf(size_t(N) * K), Xh(size_t(M) * K);
        for (size_t i = 0; i < Wf.size(); i++) Wf[i] = __float2half(Wdeq[i]);
        for (size_t i = 0; i < Xh.size(); i++) Xh[i] = __float2half(Xf[i]);
        __half *dWf, *dXh; float *dYb;
        cudaMalloc(&dWf, Wf.size() * 2); cudaMalloc(&dXh, Xh.size() * 2);
        cudaMalloc(&dYb, sizeof(float) * size_t(M) * N);
        cudaMemcpy(dWf, Wf.data(), Wf.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dXh, Xh.data(), Xh.size() * 2, cudaMemcpyHostToDevice);
        cublasHandle_t h; cublasCreate(&h);
        const float alpha = 1.f, beta = 0.f;
        auto bl = [&] {
            // Y(M,N) row-major = X(M,K) @ Wdeq(N,K)^T  ->  col-major C(N,M)
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                         dWf, CUDA_R_16F, K, dXh, CUDA_R_16F, K, &beta,
                         dYb, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };
        bl(); cudaDeviceSynchronize();
        for (int i = 0; i < warm; i++) bl();
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) bl();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float bms; cudaEventElapsedTime(&bms, t0, t1); bms /= iters;
        // sanity: baseline (fp16 activation) vs oracle (int8 activation) should be close-ish
        std::vector<float> gb(size_t(M) * N);
        cudaMemcpy(gb.data(), dYb, sizeof(float) * gb.size(), cudaMemcpyDeviceToHost);
        double bs = 0, br = 0;
        for (size_t i = 0; i < gb.size(); i++) { bs += std::abs(double(gb[i]) - double(ref[i])); br += std::abs(double(ref[i])); }
        printf("baseline dequant->fp16 + cuBLAS: rel-vs-int8oracle %.3f%% | %.4f ms  %.2f TFLOP/s\n",
               100 * bs / std::max(br, 1e-30), bms, top / (bms / 1e3));
        printf("  -> W4A8 speedup vs dequant+cuBLAS: %.2fx\n", bms / ms);
        cublasDestroy(h); cudaFree(dWf); cudaFree(dXh); cudaFree(dYb);
    }
    return rc;
}
