// Fused LM-head + sampling (argmax / Gumbel-max categorical), CUDA/SM86 port of
// ThunderMittens kernels/lm_head (argcat pair; top-k/top-p follow with the W5
// warp-topk helpers). Picks a decode token WITHOUT materializing (T,V) logits:
//   partials: grid (num_vtiles, T), one warp per (vocab tile, token); each lane
//     owns rows v = v0+lane+32r, computes <W[v],h[t]> serially, applies invtemp
//     (+bias), adds Gumbel indexed by GLOBAL vocab id (so fused == unfused draw),
//     warp-argmax -> per-tile (val,id).
//   reduce: grid (T), warp-argmax over the tile partials.
//
// Quantized variant: W packed in any FMT, dequantized on the fly (TM ships
// q8_0/q4_0; the format layer makes the rest free).
//
// Oracle: host recompute of the logits from dequantized W + THE SAME Gumbel
// (tm_rng.cuh is __host__ __device__, bit-identical) with tie tolerance.
//
// Build:
//   /usr/local/cuda/bin/nvcc lm_head.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o lm_head.out
#include "tm_qmm.cuh"
#include "tm_rng.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

using namespace tmq;

#define LMH_NEG_INF (-3.4028234663852886e38f)

__device__ __forceinline__ void warp_argmax(float& best, int& bi) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, off);
        int   oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
}

// ---- fp16 weights ----
__global__ void lm_head_argcat_partials(const half* h, const half* W, float* part_val,
                                        int* part_id, const float* bias, int V, int K,
                                        int TILE_V, int num_vtiles, float invtemp,
                                        unsigned seed, int use_gumbel, int use_bias) {
    const int vtile = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int v0 = vtile * TILE_V, v1 = min(v0 + TILE_V, V);
    const half* hrow = h + size_t(t) * K;
    float best = LMH_NEG_INF;
    int bi = (v0 + lane < v1) ? v0 + lane : v0;
    for (int v = v0 + lane; v < v1; v += 32) {
        const half* wrow = W + size_t(v) * K;
        float acc = 0.0f;
        for (int j = 0; j < K / 2; j++) {
            half2 w2 = reinterpret_cast<const half2*>(wrow)[j];
            half2 h2 = reinterpret_cast<const half2*>(hrow)[j];
            acc += __half2float(w2.x) * __half2float(h2.x) + __half2float(w2.y) * __half2float(h2.y);
        }
        float ls = acc * invtemp;
        if (use_bias) ls += bias[v];
        if (use_gumbel) ls += rng_gumbel(seed, unsigned(t), unsigned(v));
        if (ls > best || (ls == best && v < bi)) { best = ls; bi = v; }
    }
    warp_argmax(best, bi);
    if (lane == 0) {
        part_val[size_t(t) * num_vtiles + vtile] = best;
        part_id[size_t(t) * num_vtiles + vtile] = bi;
    }
}

// ---- quantized weights (any FMT) ----
template<typename FMT>
__global__ void lm_head_argcat_partials_q(const half* h, const uint8_t* Wq, float* part_val,
                                          int* part_id, const float* bias, int V, int K,
                                          int TILE_V, int num_vtiles, float invtemp,
                                          unsigned seed, int use_gumbel, int use_bias) {
    const int vtile = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int v0 = vtile * TILE_V, v1 = min(v0 + TILE_V, V);
    const int bpr = K / FMT::block_k;
    const half* hrow = h + size_t(t) * K;
    float best = LMH_NEG_INF;
    int bi = (v0 + lane < v1) ? v0 + lane : v0;
    for (int v = v0 + lane; v < v1; v += 32) {
        const uint8_t* row_base = Wq + size_t(v) * bpr * FMT::block_bytes;
        float acc = 0.0f;
        for (int kb = 0; kb < bpr; kb++) {
            const uint8_t* base = row_base + size_t(kb) * FMT::block_bytes;
            const half* xp = hrow + kb * FMT::block_k;
            for (int c0 = 0; c0 < FMT::block_k; c0 += 8) {
                float w[8];
                dequant8<FMT>(base, c0, w);
                #pragma unroll
                for (int i = 0; i < 8; i++) acc += w[i] * __half2float(xp[c0 + i]);
            }
        }
        float ls = acc * invtemp;
        if (use_bias) ls += bias[v];
        if (use_gumbel) ls += rng_gumbel(seed, unsigned(t), unsigned(v));
        if (ls > best || (ls == best && v < bi)) { best = ls; bi = v; }
    }
    warp_argmax(best, bi);
    if (lane == 0) {
        part_val[size_t(t) * num_vtiles + vtile] = best;
        part_id[size_t(t) * num_vtiles + vtile] = bi;
    }
}

__global__ void lm_head_argcat_reduce(const float* part_val, const int* part_id,
                                      int* out_idx, int num_vtiles) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const size_t base = size_t(t) * num_vtiles;
    float best = LMH_NEG_INF;
    int bi = 0x7fffffff;
    for (int j = lane; j < num_vtiles; j += 32) {
        float v = part_val[base + j];
        int id = part_id[base + j];
        if (v > best || (v == best && id < bi)) { best = v; bi = id; }
    }
    warp_argmax(best, bi);
    if (lane == 0) out_idx[t] = bi;
}

// ---- harness: golden dir gives Wq + W_deq (V=N rows) ; h = first T rows of X2 ----
static std::vector<uint8_t> read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != size_t(n)) exit(2);
    fclose(f);
    return v;
}

template<typename FMT>
int run(const std::string& dir, int V, int K) {
    const int T = 16, TILE_V = 128, num_vtiles = (V + TILE_V - 1) / TILE_V;
    const float invtemp = 1.25f;
    const unsigned seed = 1234u;
    auto Wq_h = read_file(dir + "/Wq.bin");
    auto Wd_h = read_file(dir + "/W_deq.bin");
    auto X_h  = read_file(dir + "/X2.bin");   // fp16 (64,K); take first T rows as h

    const float* Wd = reinterpret_cast<const float*>(Wd_h.data());
    // fp16 W for the non-quant kernel = fp16(W_deq)
    std::vector<half> Wf(size_t(V) * K);
    for (size_t i = 0; i < Wf.size(); i++) Wf[i] = __float2half(Wd[i]);

    uint8_t* dWq; half *dW, *dh; float* dpv; int *dpi, *dout;
    cudaMalloc(&dWq, Wq_h.size());
    cudaMalloc(&dW, sizeof(half) * Wf.size());
    cudaMalloc(&dh, sizeof(half) * T * K);
    cudaMalloc(&dpv, sizeof(float) * T * num_vtiles);
    cudaMalloc(&dpi, sizeof(int) * T * num_vtiles);
    cudaMalloc(&dout, sizeof(int) * T);
    cudaMemcpy(dWq, Wq_h.data(), Wq_h.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dW, Wf.data(), sizeof(half) * Wf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dh, X_h.data(), sizeof(half) * T * K, cudaMemcpyHostToDevice);

    const half* hh = reinterpret_cast<const half*>(X_h.data());
    int rc = 0;
    for (int mode = 0; mode < 4; mode++) {           // {fp16,quant} x {argmax,categorical}
        const bool quant = mode & 1, gumbel = mode & 2;
        dim3 grid(num_vtiles, T);
        if (quant) lm_head_argcat_partials_q<FMT><<<grid, 32>>>(dh, dWq, dpv, dpi, nullptr, V, K, TILE_V, num_vtiles, invtemp, seed, gumbel, 0);
        else       lm_head_argcat_partials<<<grid, 32>>>(dh, dW, dpv, dpi, nullptr, V, K, TILE_V, num_vtiles, invtemp, seed, gumbel, 0);
        lm_head_argcat_reduce<<<T, 32>>>(dpv, dpi, dout, num_vtiles);
        cudaDeviceSynchronize();
        std::vector<int> got(T);
        cudaMemcpy(got.data(), dout, sizeof(int) * T, cudaMemcpyDeviceToHost);

        // oracle: fp64 logits from W_deq (+ identical Gumbel), tie tolerance 1e-3
        int bad = 0;
        for (int t = 0; t < T; t++) {
            double best = -1e300; int bi = -1;
            std::vector<double> ls(V);
            for (int v = 0; v < V; v++) {
                double acc = 0;
                for (int k = 0; k < K; k++)
                    acc += double(Wd[size_t(v) * K + k]) * double(__half2float(hh[size_t(t) * K + k]));
                double l = acc * invtemp;
                if (gumbel) l += double(rng_gumbel(seed, unsigned(t), unsigned(v)));
                ls[v] = l;
                if (l > best) { best = l; bi = v; }
            }
            if (got[t] != bi && !(ls[got[t]] >= best - 1e-3)) bad++;   // tie tolerance
        }
        printf("%s/%s: %d/%d tokens match  (%s)\n", quant ? "quant" : "fp16",
               gumbel ? "categorical" : "argmax", T - bad, T, bad ? "FAIL" : "PASS");
        rc |= (bad != 0);
    }
    return rc;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <golden_dir>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    char fmt[64]; int V, K;
    FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
    if (!f || fscanf(f, "%63s %d %d", fmt, &V, &K) != 3) return 2;
    fclose(f);
    printf("== lm_head %s  V=%d K=%d T=16\n", fmt, V, K);
    std::string s(fmt);
    if (s == "q8_0") return run<q8_0>(dir, V, K);
    if (s == "q4_0") return run<q4_0>(dir, V, K);
    if (s == "mxfp4") return run<mxfp4>(dir, V, K);
    if (s == "nvfp4") return run<nvfp4>(dir, V, K);
    if (s == "mxfp8") return run<mxfp8>(dir, V, K);
    fprintf(stderr, "format %s not wired for lm_head test (add to dispatch)\n", fmt);
    return 2;
}
