/**
 * @file
 * @brief Harness for the quantized MoE grouped GEMMs (tm_moe_quant_kernels.cuh).
 * Isolates each GEMM: builds permuted A + per-expert quantized B + expert_of_tile,
 * runs the kernel, compares vs an fp64 reference over the exactly-dequantized
 * weights. (Full route->gather->gemm->finalize pipeline is covered in pytest.)
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        moe_quant_test.cu -o moe_quant_test.out -I../quant -I../serving
 * Run: CUDA_VISIBLE_DEVICES=0 ./moe_quant_test.out
 */
#include "tm_moe_quant_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace tmoeq;
static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
template <typename T> T* dnew(const std::vector<T>& h){T*d;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template <typename T> T* dzero(size_t n){T*d;CK(cudaMalloc(&d,n*sizeof(T)));CK(cudaMemset(d,0,n*sizeof(T)));return d;}
template <typename T> std::vector<T> d2h(const T*d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static void report(const char* nm, double err, double tol){bool ok=err<=tol;printf("%-34s %s (rel %.3e, tol %.1e)\n",nm,ok?"PASS":"FAIL",err,tol);if(!ok)++g_fail;}

// CPU replica of nvfp4_sf_offset (intra-expert; row < 128 -> m_tile stays 0).
static long ref_swizzle_local(int row, int group, int num_k_tiles) {
    const int m_tile = row >> 7, outer_m = row & 31, inner_m = (row >> 5) & 3;
    const int k_tile = group >> 2, inner_k = group & 3;
    return ((long)(m_tile) * num_k_tiles + k_tile) * 512L + ((outer_m << 4) | (inner_m << 2) | inner_k);
}

static std::mt19937 rng(9);
static std::vector<float> randv(size_t n,float lo,float hi){std::uniform_real_distribution<float> d(lo,hi);std::vector<float> v(n);for(auto&x:v)x=d(rng);return v;}

int main() {
    // ---- fp8 MoE GEMM ----
    {
        const int E = 4, TILES = 6, total_rows = TILES * 32, N = 128, K = 256;
        std::vector<int> eot(TILES);
        for (int t = 0; t < TILES; ++t) eot[t] = (t == TILES - 1) ? -1 : t % E;  // last tile = padding
        auto A = randv((size_t)total_rows * K, -1, 1);
        std::vector<__half> Ah((size_t)total_rows * K);
        for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = __float2half(A[i]);
        // per-expert fp8 weights (E,N,K) + rowwise scale (E,N)
        std::vector<uint8_t> B((size_t)E * N * K);
        std::vector<float> Bsc((size_t)E * N);
        auto wr = randv(B.size(), -6, 6);
        for (size_t i = 0; i < B.size(); ++i) B[i] = e4m3_encode(wr[i] * 0.1f);
        for (auto& s : Bsc) s = std::uniform_real_distribution<float>(0.02f, 0.2f)(rng);

        auto dA = dnew(Ah); auto dB = dnew(B); auto dSc = dnew(Bsc); auto dEot = dnew(eot);
        float* dY = dzero<float>((size_t)total_rows * N);
        dim3 grid(N / 16, total_rows / 16);
        moe_gemm_fp8<<<grid, 32>>>(dY, reinterpret_cast<const half*>(dA), dB, dSc, dEot,
                                   total_rows, N, K);
        CK(cudaDeviceSynchronize());
        if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERR\n"); return 1; }
        auto Y = d2h(dY, (size_t)total_rows * N);

        double gsum = 0, rsum = 0;                         // aggregate rel (matches qgemm.cu)
        for (int r = 0; r < total_rows; ++r) {
            const int e = eot[r / 32];
            for (int n = 0; n < N; ++n) {
                if (e < 0) continue;                       // padding rows: not checked (never read)
                double acc = 0;
                for (int k = 0; k < K; ++k)
                    acc += (double)A[(size_t)r * K + k]
                         * (double)e4m3_decode_host(B[((size_t)e * N + n) * K + k]);
                acc *= Bsc[(size_t)e * N + n];
                gsum += std::abs((double)Y[(size_t)r * N + n] - acc);
                rsum += std::abs(acc);
            }
        }
        report("moe_gemm_fp8 (fp64)", gsum / std::max(rsum, 1e-30), 5e-3);
    }

    // ---- nvfp4 dual-fp4 MoE GEMM ----
    {
        const int E = 3, R = 64, total_rows = E * R, N = 128, K = 256;  // R multiple of 32
        const int groups = K / 16, num_k_tiles = (K + 63) / 64, padded_groups = num_k_tiles * 4;
        const int padrows = 128;                                // R<=128 -> 1 swizzle row-tile
        std::vector<int> eot(total_rows / 32), erow0(E), sfo(E);
        for (int t = 0; t < total_rows / 32; ++t) eot[t] = t / (R / 32);
        for (int e = 0; e < E; ++e) { erow0[e] = e * R; sfo[e] = e * padrows; }
        std::uniform_int_distribution<int> nib(0, 15);
        std::vector<uint8_t> Ab((size_t)total_rows * (K / 2)), Bb((size_t)E * N * (K / 2));
        for (auto& b : Ab) b = uint8_t(nib(rng) | (nib(rng) << 4));
        for (auto& b : Bb) b = uint8_t(nib(rng) | (nib(rng) << 4));
        // swizzled A-scale (E*padrows*padded_groups) + plain B-scale (E*N*groups)
        std::vector<uint8_t> Asc((size_t)E * padrows * padded_groups, 0);
        std::vector<uint8_t> Bsc((size_t)E * N * groups);
        std::vector<float> Asf((size_t)E * R * groups), Bsf((size_t)E * N * groups);
        auto swz = [&](int lr, int g){ return ref_swizzle_local(lr, g, num_k_tiles); };
        for (int e = 0; e < E; ++e) for (int lr = 0; lr < R; ++lr) for (int g = 0; g < groups; ++g) {
            const float s = std::uniform_real_distribution<float>(0.05f, 0.5f)(rng);
            Asf[((size_t)e * R + lr) * groups + g] = e4m3_decode_host(e4m3_encode(s));
            Asc[(size_t)sfo[e] * padded_groups + swz(lr, g)] = e4m3_encode(s);
        }
        for (int e = 0; e < E; ++e) for (int n = 0; n < N; ++n) for (int g = 0; g < groups; ++g) {
            const float s = std::uniform_real_distribution<float>(0.05f, 0.5f)(rng);
            Bsf[((size_t)e * N + n) * groups + g] = e4m3_decode_host(e4m3_encode(s));
            Bsc[((size_t)e * N + n) * groups + g] = e4m3_encode(s);
        }
        std::vector<float> alpha(E);
        for (auto& a : alpha) a = std::uniform_real_distribution<float>(0.5f, 1.5f)(rng);

        auto dAb = dnew(Ab); auto dBb = dnew(Bb); auto dAsc = dnew(Asc); auto dBsc = dnew(Bsc);
        auto dal = dnew(alpha); auto deot = dnew(eot); auto der0 = dnew(erow0); auto dsfo = dnew(sfo);
        float* dY = dzero<float>((size_t)total_rows * N);
        dim3 grid(N / 16, total_rows / 16);
        moe_gemm_nvfp4<<<grid, 32>>>(dY, dAb, dBb, dAsc, dBsc, dal, deot, der0, dsfo,
                                     total_rows, N, K);
        CK(cudaDeviceSynchronize());
        if (cudaGetLastError() != cudaSuccess) { printf("NVFP4 KERNEL ERR\n"); return 1; }
        auto Y = d2h(dY, (size_t)total_rows * N);
        auto e2m1h = [](uint8_t nb){ const float cb[8]={0,0.5f,1,1.5f,2,3,4,6};
            float m=cb[nb&7]; return (nb&8)?-m:m; };
        double gsum = 0, rsum = 0;
        for (int e = 0; e < E; ++e) for (int lr = 0; lr < R; ++lr) {
            const int r = erow0[e] + lr;
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                for (int k = 0; k < K; ++k) {
                    const int g = k / 16;
                    const uint8_t ab = Ab[(size_t)r * (K/2) + k/2], bb = Bb[((size_t)e*N+n)*(K/2)+k/2];
                    const float an = e2m1h((k & 1) ? (ab >> 4) : (ab & 0xF));
                    const float bn = e2m1h((k & 1) ? (bb >> 4) : (bb & 0xF));
                    acc += (double)an * Asf[((size_t)e*R+lr)*groups+g]
                         * bn * Bsf[((size_t)e*N+n)*groups+g];
                }
                acc *= alpha[e];
                gsum += std::abs((double)Y[(size_t)r * N + n] - acc);
                rsum += std::abs(acc);
            }
        }
        report("moe_gemm_nvfp4 (fp64)", gsum / std::max(rsum, 1e-30), 5e-3);
    }

    // ---- WNA16 int4 / int8 MoE GEMM ----
    for (int BIT : {4, 8}) {
        const int E = 3, TILES = 5, total_rows = TILES * 32, N = 128, K = 256, GS = 64;
        const int PACK = 32 / BIT, groups = K / GS, packed_k = K / PACK;
        std::vector<int> eot(TILES);
        for (int t = 0; t < TILES; ++t) eot[t] = (t == TILES - 1) ? -1 : t % E;
        auto A = randv((size_t)total_rows * K, -1, 1);
        std::vector<__half> Ah((size_t)total_rows * K);
        for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = __float2half(A[i]);
        const int QMAX = (1 << BIT) - 1;
        auto order = [&](int inner) {
            if (BIT == 4) { int o[8] = {0,2,4,6,1,3,5,7}; return o[inner & 7]; }
            int o[4] = {0,2,1,3}; return o[inner & 3];
        };
        std::vector<uint32_t> qw((size_t)E * N * packed_k, 0);
        std::vector<int> q((size_t)E * N * K);              // raw codes for the reference
        std::vector<__half> sc((size_t)E * N * groups);
        std::vector<uint8_t> qz(BIT == 4 ? (size_t)E * ((N + 1) / 2) * groups
                                         : (size_t)E * N * groups, 0);
        std::uniform_int_distribution<int> qd(0, QMAX), zd(0, QMAX);
        for (auto& s : sc) s = __float2half(std::uniform_real_distribution<float>(0.01f, 0.08f)(rng));
        // random zero-points, packed like MetalForge
        std::vector<int> zpv((size_t)E * N * groups);
        for (auto& z : zpv) z = zd(rng);
        for (int e = 0; e < E; ++e) for (int n = 0; n < N; ++n) for (int g = 0; g < groups; ++g) {
            const int z = zpv[((size_t)e * N + n) * groups + g];
            if (BIT == 4) {
                const size_t base = (size_t)e * ((N + 1) / 2) * groups;
                uint8_t& b = qz[base + (n / 2) * groups + g];
                b |= uint8_t(z << ((n & 1) * 4));
            } else {
                qz[(size_t)e * N * groups + (size_t)n * groups + g] = uint8_t(z);
            }
        }
        for (int e = 0; e < E; ++e) for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) {
            const int v = qd(rng);
            q[((size_t)e * N + n) * K + k] = v;
            const int local = order(k % PACK);
            qw[((size_t)e * N + n) * packed_k + k / PACK] |= uint32_t(v) << (local * BIT);
        }
        auto dA = dnew(Ah); auto dqw = dnew(qw); auto dsc = dnew(sc); auto dqz = dnew(qz);
        auto dEot = dnew(eot);
        float* dY = dzero<float>((size_t)total_rows * N);
        dim3 grid(N / 16, total_rows / 16);
        if (BIT == 4)
            moe_gemm_wna16<4><<<grid, 32>>>(dY, reinterpret_cast<const half*>(dA), dqw,
                reinterpret_cast<const half*>(dsc), dqz, dEot, total_rows, N, K, GS, 1);
        else
            moe_gemm_wna16<8><<<grid, 32>>>(dY, reinterpret_cast<const half*>(dA), dqw,
                reinterpret_cast<const half*>(dsc), dqz, dEot, total_rows, N, K, GS, 1);
        CK(cudaDeviceSynchronize());
        if (cudaGetLastError() != cudaSuccess) { printf("WNA16 KERNEL ERR\n"); return 1; }
        auto Y = d2h(dY, (size_t)total_rows * N);
        double gsum = 0, rsum = 0;
        for (int r = 0; r < total_rows; ++r) {
            const int e = eot[r / 32];
            if (e < 0) continue;
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                for (int k = 0; k < K; ++k) {
                    const int g = k / GS;
                    const double s = __half2float(sc[((size_t)e * N + n) * groups + g]);
                    const double z = zpv[((size_t)e * N + n) * groups + g];
                    acc += (double)A[(size_t)r * K + k]
                         * ((q[((size_t)e * N + n) * K + k] - z) * s);
                }
                gsum += std::abs((double)Y[(size_t)r * N + n] - acc);
                rsum += std::abs(acc);
            }
        }
        char nm[40]; snprintf(nm, sizeof nm, "moe_gemm_wna16 int%d (fp64)", BIT);
        report(nm, gsum / std::max(rsum, 1e-30), 5e-3);
    }

    // ---- fused SiLU-and-mul fp8 quant (static + per-block) ----
    {
        const int T = 40, H = 256, GS = 128, NG = H / GS;
        auto in = randv((size_t)T * H * 2, -3, 3);          // [gate|up]
        std::vector<__half> ih((size_t)T * H * 2);
        for (size_t i = 0; i < ih.size(); ++i) ih[i] = __float2half(in[i]);
        auto silu = [](float g, float u){ return (g / (1 + std::exp(-g))) * u; };

        auto dih = dnew(ih);
        uint8_t* dout = dzero<uint8_t>((size_t)T * H);
        const float sc = 4.0f;
        silu_and_mul_quant_static<half, true><<<((size_t)T*H+255)/256, 256>>>(
            dout, reinterpret_cast<const half*>(dih), 1.0f/sc, H, T);
        CK(cudaDeviceSynchronize());
        auto cd = d2h(dout, (size_t)T*H);
        double gs = 0, rs = 0;
        for (int t = 0; t < T; ++t) for (int c = 0; c < H; ++c) {
            const double ref = silu(in[(size_t)t*H*2+c], in[(size_t)t*H*2+H+c]);
            const double got = e4m3_decode_host(cd[(size_t)t*H+c]) * sc;
            gs += std::abs(got-ref); rs += std::abs(ref);
        }
        report("silu_and_mul_fp8_static (rt)", gs/std::max(rs,1e-30), 6e-2);

        // per-block dynamic
        float* dscp = dzero<float>((size_t)T*NG);
        dim3 g2(NG, T);
        silu_and_mul_quant_perblock<half, true><<<g2, 32>>>(
            dout, dscp, reinterpret_cast<const half*>(dih), H, GS, NG);
        CK(cudaDeviceSynchronize());
        cd = d2h(dout, (size_t)T*H); auto scv = d2h(dscp, (size_t)T*NG);
        gs = 0; rs = 0;
        for (int t = 0; t < T; ++t) for (int c = 0; c < H; ++c) {
            const double ref = silu(in[(size_t)t*H*2+c], in[(size_t)t*H*2+H+c]);
            const double got = e4m3_decode_host(cd[(size_t)t*H+c]) * scv[(size_t)t*NG + c/GS];
            gs += std::abs(got-ref); rs += std::abs(ref);
        }
        report("silu_and_mul_fp8_perblock (rt)", gs/std::max(rs,1e-30), 4e-2);
    }

    // ---- per_token_group_quant_fp8 (round-trip) ----
    {
        const int T = 32, H = 512, GS = 128, NG = H / GS;
        auto in = randv((size_t)T*H, -2, 2);
        std::vector<__half> ih((size_t)T*H);
        for (size_t i = 0; i < ih.size(); ++i) ih[i] = __float2half(in[i]);
        auto dih = dnew(ih);
        uint8_t* dout = dzero<uint8_t>((size_t)T*H);
        float* dsc = dzero<float>((size_t)T*NG);
        dim3 g(NG, T);
        per_token_group_quant_fp8<half, false><<<g, 32>>>(
            dout, dsc, reinterpret_cast<const half*>(dih), H, GS, NG, 1e-6f);
        CK(cudaDeviceSynchronize());
        auto cd = d2h(dout, (size_t)T*H); auto scv = d2h(dsc, (size_t)T*NG);
        double gs = 0, rs = 0;
        for (int t = 0; t < T; ++t) for (int c = 0; c < H; ++c) {
            const double got = e4m3_decode_host(cd[(size_t)t*H+c]) * scv[(size_t)t*NG+c/GS];
            gs += std::abs(got - in[(size_t)t*H+c]); rs += std::abs(in[(size_t)t*H+c]);
        }
        report("per_token_group_quant_fp8 (rt)", gs/std::max(rs,1e-30), 5e-2);
    }

    // ---- nvfp4 experts-quant (round-trip to fp4 precision) ----
    {
        const int E = 2, R = 32, rows = E*R, K = 256, groups = K/16;
        const int num_k_tiles = (K+63)/64, padded_groups = num_k_tiles*4, padrows = 128;
        auto in = randv((size_t)rows*K, -4, 4);
        std::vector<__half> ih((size_t)rows*K);
        for (size_t i = 0; i < ih.size(); ++i) ih[i] = __float2half(in[i]);
        std::vector<int> eor(rows), er0(E), sfo(E);
        for (int r = 0; r < rows; ++r) eor[r] = r / R;
        for (int e = 0; e < E; ++e) { er0[e] = e*R; sfo[e] = e*padrows; }
        std::vector<float> gscale(E, 1.0f);
        auto dih = dnew(ih); auto dgs = dnew(gscale); auto deor = dnew(eor);
        auto der0 = dnew(er0); auto dsfo = dnew(sfo);
        uint8_t* dcode = dzero<uint8_t>((size_t)rows*(K/2));
        uint8_t* dsc = dzero<uint8_t>((size_t)E*padrows*padded_groups);
        dim3 g(groups, rows);
        nvfp4_experts_quant<half, false><<<g, 32>>>(dcode, dsc, reinterpret_cast<const half*>(dih),
            dgs, deor, der0, dsfo, K, num_k_tiles);
        CK(cudaDeviceSynchronize());
        auto code = d2h(dcode, (size_t)rows*(K/2)); auto scb = d2h(dsc, (size_t)E*padrows*padded_groups);
        auto e2m1h = [](uint8_t nb){ const float cb[8]={0,0.5f,1,1.5f,2,3,4,6};
            float m=cb[nb&7]; return (nb&8)?-m:m; };
        double gg = 0, rr2 = 0;
        for (int e = 0; e < E; ++e) for (int lr = 0; lr < R; ++lr) {
            const int r = er0[e]+lr;
            for (int k = 0; k < K; ++k) {
                const int grp = k/16;
                const uint8_t sf = scb[(size_t)sfo[e]*padded_groups + ref_swizzle_local(lr,grp,num_k_tiles)];
                const uint8_t byte = code[(size_t)r*(K/2)+k/2];
                const float nibv = e2m1h((k&1)?(byte>>4):(byte&0xF));
                const double got = nibv * e4m3_decode_host(sf) / gscale[e];
                gg += std::abs(got - in[(size_t)r*K+k]); rr2 += std::abs(in[(size_t)r*K+k]);
            }
        }
        report("nvfp4_experts_quant (rt, fp4)", gg/std::max(rr2,1e-30), 1.2e-1);
    }

    // ---- scored routing (sigmoid / softplus_sqrt) ----
    for (int mode = 0; mode < 2; ++mode) {
        const int T = 64, E = 32, K = 6;
        auto lg = randv((size_t)T*E, -4, 4);
        std::vector<__half> lh((size_t)T*E);
        for (size_t i = 0; i < lh.size(); ++i) lh[i] = __float2half(lg[i]);
        auto dlg = dnew(lh);
        int* did = dzero<int>((size_t)T*K); float* dw = dzero<float>((size_t)T*K);
        const float rsf = 2.5f;
        moe_route_scored<half><<<T, 32>>>(reinterpret_cast<const half*>(dlg), did, dw, E, K,
                                          mode, 1, rsf);
        CK(cudaDeviceSynchronize());
        auto id = d2h(did, (size_t)T*K); auto wt = d2h(dw, (size_t)T*K);
        auto score = [&](double x){ if (mode==0) return 1.0/(1.0+std::exp(-x));
            double y = x>20?x:std::log(1.0+std::exp(x)); return std::sqrt(y); };
        long mism = 0; double werr = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<int> ord(E); for (int i=0;i<E;++i) ord[i]=i;
            std::partial_sort(ord.begin(), ord.begin()+K, ord.end(), [&](int a, int b){
                double sa=score(__half2float(lh[(size_t)t*E+a])), sb=score(__half2float(lh[(size_t)t*E+b]));
                return sa>sb || (sa==sb && a<b); });
            double sum=0; for (int k=0;k<K;++k) sum += score(__half2float(lh[(size_t)t*E+ord[k]]));
            for (int k = 0; k < K; ++k) {
                mism += id[(size_t)t*K+k] != ord[k];
                double wref = score(__half2float(lh[(size_t)t*E+ord[k]]))*rsf/(sum>0?sum:1);
                werr = std::max(werr, std::abs(wt[(size_t)t*K+k]-wref));
            }
        }
        char nm[40]; snprintf(nm,sizeof nm,"moe_route_scored mode%d",mode);
        printf("%-34s %s (%ld id mism, w err %.3e)\n", nm, (mism==0&&werr<1e-3)?"PASS":"FAIL", mism, werr);
        g_fail += !(mism==0 && werr<1e-3);
    }

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
