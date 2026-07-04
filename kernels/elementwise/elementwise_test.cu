/**
 * @file
 * @brief Standalone fp64-oracle harness for the W3 elementwise/norm family
 * (tm_elementwise_kernels.cuh). Runs every kernel with T=float against CPU
 * fp64 references (analytic where independent, central finite differences for
 * the activation backwards, exact replay for dropout/embedding/add). bf16/fp16
 * I/O is exercised later by the tk_cuda pytest against torch.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        elementwise_test.cu -o elementwise_test.out
 * Run: CUDA_VISIBLE_DEVICES=6 ./elementwise_test.out
 */
#include "tm_elementwise_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace tme;

static int g_fail = 0;
#define CUCHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

template <typename T> T* dnew(const std::vector<T>& h) {
    T* d; CUCHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUCHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> T* dzero(size_t n) {
    T* d; CUCHECK(cudaMalloc(&d, n * sizeof(T)));
    CUCHECK(cudaMemset(d, 0, n * sizeof(T)));
    return d;
}
template <typename T> std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    CUCHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

static double maxrel(const std::vector<double>& ref, const std::vector<float>& got,
                     double floor_ = 1.0) {
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double s = std::max(std::abs(ref[i]), floor_);
        worst = std::max(worst, std::abs(double(got[i]) - ref[i]) / s);
    }
    return worst;
}
static void report(const char* name, double err, double tol) {
    const bool ok = err <= tol;
    printf("%-42s %s  (err %.3e, tol %.1e)\n", name, ok ? "PASS" : "FAIL", err, tol);
    if (!ok) ++g_fail;
}
static void report_exact(const char* name, long mismatches) {
    printf("%-42s %s  (%ld mismatches)\n", name, mismatches == 0 ? "PASS" : "FAIL", mismatches);
    if (mismatches) ++g_fail;
}

static std::mt19937 g_rng(1234);
static std::vector<float> randv(size_t n, float lo = -2.0f, float hi = 2.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g_rng);
    return v;
}

// ---------------------------------------------------------------------------
static void test_norms() {
    const int M = 33, D = 768;
    const float eps = 1e-5f;
    auto x = randv((size_t)M * D), w = randv(D), b = randv(D), dy = randv((size_t)M * D);

    // fp64 refs
    std::vector<double> rms_o(M * (size_t)D), ln_o(M * (size_t)D);
    std::vector<double> rms_dx(M * (size_t)D), ln_dx(M * (size_t)D);
    std::vector<double> rms_dw(D, 0.0), ln_dw(D, 0.0), ln_db(D, 0.0);
    std::vector<float> rstd_h(M), mean_h(M), ln_rstd_h(M);
    for (int r = 0; r < M; ++r) {
        const size_t base = (size_t)r * D;
        double ss = 0, sx = 0;
        for (int j = 0; j < D; ++j) { ss += (double)x[base+j]*x[base+j]; sx += x[base+j]; }
        const double rr = 1.0 / std::sqrt(ss / D + eps);
        const double mu = sx / D;
        double var = 0;
        for (int j = 0; j < D; ++j) { const double d = x[base+j]-mu; var += d*d; }
        const double rl = 1.0 / std::sqrt(var / D + eps);
        rstd_h[r] = float(rr); mean_h[r] = float(mu); ln_rstd_h[r] = float(rl);
        double s_rms = 0, s1 = 0, s2 = 0;
        for (int j = 0; j < D; ++j) {
            rms_o[base+j] = x[base+j] * rr * w[j];
            ln_o[base+j]  = (x[base+j]-mu) * rl * w[j] + b[j];
            const double m = (double)dy[base+j] * w[j];
            s_rms += m * x[base+j];
            const double xh = (x[base+j]-mu) * rl;
            s1 += m; s2 += m * xh;
        }
        s1 /= D; s2 /= D;
        for (int j = 0; j < D; ++j) {
            const double m = (double)dy[base+j] * w[j];
            rms_dx[base+j] = rr * m - (rr*rr*rr*s_rms/D) * x[base+j];
            const double xh = (x[base+j]-mu) * rl;
            ln_dx[base+j] = rl * ((double)dy[base+j]*w[j] - s1 - xh*s2);
            rms_dw[j] += (double)dy[base+j] * x[base+j] * rr;
            ln_dw[j]  += (double)dy[base+j] * xh;
            ln_db[j]  += (double)dy[base+j];
        }
    }

    float *dx_ = dnew(x), *dw_ = dnew(w), *db_ = dnew(b), *ddy = dnew(dy);
    float *dr = dnew(rstd_h), *dm = dnew(mean_h), *dlr = dnew(ln_rstd_h);
    float *o1 = dzero<float>((size_t)M*D), *o2 = dzero<float>((size_t)M*D);
    float *g1 = dzero<float>((size_t)M*D), *g2 = dzero<float>((size_t)M*D), *g3 = dzero<float>((size_t)M*D);
    float *aw = dzero<float>(D), *aw2 = dzero<float>(D), *ab = dzero<float>(D);

    rms_norm_fwd<<<M, 32>>>(dx_, dw_, o1, D, eps);
    layernorm_fwd<<<M, 32>>>(dx_, dw_, db_, o2, D, eps);
    rms_norm_bwd_dx<<<M, 32>>>(dx_, dw_, ddy, dr, g1, D);
    rms_norm_bwd_fused<<<M, 32>>>(dx_, dw_, ddy, g2, aw, D, eps);
    layernorm_bwd_dx<<<M, 32>>>(dx_, dw_, ddy, dm, dlr, g3, D);
    CUCHECK(cudaDeviceSynchronize());
    report("rms_norm_fwd", maxrel(rms_o, d2h(o1, (size_t)M*D)), 2e-6);
    report("layernorm_fwd", maxrel(ln_o, d2h(o2, (size_t)M*D)), 2e-6);
    report("rms_norm_bwd_dx", maxrel(rms_dx, d2h(g1, (size_t)M*D)), 2e-5);
    report("rms_norm_bwd_fused dx", maxrel(rms_dx, d2h(g2, (size_t)M*D)), 2e-5);
    report("rms_norm_bwd_fused dW", maxrel(rms_dw, d2h(aw, D), 4.0), 2e-5);
    report("layernorm_bwd_dx", maxrel(ln_dx, d2h(g3, (size_t)M*D)), 2e-5);

    float *g4 = dzero<float>((size_t)M*D);
    layernorm_bwd_fused<<<M, 32>>>(dx_, dw_, ddy, g4, aw2, ab, D, eps);
    CUCHECK(cudaDeviceSynchronize());
    report("layernorm_bwd_fused dx", maxrel(ln_dx, d2h(g4, (size_t)M*D)), 2e-5);
    report("layernorm_bwd_fused dW", maxrel(ln_dw, d2h(aw2, D), 4.0), 2e-5);
    report("layernorm_bwd_fused dB", maxrel(ln_db, d2h(ab, D), 4.0), 2e-5);
}

// ---------------------------------------------------------------------------
static void test_add_norm() {
    const int M = 17, D = 512;
    const float eps = 1e-5f, inv_scale = 3.0f;
    auto x = randv((size_t)M*D), res = randv((size_t)M*D), w = randv(D), b = randv(D);
    std::vector<double> sum_(M*(size_t)D), rms_o(M*(size_t)D), ln_o(M*(size_t)D);
    std::vector<uint8_t> rms_c(M*(size_t)D), ln_c(M*(size_t)D), rms_cd(M*(size_t)D), ln_cd(M*(size_t)D);
    std::vector<double> rms_s(M), ln_s(M);
    for (int r = 0; r < M; ++r) {
        const size_t base = (size_t)r * D;
        double ss = 0, sx = 0;
        for (int j = 0; j < D; ++j) {
            sum_[base+j] = (double)x[base+j] + res[base+j];
            // the kernel rounds res_out to T then re-reads it; for T=float that's exact
            ss += sum_[base+j]*sum_[base+j];
            sx += sum_[base+j];
        }
        const double rr = 1.0/std::sqrt(ss/D + eps), mu = sx/D;
        double var = 0;
        for (int j = 0; j < D; ++j) { const double d = sum_[base+j]-mu; var += d*d; }
        const double rl = 1.0/std::sqrt(var/D + eps);
        double amax_r = 0, amax_l = 0;
        for (int j = 0; j < D; ++j) {
            rms_o[base+j] = sum_[base+j]*rr*w[j];
            ln_o[base+j]  = (sum_[base+j]-mu)*rl*w[j] + b[j];
            amax_r = std::max(amax_r, std::abs(rms_o[base+j]));
            amax_l = std::max(amax_l, std::abs(ln_o[base+j]));
        }
        // fp8 codes: encoder replayed on host at fp32 input precision
        const double sr = amax_r/448.0, sl = amax_l/448.0;
        rms_s[r] = sr; ln_s[r] = sl;
        for (int j = 0; j < D; ++j) {
            rms_c[base+j] = tme_e4m3_encode(float(rms_o[base+j]*inv_scale));
            ln_c[base+j]  = tme_e4m3_encode(float(ln_o[base+j]*inv_scale));
            rms_cd[base+j] = tme_e4m3_encode(float(rms_o[base+j]*(sr>0?1.0/sr:0.0)));
            ln_cd[base+j]  = tme_e4m3_encode(float(ln_o[base+j]*(sl>0?1.0/sl:0.0)));
        }
    }
    float *dx_ = dnew(x), *drs = dnew(res), *dw = dnew(w), *db = dnew(b);
    float *o = dzero<float>((size_t)M*D), *ro = dzero<float>((size_t)M*D);
    uint8_t *c = dzero<uint8_t>((size_t)M*D);
    float *sc = dzero<float>(M);

    rms_norm_add_k<float,false,false><<<M,32>>>(dx_, drs, dw, o, nullptr, ro, nullptr, D, eps, 0);
    CUCHECK(cudaDeviceSynchronize());
    report("rms_norm_add o", maxrel(rms_o, d2h(o, (size_t)M*D)), 2e-6);
    report("rms_norm_add res_out", maxrel(sum_, d2h(ro, (size_t)M*D)), 1e-7);

    layernorm_add_k<float,false,false><<<M,32>>>(dx_, drs, dw, db, o, nullptr, ro, nullptr, D, eps, 0);
    CUCHECK(cudaDeviceSynchronize());
    report("layernorm_add o", maxrel(ln_o, d2h(o, (size_t)M*D)), 2e-6);

    long mm = 0;
    rms_norm_add_k<float,true,false><<<M,32>>>(dx_, drs, dw, nullptr, c, ro, nullptr, D, eps, inv_scale);
    CUCHECK(cudaDeviceSynchronize());
    { auto got = d2h(c, (size_t)M*D); for (size_t i = 0; i < got.size(); ++i) mm += got[i] != rms_c[i]; }
    report_exact("rms_norm_add_fp8 codes", mm);

    rms_norm_add_k<float,true,true><<<M,32>>>(dx_, drs, dw, nullptr, c, ro, sc, D, eps, 0);
    CUCHECK(cudaDeviceSynchronize());
    mm = 0;
    { auto got = d2h(c, (size_t)M*D); for (size_t i = 0; i < got.size(); ++i) mm += got[i] != rms_cd[i]; }
    report_exact("rms_norm_add_fp8_dyn codes", mm);
    report("rms_norm_add_fp8_dyn scale", maxrel(rms_s, d2h(sc, M), 1e-3), 1e-5);

    layernorm_add_k<float,true,false><<<M,32>>>(dx_, drs, dw, db, nullptr, c, ro, nullptr, D, eps, inv_scale);
    CUCHECK(cudaDeviceSynchronize());
    mm = 0;
    { auto got = d2h(c, (size_t)M*D); for (size_t i = 0; i < got.size(); ++i) mm += got[i] != ln_c[i]; }
    report_exact("layernorm_add_fp8 codes", mm);

    layernorm_add_k<float,true,true><<<M,32>>>(dx_, drs, dw, db, nullptr, c, ro, sc, D, eps, 0);
    CUCHECK(cudaDeviceSynchronize());
    mm = 0;
    { auto got = d2h(c, (size_t)M*D); for (size_t i = 0; i < got.size(); ++i) mm += got[i] != ln_cd[i]; }
    report_exact("layernorm_add_fp8_dyn codes", mm);
    report("layernorm_add_fp8_dyn scale", maxrel(ln_s, d2h(sc, M), 1e-3), 1e-5);
}

// ---------------------------------------------------------------------------
static void test_softmax_gelu() {
    const int M = 21, D = 1024;
    auto x = randv((size_t)M*D, -4.0f, 4.0f);
    std::vector<double> ref(M*(size_t)D);
    for (int r = 0; r < M; ++r) {
        const size_t base = (size_t)r*D;
        double m = -1e300, s = 0;
        for (int j = 0; j < D; ++j) m = std::max(m, (double)x[base+j]);
        for (int j = 0; j < D; ++j) s += std::exp((double)x[base+j]-m);
        for (int j = 0; j < D; ++j) ref[base+j] = std::exp((double)x[base+j]-m)/s;
    }
    float *dx_ = dnew(x), *o = dzero<float>((size_t)M*D);
    softmax_fwd<<<M,32>>>(dx_, o, D);
    CUCHECK(cudaDeviceSynchronize());
    report("softmax_fwd", maxrel(ref, d2h(o, (size_t)M*D), 1e-4), 1e-4);

    // gelu fwd vs fp64 tanh formula; bwd vs central finite differences of that formula
    const long n = 4099;
    auto gx = randv(n, -4.0f, 4.0f);
    auto gdy = randv(n);
    auto gelu64 = [](double v) {
        const double k = 0.7978845608028654, a = 0.044715;
        return 0.5*v*(1.0+std::tanh(k*(v+a*v*v*v)));
    };
    std::vector<double> gref(n), dgref(n);
    const double h = 1e-5;
    for (long i = 0; i < n; ++i) {
        gref[i] = gelu64(gx[i]);
        dgref[i] = gdy[i] * (gelu64((double)gx[i]+h) - gelu64((double)gx[i]-h)) / (2*h);
    }
    float *dgx = dnew(gx), *dgdy = dnew(gdy), *go = dzero<float>(n), *gdx = dzero<float>(n);
    gelu_fwd<<<(unsigned)((n+255)/256),256>>>(dgx, go, n);
    gelu_bwd<<<(unsigned)((n+255)/256),256>>>(dgx, dgdy, gdx, n);
    CUCHECK(cudaDeviceSynchronize());
    report("gelu_fwd", maxrel(gref, d2h(go, n)), 2e-6);
    report("gelu_bwd (finite diff)", maxrel(dgref, d2h(gdx, n)), 1e-4);
}

// ---------------------------------------------------------------------------
// fp64 replicas of glu_eval per mode (independent transcription from the definitions).
static double glu_eval64(int mode, double a, double b, double alpha, double limit) {
    auto sig = [](double v) { return 1.0/(1.0+std::exp(-v)); };
    auto erf_as = [](double v) {   // Abramowitz-Stegun 7.1.26, same constants as the kernel
        const double P=0.3275911, A1=0.254829592, A2=-0.284496736, A3=1.421413741,
                     A4=-1.453152027, A5=1.061405429;
        const double s = v < 0 ? -1.0 : 1.0, av = std::abs(v);
        const double t = 1.0/(1.0+P*av);
        const double poly = ((((A5*t+A4)*t+A3)*t+A2)*t+A1)*t;
        return s*(1.0-poly*std::exp(-av*av));
    };
    switch (mode) {
    case 0: return a > 0 ? a*b : 0.0;
    case 1: {
        const double k = 0.79788456080286535587989211986876, c = 0.044715;
        return 0.5*a*(1.0+std::tanh(k*a*(1.0+c*a*a)))*b;
    }
    case 2: return a*sig(a)*b;
    case 3: {
        const double x0 = std::min(a, limit);
        const double x1 = std::max(std::min(b, limit), -limit);
        return x0*sig(x0*alpha)*(1.0+x1);
    }
    case 4: return 0.5*a*(1.0+erf_as(a*0.70710678118654752440084436210484))*b;
    default: return a*sig(1.702*a)*b;
    }
}

static void test_glu() {
    const long n = 2051;
    const float alpha = 1.702f, limit = 7.0f;
    auto a = randv(n, -3.0f, 3.0f), b = randv(n, -3.0f, 3.0f), dc = randv(n);
    // keep finite-difference points away from the mode-0/3 kinks
    for (long i = 0; i < n; ++i) {
        if (std::abs(a[i]) < 0.05f) a[i] += 0.1f;
        if (std::abs(std::abs(a[i]) - limit) < 0.05f) a[i] += 0.1f;
        if (std::abs(std::abs(b[i]) - limit) < 0.05f) b[i] += 0.1f;
    }
    float *da_ = dnew(a), *db_ = dnew(b), *ddc = dnew(dc);
    float *o = dzero<float>(n), *ga = dzero<float>(n), *gb = dzero<float>(n);
    const double h = 1e-5;
    for (int mode = 0; mode < 6; ++mode) {
        std::vector<double> ref(n), ra(n), rb(n);
        for (long i = 0; i < n; ++i) {
            ref[i] = glu_eval64(mode, a[i], b[i], alpha, limit);
            ra[i] = dc[i]*(glu_eval64(mode,(double)a[i]+h,b[i],alpha,limit)
                          -glu_eval64(mode,(double)a[i]-h,b[i],alpha,limit))/(2*h);
            rb[i] = dc[i]*(glu_eval64(mode,a[i],(double)b[i]+h,alpha,limit)
                          -glu_eval64(mode,a[i],(double)b[i]-h,alpha,limit))/(2*h);
        }
        glu_fwd<<<(unsigned)((n+255)/256),256>>>(da_, db_, o, n, mode, alpha, limit);
        glu_bwd<<<(unsigned)((n+255)/256),256>>>(da_, db_, ddc, ga, gb, n, mode, alpha, limit);
        CUCHECK(cudaDeviceSynchronize());
        char nm[64];
        snprintf(nm, sizeof nm, "glu_fwd mode %d", mode);
        report(nm, maxrel(ref, d2h(o, n)), 5e-6);
        snprintf(nm, sizeof nm, "glu_bwd mode %d (finite diff)", mode);
        report(nm, std::max(maxrel(ra, d2h(ga, n)), maxrel(rb, d2h(gb, n))), 2e-4);
    }
}

// ---------------------------------------------------------------------------
static void test_dropout() {
    const long n = 10007;
    const uint32_t seed = 77;
    const float p = 0.3f, inv_keep = 1.0f/(1.0f-p);
    auto x = randv(n);
    float *dx_ = dnew(x), *o = dzero<float>(n), *g = dzero<float>(n);
    dropout_fwd<<<(unsigned)((n+255)/256),256>>>(dx_, o, seed, p, inv_keep, n);
    dropout_bwd<<<(unsigned)((n+255)/256),256>>>(dx_, g, seed, p, inv_keep, n);
    CUCHECK(cudaDeviceSynchronize());
    auto ho = d2h(o, n), hg = d2h(g, n);
    long mm = 0;
    for (long i = 0; i < n; ++i) {
        const float u = rng_uniform(seed, uint32_t(i), 0u);
        const float ref = (u >= p) ? x[i]*inv_keep : 0.0f;
        mm += (ho[i] != ref) + (hg[i] != ref);
    }
    report_exact("dropout fwd+bwd (exact mask replay)", mm);
}

// ---------------------------------------------------------------------------
static void ce_ref64(const std::vector<float>& logits, const std::vector<int>& tgt,
                     int Tn, int V, int ignore_index, double eps, double z_loss,
                     double softcap, const std::vector<float>& go,
                     std::vector<double>& loss, std::vector<double>& lse,
                     std::vector<double>& grad) {
    auto cap = [&](double z) { return softcap > 0 ? softcap*std::tanh(z/softcap) : z; };
    for (int r = 0; r < Tn; ++r) {
        const size_t base = (size_t)r*V;
        if (tgt[r] == ignore_index) {
            loss[r] = 0; lse[r] = 0;
            for (int i = 0; i < V; ++i) grad[base+i] = 0;
            continue;
        }
        double m = -1e300, sx = 0;
        for (int i = 0; i < V; ++i) { const double c = cap(logits[base+i]); m = std::max(m, c); sx += c; }
        double l = 0;
        for (int i = 0; i < V; ++i) l += std::exp(cap(logits[base+i])-m);
        const double L = m + std::log(l);
        lse[r] = L;
        double ls = (1.0-eps)*(L - cap(logits[base+tgt[r]]));
        if (eps > 0) ls += eps*(L - sx/V);
        if (z_loss > 0) ls += z_loss*L*L;
        loss[r] = ls;
        const double zc = 1.0 + 2.0*z_loss*L;
        for (int i = 0; i < V; ++i) {
            const double c = cap(logits[base+i]);
            const double pp = std::exp(c-L);
            double g = zc*pp - eps/V - (1.0-eps)*(i == tgt[r] ? 1.0 : 0.0);
            if (softcap > 0) { const double t = c/softcap; g *= (1.0 - t*t); }
            grad[base+i] = g*go[r];
        }
    }
}

static void test_cross_entropy() {
    struct Cfg { const char* nm; double eps, zl, cap; };
    const Cfg cfgs[] = {
        {"plain", 0, 0, 0}, {"smooth 0.1", 0.1, 0, 0},
        {"z_loss 1e-4", 0, 1e-4, 0}, {"softcap 30", 0, 0, 30.0},
        {"all", 0.1, 1e-4, 30.0},
    };
    const int Tn = 12, V = 5000, ii = -100;
    auto logits = randv((size_t)Tn*V, -6.0f, 6.0f);
    std::vector<int> tgt(Tn);
    for (int r = 0; r < Tn; ++r) tgt[r] = (r == 3 || r == 7) ? ii : int(g_rng() % V);
    auto go = randv(Tn, 0.1f, 1.0f);
    float *dl = dnew(logits), *dgo = dnew(go);
    int *dt = dnew(tgt);
    float *loss = dzero<float>(Tn), *lse = dzero<float>(Tn), *grad = dzero<float>((size_t)Tn*V);

    for (const auto& c : cfgs) {
        std::vector<double> rl(Tn), rlse(Tn), rg((size_t)Tn*V);
        ce_ref64(logits, tgt, Tn, V, ii, c.eps, c.zl, c.cap, go, rl, rlse, rg);
        char nm[80];
        // 1-warp variants
        cross_entropy_fwd<<<Tn,32>>>(dl, dt, loss, lse, V, ii, (float)c.eps, (float)c.zl, (float)c.cap);
        CUCHECK(cudaDeviceSynchronize());
        cross_entropy_bwd<<<Tn,32>>>(dl, dt, lse, dgo, grad, V, ii, (float)c.eps, (float)c.zl, (float)c.cap);
        CUCHECK(cudaDeviceSynchronize());
        snprintf(nm, sizeof nm, "cross_entropy [%s]", c.nm);
        double e = std::max(maxrel(rl, d2h(loss, Tn)), maxrel(rlse, d2h(lse, Tn)));
        e = std::max(e, maxrel(rg, d2h(grad, (size_t)Tn*V), 1e-2));
        report(nm, e, 5e-5);
        // 4-warp variants
        CUCHECK(cudaMemset(loss, 0, Tn*4)); CUCHECK(cudaMemset(lse, 0, Tn*4));
        CUCHECK(cudaMemset(grad, 0, (size_t)Tn*V*4));
        cross_entropy_fwd_mw<<<Tn,128>>>(dl, dt, loss, lse, V, ii, (float)c.eps, (float)c.zl, (float)c.cap);
        CUCHECK(cudaDeviceSynchronize());
        cross_entropy_bwd_mw<<<Tn,128>>>(dl, dt, lse, dgo, grad, V, ii, (float)c.eps, (float)c.zl, (float)c.cap);
        CUCHECK(cudaDeviceSynchronize());
        snprintf(nm, sizeof nm, "cross_entropy_mw [%s]", c.nm);
        e = std::max(maxrel(rl, d2h(loss, Tn)), maxrel(rlse, d2h(lse, Tn)));
        e = std::max(e, maxrel(rg, d2h(grad, (size_t)Tn*V), 1e-2));
        report(nm, e, 5e-5);
    }
}

// ---------------------------------------------------------------------------
static void test_embedding() {
    const int n_tok = 97, vocab = 50, D = 129;   // odd D exercises the non-vec path
    const float scale = 1.5f;
    auto table = randv((size_t)vocab*D), pos = randv((size_t)n_tok*D), dy = randv((size_t)n_tok*D);
    std::vector<int> ids(n_tok);
    for (int t = 0; t < n_tok; ++t) {
        const int r = int(g_rng() % (vocab + 6));
        ids[t] = (r >= vocab) ? ((r & 1) ? -1 : vocab + 3) : r;   // some pad / OOB
    }
    // heavy duplication for the atomic path
    for (int t = 0; t < 20; ++t) ids[t] = 7;

    std::vector<double> ref((size_t)n_tok*D), dref((size_t)vocab*D, 0.0);
    for (int t = 0; t < n_tok; ++t) {
        const bool valid = ids[t] >= 0 && ids[t] < vocab;
        for (int d = 0; d < D; ++d) {
            double v = valid ? (double)table[(size_t)ids[t]*D+d]*scale : 0.0;
            ref[(size_t)t*D+d] = v + pos[(size_t)t*D+d];
            if (valid) dref[(size_t)ids[t]*D+d] += (double)dy[(size_t)t*D+d]*scale;
        }
    }
    int *dids = dnew(ids);
    float *dtab = dnew(table), *dpos = dnew(pos), *ddy = dnew(dy);
    float *o = dzero<float>((size_t)n_tok*D), *dt1 = dzero<float>((size_t)vocab*D),
          *dt2 = dzero<float>((size_t)vocab*D);
    embedding_lookup<<<n_tok,128>>>(dids, dtab, dpos, o, D, vocab, scale, 1);
    embedding_backward<<<n_tok,128>>>(dids, ddy, dt1, D, vocab, scale);
    CUCHECK(cudaDeviceSynchronize());
    report("embedding_lookup (+pos)", maxrel(ref, d2h(o, (size_t)n_tok*D)), 1e-6);
    report("embedding_backward (atomic)", maxrel(dref, d2h(dt1, (size_t)vocab*D)), 1e-5);

    // sorted-segment backward
    std::vector<int> perm(n_tok);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int i, int j) { return ids[i] < ids[j]; });
    std::vector<int> sids(n_tok);
    for (int i = 0; i < n_tok; ++i) sids[i] = ids[perm[i]];
    int *dsi = dnew(sids), *dpm = dnew(perm);
    embedding_backward_sorted<<<n_tok,128>>>(dsi, dpm, ddy, dt2, D, vocab, n_tok, scale);
    CUCHECK(cudaDeviceSynchronize());
    report("embedding_backward_sorted", maxrel(dref, d2h(dt2, (size_t)vocab*D)), 1e-5);

    // multimodal: build src then merge
    const int n_spans = 3, n_modal = 40;
    std::vector<int> so = {5, 30, 60}, sl = {10, 4, 13}, ms = {0, 12, 20};
    std::vector<int> src_ref(n_tok, -1);
    for (int k = 0; k < n_spans; ++k)
        for (int o2 = 0; o2 < sl[k]; ++o2) src_ref[so[k]+o2] = ms[k]+o2;
    auto text = randv((size_t)n_tok*D), modal = randv((size_t)n_modal*D);
    std::vector<double> mref((size_t)n_tok*D);
    for (int t = 0; t < n_tok; ++t)
        for (int d = 0; d < D; ++d)
            mref[(size_t)t*D+d] = src_ref[t] >= 0 ? modal[(size_t)src_ref[t]*D+d] : text[(size_t)t*D+d];
    int *dso = dnew(so), *dsl = dnew(sl), *dms = dnew(ms);
    int *dsrc = dzero<int>(n_tok);
    float *dtx = dnew(text), *dmd = dnew(modal), *mo = dzero<float>((size_t)n_tok*D);
    build_multimodal_src<<<(n_tok+127)/128,128>>>(dso, dsl, dms, dsrc, n_spans, n_tok);
    CUCHECK(cudaDeviceSynchronize());
    auto hsrc = d2h(dsrc, n_tok);
    long mm = 0;
    for (int t = 0; t < n_tok; ++t) mm += hsrc[t] != src_ref[t];
    report_exact("build_multimodal_src", mm);
    merge_multimodal_spans<<<n_tok,128>>>(dtx, dmd, dsrc, mo, D, n_modal);
    CUCHECK(cudaDeviceSynchronize());
    report("merge_multimodal_spans", maxrel(mref, d2h(mo, (size_t)n_tok*D)), 0.0);
}

// ---------------------------------------------------------------------------
static void fwht64(std::vector<double>& v, int D) {
    for (int h = 1; h < D; h <<= 1)
        for (int i = 0; i < D; ++i)
            if ((i & h) == 0) {
                const double a = v[i], b = v[i+h];
                v[i] = a + b; v[i+h] = a - b;
            }
}

template <int D, int LPR>
static void run_hadamard_case(int nrows) {
    const float scale = 1.0f / std::sqrt(float(D));
    auto x = randv((size_t)nrows*D);
    std::vector<double> ref((size_t)nrows*D);
    for (int r = 0; r < nrows; ++r) {
        std::vector<double> row(D);
        for (int j = 0; j < D; ++j) row[j] = x[(size_t)r*D+j];
        fwht64(row, D);
        for (int j = 0; j < D; ++j) ref[(size_t)r*D+j] = row[j]*scale;
    }
    float *dx_ = dnew(x), *o = dzero<float>((size_t)nrows*D);
    const int RSG = 32/LPR;
    const int rows_per_block = 4*RSG;                       // 128-thread blocks, 4 warps
    hadamard_k<float,D,LPR><<<(nrows+rows_per_block-1)/rows_per_block,128>>>(dx_, o, scale, nrows);
    CUCHECK(cudaDeviceSynchronize());
    char nm[64];
    snprintf(nm, sizeof nm, "hadamard D=%d LPR=%d", D, LPR);
    report(nm, maxrel(ref, d2h(o, (size_t)nrows*D), 8.0), 1e-6);
}

static void test_misc() {
    run_hadamard_case<64,8>(37);
    run_hadamard_case<128,16>(19);
    run_hadamard_case<256,32>(11);
    run_hadamard_case<512,32>(9);

    // adamw
    const long n = 5003;
    auto p = randv(n), g = randv(n), m = randv(n, 0.0f, 1.0f), v = randv(n, 0.0f, 1.0f);
    const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    const int step = 7;
    const float bc1 = 1.0f - std::pow(b1, step), bc2 = 1.0f - std::pow(b2, step);
    std::vector<double> rp(n), rm(n), rv(n);
    for (long i = 0; i < n; ++i) {
        const double mm = b1*(double)m[i] + (1.0-b1)*g[i];
        const double vv = b2*(double)v[i] + (1.0-b2)*(double)g[i]*g[i];
        rp[i] = p[i] - lr*((mm/bc1)/(std::sqrt(vv/bc2)+eps) + wd*p[i]);
        rm[i] = mm; rv[i] = vv;
    }
    float *dp = dnew(p), *dg = dnew(g), *dm = dnew(m), *dv = dnew(v);
    float *op = dzero<float>(n), *om = dzero<float>(n), *ov = dzero<float>(n);
    adamw_step<<<(unsigned)((n+255)/256),256>>>(dp, dg, dm, dv, op, om, ov,
                                                lr, b1, b2, eps, wd, bc1, bc2, n);
    CUCHECK(cudaDeviceSynchronize());
    double e = std::max({maxrel(rp, d2h(op, n)), maxrel(rm, d2h(om, n)), maxrel(rv, d2h(ov, n))});
    report("adamw_step", e, 1e-6);

    // add
    auto ax = randv(n), ay = randv(n);
    std::vector<double> ar(n);
    for (long i = 0; i < n; ++i) ar[i] = double(ax[i] + ay[i]);   // exact: fp32 add replayed
    float *dax = dnew(ax), *day = dnew(ay), *ao = dzero<float>(n);
    add_ew<<<(unsigned)((n+255)/256),256>>>(dax, day, ao, n);
    CUCHECK(cudaDeviceSynchronize());
    report("add_ew", maxrel(ar, d2h(ao, n)), 0.0);
}

int main() {
    test_norms();
    test_add_norm();
    test_softmax_gelu();
    test_glu();
    test_dropout();
    test_cross_entropy();
    test_embedding();
    test_misc();
    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
