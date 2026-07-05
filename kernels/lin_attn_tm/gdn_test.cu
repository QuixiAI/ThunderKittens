/**
 * @file
 * @brief fp64-oracle harness for GDN gated-deltanet (gdn_kernels.cuh). Replays
 * the exact serial recurrence on the host over varlen requests with GQA and a
 * zero-initialized paged state, compares y and the written-back state.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        gdn_test.cu -o gdn_test.out -I../serving
 * Run: CUDA_VISIBLE_DEVICES=0 ./gdn_test.out
 */
#include "gdn_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace tmgdn;
static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
template <typename T> T* dnew(const std::vector<T>& h){T*d;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template <typename T> std::vector<T> d2h(const T*d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static std::mt19937 rng(11);
static std::vector<float> rv(size_t n,float lo,float hi){std::uniform_real_distribution<float> d(lo,hi);std::vector<float> v(n);for(auto&x:v)x=d(rng);return v;}

int main() {
    const int R = 3, Hk = 2, Hv = 4, Dk = 64, Dv = 64;   // GQA 2:1
    const int lens[R] = {5, 12, 8};
    std::vector<int> cu(R + 1, 0);
    for (int r = 0; r < R; ++r) cu[r + 1] = cu[r] + lens[r];
    const int T = cu[R];
    std::vector<int> slot(R); for (int r = 0; r < R; ++r) slot[r] = r;  // distinct fresh slots

    auto q = rv((size_t)T*Hk*Dk, -0.4f, 0.4f), k = rv((size_t)T*Hk*Dk, -0.4f, 0.4f);
    auto v = rv((size_t)T*Hv*Dv, -0.4f, 0.4f);
    auto gv = rv((size_t)T*Hv, 0.85f, 0.99f), bt = rv((size_t)T*Hv, 0.1f, 0.9f);
    // sigmoid-ish gates already in (0,1)
    std::vector<float> state((size_t)R*Hv*Dv*Dk, 0.0f);   // fresh (zeroed)

    auto dq=dnew(q); auto dk=dnew(k); auto dv=dnew(v); auto dg=dnew(gv); auto db=dnew(bt);
    auto dcu=dnew(cu); auto dsl=dnew(slot); auto dstate=dnew(state);
    std::vector<float> yh((size_t)T*Hv*Dv, 0.0f); auto dy=dnew(yh);

    dim3 grid(Dv, 1, R*Hv);
    gdn_linear_attention<float><<<grid, 32>>>(dq, dk, dv, dg, db, dstate, dcu, dsl, dy,
                                              R, Hk, Hv, Dk, Dv);
    CK(cudaDeviceSynchronize());
    if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERR\n"); return 1; }
    auto y = d2h(dy, (size_t)T*Hv*Dv);
    auto st = d2h(dstate, (size_t)R*Hv*Dv*Dk);

    double yerr = 0, yden = 0, serr = 0, sden = 0;
    for (int r = 0; r < R; ++r) {
        const int s0 = cu[r];
        for (int hv = 0; hv < Hv; ++hv) {
            const int hk = hv / (Hv / Hk);
            for (int dv = 0; dv < Dv; ++dv) {
                std::vector<double> S(Dk, 0.0);
                for (int t = 0; t < lens[r]; ++t) {
                    const int tok = s0 + t;
                    const double g = gv[(size_t)tok*Hv + hv], be = bt[(size_t)tok*Hv + hv];
                    double kv = 0;
                    for (int i = 0; i < Dk; ++i) { S[i] *= g; kv += S[i]*k[((size_t)tok*Hk+hk)*Dk+i]; }
                    const double delta = (v[((size_t)tok*Hv+hv)*Dv+dv] - kv) * be;
                    double out = 0;
                    for (int i = 0; i < Dk; ++i) { S[i] += k[((size_t)tok*Hk+hk)*Dk+i]*delta;
                                                   out += S[i]*q[((size_t)tok*Hk+hk)*Dk+i]; }
                    const double ref = out;
                    const double got = y[((size_t)tok*Hv+hv)*Dv+dv];
                    yerr += std::abs(got-ref); yden += std::abs(ref);
                }
                // final state row (this dv)
                for (int i = 0; i < Dk; ++i) {
                    const double got = st[(((size_t)r*Hv+hv)*Dv+dv)*Dk+i];
                    serr += std::abs(got - S[i]); sden += std::abs(S[i]);
                }
            }
        }
    }
    const double yr = yerr/std::max(yden,1e-30), sr = serr/std::max(sden,1e-30);
    printf("gdn y     rel %.3e (%s)\n", yr, yr<2e-5?"PASS":"FAIL");
    printf("gdn state rel %.3e (%s)\n", sr, sr<2e-5?"PASS":"FAIL");
    g_fail += !(yr<2e-5) + !(sr<2e-5);
    printf("\n%s (%d failures)\n", g_fail?"FAILED":"ALL PASS", g_fail);
    return g_fail?1:0;
}
