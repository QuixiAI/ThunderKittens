/**
 * @file
 * @brief fp64-oracle harness for the varlen Mamba-1 selective scan
 * (selective_scan_kernels.cuh). Replays the ragged recurrence on the host over
 * paged state slots; exercises softplus, D, delta_bias, z-gate, n_groups.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        selective_scan_test.cu -o selective_scan_test.out -I../serving
 * Run: CUDA_VISIBLE_DEVICES=0 ./selective_scan_test.out
 */
#include "selective_scan_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace tmss;
static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
template <typename T> T* dnew(const std::vector<T>& h){T*d;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template <typename T> std::vector<T> d2h(const T*d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static std::mt19937 rng(21);
static std::vector<float> rv(size_t n,float lo,float hi){std::uniform_real_distribution<float> d(lo,hi);std::vector<float> v(n);for(auto&x:v)x=d(rng);return v;}

int main() {
    const int B = 3, dim = 16, dstate = 16, ngroups = 4;
    const int lens[B] = {6, 10, 4};
    std::vector<int> qsl(B+1,0); for (int b=0;b<B;++b) qsl[b+1]=qsl[b]+lens[b];
    const int TT = qsl[B];
    std::vector<int> cache(B); for (int b=0;b<B;++b) cache[b]=b;   // distinct slots
    std::vector<uint8_t> hinit(B, 0);                              // fresh

    auto u = rv((size_t)dim*TT, -0.5f, 0.5f), delta = rv((size_t)dim*TT, -1.0f, 1.0f);
    auto A = rv((size_t)dim*dstate, -1.0f, -0.1f);                 // negative for stability
    auto Bt = rv((size_t)ngroups*dstate*TT, -0.5f, 0.5f), Ct = rv((size_t)ngroups*dstate*TT, -0.5f, 0.5f);
    auto D = rv(dim, -0.5f, 0.5f), dbias = rv(dim, -0.2f, 0.2f), z = rv((size_t)dim*TT, -1.0f, 1.0f);
    std::vector<float> state((size_t)B*dim*dstate, 0.0f);

    auto du=dnew(u); auto dd=dnew(delta); auto dA=dnew(A); auto dB=dnew(Bt); auto dC=dnew(Ct);
    auto dD=dnew(D); auto dbi=dnew(dbias); auto dz=dnew(z);
    auto dqsl=dnew(qsl); auto dcache=dnew(cache); auto dhi=dnew(hinit); auto dstate_=dnew(state);
    std::vector<float> outh((size_t)dim*TT, 0.0f); auto dout=dnew(outh);

    const int P = ((dstate+31)/32)*32;
    selective_scan_fwd_varlen<float><<<dim3(dim,B), P, ((P+31)/32)*4>>>(
        du, dd, dA, dB, dC, dD, dbi, dz, dqsl, dcache, dhi, dout, dstate_,
        B, dim, TT, dstate, ngroups, 1, 1, 1, 1, 1, 1, -1);
    CK(cudaDeviceSynchronize());
    if (cudaGetLastError() != cudaSuccess) { printf("KERNEL ERR\n"); return 1; }
    auto out = d2h(dout, (size_t)dim*TT);
    auto st = d2h(dstate_, (size_t)B*dim*dstate);

    double oerr=0, oden=0, serr=0, sden=0;
    for (int b=0;b<B;++b) for (int d=0;d<dim;++d) {
        const int grp = d/(dim/ngroups);
        std::vector<double> S(dstate, 0.0);
        for (int t=qsl[b]; t<qsl[b+1]; ++t) {
            const double uv = u[(size_t)d*TT+t];
            double dv = delta[(size_t)d*TT+t] + dbias[d];
            dv = dv<=20.0 ? std::log(1.0+std::exp(dv)) : dv;
            double sum = D[d]*uv;
            for (int s=0;s<dstate;++s) {
                const double bc_B = Bt[((size_t)grp*dstate+s)*TT+t], bc_C = Ct[((size_t)grp*dstate+s)*TT+t];
                S[s] = std::exp(dv*A[(size_t)d*dstate+s])*S[s] + bc_B*dv*uv;
                sum += S[s]*bc_C;
            }
            const double zv = z[(size_t)d*TT+t];
            sum *= zv/(1.0+std::exp(-zv));
            const double got = out[(size_t)d*TT+t];
            oerr += std::abs(got-sum); oden += std::abs(sum);
        }
        for (int s=0;s<dstate;++s) {
            const double got = st[((size_t)b*dim+d)*dstate+s];
            serr += std::abs(got-S[s]); sden += std::abs(S[s]);
        }
    }
    const double orr=oerr/std::max(oden,1e-30), srr=serr/std::max(sden,1e-30);
    printf("selective_scan_varlen out   rel %.3e (%s)\n", orr, orr<5e-6?"PASS":"FAIL");
    printf("selective_scan_varlen state rel %.3e (%s)\n", srr, srr<5e-6?"PASS":"FAIL");
    g_fail += !(orr<5e-6) + !(srr<5e-6);
    printf("\n%s (%d failures)\n", g_fail?"FAILED":"ALL PASS", g_fail);
    return g_fail?1:0;
}
