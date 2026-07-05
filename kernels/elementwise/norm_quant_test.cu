/**
 * @file
 * @brief Harness for M2 norm+quant / AZP / group-int8 (tm_norm_quant_kernels.cuh).
 * Round-trips each quantizer: dequantize the codes with the emitted scale (+zp)
 * and compare to an fp64 reference of the (rms-normed / raw) values, within the
 * format's quantization precision.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        norm_quant_test.cu -o norm_quant_test.out -I../quant -I../serving
 * Run: CUDA_VISIBLE_DEVICES=0 ./norm_quant_test.out
 */
#include "tm_norm_quant_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace tmnq;
static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
template <typename T> T* dnew(const std::vector<T>& h){T*d;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template <typename T> T* dz(size_t n){T*d;CK(cudaMalloc(&d,n*sizeof(T)));CK(cudaMemset(d,0,n*sizeof(T)));return d;}
template <typename T> std::vector<T> d2h(const T*d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static void report(const char* nm, double e, double tol){bool ok=e<=tol;printf("%-40s %s (rel %.3e, tol %.1e)\n",nm,ok?"PASS":"FAIL",e,tol);if(!ok)++g_fail;}
static std::mt19937 rng(5);
static std::vector<float> rv(size_t n,float lo,float hi){std::uniform_real_distribution<float> d(lo,hi);std::vector<float> v(n);for(auto&x:v)x=d(rng);return v;}
static double e4dec(uint8_t v){float m; if(v&0x78){int e=(v>>3)&0xF,mm=v&7;m=std::ldexp(1.0f+mm/8.0f,e-7);}else m=float(v&7)*0.001953125f; return (v&0x80)?-m:m;}

int main() {
    const int M = 40, D = 768;
    const float eps = 1e-5f;
    auto x = rv((size_t)M*D, -2, 2), res = rv((size_t)M*D, -2, 2), w = rv(D, -1, 1);
    std::vector<__half> xh((size_t)M*D), rh((size_t)M*D), wh(D);
    for (size_t i=0;i<xh.size();++i){xh[i]=__float2half(x[i]);rh[i]=__float2half(res[i]);}
    for (int i=0;i<D;++i) wh[i]=__float2half(w[i]);
    auto dx=dnew(xh); auto dr=dnew(rh); auto dw=dnew(wh);

    // reference rms-normed*weight per row (non-residual and residual)
    auto rmsref = [&](bool resid, std::vector<double>& out){
        out.resize((size_t)M*D);
        for (int r=0;r<M;++r){ double ss=0; std::vector<double> v(D);
            for (int j=0;j<D;++j){ v[j]=x[(size_t)r*D+j]+(resid?res[(size_t)r*D+j]:0.0); ss+=v[j]*v[j]; }
            double inv=1.0/std::sqrt(ss/D+eps);
            for (int j=0;j<D;++j) out[(size_t)r*D+j]=v[j]*inv*w[j]; }
    };

    // ---- rms_norm_quant fp8/int8 x dynamic, non-residual ----
    for (int fp8 = 1; fp8 >= 0; --fp8) {
        std::vector<double> ref; rmsref(false, ref);
        uint8_t* dc = dz<uint8_t>((size_t)M*D); float* dsc = dz<float>(M);
        if (fp8) rms_norm_quant<half,true,true,false><<<M,32>>>(dc,dsc,nullptr,
            reinterpret_cast<const half*>(dx),nullptr,reinterpret_cast<const half*>(dw),D,eps,0);
        else rms_norm_quant<half,false,true,false><<<M,32>>>(dc,dsc,nullptr,
            reinterpret_cast<const half*>(dx),nullptr,reinterpret_cast<const half*>(dw),D,eps,0);
        CK(cudaDeviceSynchronize());
        auto cd=d2h(dc,(size_t)M*D); auto sc=d2h(dsc,M);
        double gs=0,rs=0;
        for (int r=0;r<M;++r) for (int j=0;j<D;++j){
            double got = fp8 ? e4dec(cd[(size_t)r*D+j])*sc[r]
                             : (double)int8_t(cd[(size_t)r*D+j])*sc[r];
            gs+=std::abs(got-ref[(size_t)r*D+j]); rs+=std::abs(ref[(size_t)r*D+j]); }
        report(fp8?"rms_norm_quant fp8 dyn (rt)":"rms_norm_quant int8 dyn (rt)",
               gs/std::max(rs,1e-30), fp8?6e-2:1.5e-2);
    }

    // ---- rms_norm_add_quant int8 dynamic (residual, checks res_out too) ----
    {
        std::vector<double> ref; rmsref(true, ref);
        uint8_t* dc = dz<uint8_t>((size_t)M*D); float* dsc = dz<float>(M);
        __half* dro = dz<__half>((size_t)M*D);
        rms_norm_quant<half,false,true,true><<<M,32>>>(dc,dsc,dro,
            reinterpret_cast<const half*>(dx),reinterpret_cast<const half*>(dr),
            reinterpret_cast<const half*>(dw),D,eps,0);
        CK(cudaDeviceSynchronize());
        auto cd=d2h(dc,(size_t)M*D); auto sc=d2h(dsc,M); auto ro=d2h(dro,(size_t)M*D);
        double gs=0,rs=0,roerr=0;
        for (int r=0;r<M;++r) for (int j=0;j<D;++j){
            double got=(double)int8_t(cd[(size_t)r*D+j])*sc[r];
            gs+=std::abs(got-ref[(size_t)r*D+j]); rs+=std::abs(ref[(size_t)r*D+j]);
            roerr=std::max(roerr,(double)std::abs(__half2float(ro[(size_t)r*D+j])-(x[(size_t)r*D+j]+res[(size_t)r*D+j]))); }
        report("rms_norm_add_quant int8 dyn (rt)", gs/std::max(rs,1e-30), 1.5e-2);
        report("  res_out (x+residual)", roerr, 2e-2);
    }

    // ---- AZP int8 dynamic (round-trip) ----
    {
        int8_t* dc = dz<int8_t>((size_t)M*D); float* dsc = dz<float>(M); int* daz = dz<int>(M);
        azp_int8_quant<half,true><<<M,32>>>(dc,dsc,daz,reinterpret_cast<const half*>(dx),D,0,0);
        CK(cudaDeviceSynchronize());
        auto cd=d2h(dc,(size_t)M*D); auto sc=d2h(dsc,M); auto az=d2h(daz,M);
        double gs=0,rs=0;
        for (int r=0;r<M;++r) for (int j=0;j<D;++j){
            double got=((double)cd[(size_t)r*D+j]-az[r])*sc[r];
            gs+=std::abs(got-x[(size_t)r*D+j]); rs+=std::abs(x[(size_t)r*D+j]); }
        report("azp_int8_quant dyn (rt)", gs/std::max(rs,1e-30), 1.2e-2);
    }

    // ---- per-token-group int8 (round-trip) ----
    {
        const int GS=128, NG=D/GS;
        int8_t* dc = dz<int8_t>((size_t)M*D); float* dsc = dz<float>((size_t)M*NG);
        dim3 g(NG,M);
        per_token_group_int8_quant<half><<<g,32>>>(dc,dsc,reinterpret_cast<const half*>(dx),D,GS,NG,1e-6f);
        CK(cudaDeviceSynchronize());
        auto cd=d2h(dc,(size_t)M*D); auto sc=d2h(dsc,(size_t)M*NG);
        double gs=0,rs=0;
        for (int r=0;r<M;++r) for (int j=0;j<D;++j){
            double got=(double)cd[(size_t)r*D+j]*sc[(size_t)r*NG+j/GS];
            gs+=std::abs(got-x[(size_t)r*D+j]); rs+=std::abs(x[(size_t)r*D+j]); }
        report("per_token_group_int8 (rt)", gs/std::max(rs,1e-30), 1.5e-2);
    }

    printf("\n%s (%d failures)\n", g_fail?"FAILED":"ALL PASS", g_fail);
    return g_fail?1:0;
}
