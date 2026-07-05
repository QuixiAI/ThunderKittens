/**
 * @file
 * @brief Harness for M6 (turboquant.cuh): random-sign FWHT rotation (self-inverse
 * + parity vs a host FWHT-with-signs), permute_cols gather, per-LoRA MoE align.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        turboquant_test.cu -o turboquant_test.out
 * Run: CUDA_VISIBLE_DEVICES=0 ./turboquant_test.out
 */
#include "turboquant.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace tm6;
static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
template <typename T> T* dnew(const std::vector<T>& h){T*d;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template <typename T> T* dz(size_t n){T*d;CK(cudaMalloc(&d,n*sizeof(T)));CK(cudaMemset(d,0,n*sizeof(T)));return d;}
template <typename T> std::vector<T> d2h(const T*d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static std::mt19937 rng(61);
static std::vector<float> rv(size_t n,float lo,float hi){std::uniform_real_distribution<float> d(lo,hi);std::vector<float> v(n);for(auto&x:v)x=d(rng);return v;}
static void rep_rel(const char* nm,double e,double tol){printf("%-32s %s (rel %.3e)\n",nm,e<=tol?"PASS":"FAIL",e);if(e>tol)++g_fail;}
static void rep_ex(const char* nm,long mm){printf("%-32s %s (%ld mismatch)\n",nm,mm?"FAIL":"PASS",mm);if(mm)++g_fail;}

template <int D>
static void test_fwht() {
    const int R = 20;
    auto x = rv((size_t)R*D, -2, 2);
    std::vector<float> sign(D); std::uniform_int_distribution<int> b(0,1);
    for (auto& s : sign) s = b(rng) ? 1.0f : -1.0f;
    auto dx=dnew(x); auto dsg=dnew(sign);
    float* dfwd=dz<float>((size_t)R*D); float* dinv=dz<float>((size_t)R*D);
    fwht_rotate<float,D,false><<<R,32>>>(dx,dfwd,dsg,R); CK(cudaDeviceSynchronize());
    fwht_rotate<float,D,true><<<R,32>>>(dfwd,dinv,dsg,R); CK(cudaDeviceSynchronize());
    auto fwd=d2h(dfwd,(size_t)R*D), inv=d2h(dinv,(size_t)R*D);
    // (a) self-inverse: inverse(forward(x)) == x
    double gs=0,rs=0;
    for (size_t i=0;i<x.size();++i){ gs+=std::abs(inv[i]-x[i]); rs+=std::abs(x[i]); }
    char nm[40]; snprintf(nm,sizeof nm,"fwht_rotate D=%d self-inverse",D);
    rep_rel(nm, gs/std::max(rs,1e-30), 1e-4);
    // (b) forward parity vs a host FWHT-with-signs
    gs=0; rs=0;
    for (int r=0;r<R;++r){ std::vector<double> row(D);
        for(int i=0;i<D;++i) row[i]=x[(size_t)r*D+i]*sign[i];
        for(int h=1;h<D;h<<=1) for(int i=0;i<D;++i) if((i&h)==0){ double a=row[i],b2=row[i+h]; row[i]=a+b2; row[i+h]=a-b2; }
        const double isq=1.0/std::sqrt((double)D);
        for(int i=0;i<D;++i){ double ref=row[i]*isq; gs+=std::abs(fwd[(size_t)r*D+i]-ref); rs+=std::abs(ref); } }
    snprintf(nm,sizeof nm,"fwht_rotate D=%d fwd parity",D);
    rep_rel(nm, gs/std::max(rs,1e-30), 2e-5);
}

int main() {
    test_fwht<64>();
    test_fwht<128>();
    test_fwht<256>();
    test_fwht<512>();

    // ---- permute_cols ----
    { const int R=40, C=256; auto x=rv((size_t)R*C,-1,1);
      std::vector<uint16_t> xh((size_t)R*C); for(size_t i=0;i<xh.size();++i){ __half h=__float2half(x[i]); xh[i]=*reinterpret_cast<uint16_t*>(&h); }
      std::vector<int> perm(C); for(int i=0;i<C;++i)perm[i]=i; std::shuffle(perm.begin(),perm.end(),rng);
      auto dx=dnew(xh); auto dp=dnew(perm); uint16_t* do_=dz<uint16_t>((size_t)R*C);
      dim3 g((C+255)/256,R); permute_cols<uint16_t><<<g,256>>>(dx,do_,dp,R,C); CK(cudaDeviceSynchronize());
      auto o=d2h(do_,(size_t)R*C); long mm=0;
      for(int r=0;r<R;++r) for(int c=0;c<C;++c) mm += o[(size_t)r*C+c]!=xh[(size_t)r*C+perm[c]];
      rep_ex("permute_cols", mm); }

    // ---- moe_lora_align ----
    { const int max_loras=3, E=8, topk=2, block=4, T=24;
      const int assignments=T*topk;
      const int scap=64, ecap=32;
      std::vector<int> topk_ids(assignments); std::uniform_int_distribution<int> ed(0,E-1);
      for(auto&x:topk_ids)x=ed(rng);
      std::vector<int> tlm(T); std::uniform_int_distribution<int> ld(0,max_loras-1);
      for(auto&x:tlm)x=ld(rng);
      std::vector<int> lora_ids(max_loras); for(int i=0;i<max_loras;++i)lora_ids[i]=i;
      std::vector<uint8_t> aen(max_loras,1); aen[1]=0;   // disable lora 1
      auto dti=dnew(topk_ids); auto dtlm=dnew(tlm); auto dli=dnew(lora_ids); auto dae=dnew(aen);
      int* dsorted=dz<int>((size_t)max_loras*scap); int* dexp=dz<int>((size_t)max_loras*ecap);
      int* dntp=dz<int>(max_loras);
      moe_lora_align<<<max_loras,128>>>(dti,dtlm,dli,dae,dsorted,dexp,dntp,max_loras,E,assignments,topk,block,scap,ecap);
      CK(cudaDeviceSynchronize());
      if (cudaGetLastError()!=cudaSuccess){printf("LORA KERNEL ERR\n");return 1;}
      auto sorted=d2h(dsorted,(size_t)max_loras*scap); auto expid=d2h(dexp,(size_t)max_loras*ecap); auto ntp=d2h(dntp,max_loras);
      long mm=0;
      for(int L=0;L<max_loras;++L){
          if(!aen[L]){ mm += ntp[L]!=0; continue; }
          // host reference: count per expert (filtered), padded offsets, verify counts
          std::vector<int> cnt(E,0);
          for(int i=0;i<assignments;++i){ if(tlm[i/topk]!=L)continue; int e=topk_ids[i]; if(e<0||e>=E)e=E-1; cnt[e]++; }
          int pad=0; std::vector<int> off(E+1);
          for(int e=0;e<E;++e){ off[e]=pad; pad += ((cnt[e]+block-1)/block)*block; } off[E]=pad;
          mm += ntp[L]!=pad;
          // each expert's sorted slice [off[e], off[e]+cnt[e]) holds its routing rows; verify count by expert
          std::vector<int> seen(E,0);
          for(int e=0;e<E;++e) for(int p=off[e];p<off[e+1];++p){ int val=sorted[(size_t)L*scap+p];
              if(val<assignments){ int se=topk_ids[val]; if(se<0||se>=E)se=E-1; if(se==e)seen[e]++; } }
          for(int e=0;e<E;++e) mm += seen[e]!=cnt[e];
          // expert_ids: each padded block labeled with its expert
          for(int e=0;e<E;++e){ int sb=off[e]/block, nb=(off[e+1]-off[e])/block;
              for(int b=0;b<nb&&sb+b<ecap;++b) mm += expid[(size_t)L*ecap+sb+b]!=e; }
      }
      rep_ex("moe_lora_align", mm); }

    printf("\n%s (%d failures)\n", g_fail?"FAILED":"ALL PASS", g_fail);
    return g_fail?1:0;
}
