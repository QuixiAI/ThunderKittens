/**
 * @file
 * @brief Harness for the EAGLE spec-decode core (eagle_kernels.cuh). Exact host
 * replay of greedy/random rejection, residual recovered-token argmax, and the
 * prepare-inputs index math.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        eagle_test.cu -o eagle_test.out
 * Run: CUDA_VISIBLE_DEVICES=0 ./eagle_test.out
 */
#include "eagle_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace tmeg;
static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
template <typename T> T* dnew(const std::vector<T>& h){T*d;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template <typename T> std::vector<T> d2h(const T*d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static std::mt19937 rng(42);
static void rep(const char* nm,long mm){printf("%-26s %s (%ld mismatch)\n",nm,mm?"FAIL":"PASS",mm);if(mm)++g_fail;}

int main() {
    const int B = 5, V = 500;
    std::vector<int> nd(B); std::uniform_int_distribution<int> ndd(1,4);
    for (auto& x : nd) x = ndd(rng);
    std::vector<int> cu(B); int acc=0; for (int r=0;r<B;++r){acc+=nd[r]; cu[r]=acc;}
    const int TD = cu[B-1];
    const int OUT = 8;
    std::uniform_int_distribution<int> vd(0,V-1);

    std::vector<int64_t> draft(TD), targ_argmax(TD), bonus(B);
    for (auto& x : draft) x = vd(rng);
    for (auto& x : targ_argmax) x = vd(rng);
    for (auto& x : bonus) x = vd(rng);
    // plant some accepts: make some draft==target
    for (int i=0;i<TD;i+=3) targ_argmax[i]=draft[i];

    auto dcu=dnew(cu);
    // ---- greedy ----
    { auto dd=dnew(draft); auto dta=dnew(targ_argmax); auto dbo=dnew(bonus);
      std::vector<int64_t> out((size_t)B*OUT,-1); auto dout=dnew(out);
      rejection_greedy_sample<<<(B+31)/32,32>>>(dout,dta,dd,dbo,dcu,nullptr,0,B,OUT,1);
      CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)B*OUT); long mm=0;
      for (int r=0;r<B;++r){ int start=r==0?0:cu[r-1]; std::vector<int64_t> ref(OUT,-1); bool rej=false;
          for (int p=0;p<nd[r];++p){ ref[p]=targ_argmax[start+p]; if(draft[start+p]!=targ_argmax[start+p]){rej=true;break;} }
          if(!rej) ref[nd[r]]=bonus[r];
          for (int p=0;p<=nd[r]&&p<OUT;++p) mm += o[(size_t)r*OUT+p]!=ref[p]; }
      rep("rejection_greedy", mm); }

    // ---- random ----
    std::vector<float> tprob((size_t)TD*V), dprob((size_t)TD*V), uni(TD), invq((size_t)B*V);
    { std::uniform_real_distribution<float> ud(0,1);
      for (auto& x : tprob) x = ud(rng); for (auto& x : dprob) x = ud(rng);
      for (auto& x : uni) x = ud(rng); for (auto& x : invq) x = ud(rng)+0.1f; }
    std::vector<int64_t> recov(TD); for (auto& x : recov) x = vd(rng);
    { auto dtp=dnew(tprob); auto ddp=dnew(dprob); auto dd=dnew(draft); auto dbo=dnew(bonus);
      auto drc=dnew(recov); auto dun=dnew(uni);
      std::vector<int64_t> out((size_t)B*OUT,-1); auto dout=dnew(out);
      rejection_random_sample<<<(B+31)/32,32>>>(dout,dtp,ddp,dd,dbo,drc,dun,dcu,nullptr,0,0,B,OUT,1,V,V);
      CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)B*OUT); long mm=0;
      for (int r=0;r<B;++r){ int start=r==0?0:cu[r-1]; std::vector<int64_t> ref(OUT,-1); bool rej=false;
          for (int p=0;p<nd[r];++p){ int tok=start+p; int64_t did=draft[tok];
              float pt=tprob[(size_t)tok*V+did], q=dprob[(size_t)tok*V+did], ratio=q>0?pt/q:0;
              if(ratio>=uni[tok]) ref[p]=did; else { ref[p]=recov[tok]; rej=true; break; } }
          if(!rej) ref[nd[r]]=bonus[r];
          for (int p=0;p<=nd[r]&&p<OUT;++p) mm += o[(size_t)r*OUT+p]!=ref[p]; }
      rep("rejection_random", mm); }

    // ---- recovered tokens (residual argmax) ----
    { auto dtp=dnew(tprob); auto ddp=dnew(dprob); auto dd=dnew(draft); auto diq=dnew(invq);
      std::vector<int64_t> out(TD,-1); auto dout=dnew(out);
      sample_recovered_tokens<<<TD,32>>>(dout,dtp,ddp,dd,diq,dcu,0,B,TD,V,V,V,V);
      CK(cudaDeviceSynchronize());
      auto o=d2h(dout,TD); long mm=0;
      for (int tok=0;tok<TD;++tok){ int r=0; while (cu[r]<=tok) ++r;
          float best=-1; int bi=V;
          for (int v=0;v<V;++v){ float d=tprob[(size_t)tok*V+v]-dprob[(size_t)tok*V+v]; float prob=d>0?d:0;
              float val=prob*invq[(size_t)r*V+v]; if(val>best||(val==best&&v<bi)){best=val;bi=v;} }
          mm += o[tok]!=bi; }
      rep("recovered_tokens", mm); }

    // ---- prepare inputs ----
    { std::vector<int> valid(B), qsl(B+1,0);
      for (int r=0;r<B;++r){ valid[r]=std::uniform_int_distribution<int>(0,nd[r]+1)(rng); qsl[r+1]=qsl[r]+10; }
      auto dva=dnew(valid); auto dqsl=dnew(qsl);
      std::vector<int> tis(B,-99), nrt(B,-99); auto dtis=dnew(tis); auto dnrt=dnew(nrt);
      eagle_prepare_inputs_padded<<<(B+31)/32,32>>>(dtis,dnrt,dcu,dva,dqsl,B);
      CK(cudaDeviceSynchronize());
      auto ti=d2h(dtis,B); auto nr=d2h(dnrt,B); long mm=0;
      for (int r=0;r<B;++r){ int rej = nd[r]>0 ? nd[r]+1-valid[r] : 0; int qlast=qsl[r+1]-1;
          mm += (ti[r]!=qlast-rej) + (nr[r]!=rej); }
      rep("eagle_prepare_inputs", mm); }

    printf("\n%s (%d failures)\n", g_fail?"FAILED":"ALL PASS", g_fail);
    return g_fail?1:0;
}
