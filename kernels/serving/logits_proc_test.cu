/**
 * @file
 * @brief Harness for the M4 logits-processor zoo (logits_proc_kernels.cuh).
 * Masks (nsigma/top_a/epsilon/eta/xtc/ngram/dry) are exact-replayed on the host
 * (-inf vs passthrough must match); transforms (quadratic/skew) vs fp64.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        logits_proc_test.cu -o logits_proc_test.out
 * Run: CUDA_VISIBLE_DEVICES=0 ./logits_proc_test.out
 */
#include "logits_proc_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace tmlp;
static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
template <typename T> T* dnew(const std::vector<T>& h){T*d;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template <typename T> std::vector<T> d2h(const T*d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static std::mt19937 rng(31);
static std::vector<float> rv(size_t n,float lo,float hi){std::uniform_real_distribution<float> d(lo,hi);std::vector<float> v(n);for(auto&x:v)x=d(rng);return v;}
static bool isninf(float x){ return x < -1e30f; }
static void rep_exact(const char* nm, long mm){ printf("%-28s %s (%ld mismatch)\n", nm, mm?"FAIL":"PASS", mm); if(mm)++g_fail; }
static void rep_rel(const char* nm, double e, double tol){ printf("%-28s %s (rel %.3e)\n", nm, e<=tol?"PASS":"FAIL", e); if(e>tol)++g_fail; }

int main() {
    const int R = 6, V = 4000;
    auto lg = rv((size_t)R*V, -8, 8);
    auto dlg = dnew(lg);
    std::vector<float> out((size_t)R*V, 0.0f); auto dout = dnew(out);
    auto param = [&](float lo,float hi){ return dnew(rv(R, lo, hi)); };

    // top_nsigma
    { auto ns = rv(R, 0.5f, 2.5f); auto dns = dnew(ns);
      top_nsigma_mask<<<R,32>>>(dlg,dout,V,dns); CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)R*V); long mm=0;
      for (int r=0;r<R;++r){ double mx=-1e30,sx=0,sxx=0;
          for (int i=0;i<V;++i){double v=lg[(size_t)r*V+i];mx=std::max(mx,v);sx+=v;sxx+=v*v;}
          double var=std::max((sxx-sx*sx/V)/(V-1),0.0), thr=mx-ns[r]*std::sqrt(var);
          for (int i=0;i<V;++i){bool m=lg[(size_t)r*V+i]<thr; mm+=isninf(o[(size_t)r*V+i])!=m;} }
      rep_exact("top_nsigma", mm); }

    // top_a
    { auto ta = rv(R, 0.05f, 0.3f); auto dta = dnew(ta);
      top_a_mask<<<R,32>>>(dlg,dout,V,dta); CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)R*V); long mm=0;
      for (int r=0;r<R;++r){ double mx=-1e30; for(int i=0;i<V;++i)mx=std::max(mx,(double)lg[(size_t)r*V+i]);
          double Z=0; for(int i=0;i<V;++i)Z+=std::exp(lg[(size_t)r*V+i]-mx);
          double tp=1.0/Z, thr=tp*tp*ta[r];
          for(int i=0;i<V;++i){double p=std::exp(lg[(size_t)r*V+i]-mx)/Z; mm+=isninf(o[(size_t)r*V+i])!=(p<thr);} }
      rep_exact("top_a", mm); }

    // epsilon
    { auto ep = rv(R, 1e-4f, 5e-3f); auto dep = dnew(ep);
      epsilon_cutoff_mask<<<R,32>>>(dlg,dout,V,dep); CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)R*V); long mm=0;
      for (int r=0;r<R;++r){ double mx=-1e30; for(int i=0;i<V;++i)mx=std::max(mx,(double)lg[(size_t)r*V+i]);
          double Z=0; for(int i=0;i<V;++i)Z+=std::exp(lg[(size_t)r*V+i]-mx); Z=std::max(Z,1e-20);
          for(int i=0;i<V;++i){double v=lg[(size_t)r*V+i],p=std::exp(v-mx)/Z; bool m=(v<mx&&p<ep[r]); mm+=isninf(o[(size_t)r*V+i])!=m;} }
      rep_exact("epsilon_cutoff", mm); }

    // eta
    { auto et = rv(R, 1e-3f, 0.1f); auto det = dnew(et);
      eta_cutoff_mask<<<R,32>>>(dlg,dout,V,det); CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)R*V); long mm=0;
      for (int r=0;r<R;++r){ double mx=-1e30; for(int i=0;i<V;++i)mx=std::max(mx,(double)lg[(size_t)r*V+i]);
          double Z=0; for(int i=0;i<V;++i)Z+=std::exp(lg[(size_t)r*V+i]-mx); Z=std::max(Z,1e-20); double lZ=std::log(Z);
          double negH=0; for(int i=0;i<V;++i){double sh=lg[(size_t)r*V+i]-mx,p=std::exp(sh)/Z; if(p>0)negH+=p*(sh-lZ);}
          double e=et[r], eps=std::min((double)e, std::sqrt((double)e)*std::exp(negH));
          // exact match may differ at the eps boundary in fp32; count only clear mismatches
          for(int i=0;i<V;++i){double v=lg[(size_t)r*V+i],p=std::exp(v-mx)/Z; bool m=(v<mx&&p<eps);
              bool gm=isninf(o[(size_t)r*V+i]); if (gm!=m && std::abs(p-eps)>1e-6) mm++; } }
      rep_exact("eta_cutoff", mm); }

    // xtc
    { auto xt = rv(R, 0.05f, 0.2f); auto dxt = dnew(xt);
      std::vector<int> ax(R,1); ax[0]=0; auto dax=dnew(ax);
      xtc_mask<<<R,32>>>(dlg,dout,V,dxt,dax); CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)R*V); long mm=0;
      for (int r=0;r<R;++r){ if(!ax[r]){ for(int i=0;i<V;++i) mm+=(o[(size_t)r*V+i]!=lg[(size_t)r*V+i]); continue; }
          double mx=-1e30; for(int i=0;i<V;++i)mx=std::max(mx,(double)lg[(size_t)r*V+i]);
          double Z=0; for(int i=0;i<V;++i)Z+=std::exp(lg[(size_t)r*V+i]-mx);
          int cnt=0; double kp=1e30;
          for(int i=0;i<V;++i){double p=std::exp(lg[(size_t)r*V+i]-mx)/Z; if(p>=xt[r]){cnt++; kp=std::min(kp,p);}}
          for(int i=0;i<V;++i){double p=std::exp(lg[(size_t)r*V+i]-mx)/Z; bool rem=cnt>1&&p>=xt[r]&&p>kp;
              mm+=isninf(o[(size_t)r*V+i])!=rem;} }
      rep_exact("xtc", mm); }

    // quadratic (transform vs fp64)
    { auto fa = rv(R, 0.1f, 0.9f); auto cu = rv(R, 0.5f, 2.0f); auto dfa=dnew(fa); auto dcu=dnew(cu);
      quadratic_transform<<<R,32>>>(dlg,dout,V,dfa,dcu); CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)R*V); double gs=0,rs=0;
      for (int r=0;r<R;++r){ double mx=-1e30; for(int i=0;i<V;++i)mx=std::max(mx,(double)lg[(size_t)r*V+i]);
          double k=fa[r]*(3-cu[r])*0.5, s=fa[r]*(cu[r]-1)*0.5;
          for(int i=0;i<V;++i){double v=lg[(size_t)r*V+i],diff=v-mx; diff-=diff*diff*(s*diff-k);
              double ref=std::isfinite(diff)?v-diff:v; gs+=std::abs(o[(size_t)r*V+i]-ref); rs+=std::abs(ref);} }
      rep_rel("quadratic", gs/std::max(rs,1e-30), 1e-4); }

    // skew (on probs, vs fp64 CDF^exp)
    { std::vector<float> pr((size_t)R*V);              // softmax of the logits per row
      for (int r=0;r<R;++r){ double mx=-1e30; for(int i=0;i<V;++i)mx=std::max(mx,(double)lg[(size_t)r*V+i]);
          double Z=0; for(int i=0;i<V;++i)Z+=std::exp(lg[(size_t)r*V+i]-mx);
          for(int i=0;i<V;++i)pr[(size_t)r*V+i]=std::exp(lg[(size_t)r*V+i]-mx)/Z; }
      auto dpr=dnew(pr); auto sk=rv(R,-1.0f,1.0f); auto dsk=dnew(sk);
      skew_transform<<<R,256,((256/32)*sizeof(float))>>>(dpr,dout,V,dsk); CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)R*V); double gs=0,rs=0;
      for (int r=0;r<R;++r){ double ex=std::exp(sk[r]), cum=0, prev=0;
          for(int i=0;i<V;++i){ cum+=pr[(size_t)r*V+i]; double tr=std::pow(cum,ex); double ref=tr-prev; prev=tr;
              gs+=std::abs(o[(size_t)r*V+i]-ref); rs+=std::abs(ref);} }
      rep_rel("skew", gs/std::max(rs,1e-30), 1e-4); }

    // no_repeat_ngram (exact)
    { const int HS=32, ngram=3; std::vector<int> hist((size_t)R*HS,-1), hl(R);
      std::uniform_int_distribution<int> td(0,50);
      for (int r=0;r<R;++r){ hl[r]=20+r; for(int t=0;t<hl[r];++t) hist[(size_t)r*HS+t]=td(rng);
          // plant a repeat: make the suffix (n-1) appear earlier
          if (hl[r]>=6){ hist[(size_t)r*HS+2]=hist[(size_t)r*HS+hl[r]-2]; hist[(size_t)r*HS+3]=hist[(size_t)r*HS+hl[r]-1]; } }
      auto dh=dnew(hist); auto dhl=dnew(hl);
      no_repeat_ngram_mask<<<R,32>>>(dlg,dout,V,dh,dhl,HS,ngram); CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)R*V); long mm=0;
      for (int r=0;r<R;++r){ std::vector<char> banned(V,0); const int H=hl[r],p=ngram-1;
          if (H>=p) for(int s=0;s+p<H;++s){ bool m=true; for(int j=0;j<p;++j) if(hist[(size_t)r*HS+s+j]!=hist[(size_t)r*HS+H-p+j]){m=false;break;}
              if(m){int b=hist[(size_t)r*HS+s+p]; if(b>=0&&b<V)banned[b]=1;} }
          for(int i=0;i<V;++i){ bool gb=isninf(o[(size_t)r*V+i]); mm+=gb!=(bool)banned[i]; } }
      rep_exact("no_repeat_ngram", mm); }

    // dry_penalty (host replay of the kernel's algorithm)
    { const int HS=32, max_ngram=6, max_occ=8; std::vector<int> hist((size_t)R*HS,-1), hl(R);
      std::uniform_int_distribution<int> td(0,40);
      for (int r=0;r<R;++r){ hl[r]=18+r; for(int t=0;t<hl[r];++t) hist[(size_t)r*HS+t]=td(rng);
          if (hl[r]>=8){ hist[(size_t)r*HS+3]=hist[(size_t)r*HS+hl[r]-1];   // plant a suffix match
                         hist[(size_t)r*HS+2]=hist[(size_t)r*HS+hl[r]-2]; } }
      auto mult=rv(R,0.5f,1.5f); auto bs=rv(R,1.1f,1.8f); std::vector<int> allow(R,2); auto dal=dnew(allow);
      auto dh=dnew(hist); auto dhl=dnew(hl); auto dm=dnew(mult); auto dbs=dnew(bs);
      dry_penalty<<<R,32>>>(dlg,dout,V,dh,dhl,HS,nullptr,dm,dbs,dal,max_ngram,max_occ);
      CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)R*V); long mm=0;
      for (int r=0;r<R;++r){ std::vector<float> ref(V); for(int i=0;i<V;++i)ref[i]=lg[(size_t)r*V+i];
          const int H=hl[r]; const int last=hist[(size_t)r*HS+H-1]; int occ=0;
          for (int p=H-2;p>=0&&occ<max_occ;--p){ if(hist[(size_t)r*HS+p]!=last)continue;
              int ml=1; while(ml<max_ngram&&p-ml>=0&&(H-1-ml)>=0&&hist[(size_t)r*HS+p-ml]==hist[(size_t)r*HS+H-1-ml])++ml;
              int next=(p+1<H)?hist[(size_t)r*HS+p+1]:-1;
              if(next>=0&&next<V&&ml+1>allow[r]){ float pen=mult[r]*std::pow(bs[r],float(ml+1-allow[r]));
                  ref[next]=std::min(ref[next], lg[(size_t)r*V+next]-pen); }
              ++occ; }
          for(int i=0;i<V;++i) mm += std::abs(o[(size_t)r*V+i]-ref[i])>1e-3f; }
      rep_exact("dry_penalty", mm); }

    printf("\n%s (%d failures)\n", g_fail?"FAILED":"ALL PASS", g_fail);
    return g_fail?1:0;
}
