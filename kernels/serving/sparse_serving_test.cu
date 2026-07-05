/**
 * @file
 * @brief Harness for M5 sparse/serving extras (sparse_serving_kernels.cuh):
 * merge_attn_states + tau_tail vs fp64, lightning-indexer fp8 round-trip, and
 * the MInference vertical/slash builder vs an exact host replay of the same
 * two-pointer algorithm.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        sparse_serving_test.cu -o sparse_serving_test.out -I../quant
 * Run: CUDA_VISIBLE_DEVICES=0 ./sparse_serving_test.out
 */
#include "sparse_serving_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace tmsp;
static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
template <typename T> T* dnew(const std::vector<T>& h){T*d;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template <typename T> T* dz(size_t n){T*d;CK(cudaMalloc(&d,n*sizeof(T)));CK(cudaMemset(d,0,n*sizeof(T)));return d;}
template <typename T> std::vector<T> d2h(const T*d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static std::mt19937 rng(51);
static std::vector<float> rv(size_t n,float lo,float hi){std::uniform_real_distribution<float> d(lo,hi);std::vector<float> v(n);for(auto&x:v)x=d(rng);return v;}
static void rep_rel(const char* nm,double e,double tol){printf("%-30s %s (rel %.3e)\n",nm,e<=tol?"PASS":"FAIL",e);if(e>tol)++g_fail;}
static void rep_ex(const char* nm,long mm){printf("%-30s %s (%ld mismatch)\n",nm,mm?"FAIL":"PASS",mm);if(mm)++g_fail;}
static double e4dec(uint8_t v){float m; if(v&0x78){int e=(v>>3)&0xF,mm=v&7;m=std::ldexp(1.0f+mm/8.0f,e-7);}else m=float(v&7)*0.001953125f; return (v&0x80)?-m:m;}

int main() {
    // ---- merge_attn_states ----
    { const int T=48,H=4,D=64,PT=40;   // PT prefix tokens; tail is suffix-only
      auto po=rv((size_t)T*H*D,-1,1), so=rv((size_t)T*H*D,-1,1);
      auto pl=rv((size_t)T*H,-2,3), sl=rv((size_t)T*H,-2,3);
      auto dpo=dnew(po); auto dso=dnew(so); auto dpl=dnew(pl); auto dsl=dnew(sl);
      float* dout=dz<float>((size_t)T*H*D); float* dolse=dz<float>((size_t)T*H);
      merge_attn_states<float><<<((size_t)T*H*D+255)/256,256>>>(dout,dolse,dpo,dpl,dso,dsl,H,D,T,PT);
      CK(cudaDeviceSynchronize());
      auto o=d2h(dout,(size_t)T*H*D); auto ol=d2h(dolse,(size_t)T*H);
      double gs=0,rs=0,ls=0,lr=0;
      for (int t=0;t<T;++t) for (int h=0;h<H;++h){ double p=pl[(size_t)t*H+h],s=sl[(size_t)t*H+h];
          double reflse; std::vector<double> ref(D);
          if (t>=PT){ reflse=s; for(int d=0;d<D;++d) ref[d]=so[((size_t)t*H+h)*D+d]; }
          else { double m=std::max(p,s), pe=std::exp(p-m),se=std::exp(s-m),z=pe+se;
              reflse=std::log(z)+m; for(int d=0;d<D;++d) ref[d]=po[((size_t)t*H+h)*D+d]*(pe/z)+so[((size_t)t*H+h)*D+d]*(se/z); }
          for(int d=0;d<D;++d){ gs+=std::abs(o[((size_t)t*H+h)*D+d]-ref[d]); rs+=std::abs(ref[d]); }
          ls+=std::abs(ol[(size_t)t*H+h]-reflse); lr+=std::abs(reflse); }
      rep_rel("merge_attn_states out", gs/std::max(rs,1e-30), 2e-6);
      rep_rel("merge_attn_states lse", ls/std::max(lr,1e-30), 2e-6); }

    // ---- tau_tail ----
    { const int T=20,H=4,D=16,q_dim=H*D,PMAX=64;
      auto qkv=rv((size_t)T*3*q_dim,-1,1); auto lin=rv((size_t)T*2*H,-1,1); auto tab=rv((size_t)PMAX*H,-1,1);
      std::vector<int64_t> pos(T); for(int t=0;t<T;++t)pos[t]=t%PMAX;
      auto qkv0=qkv;
      auto dq=dnew(qkv); auto dl=dnew(lin); auto dt=dnew(tab); auto dp=dnew(pos);
      const int elements=T*H*D;
      tau_tail<float><<<(elements+255)/256,256>>>(dq,dl,dt,dp,elements,H,D,q_dim);
      CK(cudaDeviceSynchronize());
      auto o=d2h(dq,(size_t)T*3*q_dim); double gs=0,rs=0;
      for (int t=0;t<T;++t) for(int h=0;h<H;++h) for(int d=0;d<D;++d){
          double tq=std::tanh(lin[(size_t)t*2*H+h]), tv=std::tanh(lin[(size_t)t*2*H+H+h]), tp=tab[(size_t)pos[t]*H+h];
          long qi=(long)t*3*q_dim+(long)h*D+d, vi=(long)t*3*q_dim+2L*q_dim+(long)h*D+d;
          double rq=qkv0[qi]*(tq+tp), rv2=qkv0[vi]*(tv+tp);
          gs+=std::abs(o[qi]-rq)+std::abs(o[vi]-rv2); rs+=std::abs(rq)+std::abs(rv2);
          // K slice unchanged
          long ki=(long)t*3*q_dim+q_dim+(long)h*D+d; if(o[ki]!=qkv0[ki])++g_fail; }
      rep_rel("tau_tail q/v scale", gs/std::max(rs,1e-30), 2e-6); }

    // ---- lightning indexer (fp8 K-quant round-trip) ----
    { const int NT=30, HD=128, QB=64, CBS=16, NB=NT/CBS+2;
      const int nqb=HD/QB;
      const int cache_stride = HD + nqb*4;                 // data + one fp32 scale per qblock
      auto k=rv((size_t)NT*HD,-2,2); auto dk=dnew(k);
      std::vector<int64_t> slot(NT); for(int t=0;t<NT;++t)slot[t]=t; auto dsl=dnew(slot);
      uint8_t* dcache=dz<uint8_t>((size_t)NB*CBS*cache_stride);
      dim3 g(NT,nqb); indexer_k_quant_and_cache<float><<<g,32>>>(dk,dcache,dsl,NT,HD,QB,CBS,cache_stride,0);
      CK(cudaDeviceSynchronize());
      auto cache=d2h(dcache,(size_t)NB*CBS*cache_stride);
      double gs=0,rs=0;
      for (int t=0;t<NT;++t){ long block=slot[t]/CBS, off=slot[t]-block*CBS;
          long tb=block*CBS*cache_stride+off*HD;
          for(int qb=0;qb<nqb;++qb){ long so=block*CBS*cache_stride+(long)CBS*HD+(off*HD+qb*QB)*4/QB;
              float sc=reinterpret_cast<float*>(&cache[so])[0];
              for(int i=0;i<QB;++i){ double got=e4dec(cache[tb+qb*QB+i])*sc; double ref=k[(size_t)t*HD+qb*QB+i];
                  gs+=std::abs(got-ref); rs+=std::abs(ref); } } }
      rep_rel("indexer_k_quant (fp8 rt)", gs/std::max(rs,1e-30), 6e-2); }

    // ---- MInference vertical/slash builder (host-replay parity) ----
    { const int B=2,H=2,num_rows=4,nnz_v=8,nnz_s=6,bm=64,bn=64;
      std::vector<int> qsl(B),ksl(B); for(int b=0;b<B;++b){qsl[b]=200+b*40; ksl[b]=qsl[b];}
      std::vector<int> vert((size_t)B*H*nnz_v), slash((size_t)B*H*nnz_s);
      std::uniform_int_distribution<int> vd(0,300), sd(0,300);
      for (int b=0;b<B;++b) for(int h=0;h<H;++h){
          std::vector<int> vv(nnz_v),ss(nnz_s); for(auto&x:vv)x=vd(rng); for(auto&x:ss)x=sd(rng);
          std::sort(vv.begin(),vv.end()); std::sort(ss.rbegin(),ss.rend());   // slash desc
          for(int i=0;i<nnz_v;++i)vert[((size_t)b*H+h)*nnz_v+i]=vv[i];
          for(int i=0;i<nnz_s;++i)slash[((size_t)b*H+h)*nnz_s+i]=ss[i]; }
      auto dqsl=dnew(qsl); auto dksl=dnew(ksl); auto dv=dnew(vert); auto ds=dnew(slash);
      const size_t RT=(size_t)B*H*num_rows;
      int* dbc=dz<int>(RT); int* dbo=dz<int>(RT*nnz_s); int* dcc=dz<int>(RT); int* dci=dz<int>(RT*nnz_v);
      dim3 g(H,B,(num_rows+63)/64);
      convert_vertical_slash_indexes<<<g,64>>>(dqsl,dksl,dv,ds,dbc,dbo,dcc,dci,H,num_rows,nnz_v,nnz_s,bm,bn,1);
      CK(cudaDeviceSynchronize());
      if (cudaGetLastError()!=cudaSuccess){printf("MINF KERNEL ERR\n");return 1;}
      auto bc=d2h(dbc,RT); auto bo=d2h(dbo,RT*nnz_s); auto cc=d2h(dcc,RT); auto ci=d2h(dci,RT*nnz_v);
      // host replay of the exact algorithm
      auto save=[&](std::vector<int>&off,int rs,int re,int bs,int in,int kv){ if(rs>=kv)return in; if(re>kv)re=kv;
          int c=in; for(int idx=rs;idx<re;idx+=bs)off[c++]=idx; return c; };
      long mm=0;
      for (int b=0;b<B;++b) for(int h=0;h<H;++h) for(int bmi=0;bmi<num_rows;++bmi){
          const int start_m=bmi*bm; if(start_m>=qsl[b]){ continue; }
          const int end_m=start_m+bm; const int* rvp=&vert[((size_t)b*H+h)*nnz_v]; const int* rsp=&slash[((size_t)b*H+h)*nnz_s];
          std::vector<int> rbo(nnz_s,0),rci(nnz_v,0); bool hs=true; int tc=0,tb=0,s=0,v=0;
          int vi=rvp[v++],si=rsp[s++]; const int offv=ksl[b]-qsl[b];
          while(si>=end_m+offv&&s<nnz_s)si=rsp[s++]; if(si>end_m+offv)hs=false; si=std::max(offv+end_m-si,bm);
          int rs2=si-bm,re2=si; if(!hs){rs2=offv+end_m;re2=offv+end_m+bn;}
          bool sf=false;
          while(true){ if(vi<re2){ if(vi<rs2)rci[tc++]=vi; if(v<nnz_v)vi=rvp[v++]; else vi=end_m+bn+offv; }
              else { if(s<nnz_s){ si=std::max(offv+end_m-rsp[s++],bm); }
                  else { if(v==nnz_v||(vi>rs2)){ tb=save(rbo,rs2,re2,bn,tb,ksl[b]); break; }
                      else { rs2=offv+end_m; re2=offv+end_m+bn; sf=true; } }
                  if(!sf){ if(si>re2+bm){ tb=save(rbo,rs2,re2,bn,tb,ksl[b]); rs2=si-bm; re2=si; } else if(si>re2)re2+=bm; } } }
          const int ro=(b*H+h)*num_rows+bmi;
          mm += bc[ro]!=tb; mm += cc[ro]!=tc;
          for(int i=0;i<tb;++i) mm += bo[(size_t)ro*nnz_s+i]!=rbo[i];
          for(int i=0;i<tc;++i) mm += ci[(size_t)ro*nnz_v+i]!=rci[i]; }
      rep_ex("minference vert/slash", mm); }

    printf("\n%s (%d failures)\n", g_fail?"FAILED":"ALL PASS", g_fail);
    return g_fail?1:0;
}
