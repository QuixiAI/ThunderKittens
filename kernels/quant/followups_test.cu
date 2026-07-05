/**
 * @file
 * @brief Standalone harness for the MetalForge follow-up kernels: EAGLE
 * bookkeeping (eagle_kernels.cuh), lightning-indexer cp_gather + fp8-output
 * merge_attn_states (sparse_serving_kernels.cuh), TurboQuant tq_encode round-trip
 * (turboquant.cuh), and the Mamba selective_scan APC chunk-checkpoint variant
 * (selective_scan_kernels.cuh).
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        -I../serving followups_test.cu -o followups_test.out
 * Run: CUDA_VISIBLE_DEVICES=0 ./followups_test.out
 */
#include "turboquant.cuh"
#include "../serving/eagle_kernels.cuh"
#include "../serving/sparse_serving_kernels.cuh"
#include "../mamba2/selective_scan_kernels.cuh"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
template <typename T> T* dnew(const std::vector<T>& h){T*d;CK(cudaMalloc(&d,std::max<size_t>(h.size(),1)*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template <typename T> T* dz(size_t n){T*d;CK(cudaMalloc(&d,std::max<size_t>(n,1)*sizeof(T)));CK(cudaMemset(d,0,std::max<size_t>(n,1)*sizeof(T)));return d;}
template <typename T> std::vector<T> d2h(const T*d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static std::mt19937 rng(71);
static std::vector<float> rv(size_t n,float lo,float hi){std::uniform_real_distribution<float> d(lo,hi);std::vector<float> v(n);for(auto&x:v)x=d(rng);return v;}
static std::vector<float> rn(size_t n){std::normal_distribution<float> d(0,1);std::vector<float> v(n);for(auto&x:v)x=d(rng);return v;}
static void rep_rel(const char* nm,double e,double tol){printf("%-40s %s (rel %.3e)\n",nm,e<=tol?"PASS":"FAIL",e);if(e>tol)++g_fail;}
static void rep_ex(const char* nm,long mm){printf("%-40s %s (%ld mismatch)\n",nm,mm?"FAIL":"PASS",mm);if(mm)++g_fail;}

// ---------------- TurboQuant tq_encode round-trip ----------------
template <int HS>
static void test_tq(int k_bits, int k_signed, int v_bits, double k_tol, double v_tol) {
    const int num_tokens = 5, num_kv_heads = 2, block_size = 8;
    auto K = rn((size_t)num_tokens*num_kv_heads*HS);
    auto V = rn((size_t)num_tokens*num_kv_heads*HS);
    std::vector<float> sign(HS); std::uniform_int_distribution<int> b(0,1);
    for (auto& s : sign) s = b(rng) ? 1.f : -1.f;
    const int nc = 1 << v_bits;
    std::vector<float> cent(nc);                        // ascending centroids on [-2.5,2.5]
    for (int i=0;i<nc;++i) cent[i] = -2.5f + 5.0f*i/(nc-1);
    std::vector<int64_t> slot(num_tokens); for (int i=0;i<num_tokens;++i) slot[i]=i;

    const int scale_groups = HS/32;
    const int k_packed=(HS*k_bits+7)/8, v_packed=(HS*v_bits+7)/8;
    const int nslots = block_size; // slots 0..num_tokens-1 within block 0
    auto dK=dnew(K), dV=dnew(V), dsg=dnew(sign), dcent=dnew(cent);
    auto dslot=dnew(slot);
    uint8_t* dkc=dz<uint8_t>((size_t)nslots*num_kv_heads*k_packed);
    uint8_t* dvc=dz<uint8_t>((size_t)nslots*num_kv_heads*v_packed);
    __half* dks=dz<__half>((size_t)nslots*num_kv_heads*scale_groups);
    __half* dvs=dz<__half>((size_t)nslots*num_kv_heads*scale_groups);
    __half* dkz=dz<__half>((size_t)nslots*num_kv_heads*scale_groups);
    dim3 grid(num_tokens, num_kv_heads);
    tm6::tq_encode<float,HS><<<grid,HS>>>(dK,dV,dkc,dvc,dks,dvs,dkz,dslot,dcent,dsg,
        num_kv_heads,block_size,k_bits,k_signed,v_bits);
    CK(cudaDeviceSynchronize());
    if (cudaGetLastError()!=cudaSuccess){printf("TQ KERNEL ERR HS=%d\n",HS);++g_fail;return;}

    auto kc=d2h(dkc,(size_t)nslots*num_kv_heads*k_packed);
    auto vc=d2h(dvc,(size_t)nslots*num_kv_heads*v_packed);
    auto ks=d2h(dks,(size_t)nslots*num_kv_heads*scale_groups);
    auto vs=d2h(dvs,(size_t)nslots*num_kv_heads*scale_groups);
    auto kz=d2h(dkz,(size_t)nslots*num_kv_heads*scale_groups);

    auto unpack=[&](const std::vector<uint8_t>& buf,long base,int idx,int bits)->int{
        int bp=idx*bits, by=bp>>3, bo=bp&7; unsigned raw=buf[base+by];
        if (bo+bits>8) raw|=(unsigned)buf[base+by+1]<<8;
        return (raw>>bo)&((1u<<bits)-1u); };
    auto hf=[](uint16_t h)->float{ __half x; memcpy(&x,&h,2); return __half2float(x); };
    (void)hf;

    double kge=0,krs=0,vge=0,vrs=0;
    for (int tk=0; tk<num_tokens; ++tk) for (int h=0; h<num_kv_heads; ++h) {
        const long slt=slot[tk];
        const long sbase=(slt*num_kv_heads+h)*scale_groups;
        const long kbase=(slt*num_kv_heads+h)*k_packed;
        const long vbase=(slt*num_kv_heads+h)*v_packed;
        // K decode
        for (int d=0; d<HS; ++d) {
            const int g=d/32; const float sc=__half2float(ks[sbase+g]), zp=__half2float(kz[sbase+g]);
            int idx = unpack(kc,kbase,d,k_bits);
            if (k_signed) { const int mv=(1<<(k_bits-1))-1; if (idx>mv) idx-=(1<<k_bits); }
            const float rec=(idx+zp)*sc;
            const float in=K[((size_t)tk*num_kv_heads+h)*HS+d];
            kge+=std::abs(rec-in); krs+=std::abs(in);
        }
        // V decode: gather centroid*scale in rotated domain, host inverse FWHT
        std::vector<double> rot(HS);
        for (int d=0; d<HS; ++d) {
            const int g=d/32; const float sc=__half2float(vs[sbase+g]);
            const int idx=unpack(vc,vbase,d,v_bits);
            rot[d]=(double)cent[idx]*sc;
        }
        // inverse: y=(1/sqrt(HS)) H rot ; V_dec[i]=sign[i]*y[i]
        for (int hbit=1; hbit<HS; hbit<<=1) for (int i=0;i<HS;++i) if((i&hbit)==0){ double a=rot[i],c=rot[i+hbit]; rot[i]=a+c; rot[i+hbit]=a-c; }
        const double isq=1.0/std::sqrt((double)HS);
        for (int d=0; d<HS; ++d) {
            const double vdec=sign[d]*rot[d]*isq;
            const float in=V[((size_t)tk*num_kv_heads+h)*HS+d];
            vge+=std::abs(vdec-in); vrs+=std::abs((double)in);
        }
    }
    char nm[64];
    snprintf(nm,sizeof nm,"tq_encode HS=%d k%d%s K-roundtrip",HS,k_bits,k_signed?"s":"u");
    rep_rel(nm, kge/std::max(krs,1e-30), k_tol);
    snprintf(nm,sizeof nm,"tq_encode HS=%d v%d V-roundtrip",HS,v_bits);
    rep_rel(nm, vge/std::max(vrs,1e-30), v_tol);
}

// ---------------- Mamba selective_scan APC ----------------
static void test_apc() {
    const int batch=1, dim=4, dstate=8, n_groups=1, block_size=4;
    const int seqlen=10, total=seqlen;
    std::vector<int> qsl={0,seqlen};
    auto U=rn((size_t)dim*total), Dl=rv((size_t)dim*total,-0.5f,0.5f);
    auto A=rv((size_t)dim*dstate,-1.f,-0.1f);
    auto B=rn((size_t)n_groups*dstate*total), C=rn((size_t)n_groups*dstate*total);
    std::vector<float> Dd(dim,0.5f), dbias(dim,0.1f);
    auto Z=rn((size_t)dim*total);
    // paged state: block indices 0..n_blocks-1 map to physical slots
    const int n_blocks=3;              // ceil(10/4)
    std::vector<int> cache_indices={5,2,9};   // per (batch,block) physical slot
    const int cache_stride=n_blocks;
    const int max_slot=10;
    std::vector<int> bfirst={0}, blast={n_blocks-1}, initidx={0};
    std::vector<uint8_t> hinit={0};

    auto dU=dnew(U),dDl=dnew(Dl),dA=dnew(A),dB=dnew(B),dC=dnew(C),dDd=dnew(Dd),dbi=dnew(dbias),dZ=dnew(Z);
    auto dq=dnew(qsl),dci=dnew(cache_indices),dbf=dnew(bfirst),dbl=dnew(blast),dii=dnew(initidx);
    auto dhi=dnew(hinit);
    float* dSt=dz<float>((size_t)max_slot*dim*dstate);
    float* dout=dz<float>((size_t)dim*total);
    std::vector<int> dummy(1,0); auto ddummy=dnew(dummy);
    int block=(dstate+31)/32*32; size_t shbytes=((block+31)/32)*sizeof(float);
    dim3 grid(dim,batch);
    tmss::selective_scan_fwd_varlen_apc<float><<<grid,block,shbytes>>>(
        dU,dDl,dA,dB,dC,dDd,dbi,dZ,dq,dci,dhi,dout,dSt,
        dbf,dbl,dii,ddummy,ddummy,batch,dim,total,dstate,n_groups,
        1,1,1,1,-1,block_size,cache_stride,0);
    CK(cudaDeviceSynchronize());
    if (cudaGetLastError()!=cudaSuccess){printf("APC KERNEL ERR\n");++g_fail;return;}
    auto out=d2h(dout,(size_t)dim*total);
    auto st=d2h(dSt,(size_t)max_slot*dim*dstate);

    // fp64 reference: full-sequence scan (out is chunk-invariant); track running.
    double ge=0,rs=0; long smm=0;
    auto softplus=[](double x){return x<=20.0?std::log(1.0+std::exp(x)):x;};
    for (int d=0; d<dim; ++d) {
        std::vector<double> S(dstate,0.0);
        std::vector<double> chk_running;   // running state after each chunk end
        int tp=0;
        for (int chunk=0; tp<seqlen; ++chunk) {
            int ct=std::min(block_size,seqlen-tp);
            for (int off=0; off<ct; ++off) {
                int t=tp+off;
                double u=U[(size_t)d*total+t]; double dv=Dl[(size_t)d*total+t]+dbias[d]; dv=softplus(dv);
                double sum=Dd[d]*u;
                for (int s=0;s<dstate;++s){ double a=A[(size_t)d*dstate+s];
                    S[s]=std::exp(dv*a)*S[s]+B[(size_t)s*total+t]*dv*u;
                    sum+=S[s]*C[(size_t)s*total+t]; }
                double zv=Z[(size_t)d*total+t]; sum*= zv/(1.0+std::exp(-zv));
                double got=out[(size_t)d*total+t]; ge+=std::abs(got-sum); rs+=std::abs(sum);
            }
            // checkpoint slot for this chunk
            int sbi = (chunk==n_blocks-1)? blast[0] : (tp+ct-1)/block_size;
            int slot=cache_indices[sbi];
            for (int s=0;s<dstate;++s){ double got=st[((size_t)slot*dim+d)*dstate+s];
                smm += std::abs(got-S[s])>1e-3; }
            tp+=ct;
        }
    }
    rep_rel("selective_scan_apc out vs fp64", ge/std::max(rs,1e-30), 1e-5);
    rep_ex ("selective_scan_apc chunk state checkpoints", smm);
}

// ---------------- lightning-indexer cp_gather round-trip ----------------
static void test_cp_gather() {
    const int head_dim=128, qbs=128, cbs=4, num_tokens=6, num_blocks=2, batch=1;
    const int cache_stride = head_dim + head_dim*4/qbs;     // 132
    const long block_stride = (long)cbs*cache_stride;       // 528
    auto K = rn((size_t)num_tokens*head_dim);
    std::vector<int> block_table={7,3};
    std::vector<int> cu={0,num_tokens};
    std::vector<int64_t> slot(num_tokens);
    for (int t=0;t<num_tokens;++t) slot[t]=(int64_t)block_table[t/cbs]*cbs + (t%cbs);
    const int max_phys=8;
    uint8_t* dcache=dz<uint8_t>((size_t)max_phys*block_stride);
    auto dK=dnew(K); auto dslot=dnew(slot);
    // indexer fills the cache
    dim3 g1(num_tokens,(head_dim+qbs-1)/qbs);
    tmsp::indexer_k_quant_and_cache<float><<<g1,32>>>(dK,dcache,dslot,num_tokens,head_dim,qbs,cbs,cache_stride,0);
    CK(cudaDeviceSynchronize());
    // cp_gather reads them back
    auto dbt=dnew(block_table); auto dcu=dnew(cu);
    const long token_stride=head_dim;
    uint8_t* ddk=dz<uint8_t>((size_t)num_tokens*token_stride);
    uint8_t* dds=dz<uint8_t>((size_t)num_tokens*token_stride/qbs*4);
    tmsp::cp_gather_indexer_k_quant_cache<<<g1,32>>>(dcache,ddk,dds,dbt,dcu,batch,token_stride,head_dim,
        block_stride,cbs,num_blocks,num_tokens,qbs);
    CK(cudaDeviceSynchronize());
    if (cudaGetLastError()!=cudaSuccess){printf("CPGATHER KERNEL ERR\n");++g_fail;return;}
    auto cache=d2h(dcache,(size_t)max_phys*block_stride);
    auto dk=d2h(ddk,(size_t)num_tokens*token_stride);
    auto ds=d2h(dds,(size_t)num_tokens*token_stride/qbs*4);

    long mm=0;
    for (int t=0;t<num_tokens;++t) {
        const long s=slot[t]; const long blk=s/cbs, off=s-blk*cbs;
        const long dbase=blk*cbs*cache_stride + off*head_dim;
        for (int i=0;i<head_dim;++i) mm += cache[dbase+i]!=dk[(size_t)t*token_stride+i];
        const long soff=blk*cbs*cache_stride + (long)cbs*head_dim + (off*head_dim)*4/qbs;
        float cs=*reinterpret_cast<float*>(&cache[soff]);
        float gs=*reinterpret_cast<float*>(&ds[(size_t)t*token_stride/qbs*4]);
        mm += cs!=gs;
    }
    rep_ex("cp_gather_indexer vs indexer offsets", mm);
}

// ---------------- fp8-output merge_attn_states ----------------
static void test_merge_fp8() {
    const int num_tokens=6, num_heads=3, head_size=16, prefix_tokens=4;
    const float out_scale=0.5f;
    auto po=rn((size_t)num_tokens*num_heads*head_size), so=rn((size_t)num_tokens*num_heads*head_size);
    auto pl=rv((size_t)num_tokens*num_heads,-1.f,2.f), sl=rv((size_t)num_tokens*num_heads,-1.f,2.f);
    auto dpo=dnew(po),dso=dnew(so); auto dpl=dnew(pl),dsl=dnew(sl);
    long tot=(long)num_tokens*num_heads*head_size;
    uint8_t* dout=dz<uint8_t>(tot); float* dlse=dz<float>((size_t)num_tokens*num_heads);
    int threads=256, blocks=(tot+threads-1)/threads;
    tmsp::merge_attn_states_fp8<float><<<blocks,threads>>>(dout,dlse,dpo,dpl,dso,dsl,
        num_heads,head_size,num_tokens,prefix_tokens,out_scale);
    CK(cudaDeviceSynchronize());
    if (cudaGetLastError()!=cudaSuccess){printf("MERGEFP8 KERNEL ERR\n");++g_fail;return;}
    auto out=d2h(dout,tot);
    long mm=0;
    const float si=1.0f/out_scale;
    for (long idx=0; idx<tot; ++idx) {
        int hidden=idx%head_size, head=(idx/head_size)%num_heads, tok=idx/((long)head_size*num_heads);
        long li=(long)tok*num_heads+head; float val;
        if (tok>=prefix_tokens) val=so[idx];
        else { float m=std::max(pl[li],sl[li]);
            if (std::isinf(m)) val=po[idx];
            else { float pe=std::exp(pl[li]-m),se=std::exp(sl[li]-m),z=pe+se; val=po[idx]*(pe/z)+so[idx]*(se/z); } }
        mm += out[idx]!=tmq::e4m3_encode(val*si);
        (void)hidden;
    }
    rep_ex("merge_attn_states_fp8 codes", mm);
}

// ---------------- EAGLE bookkeeping (host replay) ----------------
static void test_eagle() {
    // eagle_prepare_next_token_padded
    { const int R=4, nsp=3; const int64_t vocab=100;
      std::vector<int64_t> sampled={ 5,7,-1,  -1,-1,-1,  9,200,4,  1,-1,-1 };
      std::vector<uint8_t> disc={0,0,1,0};
      std::vector<int64_t> backup={11,22,33,44};
      auto ds=dnew(sampled),db=dnew(backup); auto dd=dnew(disc);
      int64_t* dnt=dz<int64_t>(R); int64_t* dvc=dz<int64_t>(R);
      tmeg::eagle_prepare_next_token_padded<<<1,R>>>(dnt,dvc,ds,dd,db,vocab,nsp,R,nsp);
      CK(cudaDeviceSynchronize());
      auto nt=d2h(dnt,R); auto vcnt=d2h(dvc,R); long mm=0;
      for (int r=0;r<R;++r){ int64_t valid=0,last=-1;
          for(int p=0;p<nsp;++p){int64_t tok=sampled[r*nsp+p]; if(tok!=-1&&tok<vocab){valid++;last=tok;}}
          int64_t ent,evc; if(disc[r]){ent=backup[r];evc=0;} else {ent=valid>0?last:backup[r];evc=valid;}
          mm += nt[r]!=ent; mm += vcnt[r]!=evc; }
      rep_ex("eagle_prepare_next_token_padded", mm); }

    // eagle_step_slot_mapping_metadata
    { const int input_bs=5, bs=4; const int64_t block_size=8, max_len=64, pad=-1;
      const int64_t bstride=4, nbpr=4;
      std::vector<int64_t> pos={3,7,15,63,0};
      std::vector<int> bt={10,11,12,13, 20,21,22,23, 30,31,32,33, 40,41,42,43, 0,0,0,0};
      std::vector<int> seq={4,8,16,64,1};
      auto dpos=dnew(pos); auto dbt=dnew(bt); auto dseq=dnew(seq);
      int64_t* dcp=dz<int64_t>(input_bs); int64_t* dsm=dz<int64_t>(input_bs);
      tmeg::eagle_step_slot_mapping_metadata<<<1,input_bs>>>(dseq,dcp,dsm,dpos,dbt,block_size,max_len,pad,
          bs,input_bs,bstride,nbpr);
      CK(cudaDeviceSynchronize());
      auto sm=d2h(dsm,input_bs); auto cp=d2h(dcp,input_bs); auto seqo=d2h(dseq,input_bs); long mm=0;
      for (int r=0;r<input_bs;++r){
          if (r>=bs){ mm += sm[r]!=pad; continue; }
          int64_t np=pos[r]+1; bool ex=np>=max_len; int64_t cl=ex?0:np;
          mm += cp[r]!=cl;
          int64_t bn=std::min(cl/block_size,nbpr-1);
          int bid=bt[(long)r*bstride+bn]; int64_t slot=(int64_t)bid*block_size+(cl%block_size);
          mm += sm[r]!=(ex?pad:slot);
          int exp_seq= ex?1:std::min(seq[r]+1,(int)max_len); mm += seqo[r]!=exp_seq; }
      rep_ex("eagle_step_slot_mapping_metadata", mm); }

    // eagle_expand_int64
    { const int bs=4; std::vector<int64_t> cu={2,2,5,7}, in={9,-1,3,-1};
      const int64_t rf=-1, rt=77;
      auto dcu=dnew(cu),din=dnew(in); int64_t* dout=dz<int64_t>(7);
      tmeg::eagle_expand_int64<<<1,bs>>>(dout,din,dcu,rf,rt,bs);
      CK(cudaDeviceSynchronize());
      auto out=d2h(dout,7); long mm=0;
      for (int r=0;r<bs;++r){ long s=r==0?0:cu[r-1],e=cu[r]; int64_t v=in[r]==rf?rt:in[r];
          for(long i=s;i<e;++i) mm += out[i]!=v; }
      rep_ex("eagle_expand_int64", mm); }

    // copy_and_expand_eagle_inputs (shift=0)
    { const int R=2; const int64_t pad=-9, para=-8, npad=2;
      std::vector<int> qsl={0,3,7}, qel={2,5};      // req0: q[0,3) end2 ; req1: q[3,7) end5
      const int64_t total_in=7;
      std::vector<int64_t> ttok(7), tpos(7); for(int i=0;i<7;++i){ttok[i]=100+i;tpos[i]=i;}
      std::vector<int64_t> ntok={55,66};
      auto dtt=dnew(ttok),dtp=dnew(tpos),dnt=dnew(ntok); auto dqs=dnew(qsl),dqe=dnew(qel);
      const long out_len=64;
      int64_t* doi=dz<int64_t>(out_len); int64_t* dop=dz<int64_t>(out_len);
      uint8_t* drej=dz<uint8_t>(out_len); uint8_t* dmsk=dz<uint8_t>(out_len);
      int* dni=dz<int>(64); int* dhm=dz<int>(64);
      tmeg::copy_and_expand_eagle_inputs<<<1,R>>>(dtt,dtp,dnt,doi,dop,drej,dmsk,dni,dhm,dqs,dqe,
          pad,para,total_in,npad,0,R);
      CK(cudaDeviceSynchronize());
      auto oi=d2h(doi,out_len); auto op=d2h(dop,out_len); auto rej=d2h(drej,out_len); auto msk=d2h(dmsk,out_len);
      long mm=0;
      for (int req=0; req<R; ++req) {
          int qs=qsl[req], nqs=qsl[req+1], qe=qel[req];
          long num_valid=qe-qs+1, out_start=qs+(long)req*npad;
          long num_rej=nqs-qe-1, total_out=num_valid+npad+num_rej;
          long start_pos=tpos[qs], bonus=ntok[req];
          for (long j=0;j<total_out;++j){ long oidx=out_start+j;
              bool iv=j<num_valid, ib=j==num_valid, ip=(j>num_valid)&&(j<num_valid+npad), ir=j>=num_valid+npad;
              long in_idx=std::min((long)qs+0+j,total_in-1);
              int64_t tid=pad; if(iv)tid=ttok[in_idx]; else if(ib)tid=bonus; else if(ip)tid=para;
              mm += oi[oidx]!=tid; mm += op[oidx]!=(ir?0:(start_pos+j));
              mm += rej[oidx]!=(ir?1:0); mm += msk[oidx]!=(ip?1:0); }
      }
      rep_ex("copy_and_expand_eagle_inputs", mm); }
}

int main() {
    // K tolerances are the asymmetric-uniform quant floor on N(0,1) data
    // (8-bit ~5e-3, 4-bit ~9e-2); V is FWHT-rotated + centroid-quantized.
    test_tq<64>(8,1,8, 8e-3, 6e-2);
    test_tq<128>(8,1,8, 8e-3, 6e-2);
    test_tq<128>(4,0,3, 1.2e-1, 3e-1);
    test_tq<256>(8,1,4, 8e-3, 2e-1);
    test_apc();
    test_cp_gather();
    test_merge_fp8();
    test_eagle();
    printf("\n%s (%d failures)\n", g_fail?"FAILED":"ALL PASS", g_fail);
    return g_fail?1:0;
}
