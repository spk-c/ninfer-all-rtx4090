// Standalone GPU smoke test for experimental rank-one primitives (not CTest).
#include "../../../src/models/inkling2/rank1_adapter.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace ninfer::models::inkling2::rank1;
static void check(cudaError_t s) {
    if (s != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(s)); std::exit(1); }
}
template<class T> struct Buf {
    T* p = nullptr; size_t n;
    explicit Buf(size_t size):n(size) { check(cudaMalloc((void**)&p,n*sizeof(T))); }
    explicit Buf(const std::vector<T>& v):Buf(v.size()) { check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice)); }
    ~Buf() { cudaFree(p); }
    std::vector<T> read() const { std::vector<T> v(n); check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
};
static std::mt19937 rng(42);
static std::vector<float> randoms(size_t n) {
    std::uniform_real_distribution<float> d(-0.2f,0.2f);
    std::vector<float> v(n); for(float& x:v) x=d(rng); return v;
}
static void compare(const char* label, const std::vector<float>& actual, const std::vector<float>& expected) {
    float err=0;
    for(size_t i=0;i<actual.size();++i) {
        float e=std::abs(actual[i]-expected[i]); err=std::max(err,e);
        if(!std::isfinite(actual[i]) || e > 2e-5f + 2e-5f*std::abs(expected[i])) {
            std::fprintf(stderr,"%s mismatch %zu: %g vs %g\n",label,i,actual[i],expected[i]); std::exit(2);
        }
    }
    std::printf("PASS %s: %zu values, max absolute error %.9g\n",label,actual.size(),err);
}
int main() {
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,0));
    std::printf("GPU: %s sm_%d%d\n",prop.name,prop.major,prop.minor);
    constexpr int H=2304,W=3072,M=1152,E=128,T=8,K=8,P=T*K,V=201024;
    const float scale=.5f;
    {
        auto a=randoms(V), b=randoms(H), base=randoms(T*H), want=base;
        std::vector<int> ids{0,3,200057,9,42,1234,17,0};
        for(int t=0;t<T;++t)for(int j=0;j<H;++j)want[t*H+j]+=scale*a[ids[t]]*b[j];
        Buf<float> da(a),db(b),dy(base); Buf<int> di(ids);
        embedding_add<<<T,threads>>>(dy.p,di.p,da.p,db.p,T,H,scale);
        check(cudaGetLastError()); compare("embedding",dy.read(),want);
    }
    {
        auto a=randoms(W), b=randoms(H), x=randoms(T*W), base=randoms(T*H), want=base;
        for(int t=0;t<T;++t){double z=0;for(int j=0;j<W;++j)z+=double(x[t*W+j])*a[j];for(int j=0;j<H;++j)want[t*H+j]+=float(scale*z*b[j]);}
        Buf<float> da(a),db(b),dx(x),dy(base),dots(T);
        attention_dot<<<T,threads>>>(dx.p,da.p,dots.p,W);
        check(cudaGetLastError());
        attention_add<<<T,threads>>>(dy.p,dots.p,db.p,H,scale);
        check(cudaGetLastError()); compare("attention",dy.read(),want);
    }
    {
        auto a=randoms(E*M),b=randoms(E*H),x=randoms(P*M),base=randoms(P*H),want=base;
        std::vector<int> ids(P);for(int p=0;p<P;++p)ids[p]=(p*17)%E;ids[P-1]=127;
        for(int p=0;p<P;++p){int e=ids[p];double z=0;for(int j=0;j<M;++j)z+=double(x[p*M+j])*a[e*M+j];for(int j=0;j<H;++j)want[p*H+j]+=float(scale*z*b[e*H+j]);}
        Buf<float> da(a),db(b),dx(x),dy(base),dots(P);Buf<int> di(ids);
        expert_dot<<<P,threads>>>(dx.p,di.p,da.p,dots.p,M);
        check(cudaGetLastError());
        expert_add<<<P,threads>>>(dy.p,di.p,dots.p,db.p,H,scale);
        check(cudaGetLastError());compare("expert pairs",dy.read(),want);
    }
    {
        auto a=randoms(W), b=randoms(H), x=randoms(T*W), base=randoms(T*H), want=base;
        std::vector<__nv_bfloat16> xb(x.size());
        for(size_t i=0;i<x.size();++i) xb[i]=__float2bfloat16(x[i]);
        for(int t=0;t<T;++t){double z=0;for(int j=0;j<W;++j)z+=double(__bfloat162float(xb[t*W+j]))*a[j];for(int j=0;j<H;++j)want[t*H+j]+=float(z*b[j]);}
        Buf<float> da(a),db(b),dy(base),dots(T);Buf<__nv_bfloat16> dx(xb);
        dot_bf16<<<T,threads>>>(dx.p,nullptr,da.p,dots.p,W);
        check(cudaGetLastError());
        attention_add<<<T,threads>>>(dy.p,dots.p,db.p,H,1.0f);
        check(cudaGetLastError());compare("BF16 attention",dy.read(),want);
    }
    {
        auto a=randoms(E*M),b=randoms(E*H),x=randoms(P*M),base=randoms(P*H),want=base;
        std::vector<__nv_bfloat16> xb(x.size());for(size_t i=0;i<x.size();++i)xb[i]=__float2bfloat16(x[i]);
        std::vector<int> ids(P);for(int p=0;p<P;++p)ids[p]=(p*17)%E;ids[P-1]=127;
        for(int p=0;p<P;++p){int e=ids[p];double z=0;for(int j=0;j<M;++j)z+=double(__bfloat162float(xb[p*M+j]))*a[e*M+j];for(int j=0;j<H;++j)want[p*H+j]+=float(z*b[e*H+j]);}
        Buf<float> da(a),db(b),dy(base),dots(P);Buf<__nv_bfloat16> dx(xb);Buf<int> di(ids);
        dot_bf16<<<P,threads>>>(dx.p,di.p,da.p,dots.p,M);
        check(cudaGetLastError());
        expert_add<<<P,threads>>>(dy.p,di.p,dots.p,db.p,H,1.0f);
        check(cudaGetLastError());compare("BF16 expert pairs",dy.read(),want);
    }
    {
        // Use the actual planar int8 + half2 scale layout, with nonzero unused sum.
        auto a=randoms(E*M),b=randoms(E*H),base=randoms(P*H),want=base;
        std::vector<std::int8_t> qs(P*M);
        std::vector<half2> ds(P*M/32);
        std::vector<int> ids(P);
        for(int p=0;p<P;++p)ids[p]=(p*17)%E;
        ids[P-1]=127;
        for(size_t i=0;i<qs.size();++i)qs[i]=static_cast<std::int8_t>(int(i*13%255)-127);
        for(size_t i=0;i<ds.size();++i)ds[i]=__floats2half2_rn(0.001f*float(1+i%7),42.f);
        for(int p=0;p<P;++p){
            int e=ids[p];double z=0;
            for(int j=0;j<M;++j){int i=p*M+j;z+=double(qs[i])*__low2float(ds[i/32])*a[e*M+j];}
            for(int j=0;j<H;++j)want[p*H+j]+=float(z*b[e*H+j]);
        }
        Buf<std::int8_t> dq(qs);Buf<half2> dd(ds);Buf<int> di(ids);
        Buf<float> da(a),db(b),dy(base),dots(P),d_base(base);
        cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        check(cudaMemcpyAsync(dy.p,d_base.p,base.size()*sizeof(float),cudaMemcpyDeviceToDevice,stream));
        dot_q8<<<P,threads,0,stream>>>(dq.p,dd.p,di.p,da.p,dots.p,M);
        check(cudaGetLastError());
        expert_add<<<P,threads,0,stream>>>(dy.p,di.p,dots.p,db.p,H,1.0f);
        check(cudaGetLastError());
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,0));
        for(int i=0;i<3;++i){
            check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
            compare("Q8 expert graph replay",dy.read(),want);
        }
        check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    }
    {
        auto a=randoms(W),b=randoms(H),base=randoms(T*H),want=base;
        std::vector<std::int8_t> qs(T*W);std::vector<half2> ds(T*W/32);
        for(size_t i=0;i<qs.size();++i)qs[i]=static_cast<std::int8_t>(int(i*11%255)-127);
        for(size_t i=0;i<ds.size();++i)ds[i]=__floats2half2_rn(0.001f*float(1+i%5),17.f);
        for(int t=0;t<T;++t){
            double z=0;for(int j=0;j<W;++j){int i=t*W+j;z+=double(qs[i])*__low2float(ds[i/32])*a[j];}
            for(int j=0;j<H;++j)want[t*H+j]+=float(z*b[j]);
        }
        Buf<std::int8_t> dq(qs);Buf<half2> dd(ds);Buf<float> da(a),db(b),dy(base),dots(T);
        dot_q8<<<T,threads>>>(dq.p,dd.p,nullptr,da.p,dots.p,W);
        check(cudaGetLastError());attention_add<<<T,threads>>>(dy.p,dots.p,db.p,H,1.f);
        check(cudaGetLastError());compare("Q8 attention",dy.read(),want);
    }
    return 0;
}
