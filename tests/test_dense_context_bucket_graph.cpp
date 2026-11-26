#include "models/qwen3_5/program/draft_attention_buckets.h"
#include "models/qwen3_5/program/graph_bucket_site.h"
#include "ninfer/ops/softmax_attention.h"
#include "core/device_buffer.h"
#include "core/device_launch.h"
#include "cuda_availability.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::models::qwen3_5::detail;

int main() {
    int devices=0;
    const auto status=cudaGetDeviceCount(&devices);
    if(ninfer::test::cuda_unavailable(status)||(status==cudaSuccess&&devices==0))return 77;
    try {
        CUDA_CHECK(status);
        CUDA_CHECK(cudaSetDevice(0));
        constexpr int D=128,Q=32,K=8,P=64,C=196608;
        constexpr AttentionHeadGeometry geometry{D,Q,K};
        constexpr auto range=ContextBucketRange{ContextBucket::K2,ContextBucket::K256};
        static_assert(draft_full_bucket_envelope(0,C).max_context==2048);
        static_assert(draft_full_bucket_envelope(65537,C).max_context==131072);
        static_assert(draft_full_bucket_envelope(176128,C).max_context==C);
        bool rejected=false;
        try{(void)draft_full_bucket_envelope(C+1,C);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("over-capacity envelope accepted");

        // Two distinct physical pages repeated in a logical block table. Equal
        // zero logits give an analytic, nonconstant oracle for every length.
        DeviceBuffer ck(D*K*P*2*2),cv(D*K*P*2*2),table(C/P*4),length(4),valid(4),row(4);
        ck.fill(0);
        std::vector<std::uint16_t> values(D*K*P*2);
        for(int h=0;h<K;++h){
            auto begin=values.begin()+h*2*P*D;
            std::fill_n(begin,P*D,0x3f80); // 1
            std::fill_n(begin+P*D,P*D,0x4000); // 2
        }
        cv.copy_from_host(values.data(),values.size()*2);
        std::vector<std::int32_t> mapping(C/P);
        for(int i=0;i<C/P;++i)mapping[i]=i%2;
        table.copy_from_host(mapping.data(),mapping.size()*4);
        const int one=1,zero=0;
        valid.copy_from_host(&one,4);row.copy_from_host(&zero,4);
        PagedKVBatchLayerView context{
            .k_pages=Tensor(ck.p,DType::BF16,{D,P,2,K}),
            .v_pages=Tensor(cv.p,DType::BF16,{D,P,2,K}),
            .block_tables=Tensor(table.p,DType::I32,{C/P,1}),
            .head_dim=D,.num_kv_heads=K,.storage=KvCacheStorage::BFloat16};
        Tensor lt(length.p,DType::I32,{1}),vt(valid.p,DType::I32,{1}),rt(row.p,DType::I32,{1});
        cudaStream_t stream=nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        struct StreamGuard{cudaStream_t s;~StreamGuard(){(void)cudaStreamDestroy(s);}}guard{stream};
        int replays=0;
        for(int T:{2,12}){
            DeviceBuffer q(D*Q*T*2),k(D*K*T*2),v(D*K*T*2),out(D*Q*T*2);
            q.fill(0);k.fill(0);
            std::vector<std::uint16_t> query(D*K*T,0x4040); // 3
            v.copy_from_host(query.data(),query.size()*2);
            Tensor qt(q.p,DType::BF16,{D,Q,T,1}),kt(k.p,DType::BF16,{D,K,T,1}),
                   queryv(v.p,DType::BF16,{D,K,T,1}),ot(out.p,DType::BF16,{D,Q,T,1});
            const auto envelope=[&](ContextBucket bucket){
                return draft_full_bucket_envelope(std::min<std::uint32_t>(C,context_bucket_tokens(bucket)),C);
            };
            std::size_t bytes=0;
            range.for_each([&](auto b){
                bytes=std::max(bytes,context_softmax_attention_workspace_capacity_bytes(geometry,envelope(b),T,T,1));
            });
            DeviceArena workspace(bytes);
            const auto capture=[&](auto b){
                context_softmax_attention(qt,kt,queryv,lt,vt,rt,geometry,0.08838834764831844f,
                    context,envelope(b),workspace,ot,stream);
            };
            length.copy_from_host(&zero,4);
            capture(ContextBucket::K2);CUDA_CHECK(cudaStreamSynchronize(stream));
            GraphBucketSite site;
            site.prepare(stream,range,capture);
            DecodeGraphDefinition parent;
            parent.capture(stream,[&]{site.record(stream,capture);});
            DecodeGraphExecutable executable;
            executable.instantiate(parent);
            for(int L:{0,1,64,65,2048,8192,16384,32768,65536,65537,131072,176128,C,65}){
                length.copy_from_host(&L,4);
                site.select(executable,select_context_bucket(L));
                executable.launch(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<std::uint16_t> output(D*Q*T);
                out.copy_to_host(output.data(),output.size()*2);
                const int twos=(L/(2*P))*P+std::max(0,L%(2*P)-P);
                const double expected=(L+twos+3.0)/(L+1.0);
                const auto f=[](std::uint16_t x){
                    std::uint32_t bits=static_cast<std::uint32_t>(x)<<16;
                    float value;
                    std::memcpy(&value,&bits,sizeof(value));
                    return value;
                };
                if(!std::all_of(output.begin(),output.begin()+D*Q,[&](auto x){return std::abs(f(x)-expected)<0.016;})||
                   !std::all_of(output.begin()+D*Q,output.end(),[](auto x){return x==0;}))
                    throw std::runtime_error("dense bucket graph oracle mismatch at length "+std::to_string(L));
                ++replays;
            }
        }
        std::cout<<"dense context bucket graph: "<<replays<<" GPU replays with paged oracle and inactive tails passed\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
