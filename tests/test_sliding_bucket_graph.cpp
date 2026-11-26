#include "models/qwen3_5/program/graph_bucket_site.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "core/device_buffer.h"
#include "core/device_launch.h"
#include "cuda_availability.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using ninfer::models::qwen3_5::detail::GraphBucketSite;

int main() {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (ninfer::test::cuda_unavailable(status) || (status == cudaSuccess && devices == 0)) return 77;
    try {
        CUDA_CHECK(status);
        CUDA_CHECK(cudaSetDevice(0));
        constexpr int D=128, Q=32, K=8, W=4096, T=2;
        constexpr AttentionHeadGeometry geometry{D,Q,K};
        constexpr auto range = ContextBucketRange{ContextBucket::K2, ContextBucket::K128};
        DeviceBuffer q(D*Q*T*2), k(D*K*T*2), v(D*K*T*2), ck(D*K*W*2),
                     cv(D*K*W*2), pos(T*4), valid(4), lane(4), out(D*Q*T*2);
        q.fill(0); k.fill(0); ck.fill(0);
        // Equal values at every attended position make the exact output independent
        // of reduction order; test both masks and inactive query tails.
        std::vector<std::uint16_t> ones(D*K*W, 0x3f80);
        cv.copy_from_host(ones.data(), ones.size()*2);
        v.copy_from_host(ones.data(), D*K*T*2);
        const int one=1, zero=0;
        valid.copy_from_host(&one,4); lane.copy_from_host(&zero,4);
        Tensor qt(q.p,DType::BF16,{D,Q,T,1}), kt(k.p,DType::BF16,{D,K,T,1}),
               vt(v.p,DType::BF16,{D,K,T,1}), pt(pos.p,DType::I32,{T,1}),
               validt(valid.p,DType::I32,{1}), lanet(lane.p,DType::I32,{1}),
               ot(out.p,DType::BF16,{D,Q,T,1});
        CyclicKVCacheLayerView context{
            .k=Tensor(ck.p,DType::BF16,{D,W,K,1}),
            .v=Tensor(cv.p,DType::BF16,{D,W,K,1}), .capacity=W,.padded_capacity=W,
            .num_kv_heads=K,.head_dim=D,.lane_capacity=1};
        std::size_t bytes=0;
        range.for_each([&](auto bucket){
            bytes=std::max(bytes,sliding_window_attention_workspace_capacity_bytes(
                geometry,W,sliding_window_bucket_envelope(bucket),T,T,1));
        });
        DeviceArena workspace(bytes);
        cudaStream_t stream=nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        struct StreamGuard { cudaStream_t stream; ~StreamGuard(){ (void)cudaStreamDestroy(stream); } } guard{stream};
        for(auto mask:{SlidingWindowMask::Symmetric, SlidingWindowMask::Causal}){
            const auto capture=[&](auto bucket){
                sliding_window_attention(qt,kt,vt,pt,validt,lanet,geometry,W,mask,
                    0.08838834764831844f,context,sliding_window_bucket_envelope(bucket),workspace,ot,stream);
            };
            // Warm CUDA modules outside capture.
            const int first[]{64,65};
            pos.copy_from_host(first,sizeof(first));
            capture(ContextBucket::K2); CUDA_CHECK(cudaStreamSynchronize(stream));
            GraphBucketSite site;
            site.prepare(stream,range,capture);
            DecodeGraphDefinition parent;
            parent.capture(stream,[&]{site.record(stream,capture);});
            DecodeGraphExecutable executable;
            executable.instantiate(parent);
            for(int length:{64,2048,8192,16384,32768,65536,131072,64}){
                const int positions[]{length,length+1};
                pos.copy_from_host(positions,sizeof(positions));
                const auto bucket=select_context_bucket(length);
                site.select(executable,bucket);
                executable.launch(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<std::uint16_t> result(D*Q*T);
                out.copy_to_host(result.data(),result.size()*2);
                if(!std::all_of(result.begin(),result.begin()+D*Q,[](auto x){return x==0x3f80;}) ||
                   !std::all_of(result.begin()+D*Q,result.end(),[](auto x){return x==0;}))
                    throw std::runtime_error("bucket replay differs from exact constant/inactive-tail oracle");
            }
            bool rejected=false;
            try {site.select(executable,ContextBucket::M1);}catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error("unprepared bucket accepted");
        }
        std::cout<<"sliding attention bucket graph: 16 GPU replays, exact output and inactive tail passed\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
