#include "ninfer/engine.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
    const char* artifact=std::getenv("NINFER_TEST_ARTIFACT");
    if(!artifact||!*artifact)return 77;
    if(argc!=2){std::cerr<<"expected output token-file path\n";return 1;}
    try {
        ninfer::EngineOptions o;
        o.artifact_path=artifact;
        o.max_context=32768;
        o.kv_capacity=ninfer::KvCapacityPolicy::explicit_capacity(32768);
        o.prefill_chunk=1024;
        o.kv_cache=ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value;
        o.gdn_state_fp16=true;
        o.speculative.backend=ninfer::SpeculativeBackend::DFlash2;
        o.speculative.draft_tokens=5;
        o.speculative.ngram_draft_tokens=0;
        o.context_cache.enabled=false;
        o.max_concurrency=1;
        ninfer::Engine engine(o);
        // Cross the 8192/16384 boundaries with one uninterrupted request.
        std::vector<ninfer::TokenId> ids;
        const std::vector<ninfer::TokenId> motif{1001,1002,1003,1004,1005,1006,1007,1008};
        for(unsigned i=0;i<16370;++i)ids.push_back(motif[i%motif.size()]);
        ninfer::RequestOptions r;
        r.execution.requested_output_tokens=512;
        r.execution.sampling.temperature=0.0f;
        r.stop.include_model_defaults=false;
        r.output.raw=true;
        auto result=engine.generate(engine.prepare_tokens(std::move(ids)),r);
        if(result.generated_token_ids.size()!=512||
           result.finish_reason!=ninfer::FinishReason::OutputLimit)
            throw std::runtime_error("flat model test did not generate full output");
        std::ofstream f(argv[1],std::ios::binary);
        for(auto id:result.generated_token_ids)f<<id<<'\n';
        if(!f)throw std::runtime_error("cannot write generated token evidence");
        std::cout<<"flat model real: 512 generated tokens across 16384 boundary\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
