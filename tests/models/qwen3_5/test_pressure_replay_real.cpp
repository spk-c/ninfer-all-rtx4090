#include "ninfer/engine.h"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

// Run only with a real Qwen DFlash2 artifact and explicit pressure replay opt-in.
int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    const char* flag = std::getenv("NINFER_SM89_LIVE_CONTEXT_REPLAY");
    if (!artifact || !*artifact || !flag || std::string_view(flag) != "1") return 77;
    try {
        std::atomic<unsigned> replays{0};
        ninfer::EngineOptions options;
        options.artifact_path = artifact;
        options.max_context = 6144;
        // Two initial leases of 768 + 4096 tokens fit; two complete outputs do not.
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(9792);
        options.prefill_chunk = 512;
        options.max_concurrency = 2;
        options.max_pending_requests = 2;
        options.concurrent_prefill = true;
        options.kv_cache = ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value;
        options.gdn_state_fp16 = true;
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 5;
        options.speculative.ngram_draft_tokens = 0;
        options.context_cache.mode = ninfer::ContextCacheMode::Hybrid;
        options.context_cache.kv_lease_growth = true;
        options.context_cache.host_cache_budget_bytes = 256ULL << 20;
        options.context_cache.hybrid.device_snapshot_slots = 4;
        options.context_cache.hybrid.max_new_taps = 0;
        options.diagnostic_observer.callback = [&](const ninfer::Diagnostic& diagnostic) {
            std::cerr << diagnostic.message << '\n';
            if (diagnostic.message.find("Device KV pressure replay:") != std::string::npos)
                ++replays;
        };
        ninfer::Engine engine(options);
        if (!engine.options().context_cache.kv_lease_growth)
            throw std::runtime_error("fixture did not enable growing KV leases");
        ninfer::RequestOptions request;
        constexpr unsigned wanted=4352;
        request.execution.requested_output_tokens = wanted;
        request.execution.sampling.temperature = 0.0f;
        request.execution.sampling.presence_penalty = 0.0f;
        request.execution.sampling.frequency_penalty = 0.0f;
        request.execution.allow_prefix_reuse = true;
        request.stop.include_model_defaults = false;
        request.output.raw = true;
        std::vector<ninfer::GenerationSubmission> submissions;
        for(unsigned row=0;row<2;++row){
            std::vector<ninfer::TokenId> ids;
            std::uint32_t rng=17+row;
            for(unsigned i=0;i<768;++i){
                rng=rng*1664525U+1013904223U;
                ids.push_back(static_cast<ninfer::TokenId>(1000U+(rng>>8U)%30000U));
            }
            submissions.push_back({.prompt=engine.prepare_tokens(std::move(ids)),
                                   .options=request});
        }
        auto handles=engine.submit_group(std::move(submissions));
        for(auto& handle:handles){
            const auto result=handle.wait();
            std::cout<<"pressure fixture: prompt="<<result.prompt.prompt_tokens
                     <<" output="<<result.generated_token_ids.size()
                     <<" finish="<<static_cast<int>(result.finish_reason)<<'\n';
            if(result.prompt.prompt_tokens!=768 || result.generated_token_ids.size()!=wanted ||
               result.finish_reason!=ninfer::FinishReason::OutputLimit)
                throw std::runtime_error("request did not retain its original prompt and full output budget");
        }
        if(!replays.load()) throw std::runtime_error("fixture did not exercise live context replay");
        if(replays.load()>2) throw std::runtime_error("one-shot replay bound exceeded");
        std::cout<<"live pressure replay: "<<replays.load()<<" requeues, both full budgets completed\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
