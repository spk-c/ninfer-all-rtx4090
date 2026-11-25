#include "runtime/engine/pressure_replay.h"
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::runtime;
    ResolvedRequestOptions o;
    o.execution.requested_output_tokens = 100;
    PromptSummary p{.prompt_tokens=4096};
    GenerationBudget b(100, FinishReason::OutputLimit);
    b.commit(40);
    auto eligible = [&](std::size_t generated=40, bool replayed=false, bool archive=false,
                        bool other=true, bool transaction=false, unsigned reasoning=0,
                        unsigned control=0) {
        return pressure_replay_eligible(o,p,b,generated,reasoning,control,replayed,archive,
                                        other,transaction);
    };
    if (!eligible() || eligible(0) || eligible(41) || eligible(40,true) ||
        eligible(40,false,true) || eligible(40,false,false,false) ||
        eligible(40,false,false,true,true) || eligible(40,false,false,true,false,1) ||
        eligible(40,false,false,true,false,0,1)) return 1;
    p.has_media=true; if(eligible())return 2; p.has_media=false;
    p.starts_in_reasoning=true; if(eligible())return 3; p.starts_in_reasoning=false;
    o.execution.logprobs=true; if(eligible())return 4; o.execution.logprobs=false;
    o.execution.sampling.temperature=0.5f; if(eligible())return 5; o.execution.sampling.temperature=0;
    o.execution.sampling.frequency_penalty=0.1f; if(eligible())return 6;
    o.execution.sampling.frequency_penalty=0;
    o.execution.structured_output.kind=StructuredOutputKind::JsonObject;
    if(eligible())return 7; o.execution.structured_output.kind=StructuredOutputKind::None;
    o.execution.bad_words={"x"}; if(eligible())return 8; o.execution.bad_words.clear();
    o.execution.thinking.budget=5; if(eligible())return 9; o.execution.thinking.budget.reset();
    o.ngram_session.key="session"; if(eligible())return 10; o.ngram_session.key.clear();
    if(!eligible())return 11;
    b.cap_remaining(59); if(eligible())return 12;
    std::cout<<"pressure replay: safe-mode eligibility, exact remaining budget and bounded retry passed\n";
}
