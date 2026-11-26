#pragma once

#include "runtime/contract/request.h"
#include "runtime/engine/generation_budget.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::runtime {

// Only replay committed text at an idle decode boundary. More complicated
// output modes have extra model/grammar state not restored by this path.
inline bool pressure_replay_options_eligible(const ResolvedRequestOptions& options,
                                              const PromptSummary& prompt) noexcept {
    const auto& e = options.execution;
    const auto& s = e.sampling;
    return !prompt.has_media && !prompt.starts_in_reasoning &&
           !e.branches && !e.grammar && e.bad_words.empty() &&
           e.structured_output.kind == StructuredOutputKind::None &&
           !e.post_thinking_sampling && !e.thinking.budget && !e.thinking.effective_budget &&
           !e.thinking.answer_now && e.thinking.loop == ReasoningLoopAction::Off &&
           !e.logprobs && e.readout_tokens.empty() &&
           options.ngram_session.key.empty() && options.ngram_session.parent.empty() &&
           s.temperature == 0.0F && s.presence_penalty == 0.0F && s.frequency_penalty == 0.0F &&
           !s.gumbel && e.allow_prefix_reuse && e.requested_output_tokens > 1;
}

inline bool pressure_replay_eligible(const ResolvedRequestOptions& options,
                                      const PromptSummary& prompt, const GenerationBudget& budget,
                                      std::size_t generated, std::uint32_t reasoning_tokens,
                                      std::uint32_t control_suffix, bool already_requeued,
                                      bool has_archive, bool other_active, bool transaction) noexcept {
    const auto& e = options.execution;
    return pressure_replay_options_eligible(options, prompt) &&
           other_active && !transaction && !already_requeued && !has_archive &&
           reasoning_tokens == 0 && control_suffix == 0 &&
           generated != 0 && budget.remaining() != 0 &&
           budget.limit_reason() == FinishReason::OutputLimit &&
           generated < e.requested_output_tokens &&
           budget.remaining() == e.requested_output_tokens - generated;
}
} // namespace ninfer::runtime
