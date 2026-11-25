#pragma once

#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention_envelope.h"
#include <algorithm>
#include <cstdlib>
#include <string_view>

namespace ninfer::models::qwen3_5::detail {

// One process-wide selection for both workspace planning and graph capture.
inline bool draft_attention_buckets_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SM89_ATTENTION_BUCKETS");
        return value && std::string_view(value) == "1";
    }();
    return enabled;
}

inline ops::SlidingWindowAttentionExecutionEnvelope draft_sliding_envelope(std::uint32_t maximum) {
    if (draft_attention_buckets_enabled() && maximum <= ops::kContextBucketTokens.back())
        return ops::sliding_window_bucket_envelope(ops::select_context_bucket(maximum));
    return {0, maximum};
}

// Full-context BF16 draft attention uses a paged view, not the cyclic cache.
// Round only the resource promise and clamp the final bucket to the logical capacity;
// the device lengths still determine exactly which keys are read.
inline constexpr ops::ContextAttentionExecutionEnvelope
draft_full_bucket_envelope(std::uint32_t maximum, std::uint32_t capacity) {
    if (maximum > capacity) throw std::invalid_argument("draft context exceeds capacity");
    if (maximum > ops::kContextBucketTokens.back()) return {0, maximum};
    return {0, std::min(capacity, ops::context_bucket_tokens(ops::select_context_bucket(maximum)))};
}

inline ops::ContextAttentionExecutionEnvelope
draft_full_envelope(std::uint32_t maximum, std::uint32_t capacity) {
    if (draft_attention_buckets_enabled()) return draft_full_bucket_envelope(maximum, capacity);
    return {0, maximum};
}

} // namespace ninfer::models::qwen3_5::detail
