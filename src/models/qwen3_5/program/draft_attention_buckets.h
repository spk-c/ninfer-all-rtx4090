#pragma once

#include "ninfer/ops/sliding_window_attention.h"
#include <cstdlib>
#include <string_view>

namespace ninfer::models::qwen3_5::detail {

// One process-wide selection for both workspace planning and graph capture.
inline ops::SlidingWindowAttentionExecutionEnvelope draft_sliding_envelope(std::uint32_t maximum) {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SM89_ATTENTION_BUCKETS");
        return value && std::string_view(value) == "1";
    }();
    if (enabled && maximum <= ops::kContextBucketTokens.back())
        return ops::sliding_window_bucket_envelope(ops::select_context_bucket(maximum));
    return {0, maximum};
}

} // namespace ninfer::models::qwen3_5::detail
