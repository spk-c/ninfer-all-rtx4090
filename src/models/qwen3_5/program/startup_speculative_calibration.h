#pragma once

#include "models/qwen3_5/program/graph_profile_index.h"
#include "ninfer/ops/context_bucket.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct SpeculativeProbe {
    std::uint32_t batch, window, frontier;
    bool operator==(const SpeculativeProbe&) const = default;
};

// Use only actual captured profiles. Short/medium contexts exercise the same
// startup backing as graph warmup; do not pretend repeated pages are real 128K.
inline std::vector<SpeculativeProbe> startup_speculative_probes(
    std::span<const GraphProfileIndex::Entry> profiles, std::uint32_t capacity,
    std::uint32_t concurrency) {
    std::vector<SpeculativeProbe> out;
    const std::uint32_t corners[]{1U, std::min(4U, concurrency), concurrency};
    for (auto batch : corners) {
        if (!batch) continue;
        for (auto frontier : {64U, 2112U}) {
            for (const auto& p : profiles) {
                if (p.batch != batch || !p.window || frontier < p.first ||
                    frontier > p.last || std::uint64_t(frontier) + p.window + 1 > capacity)
                    continue;
                const SpeculativeProbe probe{batch, p.window, frontier};
                if (std::ranges::find(out, probe) == out.end()) out.push_back(probe);
            }
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::detail
