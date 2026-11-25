#pragma once

#include "models/qwen3_5/program/graph_profile_index.h"
#include "ninfer/ops/context_bucket.h"

#include <algorithm>
#include <array>
#include <limits>
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
    std::uint32_t concurrency, bool physical = false) {
    std::vector<SpeculativeProbe> out;
    const std::uint32_t corners[]{1U, std::min(4U, concurrency), concurrency};
    constexpr std::array frontiers{64U, 2112U, 8192U, 32768U};
    for (auto batch : corners) {
        if (!batch) continue;
        for (auto frontier : std::span(frontiers).first(physical ? frontiers.size() : 2U)) {
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

// Limit distinct physical backing, including any pages retained by earlier probes.
// This is a page count, not a new GPU allocation: the existing pool owns it.
inline constexpr std::uint32_t kStartupPhysicalPageBudget = 1024;

inline bool startup_physical_backing_fits(std::span<const std::uint32_t> mapped,
                                         std::uint32_t batch, std::uint32_t required,
                                         std::uint32_t free_pages,
                                         std::uint32_t row_capacity,
                                         std::uint32_t budget = kStartupPhysicalPageBudget) {
    if (!batch || batch > mapped.size() || !required || required > row_capacity) return false;
    std::uint64_t total = 0, growth = 0;
    for (std::size_t i = 0; i < mapped.size(); ++i) {
        const auto after = i < batch ? std::max(mapped[i], required) : mapped[i];
        total += after;
        growth += after - mapped[i];
    }
    return total <= budget && growth <= free_pages;
}

} // namespace ninfer::models::qwen3_5::detail
