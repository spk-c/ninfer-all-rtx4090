#pragma once

#include <cstddef>
#include <optional>

namespace ninfer {

// The free-byte counters are device-wide, not a graph allocator accounting.
// A saturated budget or a nonpositive net change cannot establish the graph cost:
// another process or WDDM budget/eviction changes may mask the allocation.
inline constexpr std::optional<std::size_t> graph_memory_net_decrease(
    std::size_t free_before, std::size_t free_after) noexcept {
    if (free_after == 0 || free_before <= free_after) { return std::nullopt; }
    return free_before - free_after;
}

} // namespace ninfer
