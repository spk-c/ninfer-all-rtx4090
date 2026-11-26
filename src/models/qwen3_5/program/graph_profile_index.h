#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Host-only index shared by all decode graph families. The captured profile bounds
// are authoritative; canonical context buckets never silently replace kernel envelopes.
class GraphProfileIndex {
public:
    struct Entry {
        std::uint32_t batch, window, first, last, topology;
        std::size_t profile;
    };

    void build(std::vector<Entry> entries, std::uint32_t max_frontier) {
        std::ranges::sort(entries, {}, [](const Entry& e) {
            return std::tuple{e.batch, e.window, e.first};
        });
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const auto& e = entries[i];
            const auto same_group = [](const Entry& a, const Entry& b) {
                return a.batch == b.batch && a.window == b.window;
            };
            const bool first = i == 0 || !same_group(entries[i-1], e);
            const bool last = i + 1 == entries.size() || !same_group(e, entries[i+1]);
            if (!e.batch || e.first > e.last || e.last > max_frontier ||
                (first ? e.first != 0 : std::uint64_t(entries[i-1].last) + 1 != e.first) ||
                (last && e.last != max_frontier))
                throw std::logic_error("decode graph index has incomplete or overlapping coverage");
        }
        entries_ = std::move(entries);
    }

    [[nodiscard]] const Entry& select(std::uint32_t batch, std::uint32_t window,
                                      std::uint32_t frontier) const {
        const auto key = std::tuple{batch, window, frontier};
        const auto it = std::ranges::upper_bound(entries_, key, {}, [](const Entry& e) {
            return std::tuple{e.batch, e.window, e.first};
        });
        if (it == entries_.begin())
            throw std::logic_error("decode graph profile unavailable");
        const auto& entry = *std::prev(it);
        if (entry.batch != batch || entry.window != window || frontier > entry.last)
            throw std::logic_error("decode graph profile unavailable");
        return entry;
    }

private:
    std::vector<Entry> entries_;
};

} // namespace ninfer::models::qwen3_5::detail
