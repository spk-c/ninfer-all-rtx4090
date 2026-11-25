#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

namespace ninfer::models::qwen3_5::detail {

// A bounded, non-owning shortlist for immutable capture metadata. A digest is
// never proof of identity: callers must verify exact tokens and model metadata.
template <class Key, class Backing, std::size_t Slots = 128>
class CaptureBackingPool {
    static_assert(Slots > 0);
    struct Entry {
        Key key{};
        std::weak_ptr<const Backing> backing;
    };
    std::array<Entry, Slots> entries_{};
    std::size_t next_ = 0;
    std::mutex mutex_;

public:
    template <class Match, class Make>
    std::shared_ptr<const Backing> get_or_create(const Key& key, Match&& match, Make&& make) {
        std::lock_guard lock(mutex_);
        std::size_t empty = Slots;
        for (std::size_t i = 0; i < Slots; ++i) {
            auto candidate = entries_[i].backing.lock();
            if (!candidate) {
                if (empty == Slots) empty = i;
            } else if (entries_[i].key == key && match(*candidate)) {
                return candidate;
            }
        }
        std::shared_ptr<const Backing> created = make();
        if (empty == Slots) {
            empty = next_;
            next_ = (next_ + 1) % Slots;
        }
        entries_[empty] = Entry{key, created};
        return created;
    }
};

} // namespace ninfer::models::qwen3_5::detail
