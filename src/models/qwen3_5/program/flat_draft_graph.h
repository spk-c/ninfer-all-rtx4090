#pragma once

#include "core/arena.h"
#include "models/qwen3_5/program/graph_bucket_site.h"

#include <array>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Explicitly recorded proposal readouts. The target verifier and the rest of the
// round retain their existing graph profile and measured routing. During the
// first capture we collect value-owned operand bindings; standalone patches use
// a borrowed arena at exactly the original offset. The second capture records
// only those readouts as inline slices of the parent.
class FlatDraftGraph {
public:
    using BindingKey = std::array<const void*, 10>;
    using Capture = std::function<void(ops::ContextBucket, WorkspaceArena&, cudaStream_t)>;

    explicit FlatDraftGraph(ops::ContextBucketRange range) : range_(range), selected_(range.last) {
        range_.validate();
    }

    void readout(WorkspaceArena& work, cudaStream_t stream, BindingKey key, Capture capture) {
        if (work.rank_count() != 1) throw std::logic_error("flat draft readout requires one rank");
        if (collecting_) {
            bindings_.push_back({work.base(), work.capacity(), work.used(), key, std::move(capture), {}});
            bindings_.back().capture(range_.last, work, stream);
            return;
        }
        if (cursor_ >= bindings_.size()) throw std::logic_error("extra flat draft readout");
        auto& binding = bindings_[cursor_++];
        if (binding.base != work.base() || binding.bytes != work.capacity() ||
            binding.offset != work.used() || binding.key != key) {
            throw std::logic_error("flat draft readout workspace layout changed between captures");
        }
        binding.site.record(stream, [&](auto bucket) { capture(bucket, work, stream); }, range_.last);
    }

    void prepare(cudaStream_t stream) {
        if (!collecting_ || bindings_.empty()) throw std::logic_error("no flat draft readouts collected");
        for (auto& binding : bindings_) {
            binding.site.prepare(stream, range_, [&](auto bucket) {
                WorkspaceArena borrowed(DeviceSpan{binding.base, binding.bytes});
                if (binding.offset) (void)borrowed.alloc_bytes(binding.offset, 1);
                binding.capture(bucket, borrowed, stream);
            });
        }
        collecting_ = false;
        cursor_ = 0;
    }

    void finish() {
        if (collecting_ || cursor_ != bindings_.size())
            throw std::logic_error("missing flat draft readout during parent capture");
        for (auto& binding : bindings_) binding.capture = {};
        ready_ = true;
    }

    void reset_selection() noexcept { selected_ = range_.last; }

    // The original parent always uses the largest envelope, so a failed patch
    // can safely revert to it. Reinstantiate before replaying any partially
    // patched executable. Disable this site's optimization for its lifetime.
    void select(DecodeGraphExecutable& executable, const DecodeGraphDefinition& parent,
                ops::ContextBucket bucket) {
        if (!ready_ || !ops::valid_context_bucket(bucket) ||
            bucket < range_.first || bucket > range_.last)
            throw std::logic_error("flat draft bucket outside prepared profile");
        if (disabled_ || selected_ == bucket) return;
        try {
            for (auto& binding : bindings_) binding.site.select(executable, bucket);
            selected_ = bucket;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "flat draft graph fallback: %s\n", error.what());
            executable.instantiate(parent);
            selected_ = range_.last;
            disabled_ = true;
        }
    }

    [[nodiscard]] bool ready() const noexcept { return ready_; }
    [[nodiscard]] bool disabled() const noexcept { return disabled_; }
    [[nodiscard]] ops::ContextBucketRange range() const noexcept { return range_; }
    [[nodiscard]] std::size_t sites() const noexcept { return bindings_.size(); }

private:
    struct Binding {
        void* base;
        std::size_t bytes, offset;
        BindingKey key;
        Capture capture;
        GraphBucketSite site;
    };
    ops::ContextBucketRange range_;
    ops::ContextBucket selected_;
    std::vector<Binding> bindings_;
    std::size_t cursor_ = 0;
    bool collecting_ = true, ready_ = false, disabled_ = false;
};

} // namespace ninfer::models::qwen3_5::detail
