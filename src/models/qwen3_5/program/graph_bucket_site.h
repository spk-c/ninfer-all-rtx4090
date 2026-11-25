#pragma once

#include "core/decode_graph.h"
#include "ninfer/ops/context_bucket.h"

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// A fixed kernel chain with one captured parameter set per context bucket.
// The caller owns operands/workspace and must keep their addresses alive.
// Unlike whole-graph updates, this never changes unrelated graph nodes.
class GraphBucketSite {
public:
    using Capture = std::function<void(ops::ContextBucket)>;

    void prepare(cudaStream_t stream, ops::ContextBucketRange range, const Capture& capture) {
        if (!patches_.empty()) throw std::logic_error("bucket site already prepared");
        range.validate();
        std::vector<DecodeGraphKernelPatch> patches;
        range.for_each([&](auto bucket) {
            patches.emplace_back();
            patches.back().capture(stream, [&] { capture(bucket); });
            if (patches.back().params().size() != patches.front().params().size())
                throw std::logic_error("bucket variants changed kernel topology");
        });
        patches_ = std::move(patches);
        range_ = range;
    }

    void record(cudaStream_t stream, const Capture& capture) {
        record(stream, capture, range_.first);
    }

    void record(cudaStream_t stream, const Capture& capture, ops::ContextBucket initial) {
        if (patches_.empty() || !slice_.empty())
            throw std::logic_error("bucket site must be prepared and recorded once");
        if (!ops::valid_context_bucket(initial) || initial < range_.first || initial > range_.last)
            throw std::invalid_argument("initial bucket outside prepared range");
        slice_.capture(stream, [&] { capture(initial); });
        if (slice_.size() != patches_.front().params().size())
            throw std::logic_error("bucket site parent differs from variants");
    }

    void select(DecodeGraphExecutable& executable, ops::ContextBucket bucket) const {
        if (patches_.empty() || !ops::valid_context_bucket(bucket) ||
            bucket < range_.first || bucket > range_.last)
            throw std::invalid_argument("bucket site was not prepared for this context");
        const auto i = static_cast<std::size_t>(bucket) - static_cast<std::size_t>(range_.first);
        executable.update_slice(slice_, patches_.at(i));
    }

    [[nodiscard]] std::size_t variants() const noexcept { return patches_.size(); }

private:
    ops::ContextBucketRange range_;
    std::vector<DecodeGraphKernelPatch> patches_;
    DecodeGraphKernelSlice slice_;
};

} // namespace ninfer::models::qwen3_5::detail
