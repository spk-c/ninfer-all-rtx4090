#pragma once
// Diagnostic patch only: synchronous raw copies perturb execution and are not benchmarks.
#include "qwen_target_trace.h"
#include "models/qwen3_5/state/state_image.h"
#include "models/qwen3_5/state/decoder_state.h"
#include <algorithm>
#include <sstream>

namespace ninfer::diagnostic {
inline void cache_trace_split(unsigned cursor, std::optional<unsigned>& split) {
    const char* forced = std::getenv("NINFER_DIAGNOSTIC_SPLITS");
    if (!forced) return;
    std::istringstream input(forced);
    std::string value;
    while (std::getline(input, value, ',')) {
        const auto frontier = static_cast<unsigned>(std::stoul(value));
        if (frontier > cursor && (!split || frontier < *split)) split = frontier;
    }
}
inline void cache_state_trace(const std::string& kind,
                              models::qwen3_5::StateImageDevicePool& pool, int slot,
                              const models::qwen3_5::PagedKVCacheView& cache,
                              unsigned attention_layers, unsigned frontier, cudaStream_t stream) {
    if (!std::getenv("NINFER_TARGET_TRACE")) return;
    if (pool.shard_count() != 1) throw std::runtime_error("cache diagnostic requires single rank");
    auto& linear = pool.linear();
    for (unsigned layer = 0; layer < linear.layer_count(); ++layer) {
        target_trace(kind + "-gdn-" + std::to_string(layer),
                     {{"conv", linear.conv_slot(layer, slot)},
                      {"recurrent", linear.recurrent_slot(layer, slot)}}, stream);
    }
    const auto state = pool.slot_view(slot);
    target_trace(kind + "-state", {{"hidden", state.continuation_hidden}}, stream);
    if (const auto* local = pool.dflash_local()) {
        for (unsigned layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            target_trace(kind + "-draft-" + std::to_string(layer),
                         {{"k", view.k.slice(3, slot, 1)}, {"v", view.v.slice(3, slot, 1)}}, stream);
        }
    }
    // Read only valid logical tokens, never unused physical pages or partial-page tails.
    for (unsigned layer = 0; layer < attention_layers; ++layer) {
        const auto view = cache.layer_view(layer);
        const unsigned pages = (frontier + kPagedKVPageSize - 1) / kPagedKVPageSize;
        std::vector<int> table(pages);
        if (pages && (cudaMemcpyAsync(table.data(), view.block_table.data, pages * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                      cudaStreamSynchronize(stream) != cudaSuccess))
            throw std::runtime_error("cache trace table transfer failed");
        for (const auto& [name, plane] : std::initializer_list<std::pair<const char*, Tensor>>{
                 {"k", view.k_pages}, {"v", view.v_pages},
                 {"kscale", view.k_scale_pages}, {"vscale", view.v_scale_pages}}) {
            if (!plane.data || !frontier) continue;
            const bool page_major = plane.ne[2] == view.num_kv_heads;
            const auto row_bytes = static_cast<std::size_t>(plane.nb[1]);
            std::vector<char> bytes(row_bytes * frontier * view.num_kv_heads);
            for (unsigned page = 0; page < pages; ++page) {
                if (table[page] < 0) throw std::runtime_error("unmapped cache trace page");
                const unsigned count = std::min<unsigned>(kPagedKVPageSize, frontier - page * kPagedKVPageSize);
                for (int head = 0; head < view.num_kv_heads; ++head) {
                    const auto* source = static_cast<const char*>(plane.data) +
                        table[page] * plane.nb[page_major ? 3 : 2] + head * plane.nb[page_major ? 2 : 3];
                    auto* destination = bytes.data() + (head * frontier + page * kPagedKVPageSize) * row_bytes;
                    if (cudaMemcpyAsync(destination, source, count * row_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                        throw std::runtime_error("cache trace KV transfer failed");
                }
            }
            if (cudaStreamSynchronize(stream) != cudaSuccess) throw std::runtime_error("cache trace sync failed");
            // Upload-free host output has the same metadata framing as target_trace.
            static std::atomic<unsigned> serial{0};
            const auto id = serial++;
            const auto root = std::filesystem::path(std::getenv("NINFER_TARGET_TRACE"));
            const auto filename = "kv-" + std::to_string(id) + "-" + kind + "-" + std::to_string(layer) + "-" + name + ".bin";
            std::ofstream output(root / filename, std::ios::binary);
            output.write(bytes.data(), bytes.size());
            if (!output) throw std::runtime_error("cache trace KV write failed");
            std::ofstream index(root / "kv.jsonl", std::ios::app);
            index << "{\"kind\":\"" << kind << "\",\"layer\":" << layer << ",\"name\":\"" << name
                  << "\",\"file\":\"" << filename << "\",\"dtype\":" << int(plane.dtype)
                  << ",\"leading\":" << plane.ne[0] << ",\"frontier\":" << frontier
                  << ",\"heads\":" << view.num_kv_heads << "}\n";
        }
    }
}
} // namespace ninfer::diagnostic
