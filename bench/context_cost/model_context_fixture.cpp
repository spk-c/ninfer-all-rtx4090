#include "model_context_fixture.h"
#include "core/wide_math.h"

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/device_buffer.h"
#include "core/host_kv_arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/pinned_memory.h"
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ninfer::bench::context_cost {
namespace {

std::size_t checked_size_mul(std::size_t left, std::size_t right, const char* label) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(label);
    }
    return left * right;
}

std::filesystem::path existing_input_path(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) { return path; }
#ifdef NINFER_SOURCE_DIR
    if (path.is_relative()) {
        const std::filesystem::path source = std::filesystem::path(NINFER_SOURCE_DIR) / path;
        if (std::filesystem::exists(source)) { return source; }
    }
#endif
    throw std::runtime_error("input path does not exist: " + path.string());
}

std::vector<TokenId> load_corpus(const std::filesystem::path& path) {
    const std::filesystem::path resolved = existing_input_path(path);
    std::ifstream input(resolved);
    if (!input) { throw std::runtime_error("failed to open token corpus: " + resolved.string()); }
    std::vector<TokenId> tokens;
    std::int64_t value = 0;
    while (input >> value) {
        if (value < 0 || value > std::numeric_limits<TokenId>::max()) {
            throw std::invalid_argument("token corpus contains an out-of-range token id");
        }
        tokens.push_back(static_cast<TokenId>(value));
    }
    if (tokens.empty()) { throw std::invalid_argument("token corpus is empty"); }
    return tokens;
}

std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, std::uint32_t count) {
    if (count == 0 || count > corpus.size()) {
        throw std::invalid_argument("requested prompt slice exceeds the token corpus");
    }
    return {corpus.begin(), corpus.begin() + static_cast<std::ptrdiff_t>(count)};
}

RequestOptions one_token_request() {
    RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.allow_prefix_reuse      = false;
    request.execution.sampling.temperature    = 0.0F;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    request.output.preserve_special_tokens    = true;
    return request;
}

struct TextCase {
    std::uint32_t prefix = 0;
    std::uint32_t suffix = 0;
    bool validation      = false;
};

std::vector<TextCase> text_cases(std::uint32_t chunk) {
    if (chunk < 8) { throw std::invalid_argument("prefill chunk is too small for calibration"); }
    return {
        {.prefix = 0, .suffix = chunk / 8, .validation = false},
        {.prefix = 0, .suffix = chunk / 2, .validation = false},
        {.prefix = 0, .suffix = chunk, .validation = false},
        {.prefix = 0, .suffix = 2 * chunk, .validation = false},
        {.prefix = chunk, .suffix = chunk / 8, .validation = false},
        {.prefix = chunk, .suffix = chunk / 2, .validation = false},
        {.prefix = 4 * chunk, .suffix = chunk / 8, .validation = false},
        {.prefix = 4 * chunk, .suffix = chunk, .validation = false},
        {.prefix = 0, .suffix = 3 * chunk / 4, .validation = true},
        {.prefix = 2 * chunk, .suffix = chunk / 4, .validation = true},
        {.prefix = 2 * chunk, .suffix = 3 * chunk / 2, .validation = true},
        {.prefix = 6 * chunk, .suffix = chunk / 2, .validation = true},
    };
}

std::uint64_t attention_pairs(std::uint32_t prefix, std::uint32_t suffix) {
    // Both inputs are 32-bit, so the triangular term stays below 2^63 and needs no wide
    // intermediate; only the sum with prefix*suffix can leave uint64.
    const std::uint64_t triangular = static_cast<std::uint64_t>(suffix) * (suffix + 1ULL) / 2ULL;
    bool overflowed                = false;
    const ninfer::core::Uint128 pairs =
        ninfer::core::wide_add(ninfer::core::wide_multiply(prefix, suffix),
                               ninfer::core::Uint128{triangular, 0}, overflowed);
    if (overflowed || !ninfer::core::wide_fits_uint64(pairs)) {
        throw std::overflow_error("prefill attention-pair count exceeds uint64");
    }
    return pairs.low;
}

std::vector<std::uint8_t> block_ppm(int width, int height, std::uint8_t value) {
    if (width <= 0 || height <= 0) { throw std::invalid_argument("invalid synthetic image size"); }
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    const std::size_t pixels =
        checked_size_mul(static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                         "synthetic image pixel count");
    std::vector<std::uint8_t> ppm;
    ppm.reserve(header.size() + checked_size_mul(pixels, 3, "synthetic image bytes"));
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    ppm.insert(ppm.end(), pixels * 3U, value);
    return ppm;
}

PromptInput image_prompt(int dimension) {
    MessagePart image;
    image.kind       = MessagePartKind::Media;
    image.media.kind = MediaKind::Image;
    image.media.bytes =
        block_ppm(dimension, dimension, static_cast<std::uint8_t>(dimension & 0xff));
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "context-cost-" + std::to_string(dimension) + ".ppm";

    ChatMessage message;
    message.role = ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(MessagePart{
        .kind = MessagePartKind::Text, .text = "Describe the image briefly.", .media = {}});
    PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

} // namespace

ArtifactProfile inspect_artifact(const std::filesystem::path& artifact_path) {
    const std::filesystem::path path = existing_input_path(artifact_path);
    ninfer::artifact::Reader reader(path);
    return ArtifactProfile{.path = path};
}

PrefillSuiteResult measure_prefill(const ArtifactProfile& artifact,
                                   const MeasurementOptions& options) {
    const std::vector<TextCase> cases = text_cases(options.prefill_chunk);
    std::uint32_t required_context    = 0;
    std::set<std::uint32_t> root_lengths;
    for (const TextCase& test : cases) {
        if (test.prefix != 0) { root_lengths.insert(test.prefix); }
        root_lengths.insert(test.prefix + test.suffix);
        required_context = std::max(required_context, test.prefix + test.suffix);
    }
    if (required_context > options.max_context) {
        throw std::invalid_argument("prefill calibration grid requires max_context >= " +
                                    std::to_string(required_context));
    }
    const std::vector<TokenId> corpus = load_corpus(options.corpus);
    if (corpus.size() < required_context) {
        throw std::invalid_argument("token corpus is shorter than the prefill calibration grid");
    }

    EngineOptions engine_options;
    engine_options.artifact_path         = artifact.path;
    engine_options.device                = options.device;
    engine_options.max_context           = options.max_context;
    engine_options.kv_capacity           = KvCapacityPolicy::explicit_capacity(options.max_context);
    engine_options.max_concurrency       = 1;
    engine_options.max_pending_requests  = 1;
    engine_options.prefill_chunk         = options.prefill_chunk;
    engine_options.kv_cache              = KvCacheStorage::BFloat16;
    engine_options.speculative.backend   = SpeculativeBackend::None;
    engine_options.enable_vision         = true;
    engine_options.use_cuda_graph        = true;
    engine_options.context_cache.enabled = false;
    engine_options.context_cache.host_state_slots       = 0;
    engine_options.context_cache.host_kv_capacity_bytes = 0;

    Engine engine(engine_options);
    const auto run_root = [&](std::uint32_t tokens) {
        auto prompt                = engine.prepare_tokens(prompt_slice(corpus, tokens), false);
        GenerationResult generated = engine.generate(std::move(prompt), one_token_request());
        if (generated.generated_token_ids.size() != 1 ||
            generated.finish_reason != FinishReason::OutputLimit ||
            generated.prefix_reuse_path != PrefixReusePath::Root ||
            generated.reused_prompt_tokens != 0 || !(generated.timings.prefill_seconds > 0.0)) {
            throw std::runtime_error("text prefill calibration left the expected root path");
        }
        return generated.timings.prefill_seconds * 1'000'000'000.0;
    };

    const std::vector<std::uint32_t> lengths(root_lengths.begin(), root_lengths.end());
    (void)run_root(lengths.front());
    if (lengths.back() != lengths.front()) { (void)run_root(lengths.back()); }
    std::map<std::uint32_t, std::vector<double>> root_samples;
    for (const std::uint32_t length : lengths) {
        root_samples.emplace(length, std::vector<double>{});
    }
    for (int repetition = 0; repetition < options.prefill_repetitions; ++repetition) {
        for (std::size_t offset = 0; offset < lengths.size(); ++offset) {
            const std::size_t index =
                (offset + static_cast<std::size_t>(repetition)) % lengths.size();
            root_samples.at(lengths[index]).push_back(run_root(lengths[index]));
        }
    }

    PrefillSuiteResult result;
    result.load = engine.load_summary();
    for (const TextCase& test : cases) {
        std::vector<double> samples;
        const auto& endpoint = root_samples.at(test.prefix + test.suffix);
        if (test.prefix == 0) {
            samples = endpoint;
        } else {
            const auto& base = root_samples.at(test.prefix);
            for (std::size_t index = 0; index < endpoint.size(); ++index) {
                const double difference = endpoint[index] - base[index];
                if (!(difference > 0.0)) {
                    throw std::runtime_error(
                        "paired root prefill produced a nonpositive suffix time");
                }
                samples.push_back(difference);
            }
        }
        result.measurements.push_back(PrefillMeasurement{
            .label = "text-b" + std::to_string(test.prefix) + "-s" + std::to_string(test.suffix),
            .prefix_tokens   = test.prefix,
            .suffix_tokens   = test.suffix,
            .chunks          = (test.suffix + options.prefill_chunk - 1U) / options.prefill_chunk,
            .attention_pairs = attention_pairs(test.prefix, test.suffix),
            .vision_items    = 0,
            .vision_patches  = 0,
            .validation      = test.validation,
            .timing          = summarize_samples(std::move(samples)),
        });
    }

    struct VisionCase {
        int dimension;
        bool validation;
    };

    constexpr std::array<VisionCase, 4> vision_cases{{
        {.dimension = 224, .validation = false},
        {.dimension = 448, .validation = false},
        {.dimension = 672, .validation = false},
        {.dimension = 560, .validation = true},
    }};
    const auto run_vision = [&](int dimension) {
        PreparedPrompt prompt       = engine.prepare(image_prompt(dimension));
        const std::uint64_t patches = prompt.preparation_stats().raw_patches;
        GenerationResult generated  = engine.generate(std::move(prompt), one_token_request());
        if (patches == 0 || generated.generated_token_ids.size() != 1 ||
            generated.finish_reason != FinishReason::OutputLimit || !generated.prompt.has_media ||
            !(generated.timings.vision_seconds > 0.0)) {
            throw std::runtime_error("Vision prefill calibration did not execute Vision");
        }
        return std::pair{patches, generated.timings.vision_seconds * 1'000'000'000.0};
    };
    (void)run_vision(vision_cases.back().dimension);
    std::array<std::vector<double>, vision_cases.size()> vision_samples;
    std::array<std::uint64_t, vision_cases.size()> patches{};
    for (int repetition = 0; repetition < options.prefill_repetitions; ++repetition) {
        for (std::size_t offset = 0; offset < vision_cases.size(); ++offset) {
            const std::size_t index =
                (offset + static_cast<std::size_t>(repetition)) % vision_cases.size();
            const auto [count, nanoseconds] = run_vision(vision_cases[index].dimension);
            if (patches[index] != 0 && patches[index] != count) {
                throw std::runtime_error("synthetic image patch count changed between runs");
            }
            patches[index] = count;
            vision_samples[index].push_back(nanoseconds);
        }
    }
    for (std::size_t index = 0; index < vision_cases.size(); ++index) {
        result.measurements.push_back(PrefillMeasurement{
            .label           = "vision-" + std::to_string(vision_cases[index].dimension),
            .prefix_tokens   = 0,
            .suffix_tokens   = 0,
            .chunks          = 0,
            .attention_pairs = 0,
            .vision_items    = 1,
            .vision_patches  = patches[index],
            .validation      = vision_cases[index].validation,
            .timing          = summarize_samples(std::move(vision_samples[index])),
        });
    }
    return result;
}

} // namespace ninfer::bench::context_cost
