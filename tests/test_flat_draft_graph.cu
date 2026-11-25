#include "models/qwen3_5/program/flat_draft_graph.h"
#include "core/device_launch.h"
#include "cuda_availability.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace {
using ninfer::models::qwen3_5::detail::FlatDraftGraph;
using ninfer::ops::ContextBucket;
using ninfer::ops::ContextBucketRange;

__global__ void write_digit(unsigned* p, unsigned digit) { *p = digit; }
__global__ void append_digit(unsigned* p, unsigned digit) { *p = *p * 10U + digit; }
__global__ void append_scratch(unsigned* p, const unsigned* scratch) { *p = *p * 10U + *scratch; }

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Resources {
    cudaStream_t stream = nullptr;
    unsigned* value = nullptr;
    ~Resources() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (value) (void)cudaFree(value);
        if (stream) (void)cudaStreamDestroy(stream);
    }
};
}

int main() {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (ninfer::test::cuda_unavailable(status) || (status == cudaSuccess && count == 0)) return 77;
    try {
        CUDA_CHECK(status);
        CUDA_CHECK(cudaSetDevice(0));
        Resources r;
        CUDA_CHECK(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&r.value, sizeof(unsigned)));
        ninfer::WorkspaceArena work(4096);
        FlatDraftGraph flat(ContextBucketRange{ContextBucket::K2, ContextBucket::K8});
        void* scratch_addresses[2]{};
        const auto body = [&] {
            work.reset();
            append_digit<<<1, 1, 0, r.stream>>>(r.value, 1);
            for (unsigned site = 0; site != 2; ++site) {
                auto scope = work.scope();
                (void)work.alloc_bytes(site ? 401 : 13, 1);
                auto* const expected_slot = &scratch_addresses[site];
                auto* const result = r.value;
                flat.readout(work, r.stream, {result, expected_slot},
                    [=](ContextBucket bucket, ninfer::WorkspaceArena& arena, cudaStream_t stream) {
                        auto inner = arena.scope();
                        auto* scratch = static_cast<unsigned*>(arena.alloc_bytes(sizeof(unsigned)).data);
                        if (!*expected_slot) *expected_slot = scratch;
                        require(*expected_slot == scratch, "borrowed arena changed an operand address");
                        const unsigned digit = (bucket == ContextBucket::K2 ? 2U : 7U) + site;
                        write_digit<<<1, 1, 0, stream>>>(scratch, digit);
                        append_scratch<<<1, 1, 0, stream>>>(result, scratch);
                    });
            }
            append_digit<<<1, 1, 0, r.stream>>>(r.value, 4);
        };
        // Load all modules before the first capture.
        append_digit<<<1,1,0,r.stream>>>(r.value,0);
        write_digit<<<1,1,0,r.stream>>>(r.value,0);
        append_scratch<<<1,1,0,r.stream>>>(r.value,r.value);
        CUDA_CHECK(cudaStreamSynchronize(r.stream));
        ninfer::DecodeGraphDefinition parent;
        parent.capture(r.stream, body);
        flat.prepare(r.stream);
        parent.capture(r.stream, body);
        flat.finish();
        require(flat.ready() && flat.sites() == 2, "missing flat readouts");
        require(scratch_addresses[0] != scratch_addresses[1], "test scratch addresses overlap");
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(parent);
        const auto run = [&](unsigned expected) {
            CUDA_CHECK(cudaMemsetAsync(r.value, 0, sizeof(unsigned), r.stream));
            executable.launch(r.stream);
            CUDA_CHECK(cudaStreamSynchronize(r.stream));
            unsigned value{};
            CUDA_CHECK(cudaMemcpy(&value, r.value, sizeof(value), cudaMemcpyDeviceToHost));
            require(value == expected, "flat graph result or non-readout node changed");
        };
        run(1784);
        flat.select(executable, parent, ContextBucket::K2);
        run(1234);
        flat.select(executable, parent, ContextBucket::K2);
        run(1234);
        flat.select(executable, parent, ContextBucket::K8);
        run(1784);
        // Wrong executable: the first attempted patch is rejected, restoring the
        // safe largest-envelope parent and disabling the patch path.
        ninfer::DecodeGraphDefinition wrong;
        wrong.capture(r.stream, [&]{ append_digit<<<1,1,0,r.stream>>>(r.value,9); });
        executable.instantiate(wrong);
        flat.select(executable, parent, ContextBucket::K2);
        run(1784);
        flat.select(executable, parent, ContextBucket::K2);
        run(1784);
        // The second parent capture must reproduce both operand keys and offsets.
        FlatDraftGraph mismatch(ContextBucketRange{ContextBucket::K2, ContextBucket::K8});
        ninfer::DecodeGraphDefinition discarded;
        const auto collect = [&](bool changed) {
            work.reset();
            mismatch.readout(work, r.stream, {changed ? nullptr : r.value},
                [&](ContextBucket, ninfer::WorkspaceArena&, cudaStream_t stream) {
                    append_digit<<<1,1,0,stream>>>(r.value,1);
                });
        };
        discarded.capture(r.stream,[&]{collect(false);});
        mismatch.prepare(r.stream);
        bool rejected = false;
        try { discarded.capture(r.stream,[&]{collect(true);}); }
        catch(const std::logic_error&) { rejected = true; }
        require(rejected && !discarded.ready(), "changed operands accepted in parent capture");
        std::cout << "flat draft graph: two readouts, stable borrowed scratch, six replays and fallback guards passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
