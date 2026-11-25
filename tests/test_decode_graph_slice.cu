#include "core/decode_graph.h"
#include "core/device_launch.h"
#include "cuda_availability.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace {
__global__ void append_digit(unsigned* p, unsigned digit) {
    if (blockIdx.x == 0 && threadIdx.x == 0) p[0] = 10U * p[0] + digit;
    if (threadIdx.x == 0) atomicAdd(p + 1, 1U);
}
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Resources {
    cudaStream_t stream = nullptr;
    unsigned* values = nullptr;
    ~Resources() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (values) (void)cudaFree(values);
        if (stream) (void)cudaStreamDestroy(stream);
    }
};
} // namespace

int main() {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (ninfer::test::cuda_unavailable(status) || (status == cudaSuccess && count == 0)) return 77;
    try {
        CUDA_CHECK(status);
        CUDA_CHECK(cudaSetDevice(0));
        Resources r;
        CUDA_CHECK(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&r.values, 2 * sizeof(unsigned)));
        CUDA_CHECK(cudaMemsetAsync(r.values, 0, 2 * sizeof(unsigned), r.stream));
        append_digit<<<1, 1, 0, r.stream>>>(r.values, 0);
        CUDA_CHECK(cudaStreamSynchronize(r.stream));
        ninfer::DecodeGraphDefinition parent;
        ninfer::DecodeGraphKernelSlice slice;
        parent.capture(r.stream, [&] {
            append_digit<<<1, 1, 0, r.stream>>>(r.values, 1);
            slice.capture(r.stream, [&] {
                append_digit<<<1, 1, 0, r.stream>>>(r.values, 2);
                append_digit<<<1, 1, 0, r.stream>>>(r.values, 3);
            });
            append_digit<<<1, 1, 0, r.stream>>>(r.values, 4);
        });
        ninfer::DecodeGraphKernelPatch patch;
        patch.capture(r.stream, [&] {
            append_digit<<<2, 1, 0, r.stream>>>(r.values, 7);
            append_digit<<<3, 1, 0, r.stream>>>(r.values, 8);
        });
        require(slice.size() == 2 && patch.params().size() == 2, "kernel chain length");
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(parent);
        const auto run = [&](unsigned digits, unsigned blocks) {
            CUDA_CHECK(cudaMemsetAsync(r.values, 0, 2 * sizeof(unsigned), r.stream));
            executable.launch(r.stream);
            CUDA_CHECK(cudaStreamSynchronize(r.stream));
            unsigned out[2]{};
            CUDA_CHECK(cudaMemcpy(out, r.values, sizeof(out), cudaMemcpyDeviceToHost));
            require(out[0] == digits && out[1] == blocks, "patched graph values or grid");
        };
        run(1234, 4);
        executable.update_slice(slice, patch);
        run(1784, 7);
        ninfer::DecodeGraphDefinition wrong;
        wrong.capture(r.stream, [&] { append_digit<<<1, 1, 0, r.stream>>>(r.values, 0); });
        executable.instantiate(wrong);
        bool rejected = false;
        try { executable.update_slice(slice, patch); } catch (const std::logic_error&) { rejected = true; }
        require(rejected, "slice of another graph accepted");
        for (bool empty : {true, false}) {
            rejected = false;
            try {
                patch.capture(r.stream, [&] {
                    if (!empty) CUDA_CHECK(cudaMemsetAsync(r.values, 0, sizeof(unsigned), r.stream));
                });
            } catch (const std::logic_error&) { rejected = true; }
            require(rejected && !patch.ready(), "invalid kernel patch retained");
        }
        std::cout << "CUDA kernel slice: patch args/grid/order and invalid graph guards passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
