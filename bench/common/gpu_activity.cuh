#pragma once

// Adapted from Neroued/ninfer bench/common/gpu_activity.cuh (Apache-2.0).
// Register-only conditioning before an Op: does not adjust clocks or power limits.
#include "core/device.h"
#include <cstdint>

namespace ninfer::bench {
namespace detail {
__device__ inline unsigned long long gpu_global_nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
static __global__ void gpu_condition_kernel(unsigned long long nanoseconds) {
    const auto begin = gpu_global_nanoseconds();
    float value = float(threadIdx.x) + 1.F;
    do {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            value = __fmaf_rn(value, 1.000001F, 0.000001F);
        }
        asm volatile("" : "+f"(value));
    } while (gpu_global_nanoseconds() - begin < nanoseconds);
}
} // namespace detail

inline void condition_gpu(DeviceExecutionView execution, int microseconds) {
    if (microseconds <= 0 || execution.multiprocessor_count <= 0) { return; }
    detail::gpu_condition_kernel<<<2 * execution.multiprocessor_count, 256, 0,
                                   execution.stream>>>(
        static_cast<unsigned long long>(microseconds) * 1000ULL);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::bench
