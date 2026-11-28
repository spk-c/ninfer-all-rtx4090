#pragma once
// Experimental Inkling2 rank-one CUDA math. Not connected to executor or cache.
// Token-major FP32 rows, already selected/validated IDs, immutable factors.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::models::inkling2::rank1 {
constexpr int threads = 128;
static __global__ void embedding_add(float* y, const int* ids,
                                    const float* a, const float* b,
                                    int tokens, int hidden, float scale) {
    const int t = blockIdx.x;
    if (t >= tokens) return;
    const float z = scale * a[ids[t]];
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
        y[(std::int64_t)t * hidden + j] += z * b[j];
}

// One reduction per token, reused by all output columns.
static __global__ void attention_dot(const float* x, const float* a,
                                    float* dots, int width) {
    __shared__ float sums[threads];
    const int t = blockIdx.x;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += threads)
        sum += x[(std::int64_t)t * width + j] * a[j];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int k = threads / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sums[threadIdx.x] += sums[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[t] = sums[0];
}

static __global__ void attention_add(float* y, const float* dots,
                                     const float* b, int hidden, float scale) {
    const int t = blockIdx.x;
    const float z = scale * dots[t];
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
        y[(std::int64_t)t * hidden + j] += z * b[j];
}

// Pair-major FP32 expert activations; a separate factor for each selected
// expert. Route weights (including the existing down_scale) are NOT applied here.
static __global__ void expert_dot(const float* middle, const int* ids,
                                 const float* a, float* dots, int width) {
    __shared__ float sums[threads];
    const int p = blockIdx.x, e = ids[p];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += threads)
        sum += middle[(std::int64_t)p * width + j] * a[(std::int64_t)e * width + j];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int k = threads / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sums[threadIdx.x] += sums[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[p] = sums[0];
}

// Add to unweighted per-pair expert products before existing route reduction.
// An executor using the q8_1/fixed-point fused path needs a separate integration
// and numerical test; do not use this kernel on its accumulator.
static __global__ void expert_add(float* rows, const int* ids, const float* dots,
                                  const float* b, int hidden, float scale) {
    const int p = blockIdx.x, e = ids[p];
    const float z = scale * dots[p];
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
        rows[(std::int64_t)p * hidden + j] += z * b[(std::int64_t)e * hidden + j];
}
// BF16 operands match the executor's matrix projection path.
static __global__ void dot_bf16(const __nv_bfloat16* x, const int* ids,
                                const float* a, float* dots, int width) {
    __shared__ float sums[threads];
    const int row = blockIdx.x;
    const int e = ids ? ids[row] : 0;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += threads)
        sum += __bfloat162float(x[(std::int64_t)row * width + j]) *
               a[(std::int64_t)e * width + j];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int k = threads / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sums[threadIdx.x] += sums[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[row] = sums[0];
}
// Planar Q8_1 values use the same stored half scale as the base vector product.
// This is deliberately not the transposed PTQ1_0 representation.
static __global__ void dot_q8(const std::int8_t* qs, const half2* ds, const int* ids,
                              const float* a, float* dots, int width) {
    __shared__ float sums[threads];
    const int row = blockIdx.x;
    const int e = ids ? ids[row] : 0;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += threads) {
        const auto i = std::int64_t(row) * width + j;
        const float d = __low2float(ds[i / 32]);
        sum += (float(qs[i]) * d) * a[std::int64_t(e) * width + j];
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int k = threads / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sums[threadIdx.x] += sums[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[row] = sums[0];
}
} // namespace ninfer::models::inkling2::rank1
