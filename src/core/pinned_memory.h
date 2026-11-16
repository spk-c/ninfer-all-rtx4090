#pragma once

// Pinned host memory: owning page-locked buffers, and the registration of existing host
// allocations for transfers and for kernels that read them in place.

#include <cuda_runtime_api.h>

#include <cstddef>

namespace ninfer {

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t size_bytes, bool mapped = false);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() const noexcept;
    std::size_t size() const noexcept;
    // The optional mapped allocation can be read by a CUDA kernel during graph capture.
    // Preserve the existing RTX4090 mapped-transfer implementation in arena.cu.
    void copy_to_device_using_kernel(void* destination, std::size_t bytes,
                                     cudaStream_t stream) const;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
};

// Registers an existing host allocation without moving its bytes. The allocation must outlive
// the registration and every queued transfer; portable registration serves every device rank.
// `mapped` also maps it into the device address space, for kernels that read it in place
// (cudaHostGetDevicePointer gives the address they use).
class HostMemoryRegistration {
public:
    HostMemoryRegistration(void* data, std::size_t bytes, bool mapped = false);
    ~HostMemoryRegistration();
    HostMemoryRegistration(const HostMemoryRegistration&) = delete;
    HostMemoryRegistration& operator=(const HostMemoryRegistration&) = delete;

private:
    void* data_ = nullptr;
};

} // namespace ninfer
