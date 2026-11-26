#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "spatial/core/spat_format.hpp"
#include "spatial/core/tensor.hpp"

namespace spatial {

struct DeviceInfo {
    std::string name;
    uint64_t    max_buffer_bytes      = 0;
    uint64_t    recommended_ws_bytes  = 0;
    bool        unified_memory        = true;
};

// Metal-backed unified compute engine (cleanly separable for CUDA/Vulkan backends later)
class MetalBackend {
public:
    MetalBackend();
    ~MetalBackend();

    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    DeviceInfo device_info() const;

    // Allocate unified-memory shared buffer (owned by backend, freed on destruction)
    Buffer alloc_buffer(size_t size_bytes);

    // Wrap an existing 16KB page-aligned mmap region with ZERO copies into Metal!
    Buffer wrap_mmap_buffer(void* page_aligned_ptr, size_t size_bytes);

    // Load and compile a .metal shader file (or MSL source string) at runtime
    void load_shader_file(const std::string& path);
    void load_shader_source(const std::string& name, const std::string& msl_source);

    // Command buffer lifecycle (batch multiple dispatches into 1 GPU submission)
    void begin_batch();
    void end_batch_and_wait();  // Commits and waits; updates last_gpu_ms()
    double last_gpu_ms() const;

    // Low-level generic kernel dispatch (used by all spatial ops)
    struct Grid { uint32_t x = 1, y = 1, z = 1; };
    struct BufArg { const Buffer* buf; uint32_t index; };
    struct BytesArg { const void* ptr; size_t size; uint32_t index; };

    void dispatch(
        const std::string& kernel_name,
        Grid grid_threads,
        Grid threadgroup_size,
        const std::vector<BufArg>& buffers,
        const std::vector<BytesArg>& constants = {},
        size_t threadgroup_mem_bytes = 0);

    // Dispatch specifying exact threadgroups (for tiled simdgroup GEMM / FlashAttention)
    void dispatch_threadgroups(
        const std::string& kernel_name,
        Grid threadgroups,
        Grid threads_per_group,
        const std::vector<BufArg>& buffers,
        const std::vector<BytesArg>& constants = {},
        size_t threadgroup_mem_bytes = 0);

    // Raw Metal handles for zero-copy CAMetalLayer presentation in M4
    void* raw_device() const;
    void* raw_command_queue() const;

    // Helper: zero a buffer on GPU/CPU
    void zero_buffer(const Buffer& buf);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace spatial
