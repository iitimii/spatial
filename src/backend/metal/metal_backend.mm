#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "spatial/backend/metal_backend.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace spatial {

struct MetalBackend::Impl {
    id<MTLDevice>              device = nil;
    id<MTLCommandQueue>        queue  = nil;
    id<MTLCommandBuffer>       cmd    = nil;
    id<MTLComputeCommandEncoder> enc  = nil;
    double                     last_gpu_ms = 0.0;

    std::vector<id<MTLBuffer>> owned_buffers;
    std::vector<id<MTLLibrary>> libraries;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    id<MTLComputePipelineState> get_pipeline(const std::string& name) {
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return it->second;

        NSString* fn_name = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> fn = nil;
        for (auto rit = libraries.rbegin(); rit != libraries.rend(); ++rit) {
            fn = [*rit newFunctionWithName:fn_name];
            if (fn) break;
        }
        if (!fn) {
            throw std::runtime_error("Metal kernel not found: " + name);
        }
        NSError* err = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            std::string msg = err ? err.localizedDescription.UTF8String : "unknown error";
            throw std::runtime_error("Metal PSO creation failed for " + name + ": " + msg);
        }
        pipelines[name] = pso;
        return pso;
    }

    void ensure_encoder() {
        if (!cmd) {
            cmd = [queue commandBuffer];
            if (!cmd) throw std::runtime_error("Failed to create MTLCommandBuffer");
        }
        if (!enc) {
            enc = [cmd computeCommandEncoder];
            if (!enc) throw std::runtime_error("Failed to create MTLComputeCommandEncoder");
        }
    }
};

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw std::runtime_error("No Metal device available");
        }
        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) {
            throw std::runtime_error("Failed to create Metal command queue");
        }
    }
}

MetalBackend::~MetalBackend() {
    @autoreleasepool {
        if (impl_ && impl_->enc) {
            [impl_->enc endEncoding];
            impl_->enc = nil;
        }
        if (impl_ && impl_->cmd) {
            [impl_->cmd commit];
            [impl_->cmd waitUntilCompleted];
            impl_->cmd = nil;
        }
        impl_.reset();
    }
}

DeviceInfo MetalBackend::device_info() const {
    DeviceInfo di;
    di.name                 = impl_->device.name.UTF8String;
    di.max_buffer_bytes     = impl_->device.maxBufferLength;
    di.recommended_ws_bytes = impl_->device.recommendedMaxWorkingSetSize;
    di.unified_memory       = impl_->device.hasUnifiedMemory;
    return di;
}

Buffer MetalBackend::alloc_buffer(size_t size_bytes) {
    if (size_bytes == 0) size_bytes = 256;
    size_t aligned = (size_bytes + 255) & ~size_t(255);
    id<MTLBuffer> mb = [impl_->device newBufferWithLength:aligned
                                                  options:MTLResourceStorageModeShared];
    if (!mb) {
        throw std::runtime_error("Metal alloc_buffer failed for " + std::to_string(aligned) + " bytes");
    }
    impl_->owned_buffers.push_back(mb);
    Buffer b;
    b.cpu_ptr     = mb.contents;
    b.gpu_handle  = (__bridge void*)mb;
    b.byte_offset = 0;
    b.size_bytes  = aligned;
    return b;
}

Buffer MetalBackend::wrap_mmap_buffer(void* page_aligned_ptr, size_t size_bytes) {
    constexpr size_t PAGE_SIZE_ARM64 = 16384;
    if ((reinterpret_cast<uintptr_t>(page_aligned_ptr) % PAGE_SIZE_ARM64) != 0) {
        throw std::runtime_error("wrap_mmap_buffer pointer is not 16KB page aligned");
    }
    // Metal requires length to be page-aligned for NoCopy; round up to page boundary ONLY if
    // the underlying mmap covers it (mmap always rounds up to page size!).
    size_t page_len = (size_bytes + PAGE_SIZE_ARM64 - 1) & ~(PAGE_SIZE_ARM64 - 1);
    id<MTLBuffer> mb = [impl_->device newBufferWithBytesNoCopy:page_aligned_ptr
                                                        length:page_len
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
    if (!mb) {
        // Fallback: copy into a regular shared buffer if NoCopy fails (e.g. odd size or RO prot)
        Buffer copy = alloc_buffer(size_bytes);
        std::memcpy(copy.cpu_ptr, page_aligned_ptr, size_bytes);
        return copy;
    }
    impl_->owned_buffers.push_back(mb);
    Buffer b;
    b.cpu_ptr     = mb.contents;
    b.gpu_handle  = (__bridge void*)mb;
    b.byte_offset = 0;
    b.size_bytes  = size_bytes;
    return b;
}

void MetalBackend::load_shader_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Cannot open Metal shader file: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    load_shader_source(path, oss.str());
}

void MetalBackend::load_shader_source(const std::string& name, const std::string& msl_source) {
    @autoreleasepool {
        NSString* src = [NSString stringWithUTF8String:msl_source.c_str()];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        NSError* err = nil;
        id<MTLLibrary> lib = [impl_->device newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            std::string msg = err ? err.localizedDescription.UTF8String : "unknown error";
            throw std::runtime_error("Metal compile error in " + name + ":\n" + msg);
        }
        impl_->libraries.push_back(lib);
    }
}

void MetalBackend::begin_batch() {
    @autoreleasepool {
        if (impl_->cmd) {
            throw std::runtime_error("MetalBackend::begin_batch called while batch active");
        }
        impl_->cmd = [impl_->queue commandBuffer];
        impl_->enc = [impl_->cmd computeCommandEncoder];
    }
}

void MetalBackend::end_batch_and_wait() {
    @autoreleasepool {
        if (!impl_->cmd) return;
        if (impl_->enc) {
            [impl_->enc endEncoding];
            impl_->enc = nil;
        }
        id<MTLCommandBuffer> cb = impl_->cmd;
        impl_->cmd = nil;
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            std::string msg = cb.error ? cb.error.localizedDescription.UTF8String : "unknown GPU error";
            throw std::runtime_error("Metal GPU execution failed: " + msg);
        }
        double start_t = cb.GPUStartTime;
        double end_t   = cb.GPUEndTime;
        impl_->last_gpu_ms = (end_t > start_t) ? (end_t - start_t) * 1000.0 : 0.0;
    }
}

double MetalBackend::last_gpu_ms() const {
    return impl_->last_gpu_ms;
}

void MetalBackend::dispatch(
    const std::string& kernel_name,
    Grid grid_threads,
    Grid threadgroup_size,
    const std::vector<BufArg>& buffers,
    const std::vector<BytesArg>& constants,
    size_t threadgroup_mem_bytes) {
    @autoreleasepool {
        bool auto_commit = (impl_->cmd == nil);
        impl_->ensure_encoder();
        id<MTLComputePipelineState> pso = impl_->get_pipeline(kernel_name);
        [impl_->enc setComputePipelineState:pso];
        for (const auto& ba : buffers) {
            id<MTLBuffer> mb = (__bridge id<MTLBuffer>)ba.buf->gpu_handle;
            [impl_->enc setBuffer:mb offset:ba.buf->byte_offset atIndex:ba.index];
        }
        for (const auto& ca : constants) {
            [impl_->enc setBytes:ca.ptr length:ca.size atIndex:ca.index];
        }
        if (threadgroup_mem_bytes > 0) {
            [impl_->enc setThreadgroupMemoryLength:threadgroup_mem_bytes atIndex:0];
        }
        MTLSize grid = MTLSizeMake(grid_threads.x, grid_threads.y, grid_threads.z);
        MTLSize tg   = MTLSizeMake(threadgroup_size.x, threadgroup_size.y, threadgroup_size.z);
        [impl_->enc dispatchThreads:grid threadsPerThreadgroup:tg];
        if (auto_commit) {
            end_batch_and_wait();
        }
    }
}

void MetalBackend::dispatch_threadgroups(
    const std::string& kernel_name,
    Grid threadgroups,
    Grid threads_per_group,
    const std::vector<BufArg>& buffers,
    const std::vector<BytesArg>& constants,
    size_t threadgroup_mem_bytes) {
    @autoreleasepool {
        bool auto_commit = (impl_->cmd == nil);
        impl_->ensure_encoder();
        id<MTLComputePipelineState> pso = impl_->get_pipeline(kernel_name);
        [impl_->enc setComputePipelineState:pso];
        for (const auto& ba : buffers) {
            id<MTLBuffer> mb = (__bridge id<MTLBuffer>)ba.buf->gpu_handle;
            [impl_->enc setBuffer:mb offset:ba.buf->byte_offset atIndex:ba.index];
        }
        for (const auto& ca : constants) {
            [impl_->enc setBytes:ca.ptr length:ca.size atIndex:ca.index];
        }
        if (threadgroup_mem_bytes > 0) {
            [impl_->enc setThreadgroupMemoryLength:threadgroup_mem_bytes atIndex:0];
        }
        MTLSize tgs = MTLSizeMake(threadgroups.x, threadgroups.y, threadgroups.z);
        MTLSize tpg = MTLSizeMake(threads_per_group.x, threads_per_group.y, threads_per_group.z);
        [impl_->enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];
        if (auto_commit) {
            end_batch_and_wait();
        }
    }
}

void* MetalBackend::raw_device() const {
    return (__bridge void*)impl_->device;
}

void* MetalBackend::raw_command_queue() const {
    return (__bridge void*)impl_->queue;
}

void MetalBackend::zero_buffer(const Buffer& buf) {
    if (impl_->cmd) {
        // Flush or zero via blit if inside batch
        if (impl_->enc) {
            [impl_->enc endEncoding];
            impl_->enc = nil;
        }
        id<MTLBlitCommandEncoder> blit = [impl_->cmd blitCommandEncoder];
        id<MTLBuffer> mb = (__bridge id<MTLBuffer>)buf.gpu_handle;
        [blit fillBuffer:mb range:NSMakeRange(buf.byte_offset, buf.size_bytes) value:0];
        [blit endEncoding];
        impl_->enc = [impl_->cmd computeCommandEncoder];
    } else if (buf.cpu_ptr) {
        std::memset(buf.cpu_ptr, 0, buf.size_bytes);
    }
}

}  // namespace spatial
