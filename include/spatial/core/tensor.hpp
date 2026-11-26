#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "spatial/core/dtype.hpp"

namespace spatial {

constexpr int MAX_DIMS = 5;

// Unified device/host buffer handle (zero-copy on Apple Silicon Unified Memory)
struct Buffer {
    void*    cpu_ptr     = nullptr;  // Direct CPU pointer (shared storage on Metal)
    void*    gpu_handle  = nullptr;  // Opaque id<MTLBuffer> (non-owning or backend-owned)
    size_t   byte_offset = 0;        // Offset inside underlying gpu_handle
    size_t   size_bytes  = 0;

    bool valid() const { return cpu_ptr != nullptr || gpu_handle != nullptr; }

    Buffer sub(size_t offset, size_t len) const {
        assert(offset + len <= size_bytes);
        Buffer b;
        b.cpu_ptr     = cpu_ptr ? static_cast<uint8_t*>(cpu_ptr) + offset : nullptr;
        b.gpu_handle  = gpu_handle;
        b.byte_offset = byte_offset + offset;
        b.size_bytes  = len;
        return b;
    }
};

// Zero-copy N-dimensional strided tensor view over a unified Buffer
struct TensorView {
    Buffer                     buf{};
    DType                      dtype = DType::F32;
    int                        ndim  = 0;
    std::array<int64_t, MAX_DIMS> shape{};
    std::array<int64_t, MAX_DIMS> stride{};

    TensorView() = default;

    TensorView(Buffer b, DType dt, std::initializer_list<int64_t> dims)
        : buf(b), dtype(dt), ndim(static_cast<int>(dims.size())) {
        assert(ndim <= MAX_DIMS);
        int i = 0;
        for (int64_t d : dims) shape[i++] = d;
        compute_contiguous_strides();
    }

    TensorView(Buffer b, DType dt, const std::vector<int64_t>& dims)
        : buf(b), dtype(dt), ndim(static_cast<int>(dims.size())) {
        assert(ndim <= MAX_DIMS);
        for (int i = 0; i < ndim; ++i) shape[i] = dims[i];
        compute_contiguous_strides();
    }

    void compute_contiguous_strides() {
        int64_t s = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            stride[i] = s;
            s *= shape[i];
        }
    }

    size_t numel() const {
        if (ndim == 0) return 0;
        size_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= static_cast<size_t>(shape[i]);
        return n;
    }

    size_t nbytes() const { return dtype_nbytes(dtype, numel()); }

    bool is_contiguous() const {
        int64_t s = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            if (shape[i] != 1 && stride[i] != s) return false;
            s *= shape[i];
        }
        return true;
    }

    TensorView view(std::initializer_list<int64_t> new_dims) const {
        if (!is_contiguous()) {
            throw std::runtime_error("TensorView::view requires contiguous tensor");
        }
        TensorView out(buf, dtype, new_dims);
        if (out.numel() != numel()) {
            throw std::runtime_error("TensorView::view numel mismatch");
        }
        return out;
    }

    TensorView permute(std::initializer_list<int> order) const {
        assert(static_cast<int>(order.size()) == ndim);
        TensorView out = *this;
        int i = 0;
        for (int idx : order) {
            out.shape[i]  = shape[idx];
            out.stride[i] = stride[idx];
            ++i;
        }
        return out;
    }

    // Zero-copy slice along outermost dimension (for contiguous element dtypes)
    TensorView slice_dim0(int64_t start, int64_t count) const {
        assert(ndim >= 1 && start >= 0 && start + count <= shape[0]);
        assert(dtype != DType::Q4_0 && dtype != DType::Q8_0);
        size_t elem_bytes = dtype_nbytes(dtype, 1);
        size_t off = static_cast<size_t>(start * stride[0]) * elem_bytes;
        size_t len = static_cast<size_t>(count * stride[0]) * elem_bytes;
        TensorView out = *this;
        out.buf = buf.sub(off, len);
        out.shape[0] = count;
        return out;
    }

    template <typename T>
    T* data() const { return static_cast<T*>(buf.cpu_ptr); }

    std::string shape_str() const {
        std::string s = "[";
        for (int i = 0; i < ndim; ++i) {
            if (i) s += ", ";
            s += std::to_string(shape[i]);
        }
        s += "]";
        return s;
    }
};

// Scoped bump allocator over a pre-allocated unified Buffer (zero mallocs during inference)
class Arena {
public:
    Arena() = default;
    explicit Arena(Buffer backing, size_t alignment = 256)
        : backing_(backing), align_(alignment) {}

    size_t mark() const { return offset_; }
    void   rewind(size_t m) { offset_ = m; }
    void   reset() { offset_ = 0; }
    size_t used() const { return offset_; }
    size_t peak() const { return peak_; }
    size_t capacity() const { return backing_.size_bytes; }

    Buffer alloc_bytes(size_t bytes) {
        size_t aligned = (offset_ + align_ - 1) & ~(align_ - 1);
        if (aligned + bytes > backing_.size_bytes) {
            throw std::runtime_error(
                "Arena OOM: requested " + std::to_string(bytes) +
                " bytes at offset " + std::to_string(aligned) +
                " / cap " + std::to_string(backing_.size_bytes));
        }
        Buffer out = backing_.sub(aligned, bytes);
        offset_ = aligned + bytes;
        if (offset_ > peak_) peak_ = offset_;
        return out;
    }

    TensorView alloc_tensor(DType dt, std::initializer_list<int64_t> dims) {
        TensorView tmp(Buffer{}, dt, dims);
        Buffer b = alloc_bytes(tmp.nbytes());
        return TensorView(b, dt, dims);
    }

    TensorView alloc_tensor(DType dt, const std::vector<int64_t>& dims) {
        TensorView tmp(Buffer{}, dt, dims);
        Buffer b = alloc_bytes(tmp.nbytes());
        return TensorView(b, dt, dims);
    }

private:
    Buffer backing_{};
    size_t align_  = 256;
    size_t offset_ = 0;
    size_t peak_   = 0;
};

struct ArenaScope {
    Arena& arena;
    size_t saved;
    explicit ArenaScope(Arena& a) : arena(a), saved(a.mark()) {}
    ~ArenaScope() { arena.rewind(saved); }
};

}  // namespace spatial
