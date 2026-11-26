#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "spatial/core/dtype.hpp"
#include "spatial/core/tensor.hpp"

namespace spatial {

// .spat file binary layout (v1):
//   [0..7]   Magic: "SPAT\0\0\0\0"
//   [8..11]  uint32_t version = 1
//   [12..15] uint32_t flags   = 0
//   [16..23] uint64_t header_json_size
//   [24..31] uint64_t data_offset (16384-byte page aligned for zero-copy Metal wrap!)
//   [32..]   UTF-8 JSON header (metadata + tensors array)
//   [data_offset..] 256-byte aligned raw tensor payloads

struct SpatTensorInfo {
    std::string          name;
    DType                dtype = DType::F32;
    std::vector<int64_t> shape;
    uint64_t             offset = 0;  // relative to data_offset
    uint64_t             nbytes = 0;
};

class SpatFile {
public:
    SpatFile() = default;
    ~SpatFile();

    SpatFile(const SpatFile&) = delete;
    SpatFile& operator=(const SpatFile&) = delete;
    SpatFile(SpatFile&& other) noexcept;
    SpatFile& operator=(SpatFile&& other) noexcept;

    static SpatFile open(const std::string& path);

    bool has_tensor(const std::string& name) const;
    const SpatTensorInfo& info(const std::string& name) const;
    const std::unordered_map<std::string, SpatTensorInfo>& tensors() const { return tensors_; }

    const std::string& header_json() const { return header_json_; }
    std::string meta_string(const std::string& key, const std::string& def = "") const;
    int64_t     meta_int(const std::string& key, int64_t def = 0) const;
    double      meta_float(const std::string& key, double def = 0.0) const;

    // Raw page-aligned pointer & size of the tensor payload section (for zero-copy Metal wrapping)
    void*  data_ptr() const;
    size_t data_size() const { return data_size_; }

    // Create a TensorView referencing a bound unified Buffer covering [data_ptr(), data_size()]
    TensorView get_tensor(const std::string& name, const Buffer& data_buf) const;

private:
    int    fd_         = -1;
    void*  mmap_ptr_   = nullptr;
    size_t mmap_size_  = 0;
    size_t data_off_   = 0;
    size_t data_size_  = 0;

    std::string header_json_;
    std::unordered_map<std::string, SpatTensorInfo> tensors_;
    std::unordered_map<std::string, std::string>    meta_;
};

}  // namespace spatial
