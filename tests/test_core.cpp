#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "spatial/backend/metal_backend.hpp"
#include "spatial/core/dtype.hpp"
#include "spatial/core/profiler.hpp"
#include "spatial/core/spat_format.hpp"
#include "spatial/core/tensor.hpp"

using namespace spatial;

static void test_dtype_and_tensor() {
    for (float x : {0.0f, 1.0f, -2.5f, 0.125f, 123.5f}) {
        uint16_t h = f32_to_f16(x);
        float y = f16_to_f32(h);
        if (std::fabs(x - y) >= 1e-2f) std::abort();
    }

    MetalBackend mb;
    auto info = mb.device_info();
    std::printf("[M0] Metal device: %s | unified=%d | recWS=%.0f MB\n",
                info.name.c_str(), info.unified_memory,
                info.recommended_ws_bytes / (1024.0 * 1024.0));

    Buffer backing = mb.alloc_buffer(4 * 1024 * 1024);
    Arena arena(backing);
    {
        ArenaScope scope(arena);
        auto t = arena.alloc_tensor(DType::F32, {2, 3, 4});
        if (t.numel() != 24 || !t.is_contiguous()) std::abort();
        auto tp = t.permute({0, 2, 1});
        if (tp.is_contiguous() || tp.shape[1] != 4 || tp.shape[2] != 3) std::abort();
        auto tv = t.view({6, 4});
        if (!tv.is_contiguous() || tv.shape[0] != 6) std::abort();
        auto ts = tv.slice_dim0(2, 3);
        if (ts.shape[0] != 3 || ts.shape[1] != 4) std::abort();
    }
    if (arena.used() != 0 || arena.peak() < 96) std::abort();
    std::printf("[M0] TensorView + Arena tests PASSED (peak=%zu B)\n", arena.peak());
}

static void test_metal_compute_and_spat() {
    // 1. Create a tiny .spat file on disk and mmap + zero-copy wrap into Metal
    const std::string path = "/tmp/spatial_m0_test.spat";
    std::vector<float> payload(1024);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<float>(i) * 0.5f;

    const std::string json =
        R"({"metadata":{"model":"test","layers":24},"tensors":[{"name":"w","dtype":"F32","shape":[32,32],"offset":0,"nbytes":4096}]})";
    constexpr uint64_t DATA_OFF = 16384;  // 16 KB page aligned!
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs.write("SPAT\0\0\0\0", 8);
        uint32_t ver = 1, flags = 0;
        uint64_t jlen = json.size(), doff = DATA_OFF;
        ofs.write(reinterpret_cast<const char*>(&ver), 4);
        ofs.write(reinterpret_cast<const char*>(&flags), 4);
        ofs.write(reinterpret_cast<const char*>(&jlen), 8);
        ofs.write(reinterpret_cast<const char*>(&doff), 8);
        ofs.write(json.data(), json.size());
        std::vector<char> pad(DATA_OFF - 32 - json.size(), 0);
        ofs.write(pad.data(), pad.size());
        ofs.write(reinterpret_cast<const char*>(payload.data()), payload.size() * 4);
        // Pad file to page multiple for clean NoCopy wrap
        std::vector<char> tail(16384 - payload.size() * 4, 0);
        ofs.write(tail.data(), tail.size());
    }

    SpatFile sf = SpatFile::open(path);
    assert(sf.meta_string("model") == "test");
    assert(sf.meta_int("layers") == 24);
    assert(sf.has_tensor("w"));

    MetalBackend mb;
    Buffer wbuf = mb.wrap_mmap_buffer(sf.data_ptr(), sf.data_size());
    TensorView w = sf.get_tensor("w", wbuf);
    Buffer out_buf = mb.alloc_buffer(4096);

    mb.load_shader_source("m0_test", R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void scale2(device const float* inp [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant float& k [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
            out[id] = inp[id] * k;
        }
    )");

    float k = 3.0f;
    mb.begin_batch();
    mb.dispatch("scale2", {1024, 1, 1}, {256, 1, 1},
                {{&w.buf, 0}, {&out_buf, 1}}, {{&k, sizeof(k), 2}});
    mb.end_batch_and_wait();

    const float* out = static_cast<const float*>(out_buf.cpu_ptr);
    for (size_t i = 0; i < 1024; ++i) {
        if (std::fabs(out[i] - payload[i] * 3.0f) >= 1e-5f) std::abort();
    }
    auto mem = get_memory_stats();
    std::printf("[M0] .spat mmap zero-copy + Metal kernel PASSED (gpu=%.3f ms, footprint=%.1f MB)\n",
                mb.last_gpu_ms(), mem.footprint_mb);
}

int main() {
    test_dtype_and_tensor();
    test_metal_compute_and_spat();
    std::printf("[M0] ALL CORE TESTS PASSED!\n");
    return 0;
}
