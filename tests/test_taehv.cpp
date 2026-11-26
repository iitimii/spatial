#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "spatial/backend/metal_backend.hpp"
#include "spatial/core/dtype.hpp"
#include "spatial/core/profiler.hpp"
#include "spatial/models/taehv.hpp"

using namespace spatial;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

template <typename T>
static std::vector<T> read_bin(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Cannot open " + path);
    size_t sz = ifs.tellg();
    ifs.seekg(0);
    std::vector<T> out(sz / sizeof(T));
    ifs.read(reinterpret_cast<char*>(out.data()), sz);
    return out;
}

int main() {
    MetalBackend mb;
    TAEHV vae(mb, "/Users/timi/Developer/spatial/weights/taehv1_5.spat", 256, 512);

    auto in_rgb  = read_bin<uint8_t>("/tmp/spatial_taehv_ref/in_rgb.bin");
    auto ref_lat = read_bin<float>("/tmp/spatial_taehv_ref/ref_lat_f32.bin");
    auto ref_dec = read_bin<uint8_t>("/tmp/spatial_taehv_ref/ref_dec_rgb.bin");

    Buffer lat_buf = mb.alloc_buffer(32 * 16 * 32 * 2);
    vae.encode_chunk(in_rgb.data(), lat_buf);

    // Compare Metal F16 latent vs PyTorch F32 reference
    const uint16_t* lat_f16 = static_cast<const uint16_t*>(lat_buf.cpu_ptr);
    float max_lat_err = 0.0f;
    double sum_lat_err = 0.0;
    for (size_t i = 0; i < ref_lat.size(); ++i) {
        float v = f16_to_f32(lat_f16[i]);
        float e = std::fabs(v - ref_lat[i]);
        if (e > max_lat_err) max_lat_err = e;
        sum_lat_err += e;
    }
    double mean_lat_err = sum_lat_err / ref_lat.size();
    std::printf("[M1] TAEHV Encode: gpu=%.2f ms | max_err=%.5f | mean_err=%.6f\n",
                vae.last_encode_gpu_ms(), max_lat_err, mean_lat_err);
    CHECK(mean_lat_err < 0.01, "Encode mean error too high!");
    CHECK(max_lat_err < 0.1, "Encode max error too high!");

    // Decode chunk (includes 3-pass warmup on first call)
    std::vector<uint8_t> dec_rgb;
    vae.decode_chunk_rgb(lat_buf, dec_rgb);
    CHECK(dec_rgb.size() == ref_dec.size(), "Decode size mismatch!");

    int max_px_diff = 0;
    double sum_px_diff = 0.0;
    for (size_t i = 0; i < ref_dec.size(); ++i) {
        int d = std::abs(int(dec_rgb[i]) - int(ref_dec[i]));
        if (d > max_px_diff) max_px_diff = d;
        sum_px_diff += d;
    }
    double mean_px_diff = sum_px_diff / ref_dec.size();
    std::printf("[M1] TAEHV Decode (warmup+4f): gpu=%.2f ms | max_px_diff=%d/255 | mean_px_diff=%.3f/255\n",
                vae.last_decode_gpu_ms(), max_px_diff, mean_px_diff);

    // Steady-state decode benchmark (1 latent -> 4 RGB frames!)
    vae.decode_chunk_rgb(lat_buf, dec_rgb);
    double steady_ms = vae.last_decode_gpu_ms();
    auto mem = get_memory_stats();
    std::printf("[M1] TAEHV Steady Decode (4 frames): gpu=%.2f ms (%.2f ms/frame = %.1f FPS) | footprint=%.1f MB\n",
                steady_ms, steady_ms / 4.0, 4000.0 / std::max(steady_ms, 0.01), mem.footprint_mb);

    CHECK(mean_px_diff < 2.0, "Decode pixel error too high!");
    std::printf("[M1] TAEHV1.5 METAL PARITY & BENCHMARK PASSED!\n");
    return 0;
}
