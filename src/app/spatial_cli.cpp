#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "spatial/backend/metal_backend.hpp"
#include "spatial/core/dtype.hpp"
#include "spatial/core/profiler.hpp"
#include "spatial/models/waypoint.hpp"

using namespace spatial;

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

static void write_ppm(const std::string& path, const uint8_t* rgb, uint32_t w, uint32_t h) {
    std::ofstream ofs(path, std::ios::binary);
    std::string hdr = "P6\n" + std::to_string(w) + " " + std::to_string(h) + "\n255\n";
    ofs.write(hdr.data(), hdr.size());
    ofs.write(reinterpret_cast<const char*>(rgb), static_cast<size_t>(w) * h * 3);
}

static std::vector<uint8_t> read_ppm(const std::string& path, uint32_t& w, uint32_t& h) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open PPM: " + path);
    std::string magic;
    int maxv = 0;
    ifs >> magic >> w >> h >> maxv;
    if (magic != "P6" || maxv != 255) throw std::runtime_error("Only P6 255 PPM supported");
    ifs.get();  // single whitespace byte
    std::vector<uint8_t> data(static_cast<size_t>(w) * h * 3);
    ifs.read(reinterpret_cast<char*>(data.data()), data.size());
    return data;
}

static int run_verify(const std::string& dit_path) {
    MetalBackend mb;
    Timer load_t;
    WaypointModel dit(mb, dit_path);
    double load_ms = load_t.elapsed_ms();
    auto mem0 = get_memory_stats();
    std::printf("[M3 Verify] Loaded %s in %.1f ms (mmap zero-copy!) | footprint=%.1f MB | KV=%.1f MB\n",
                dit_path.c_str(), load_ms, mem0.footprint_mb,
                dit.kv_cache().total_bytes() / (1024.0 * 1024.0));

    auto seed_f32  = read_bin<float>("/tmp/spatial_wp_ref/seed_lat_f32.bin");
    auto noisy_f32 = read_bin<float>("/tmp/spatial_wp_ref/noisy_lat_f32.bin");
    auto ref_v_f32 = read_bin<float>("/tmp/spatial_wp_ref/ref_v_f32.bin");

    Buffer seed_buf  = mb.alloc_buffer(seed_f32.size() * 2);
    Buffer noisy_buf = mb.alloc_buffer(noisy_f32.size() * 2);
    Buffer out_v_buf = mb.alloc_buffer(ref_v_f32.size() * 2);
    for (size_t i = 0; i < seed_f32.size(); ++i) {
        static_cast<uint16_t*>(seed_buf.cpu_ptr)[i]  = f32_to_f16(seed_f32[i]);
        static_cast<uint16_t*>(noisy_buf.cpu_ptr)[i] = f32_to_f16(noisy_f32[i]);
    }

    ControlState empty_ctrl;
    dit.cache_seed_latent(seed_buf, 0, empty_ctrl);
    double pass0_ms = dit.last_forward_gpu_ms();

    ControlState act_ctrl;
    act_ctrl.mouse_dx = 0.25f;
    act_ctrl.mouse_dy = -0.15f;
    act_ctrl.buttons = {87};
    dit.kv_cache().set_frozen(true);
    dit.forward(noisy_buf, /*sigma_idx=*/0, /*frame_idx=*/1, act_ctrl, out_v_buf);
    double pass1_ms = dit.last_forward_gpu_ms();

    const uint16_t* v16 = static_cast<const uint16_t*>(out_v_buf.cpu_ptr);
    double dot = 0.0, n_ref = 0.0, n_gpu = 0.0, sum_err = 0.0, max_err = 0.0;
    for (size_t i = 0; i < ref_v_f32.size(); ++i) {
        double r = ref_v_f32[i];
        double g = f16_to_f32(v16[i]);
        dot   += r * g;
        n_ref += r * r;
        n_gpu += g * g;
        double e = std::fabs(r - g);
        sum_err += e;
        if (e > max_err) max_err = e;
    }
    double cos_sim = dot / (std::sqrt(n_ref) * std::sqrt(n_gpu) + 1e-12);
    auto mem1 = get_memory_stats();
    std::printf("[M3 Verify] Pass0=%.2f ms | Pass1=%.2f ms | cos_sim=%.6f | mean_err=%.5f | max_err=%.4f | footprint=%.1f MB\n",
                pass0_ms, pass1_ms, cos_sim, sum_err / ref_v_f32.size(), max_err, mem1.footprint_mb);

    if (cos_sim < 0.96) {
        std::fprintf(stderr, "PARITY CHECK FAILED: cos_sim=%.6f < 0.96\n", cos_sim);
        return 1;
    }
    std::printf("[M3 Verify] FULL 24-LAYER WAYPOINT-1.5 PARITY PASSED!\n");
    return 0;
}

static int run_rollout(const std::string& dit_path,
                       const std::string& vae_path,
                       const std::string& seed_ppm,
                       const std::string& out_dir,
                       int chunks,
                       uint32_t steps) {
    MetalBackend mb;
    auto info = mb.device_info();
    std::printf("[Spatial Engine] Device: %s (unified=%d, recWS=%.0f MB)\n",
                info.name.c_str(), info.unified_memory,
                info.recommended_ws_bytes / (1024.0 * 1024.0));

    Timer load_t;
    WorldSession sess(mb, dit_path, vae_path, /*rng_seed=*/42);
    std::printf("[Spatial Engine] Loaded DiT + TAEHV in %.1f ms | footprint=%.1f MB\n",
                load_t.elapsed_ms(), get_memory_stats().footprint_mb);

    uint32_t sw = 0, sh = 0;
    std::vector<uint8_t> seed_rgb = read_ppm(seed_ppm, sw, sh);
    if (sw != sess.width() || sh != sess.height()) {
        throw std::runtime_error("Seed PPM must be " + std::to_string(sess.width()) +
                                 "x" + std::to_string(sess.height()));
    }

    Timer seed_t;
    sess.seed_from_rgb(seed_rgb.data());
    std::printf("[Spatial Engine] Seeded frame 0 in %.1f ms (VAE=%.1f ms, DiT_cache=%.1f ms)\n",
                seed_t.elapsed_ms(), sess.last_vae_ms(), sess.last_dit_ms());
    write_ppm(out_dir + "/frame_000.ppm", seed_rgb.data(), sw, sh);

    std::vector<uint8_t> chunk_rgb;
    double total_wall_ms = 0.0, total_dit_ms = 0.0, total_vae_ms = 0.0;
    int frame_counter = 1;

    for (int c = 0; c < chunks; ++c) {
        ControlState ctrl;
        // Walk forward ('W' = 87) and gently pan right
        ctrl.buttons = {87};
        ctrl.mouse_dx = (c >= chunks / 2) ? 0.15f : 0.05f;
        ctrl.mouse_dy = 0.0f;

        Timer t;
        sess.step_rgb(ctrl, chunk_rgb, steps);
        double wall_ms = t.elapsed_ms();
        total_wall_ms += wall_ms;
        total_dit_ms  += sess.last_dit_ms();
        total_vae_ms  += sess.last_vae_ms();

        size_t fbytes = static_cast<size_t>(sw) * sh * 3;
        for (int f = 0; f < 4; ++f) {
            char fname[64];
            std::snprintf(fname, sizeof(fname), "/frame_%03d.ppm", frame_counter++);
            write_ppm(out_dir + fname, chunk_rgb.data() + f * fbytes, sw, sh);
        }

        double chunk_fps = 4000.0 / wall_ms;
        std::printf("  Chunk %2d/%d (4 frames): wall=%6.1f ms | DiT(%u-step+cache)=%6.1f ms | VAE=%5.1f ms | %.2f FPS\n",
                    c + 1, chunks, wall_ms, steps, sess.last_dit_ms(), sess.last_vae_ms(), chunk_fps);
    }

    auto mem = get_memory_stats();
    double avg_fps = (4.0 * chunks * 1000.0) / total_wall_ms;
    std::printf("\n=== M3 Rollout Summary (%d chunks = %d frames, %u denoise steps) ===\n",
                chunks, chunks * 4, steps);
    std::printf("  Avg throughput: %.2f FPS (%.1f ms/chunk | DiT=%.1f ms, VAE=%.1f ms)\n",
                avg_fps, total_wall_ms / chunks, total_dit_ms / chunks, total_vae_ms / chunks);
    std::printf("  Memory: phys_footprint=%.1f MB | RSS=%.1f MB\n", mem.footprint_mb, mem.rss_mb);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Usage:\n"
                    "  %s --verify <dit.spat>\n"
                    "  %s --rollout <dit.spat> <taehv.spat> <seed.ppm> <out_dir> [chunks=4] [steps=4]\n",
                    argv[0], argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "--verify" && argc >= 3) {
        return run_verify(argv[2]);
    }
    if (mode == "--rollout" && argc >= 6) {
        int chunks = argc >= 7 ? std::stoi(argv[6]) : 4;
        uint32_t steps = argc >= 8 ? static_cast<uint32_t>(std::stoi(argv[7])) : 4;
        return run_rollout(argv[2], argv[3], argv[4], argv[5], chunks, steps);
    }
    std::fprintf(stderr, "Invalid arguments.\n");
    return 1;
}
