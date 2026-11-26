#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "spatial/backend/metal_backend.hpp"
#include "spatial/cache/spatial_kv_cache.hpp"
#include "spatial/core/spat_format.hpp"
#include "spatial/core/tensor.hpp"
#include "spatial/models/taehv.hpp"

namespace spatial {

struct ControlState {
    float mouse_dx = 0.0f;
    float mouse_dy = 0.0f;
    float scroll   = 0.0f;
    std::vector<uint8_t> buttons;  // active button IDs in [0..255]
};

struct WaypointConfig {
    uint32_t n_layers               = 24;
    uint32_t d_model                = 2048;
    uint32_t n_heads                = 32;
    uint32_t n_kv_heads             = 16;
    uint32_t mlp_ratio              = 4;
    uint32_t channels               = 32;
    uint32_t height                 = 8;
    uint32_t width                  = 16;
    uint32_t patch_h                = 2;
    uint32_t patch_w                = 2;
    uint32_t tokens_per_frame       = 128;
    uint32_t local_window           = 16;
    uint32_t global_window          = 128;
    uint32_t global_attn_period     = 4;
    int32_t  global_attn_offset     = -1;
    uint32_t global_pinned_dilation = 8;
    uint32_t ctrl_cond_period       = 3;
    uint32_t n_buttons              = 256;
    uint32_t temporal_compression   = 4;
    uint32_t base_fps               = 15;
    uint32_t inference_fps          = 60;
    float    rope_nyquist_frac      = 0.8f;
    float    rope_theta             = 10000.0f;
    std::vector<float> sigmas       = {1.0f, 0.9f, 0.75f, 0.3f, 0.0f};
};

// Clean-room C++20/Metal Waypoint-1.5 DiT runtime
class WaypointModel {
public:
    WaypointModel(MetalBackend& backend, const std::string& spat_path);

    const WaypointConfig& config() const { return cfg_; }
    SpatialKVCache& kv_cache() { return *kv_cache_; }
    void reset() { kv_cache_->reset(); }

    // Execute a single DiT forward pass for latent `x_in` [C, H, W] (F16) at discrete `sigma_idx` (0..4)
    // Writes velocity prediction into `v_out` [C, H, W] (F16).
    void forward(const Buffer& x_in,
                 uint32_t sigma_idx,
                 uint64_t frame_idx,
                 const ControlState& ctrl,
                 Buffer& v_out);

    // Run full N-step rectified flow denoising + cache pass for 1 chunk (= 4 video frames)
    // `noise_in_out` starts as N(0, 1) F16 [C, H, W] and is denoised in-place.
    // `denoise_steps` defaults to 4 (can be set to 2 for ultra-fast interactive preview!).
    void denoise_and_cache(Buffer& noise_in_out,
                           uint64_t frame_idx,
                           const ControlState& ctrl,
                           uint32_t denoise_steps = 4);

    // Cache a clean encoded seed latent (sigma=0.0, unfrozen)
    void cache_seed_latent(const Buffer& clean_latent,
                           uint64_t frame_idx,
                           const ControlState& ctrl);

    double last_forward_gpu_ms() const { return last_forward_gpu_ms_; }

private:
    struct BlockWeights {
        TensorView qkv;
        TensorView out;
        float      v_lamb = 0.5f;
        bool       has_ctrl = false;
        TensorView ctrl_fc1_x;
        TensorView ctrl_fc1_c;
        TensorView ctrl_fc2;
        TensorView mlp_fc1;
        TensorView mlp_fc2;
    };

    void dispatch_linear(const Buffer& A, const TensorView& W, const Buffer& C,
                         uint32_t M, bool apply_silu = false);
    void compute_rope_angles(uint64_t t_pos);

    MetalBackend& backend_;
    SpatFile      file_;
    Buffer        weights_buf_{};
    WaypointConfig cfg_;
    std::unique_ptr<SpatialKVCache> kv_cache_;

    // Folded conditioning LUTs (replaces 633M params!)
    TensorView cond_table_;      // [S, n_layers, 6, D] F16
    TensorView out_norm_table_;  // [S, 2, D] F16
    TensorView patchify_w_;      // [D, C*4] F16
    TensorView unpatchify_w_;    // [C*4, D] F16
    TensorView unpatchify_b_;    // [C] F16
    TensorView ctrl_fc1_;        // [D*4, 259] F16
    TensorView ctrl_fc2_;        // [D, D*4] F16
    std::vector<BlockWeights> blocks_;

    // Precomputed OrthoRoPE base frequencies
    std::vector<float> rope_xy_;
    std::vector<float> rope_inv_t_;
    Buffer rope_cos_buf_{};
    Buffer rope_sin_buf_{};

    // Static activation arena (zero runtime mallocs!)
    Buffer arena_pool_{};
    Arena  arena_;
    Buffer ctrl_in_buf_{};
    Buffer dummy_v_out_{};

    double last_forward_gpu_ms_ = 0.0;
};

// End-to-end interactive world session (seed image -> real-time chunk rollouts -> RGB/BGRA frames)
class WorldSession {
public:
    WorldSession(MetalBackend& backend,
                 const std::string& dit_spat_path,
                 const std::string& taehv_spat_path,
                 uint64_t rng_seed = 42);

    // Reset world state and seed from an RGB uint8 image [H, W, 3]
    void seed_from_rgb(const uint8_t* rgb_hwc);

    // Step world forward by 1 latent chunk (= 4 video frames)
    void step_rgb(const ControlState& ctrl, std::vector<uint8_t>& out_rgb_4frames,
                  uint32_t denoise_steps = 4);
    void step_bgra(const ControlState& ctrl, std::array<Buffer, 4>& out_bgra_4frames,
                   uint32_t denoise_steps = 4);

    uint32_t width() const { return vae_.pixel_w(); }
    uint32_t height() const { return vae_.pixel_h(); }
    uint64_t chunk_index() const { return frame_idx_; }
    WaypointModel& dit() { return dit_; }
    TAEHV& vae() { return vae_; }

    double last_dit_ms() const { return last_dit_ms_; }
    double last_vae_ms() const { return last_vae_ms_; }

private:
    void fill_gaussian_noise(Buffer& latent_f16);

    MetalBackend& backend_;
    WaypointModel dit_;
    TAEHV         vae_;
    Buffer        latent_buf_{};
    uint64_t      frame_idx_ = 0;
    uint64_t      rng_state_ = 42;
    double        last_dit_ms_ = 0.0;
    double        last_vae_ms_ = 0.0;
};

}  // namespace spatial
