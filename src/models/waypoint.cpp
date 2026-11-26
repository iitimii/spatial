#include "spatial/models/waypoint.hpp"

#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace spatial {

namespace {

struct GemmParams { uint32_t M, N, K, flags; };
struct NormParams { uint32_t T, D, flags; };
struct GateParams { uint32_t T, D; };
struct QkvPostParams {
    uint32_t T, n_heads, n_kv_heads, d_head, layer_idx, tail_slot, ring_slot, write_ring, max_slots;
    float    v_lamb;
};
struct AttnParams {
    uint32_t T, n_heads, n_kv_heads, d_head, max_slots, num_active_slots;
    float    scale;
    uint32_t active_slots[32];
};
struct PatchParams { uint32_t C, Hp, Wp; };
struct FlowStepParams { uint32_t n; float dsigma; };

inline uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

}  // namespace

WaypointModel::WaypointModel(MetalBackend& backend, const std::string& spat_path)
    : backend_(backend), file_(SpatFile::open(spat_path)) {
    backend_.load_shader_file(std::string(SPATIAL_SHADER_DIR) + "/dit.metal");
    weights_buf_ = backend_.wrap_mmap_buffer(file_.data_ptr(), file_.data_size());

    // Load config from .spat metadata
    cfg_.n_layers               = static_cast<uint32_t>(file_.meta_int("n_layers", 24));
    cfg_.d_model                = static_cast<uint32_t>(file_.meta_int("d_model", 2048));
    cfg_.n_heads                = static_cast<uint32_t>(file_.meta_int("n_heads", 32));
    cfg_.n_kv_heads             = static_cast<uint32_t>(file_.meta_int("n_kv_heads", 16));
    cfg_.mlp_ratio              = static_cast<uint32_t>(file_.meta_int("mlp_ratio", 4));
    cfg_.channels               = static_cast<uint32_t>(file_.meta_int("channels", 32));
    cfg_.height                 = static_cast<uint32_t>(file_.meta_int("height", 8));
    cfg_.width                  = static_cast<uint32_t>(file_.meta_int("width", 16));
    cfg_.patch_h                = static_cast<uint32_t>(file_.meta_int("patch_h", 2));
    cfg_.patch_w                = static_cast<uint32_t>(file_.meta_int("patch_w", 2));
    cfg_.tokens_per_frame       = static_cast<uint32_t>(file_.meta_int("tokens_per_frame", 128));
    cfg_.local_window           = static_cast<uint32_t>(file_.meta_int("local_window", 16));
    cfg_.global_window          = static_cast<uint32_t>(file_.meta_int("global_window", 128));
    cfg_.global_attn_period     = static_cast<uint32_t>(file_.meta_int("global_attn_period", 4));
    cfg_.global_attn_offset     = static_cast<int32_t>(file_.meta_int("global_attn_offset", -1));
    cfg_.global_pinned_dilation = static_cast<uint32_t>(file_.meta_int("global_pinned_dilation", 8));
    cfg_.ctrl_cond_period       = static_cast<uint32_t>(file_.meta_int("ctrl_conditioning_period", 3));
    cfg_.n_buttons              = static_cast<uint32_t>(file_.meta_int("n_buttons", 256));
    cfg_.temporal_compression   = static_cast<uint32_t>(file_.meta_int("temporal_compression", 4));
    cfg_.base_fps               = static_cast<uint32_t>(file_.meta_int("base_fps", 15));
    cfg_.inference_fps          = static_cast<uint32_t>(file_.meta_int("inference_fps", 60));
    cfg_.rope_nyquist_frac      = static_cast<float>(file_.meta_float("rope_nyquist_frac", 0.8));
    cfg_.rope_theta             = static_cast<float>(file_.meta_float("rope_theta", 10000.0));

    // Parse scheduler_sigmas CSV string from metadata
    std::string sig_csv = file_.meta_string("scheduler_sigmas", "1.0,0.9,0.75,0.3,0.0");
    cfg_.sigmas.clear();
    std::stringstream ss(sig_csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) cfg_.sigmas.push_back(std::stof(tok));
    }

    auto T = [&](const std::string& name) { return file_.get_tensor(name, weights_buf_); };
    cond_table_     = T("folded.cond_table");
    out_norm_table_ = T("folded.out_norm_table");
    patchify_w_     = T("patchify.weight");
    unpatchify_w_   = T("unpatchify.weight");
    unpatchify_b_   = T("unpatchify.bias");
    ctrl_fc1_       = T("ctrl_emb.fc1");
    ctrl_fc2_       = T("ctrl_emb.fc2");

    blocks_.resize(cfg_.n_layers);
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        std::string p = "blk." + std::to_string(l);
        auto& b = blocks_[l];
        b.qkv = T(p + ".attn.qkv");
        b.out = T(p + ".attn.out");
        TensorView vl = T(p + ".attn.v_lamb");
        b.v_lamb = *vl.data<float>();
        b.has_ctrl = file_.has_tensor(p + ".ctrl.fc1_x");
        if (b.has_ctrl) {
            b.ctrl_fc1_x = T(p + ".ctrl.fc1_x");
            b.ctrl_fc1_c = T(p + ".ctrl.fc1_c");
            b.ctrl_fc2   = T(p + ".ctrl.fc2");
        }
        b.mlp_fc1 = T(p + ".mlp.fc1");
        b.mlp_fc2 = T(p + ".mlp.fc2");
    }

    // Initialize Hierarchical Spatial-Temporal KV Cache
    SpatialKVConfig kvc;
    kvc.n_layers               = cfg_.n_layers;
    kvc.n_kv_heads             = cfg_.n_kv_heads;
    kvc.d_head                 = cfg_.d_model / cfg_.n_heads;
    kvc.tokens_per_frame       = cfg_.tokens_per_frame;
    kvc.local_window           = cfg_.local_window;
    kvc.global_window          = cfg_.global_window;
    kvc.global_attn_period     = cfg_.global_attn_period;
    kvc.global_attn_offset     = cfg_.global_attn_offset;
    kvc.global_pinned_dilation = cfg_.global_pinned_dilation;
    kv_cache_ = std::make_unique<SpatialKVCache>(backend_, kvc);

    // Precompute OrthoRoPE frequency basis (d_head=64 -> d_xy=8, d_t=16)
    uint32_t d_head = cfg_.d_model / cfg_.n_heads;
    uint32_t d_xy = d_head / 8;
    uint32_t d_t  = d_head / 4;
    float max_freq = static_cast<float>(std::min(cfg_.height, cfg_.width)) * cfg_.rope_nyquist_frac;
    uint32_t n_lin = (d_xy + 1) / 2;
    rope_xy_.resize(d_xy);
    for (uint32_t i = 0; i < d_xy; ++i) {
        uint32_t idx = i / 2;
        float frac = (n_lin > 1) ? (static_cast<float>(idx) / static_cast<float>(n_lin - 1)) : 0.0f;
        float val = (1.0f + frac * (max_freq * 0.5f - 1.0f)) * static_cast<float>(M_PI);
        rope_xy_[i] = val;
    }
    rope_inv_t_.resize(d_t);
    for (uint32_t i = 0; i < d_t; ++i) {
        uint32_t k = (i / 2) * 2;
        rope_inv_t_[i] = 1.0f / std::pow(cfg_.rope_theta, static_cast<float>(k) / static_cast<float>(d_t));
    }

    rope_cos_buf_ = backend_.alloc_buffer(cfg_.tokens_per_frame * (d_head / 2) * sizeof(float));
    rope_sin_buf_ = backend_.alloc_buffer(cfg_.tokens_per_frame * (d_head / 2) * sizeof(float));

    // Pre-allocate static activation arena (32 MB easily fits all 360P/720P intermediates)
    arena_pool_  = backend_.alloc_buffer(32 * 1024 * 1024);
    arena_       = Arena(arena_pool_);
    ctrl_in_buf_ = backend_.alloc_buffer((cfg_.n_buttons + 3) * sizeof(uint16_t));
    size_t lat_bytes = static_cast<size_t>(cfg_.channels) *
                       (cfg_.height * cfg_.patch_h) * (cfg_.width * cfg_.patch_w) * sizeof(uint16_t);
    dummy_v_out_ = backend_.alloc_buffer(lat_bytes);
}

void WaypointModel::compute_rope_angles(uint64_t t_pos) {
    uint32_t H = cfg_.height;
    uint32_t W = cfg_.width;
    uint32_t d_xy = static_cast<uint32_t>(rope_xy_.size());
    uint32_t d_t  = static_cast<uint32_t>(rope_inv_t_.size());
    uint32_t half_d = 2 * d_xy + d_t;

    float* cos_p = static_cast<float*>(rope_cos_buf_.cpu_ptr);
    float* sin_p = static_cast<float*>(rope_sin_buf_.cpu_ptr);
    float tf = static_cast<float>(t_pos);

    for (uint32_t tok = 0; tok < cfg_.tokens_per_frame; ++tok) {
        uint32_t y = tok / W;
        uint32_t x = tok % W;
        float xn = (2.0f * static_cast<float>(x) + 1.0f) / static_cast<float>(W) - 1.0f;
        float yn = (2.0f * static_cast<float>(y) + 1.0f) / static_cast<float>(H) - 1.0f;

        float* c_row = cos_p + tok * half_d;
        float* s_row = sin_p + tok * half_d;
        uint32_t k = 0;
        for (uint32_t i = 0; i < d_xy; ++i, ++k) {
            float a = xn * rope_xy_[i];
            c_row[k] = std::cos(a);
            s_row[k] = std::sin(a);
        }
        for (uint32_t i = 0; i < d_xy; ++i, ++k) {
            float a = yn * rope_xy_[i];
            c_row[k] = std::cos(a);
            s_row[k] = std::sin(a);
        }
        for (uint32_t i = 0; i < d_t; ++i, ++k) {
            float a = tf * rope_inv_t_[i];
            c_row[k] = std::cos(a);
            s_row[k] = std::sin(a);
        }
    }
}

void WaypointModel::dispatch_linear(const Buffer& A, const TensorView& W, const Buffer& C,
                                    uint32_t M, bool apply_silu) {
    uint32_t N = static_cast<uint32_t>(W.shape[0]);
    uint32_t K = static_cast<uint32_t>(W.shape[1]);
    GemmParams gp{M, N, K, apply_silu ? 1u : 0u};

    if (M == 1) {
        const char* kname = (W.dtype == DType::Q4_0) ? "gemv_q4_0_f16"
                          : (W.dtype == DType::Q8_0) ? "gemv_q8_0_f16"
                                                     : "gemv_f16_f16";
        backend_.dispatch_threadgroups(kname, {N, 1, 1}, {32, 1, 1},
                                       {{&A, 0}, {&W.buf, 1}, {&C, 2}},
                                       {{&gp, sizeof(gp), 3}});
    } else {
        const char* kname = (W.dtype == DType::Q4_0) ? "gemm_q4_0_f16"
                          : (W.dtype == DType::Q8_0) ? "gemm_q8_0_f16"
                                                     : "gemm_f16_f16";
        backend_.dispatch_threadgroups(kname, {div_up(N, 64), div_up(M, 32), 1}, {128, 1, 1},
                                       {{&A, 0}, {&W.buf, 1}, {&C, 2}},
                                       {{&gp, sizeof(gp), 3}});
    }
}

void WaypointModel::forward(const Buffer& x_in,
                            uint32_t sigma_idx,
                            uint64_t frame_idx,
                            const ControlState& ctrl,
                            Buffer& v_out) {
    const uint32_t T = cfg_.tokens_per_frame;
    const uint32_t D = cfg_.d_model;
    const uint32_t d_head = D / cfg_.n_heads;
    const uint32_t qkv_dim = (cfg_.n_heads + 2 * cfg_.n_kv_heads) * d_head;
    const uint32_t mlp_mid = D * cfg_.mlp_ratio;

    // Timestamp multiplier: base_fps // (inference_fps // temporal_compression)
    uint32_t latent_fps = cfg_.inference_fps / cfg_.temporal_compression;
    uint64_t ts_mult = (latent_fps > 0) ? (cfg_.base_fps / latent_fps) : 1;
    compute_rope_angles(frame_idx * ts_mult);

    // Pack controller input [mouse_dx, mouse_dy, button[0..255], scroll] into F16
    uint16_t* c_ptr = static_cast<uint16_t*>(ctrl_in_buf_.cpu_ptr);
    std::memset(c_ptr, 0, (cfg_.n_buttons + 3) * sizeof(uint16_t));
    c_ptr[0] = f32_to_f16(ctrl.mouse_dx);
    c_ptr[1] = f32_to_f16(ctrl.mouse_dy);
    for (uint8_t btn : ctrl.buttons) {
        if (btn < cfg_.n_buttons) c_ptr[2 + btn] = f32_to_f16(1.0f);
    }
    float sgn = (ctrl.scroll > 0.0f ? 1.0f : (ctrl.scroll < 0.0f ? -1.0f : 0.0f));
    c_ptr[2 + cfg_.n_buttons] = f32_to_f16(sgn);

    // Pre-plan all 24 KV cache slots on CPU before encoding the Metal command buffer
    std::vector<LayerSlotPlan> plans(cfg_.n_layers);
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        plans[l] = kv_cache_->prepare_layer(l, frame_idx);
    }

    ArenaScope scope(arena_);
    Buffer patch_in  = arena_.alloc_bytes(T * (cfg_.channels * 4) * 2);
    Buffer x_buf     = arena_.alloc_bytes(T * D * 2);
    Buffer h_buf     = arena_.alloc_bytes(T * D * 2);
    Buffer qkv_buf   = arena_.alloc_bytes(T * qkv_dim * 2);
    Buffer q_buf     = arena_.alloc_bytes(cfg_.n_heads * T * d_head * 2);
    Buffer v1_buf    = arena_.alloc_bytes(cfg_.n_kv_heads * T * d_head * 2);
    Buffer attn_out  = arena_.alloc_bytes(T * D * 2);
    Buffer proj_out  = arena_.alloc_bytes(T * D * 2);
    Buffer mlp_mid_b = arena_.alloc_bytes(T * mlp_mid * 2);
    Buffer ctrl_mid  = arena_.alloc_bytes(mlp_mid * 2);
    Buffer ctrl_emb  = arena_.alloc_bytes(D * 2);
    Buffer ctrl_norm = arena_.alloc_bytes(D * 2);
    Buffer ctrl_c    = arena_.alloc_bytes(D * 2);
    Buffer ctrl_x    = arena_.alloc_bytes(T * D * 2);
    Buffer unpatch_i = arena_.alloc_bytes(T * (cfg_.channels * 4) * 2);

    backend_.begin_batch();

    // 1. Controller embedding MLP (M=1): [1, 259] -> SiLU -> [1, 8192] -> [1, 2048] + RMSNorm
    dispatch_linear(ctrl_in_buf_, ctrl_fc1_, ctrl_mid, 1, /*apply_silu=*/true);
    dispatch_linear(ctrl_mid, ctrl_fc2_, ctrl_emb, 1, /*apply_silu=*/false);
    {
        NormParams np{1, D, 0u};
        backend_.dispatch_threadgroups("rmsnorm_ada_f16", {1, 1, 1}, {256, 1, 1},
                                       {{&ctrl_emb, 0}, {&ctrl_emb, 1}, {&ctrl_emb, 2}, {&ctrl_norm, 3}},
                                       {{&np, sizeof(np), 4}});
    }

    // 2. Patchify: rearrange [C, 16, 32] -> [128, 128] -> Linear -> [128, 2048]
    {
        PatchParams pp{cfg_.channels, cfg_.height, cfg_.width};
        backend_.dispatch("patchify_rearrange_f16", {T, cfg_.channels, 1}, {32, 8, 1},
                          {{&x_in, 0}, {&patch_in, 1}}, {{&pp, sizeof(pp), 2}});
        dispatch_linear(patch_in, patchify_w_, x_buf, T, false);
    }

    // 3. 24 WorldDiT Blocks
    size_t vec_bytes = D * sizeof(uint16_t);
    size_t layer_cond_stride = 6 * vec_bytes;
    size_t sigma_cond_offset = static_cast<size_t>(sigma_idx) * cfg_.n_layers * layer_cond_stride;

    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const auto& blk = blocks_[l];
        const auto& plan = plans[l];

        size_t base_c = sigma_cond_offset + l * layer_cond_stride;
        Buffer s0 = cond_table_.buf.sub(base_c + 0 * vec_bytes, vec_bytes);
        Buffer b0 = cond_table_.buf.sub(base_c + 1 * vec_bytes, vec_bytes);
        Buffer g0 = cond_table_.buf.sub(base_c + 2 * vec_bytes, vec_bytes);
        Buffer s1 = cond_table_.buf.sub(base_c + 3 * vec_bytes, vec_bytes);
        Buffer b1 = cond_table_.buf.sub(base_c + 4 * vec_bytes, vec_bytes);
        Buffer g1 = cond_table_.buf.sub(base_c + 5 * vec_bytes, vec_bytes);

        // 3a. AdaRMSNorm -> h_buf
        {
            NormParams np{T, D, 1u};
            backend_.dispatch_threadgroups("rmsnorm_ada_f16", {T, 1, 1}, {256, 1, 1},
                                           {{&x_buf, 0}, {&s0, 1}, {&b0, 2}, {&h_buf, 3}},
                                           {{&np, sizeof(np), 4}});
        }

        // 3b. Merged QKV GEMM
        dispatch_linear(h_buf, blk.qkv, qkv_buf, T, false);

        // 3c. Fused Q/K RMSNorm + OrthoRoPE + Value Residual + KV Cache Upsert
        {
            QkvPostParams qp{
                T, cfg_.n_heads, cfg_.n_kv_heads, d_head,
                l, plan.tail_slot, plan.ring_slot, plan.write_ring ? 1u : 0u,
                kv_cache_->max_slots(l), blk.v_lamb
            };
            backend_.dispatch_threadgroups("q_norm_rope_f16", {T, cfg_.n_heads, 1}, {32, 1, 1},
                                           {{&qkv_buf, 0}, {&rope_cos_buf_, 1}, {&rope_sin_buf_, 2}, {&q_buf, 3}},
                                           {{&qp, sizeof(qp), 4}});
            backend_.dispatch_threadgroups("kv_post_upsert_f16", {T, cfg_.n_kv_heads, 1}, {32, 1, 1},
                                           {{&qkv_buf, 0}, {&rope_cos_buf_, 1}, {&rope_sin_buf_, 2},
                                            {&v1_buf, 3}, {&kv_cache_->k_buffer(l), 4}, {&kv_cache_->v_buffer(l), 5}},
                                           {{&qp, sizeof(qp), 6}});
        }

        // 3d. GQA FlashAttention over active KV cache slots
        {
            AttnParams ap{};
            ap.T = T;
            ap.n_heads = cfg_.n_heads;
            ap.n_kv_heads = cfg_.n_kv_heads;
            ap.d_head = d_head;
            ap.max_slots = kv_cache_->max_slots(l);
            ap.num_active_slots = static_cast<uint32_t>(plan.active_slots.size());
            ap.scale = 1.0f / std::sqrt(static_cast<float>(d_head));
            for (uint32_t i = 0; i < ap.num_active_slots; ++i) {
                ap.active_slots[i] = plan.active_slots[i];
            }
            backend_.dispatch_threadgroups("flash_attn_gqa_f16", {T, cfg_.n_heads, 1}, {32, 1, 1},
                                           {{&q_buf, 0}, {&kv_cache_->k_buffer(l), 1},
                                            {&kv_cache_->v_buffer(l), 2}, {&attn_out, 3}},
                                           {{&ap, sizeof(ap), 4}});
        }

        // 3e. Output projection + AdaGate + residual
        dispatch_linear(attn_out, blk.out, proj_out, T, false);
        {
            GateParams gp{T, D};
            backend_.dispatch("ada_gate_add_f16", {D, T, 1}, {32, 8, 1},
                              {{&proj_out, 0}, {&g0, 1}, {&x_buf, 2}}, {{&gp, sizeof(gp), 3}});
        }

        // 3f. Optional Control MLPFusion (every 3rd layer)
        if (blk.has_ctrl) {
            NormParams np{T, D, 0u};
            backend_.dispatch_threadgroups("rmsnorm_ada_f16", {T, 1, 1}, {256, 1, 1},
                                           {{&x_buf, 0}, {&x_buf, 1}, {&x_buf, 2}, {&h_buf, 3}},
                                           {{&np, sizeof(np), 4}});
            dispatch_linear(h_buf, blk.ctrl_fc1_x, ctrl_x, T, false);
            dispatch_linear(ctrl_norm, blk.ctrl_fc1_c, ctrl_c, 1, false);
            GateParams gp{T, D};
            backend_.dispatch("add_vec_silu_f16", {D, T, 1}, {32, 8, 1},
                              {{&ctrl_x, 0}, {&ctrl_c, 1}, {&h_buf, 2}}, {{&gp, sizeof(gp), 3}});
            dispatch_linear(h_buf, blk.ctrl_fc2, proj_out, T, false);
            uint32_t n_td = T * D;
            backend_.dispatch("add_inplace_f16", {n_td, 1, 1}, {256, 1, 1},
                              {{&proj_out, 0}, {&x_buf, 1}}, {{&n_td, sizeof(n_td), 2}});
        }

        // 3g. DiT MLP: AdaRMSNorm -> fc1+SiLU -> fc2 -> AdaGate + residual
        {
            NormParams np{T, D, 1u};
            backend_.dispatch_threadgroups("rmsnorm_ada_f16", {T, 1, 1}, {256, 1, 1},
                                           {{&x_buf, 0}, {&s1, 1}, {&b1, 2}, {&h_buf, 3}},
                                           {{&np, sizeof(np), 4}});
        }
        dispatch_linear(h_buf, blk.mlp_fc1, mlp_mid_b, T, /*apply_silu=*/true);
        dispatch_linear(mlp_mid_b, blk.mlp_fc2, proj_out, T, false);
        {
            GateParams gp{T, D};
            backend_.dispatch("ada_gate_add_f16", {D, T, 1}, {32, 8, 1},
                              {{&proj_out, 0}, {&g1, 1}, {&x_buf, 2}}, {{&gp, sizeof(gp), 3}});
        }
    }

    // 4. Final OutNorm + SiLU + Unpatchify -> v_out [C, 16, 32]
    // (Skip unpatchify on cache-only pass if caller passed dummy_v_out_? Keeping it is < 0.05 ms!)
    {
        size_t out_off = static_cast<size_t>(sigma_idx) * 2 * vec_bytes;
        Buffer os = out_norm_table_.buf.sub(out_off, vec_bytes);
        Buffer ob = out_norm_table_.buf.sub(out_off + vec_bytes, vec_bytes);
        NormParams np{T, D, 1u | 2u};  // AdaLN + SiLU!
        backend_.dispatch_threadgroups("rmsnorm_ada_f16", {T, 1, 1}, {256, 1, 1},
                                       {{&x_buf, 0}, {&os, 1}, {&ob, 2}, {&h_buf, 3}},
                                       {{&np, sizeof(np), 4}});

        dispatch_linear(h_buf, unpatchify_w_, unpatch_i, T, false);
        PatchParams pp{cfg_.channels, cfg_.height, cfg_.width};
        backend_.dispatch("unpatchify_rearrange_f16", {T, cfg_.channels, 1}, {32, 8, 1},
                          {{&unpatch_i, 0}, {&unpatchify_b_.buf, 1}, {&v_out, 2}},
                          {{&pp, sizeof(pp), 3}});
    }

    backend_.end_batch_and_wait();
    last_forward_gpu_ms_ = backend_.last_gpu_ms();
}

void WaypointModel::cache_seed_latent(const Buffer& clean_latent,
                                      uint64_t frame_idx,
                                      const ControlState& ctrl) {
    kv_cache_->set_frozen(false);
    uint32_t sigma0_idx = static_cast<uint32_t>(cfg_.sigmas.size() - 1);  // sigma == 0.0
    forward(clean_latent, sigma0_idx, frame_idx, ctrl, dummy_v_out_);
    kv_cache_->set_frozen(true);
}

void WaypointModel::denoise_and_cache(Buffer& noise_in_out,
                                      uint64_t frame_idx,
                                      const ControlState& ctrl,
                                      uint32_t denoise_steps) {
    // Support full 4-step schedule [1.0, 0.9, 0.75, 0.3, 0.0] or fast 2-step [1.0, 0.75, 0.0]
    std::vector<uint32_t> step_indices;
    if (denoise_steps == 2 && cfg_.sigmas.size() == 5) {
        step_indices = {0, 2, 4};  // 1.0 -> 0.75 -> 0.0
    } else if (denoise_steps == 3 && cfg_.sigmas.size() == 5) {
        step_indices = {0, 2, 3, 4};  // 1.0 -> 0.75 -> 0.3 -> 0.0
    } else {
        for (uint32_t i = 0; i < cfg_.sigmas.size(); ++i) step_indices.push_back(i);
    }

    uint32_t lat_n = cfg_.channels * (cfg_.height * cfg_.patch_h) * (cfg_.width * cfg_.patch_w);
    double total_gpu_ms = 0.0;

    kv_cache_->set_frozen(true);
    for (size_t i = 0; i + 1 < step_indices.size(); ++i) {
        uint32_t s_idx = step_indices[i];
        uint32_t next_idx = step_indices[i + 1];
        float dsigma = cfg_.sigmas[next_idx] - cfg_.sigmas[s_idx];

        forward(noise_in_out, s_idx, frame_idx, ctrl, dummy_v_out_);
        total_gpu_ms += last_forward_gpu_ms_;

        FlowStepParams fp{lat_n, dsigma};
        backend_.begin_batch();
        backend_.dispatch("flow_euler_step_f16", {lat_n, 1, 1}, {256, 1, 1},
                          {{&noise_in_out, 0}, {&dummy_v_out_, 1}}, {{&fp, sizeof(fp), 2}});
        backend_.end_batch_and_wait();
        total_gpu_ms += backend_.last_gpu_ms();
    }

    // Cache pass at sigma == 0.0 (unfrozen) to persist clean latent into KV cache
    kv_cache_->set_frozen(false);
    uint32_t sigma0_idx = static_cast<uint32_t>(cfg_.sigmas.size() - 1);
    forward(noise_in_out, sigma0_idx, frame_idx, ctrl, dummy_v_out_);
    total_gpu_ms += last_forward_gpu_ms_;
    kv_cache_->set_frozen(true);

    last_forward_gpu_ms_ = total_gpu_ms;
}

// ============================================================================
// WorldSession: End-to-End Streaming Interactive World Engine
// ============================================================================
WorldSession::WorldSession(MetalBackend& backend,
                           const std::string& dit_spat_path,
                           const std::string& taehv_spat_path,
                           uint64_t rng_seed)
    : backend_(backend),
      dit_(backend, dit_spat_path),
      vae_(backend, taehv_spat_path,
           dit_.config().height * dit_.config().patch_h * 16,
           dit_.config().width * dit_.config().patch_w * 16),
      rng_state_(rng_seed) {
    size_t lat_bytes = static_cast<size_t>(dit_.config().channels) *
                       (dit_.config().height * dit_.config().patch_h) *
                       (dit_.config().width * dit_.config().patch_w) * sizeof(uint16_t);
    latent_buf_ = backend_.alloc_buffer(lat_bytes);
}

void WorldSession::fill_gaussian_noise(Buffer& latent_f16) {
    // Fast deterministic SplitMix64 + Box-Muller Gaussian noise
    auto next_u64 = [&]() -> uint64_t {
        uint64_t z = (rng_state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    auto next_u01 = [&]() -> float {
        return (static_cast<float>(next_u64() >> 40) + 0.5f) / 16777216.0f;
    };

    size_t n = latent_f16.size_bytes / sizeof(uint16_t);
    uint16_t* dst = static_cast<uint16_t*>(latent_f16.cpu_ptr);
    for (size_t i = 0; i < n; i += 2) {
        float u1 = std::max(next_u01(), 1e-7f);
        float u2 = next_u01();
        float r = std::sqrt(-2.0f * std::log(u1));
        float th = 2.0f * static_cast<float>(M_PI) * u2;
        dst[i] = f32_to_f16(r * std::cos(th));
        if (i + 1 < n) dst[i + 1] = f32_to_f16(r * std::sin(th));
    }
}

void WorldSession::seed_from_rgb(const uint8_t* rgb_hwc) {
    dit_.reset();
    vae_.reset();
    frame_idx_ = 0;

    // Repeat single seed image 4x for temporal VAE encoder
    size_t frame_bytes = static_cast<size_t>(vae_.pixel_h()) * vae_.pixel_w() * 3;
    std::vector<uint8_t> chunk4(4 * frame_bytes);
    for (int i = 0; i < 4; ++i) {
        std::memcpy(chunk4.data() + i * frame_bytes, rgb_hwc, frame_bytes);
    }

    vae_.encode_chunk(chunk4.data(), latent_buf_);
    last_vae_ms_ = vae_.last_encode_gpu_ms();

    ControlState empty_ctrl;
    dit_.cache_seed_latent(latent_buf_, frame_idx_, empty_ctrl);
    last_dit_ms_ = dit_.last_forward_gpu_ms();
    frame_idx_ = 1;
}

void WorldSession::step_rgb(const ControlState& ctrl,
                            std::vector<uint8_t>& out_rgb_4frames,
                            uint32_t denoise_steps) {
    fill_gaussian_noise(latent_buf_);
    dit_.denoise_and_cache(latent_buf_, frame_idx_, ctrl, denoise_steps);
    last_dit_ms_ = dit_.last_forward_gpu_ms();
    frame_idx_ += 1;

    vae_.decode_chunk_rgb(latent_buf_, out_rgb_4frames);
    last_vae_ms_ = vae_.last_decode_gpu_ms();
}

void WorldSession::step_bgra(const ControlState& ctrl,
                             std::array<Buffer, 4>& out_bgra_4frames,
                             uint32_t denoise_steps) {
    fill_gaussian_noise(latent_buf_);
    dit_.denoise_and_cache(latent_buf_, frame_idx_, ctrl, denoise_steps);
    last_dit_ms_ = dit_.last_forward_gpu_ms();
    frame_idx_ += 1;

    vae_.decode_chunk_bgra(latent_buf_, out_bgra_4frames);
    last_vae_ms_ = vae_.last_decode_gpu_ms();
}

}  // namespace spatial
