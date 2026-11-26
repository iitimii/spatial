#include "spatial/models/taehv.hpp"

#include <cstring>
#include <stdexcept>

namespace spatial {

namespace {

struct Conv3x3Params {
    uint32_t C_in, C_out, H_in, W_in, H_out, W_out, stride, flags;
};

struct ConvGemmParams {
    uint32_t M, N, K, flags;
};

struct Conv1x1Params {
    uint32_t C_in, C_out, HW;
};

struct UpsampleParams {
    uint32_t C, H_in, W_in;
};

struct ImageParams {
    uint32_t H, W;
};

inline uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

inline MetalBackend::Grid tg2d(uint32_t w, uint32_t h) {
    return {std::min<uint32_t>(16, w), std::min<uint32_t>(16, h), 1};
}

// High-speed im2col + simdgroup_matrix 32x64 Conv3x3 helper
void dispatch_conv3x3(MetalBackend& backend, Arena& scratch,
                      const Buffer& inp, const Buffer& weight, const Buffer& bias,
                      const Buffer& resid, const Buffer& out,
                      uint32_t C_in, uint32_t C_out,
                      uint32_t H_in, uint32_t W_in,
                      uint32_t H_out, uint32_t W_out,
                      uint32_t stride, uint32_t flags) {
    ArenaScope sc(scratch);
    uint32_t K = C_in * 9u;
    uint32_t HW_out = H_out * W_out;
    Buffer col = scratch.alloc_bytes(static_cast<size_t>(HW_out) * K * 2);

    Conv3x3Params ip{C_in, C_out, H_in, W_in, H_out, W_out, stride, flags};
    backend.dispatch("im2col_3x3_f16", {W_out, H_out, C_in}, {8, 8, 1},
                     {{&inp, 0}, {&col, 1}}, {{&ip, sizeof(ip), 2}});

    ConvGemmParams gp{C_out, HW_out, K, flags};
    backend.dispatch_threadgroups("conv_gemm_f16", {div_up(HW_out, 64), div_up(C_out, 32), 1}, {128, 1, 1},
                                  {{&weight, 0}, {&col, 1}, {&bias, 2}, {&resid, 3}, {&out, 4}},
                                  {{&gp, sizeof(gp), 5}});
}

void dispatch_conv3x3_cat2(MetalBackend& backend, Arena& scratch,
                           const Buffer& x, const Buffer& past,
                           const Buffer& weight, const Buffer& bias, const Buffer& out,
                           uint32_t C_in, uint32_t C_out, uint32_t H, uint32_t W, uint32_t flags) {
    ArenaScope sc(scratch);
    uint32_t K = (C_in * 2u) * 9u;
    uint32_t HW = H * W;
    Buffer col = scratch.alloc_bytes(static_cast<size_t>(HW) * K * 2);

    Conv3x3Params ip{C_in, C_out, H, W, H, W, 1, flags};
    backend.dispatch("im2col_3x3_cat2_f16", {W, H, C_in}, {8, 8, 1},
                     {{&x, 0}, {&past, 1}, {&col, 2}}, {{&ip, sizeof(ip), 3}});

    ConvGemmParams gp{C_out, HW, K, flags & 7u};
    backend.dispatch_threadgroups("conv_gemm_f16", {div_up(HW, 64), div_up(C_out, 32), 1}, {128, 1, 1},
                                  {{&weight, 0}, {&col, 1}, {&bias, 2}, {&x, 3}, {&out, 4}},
                                  {{&gp, sizeof(gp), 5}});
}

}  // namespace

TAEHV::TAEHV(MetalBackend& backend, const std::string& spat_path,
             uint32_t pixel_h, uint32_t pixel_w)
    : backend_(backend),
      file_(SpatFile::open(spat_path)),
      pixel_h_(pixel_h),
      pixel_w_(pixel_w) {
    backend_.load_shader_file(std::string(SPATIAL_SHADER_DIR) + "/taehv.metal");
    weights_buf_ = backend_.wrap_mmap_buffer(file_.data_ptr(), file_.data_size());

    auto T = [&](const std::string& name) { return file_.get_tensor(name, weights_buf_); };
    auto load_mb = [&](const std::string& prefix) -> MemBlockWeights {
        return {
            T(prefix + ".conv.0.weight"), T(prefix + ".conv.0.bias"),
            T(prefix + ".conv.2.weight"), T(prefix + ".conv.2.bias"),
            T(prefix + ".conv.4.weight"), T(prefix + ".conv.4.bias"),
        };
    };

    // Encoder weights
    enc0_w_  = T("encoder.0.weight");  enc0_b_ = T("encoder.0.bias");
    enc2_w_  = T("encoder.2.conv.weight");
    enc3_w_  = T("encoder.3.weight");
    enc_mb1_ = {load_mb("encoder.4"), load_mb("encoder.5"), load_mb("encoder.6")};
    enc7_w_  = T("encoder.7.conv.weight");
    enc8_w_  = T("encoder.8.weight");
    enc_mb2_ = {load_mb("encoder.9"), load_mb("encoder.10"), load_mb("encoder.11")};
    enc12_w_ = T("encoder.12.conv.weight");
    enc13_w_ = T("encoder.13.weight");
    enc_mb3_ = {load_mb("encoder.14"), load_mb("encoder.15"), load_mb("encoder.16")};
    enc17_w_ = T("encoder.17.weight"); enc17_b_ = T("encoder.17.bias");

    // Decoder weights
    dec1_w_  = T("decoder.1.weight");  dec1_b_ = T("decoder.1.bias");
    dec_mb1_ = {load_mb("decoder.3"), load_mb("decoder.4"), load_mb("decoder.5")};
    dec7_w_  = T("decoder.7.conv.weight");
    dec8_w_  = T("decoder.8.weight");
    dec_mb2_ = {load_mb("decoder.9"), load_mb("decoder.10"), load_mb("decoder.11")};
    dec13_w_ = T("decoder.13.conv.weight");
    dec14_w_ = T("decoder.14.weight");
    dec_mb3_ = {load_mb("decoder.15"), load_mb("decoder.16"), load_mb("decoder.17")};
    dec19_w_ = T("decoder.19.conv.weight");
    dec20_w_ = T("decoder.20.weight");
    dec22_w_ = T("decoder.22.weight"); dec22_b_ = T("decoder.22.bias");

    uint32_t h4 = pixel_h_ / 4, w4 = pixel_w_ / 4;
    uint32_t h8 = pixel_h_ / 8, w8 = pixel_w_ / 8;
    uint32_t h16 = pixel_h_ / 16, w16 = pixel_w_ / 16;

    for (int i = 0; i < 3; ++i) enc_mem_[i]     = backend_.alloc_buffer(64 * h4 * w4 * 2);
    for (int i = 0; i < 3; ++i) enc_mem_[3 + i] = backend_.alloc_buffer(64 * h8 * w8 * 2);
    for (int i = 0; i < 3; ++i) enc_mem_[6 + i] = backend_.alloc_buffer(64 * h16 * w16 * 2);

    for (int i = 0; i < 3; ++i) dec_mem_[i]     = backend_.alloc_buffer(256 * h16 * w16 * 2);
    for (int i = 0; i < 3; ++i) dec_mem_[3 + i] = backend_.alloc_buffer(128 * h8 * w8 * 2);
    for (int i = 0; i < 3; ++i) dec_mem_[6 + i] = backend_.alloc_buffer(64 * h4 * w4 * 2);

    // 96 MB scratch arena (accommodates im2col tile buffers cleanly!)
    scratch_pool_ = backend_.alloc_buffer(96 * 1024 * 1024);
    scratch_ = Arena(scratch_pool_);

    enc_rgb_staging_ = backend_.alloc_buffer(4 * pixel_h_ * pixel_w_ * 3);
    for (int i = 0; i < 4; ++i) {
        dec_rgb_staging_[i] = backend_.alloc_buffer(pixel_h_ * pixel_w_ * 4);
    }
    reset();
}

void TAEHV::reset() {
    enc_mem_valid_.fill(false);
    dec_mem_valid_.fill(false);
    n_frames_decoded_ = 0;
}

void TAEHV::run_memblock(const MemBlockWeights& mb, const Buffer& x_in,
                         Buffer& past_buf, bool& past_valid,
                         Buffer& x_out, Buffer& tmp0, Buffer& tmp1,
                         uint32_t C, uint32_t H, uint32_t W) {
    dispatch_conv3x3_cat2(backend_, scratch_, x_in, past_buf, mb.w0.buf, mb.b0.buf, tmp0,
                          C, C, H, W, 1u | 2u | (past_valid ? 0u : 8u));
    dispatch_conv3x3(backend_, scratch_, tmp0, mb.w2.buf, mb.b2.buf, tmp0, tmp1,
                     C, C, H, W, H, W, 1, 1u | 2u);
    dispatch_conv3x3(backend_, scratch_, tmp1, mb.w4.buf, mb.b4.buf, x_in, x_out,
                     C, C, H, W, H, W, 1, 1u | 2u | 4u);

    uint32_t n = C * H * W;
    backend_.dispatch("copy_f16", {n, 1, 1}, {256, 1, 1},
                      {{&x_in, 0}, {&past_buf, 1}}, {{&n, sizeof(n), 2}});
    past_valid = true;
}

void TAEHV::encode_chunk(const uint8_t* rgb_4frames, Buffer& out_latent_f16) {
    const size_t frame_bytes = static_cast<size_t>(pixel_h_) * pixel_w_ * 3;
    std::memcpy(enc_rgb_staging_.cpu_ptr, rgb_4frames, 4 * frame_bytes);

    ArenaScope scope(scratch_);
    uint32_t h2 = pixel_h_ / 2, w2 = pixel_w_ / 2;
    uint32_t h4 = pixel_h_ / 4, w4 = pixel_w_ / 4;
    uint32_t h8 = pixel_h_ / 8, w8 = pixel_w_ / 8;
    uint32_t h16 = pixel_h_ / 16, w16 = pixel_w_ / 16;

    Buffer unshuf = scratch_.alloc_bytes(12 * h2 * w2 * 2);
    Buffer f_pair[2] = {scratch_.alloc_bytes(64 * h2 * w2 * 2),
                        scratch_.alloc_bytes(64 * h2 * w2 * 2)};
    Buffer tpool1_out = scratch_.alloc_bytes(64 * h2 * w2 * 2);
    Buffer s4_a = scratch_.alloc_bytes(64 * h4 * w4 * 2);
    Buffer s4_b = scratch_.alloc_bytes(64 * h4 * w4 * 2);
    Buffer s4_t0 = scratch_.alloc_bytes(64 * h4 * w4 * 2);
    Buffer s4_t1 = scratch_.alloc_bytes(64 * h4 * w4 * 2);
    Buffer g_pair[2] = {scratch_.alloc_bytes(64 * h4 * w4 * 2),
                        scratch_.alloc_bytes(64 * h4 * w4 * 2)};
    Buffer tpool2_out = scratch_.alloc_bytes(64 * h4 * w4 * 2);
    Buffer s8_a = scratch_.alloc_bytes(64 * h8 * w8 * 2);
    Buffer s8_b = scratch_.alloc_bytes(64 * h8 * w8 * 2);
    Buffer s8_t0 = scratch_.alloc_bytes(64 * h8 * w8 * 2);
    Buffer s8_t1 = scratch_.alloc_bytes(64 * h8 * w8 * 2);
    Buffer tpool3_out = scratch_.alloc_bytes(64 * h8 * w8 * 2);
    Buffer s16_a = scratch_.alloc_bytes(64 * h16 * w16 * 2);
    Buffer s16_b = scratch_.alloc_bytes(64 * h16 * w16 * 2);
    Buffer s16_t0 = scratch_.alloc_bytes(64 * h16 * w16 * 2);
    Buffer s16_t1 = scratch_.alloc_bytes(64 * h16 * w16 * 2);

    backend_.begin_batch();

    for (int pair = 0; pair < 2; ++pair) {
        for (int sub = 0; sub < 2; ++sub) {
            int f_idx = pair * 2 + sub;
            Buffer rgb_slice = enc_rgb_staging_.sub(f_idx * frame_bytes, frame_bytes);
            ImageParams ip{pixel_h_, pixel_w_};
            backend_.dispatch("rgb_u8_to_unshuffle_f16", {w2, h2, 1}, tg2d(w2, h2),
                              {{&rgb_slice, 0}, {&unshuf, 1}}, {{&ip, sizeof(ip), 2}});

            dispatch_conv3x3(backend_, scratch_, unshuf, enc0_w_.buf, enc0_b_.buf, unshuf, f_pair[sub],
                             12, 64, h2, w2, h2, w2, 1, 1u | 2u);
        }

        Conv1x1Params tp1{64, 64, h2 * w2};
        backend_.dispatch("conv2d_1x1_cat2_f16", {h2 * w2, 64, 1}, {64, 4, 1},
                          {{&f_pair[0], 0}, {&f_pair[1], 1}, {&enc2_w_.buf, 2}, {&tpool1_out, 3}},
                          {{&tp1, sizeof(tp1), 4}});

        dispatch_conv3x3(backend_, scratch_, tpool1_out, enc3_w_.buf, enc3_w_.buf, tpool1_out, s4_a,
                         64, 64, h2, w2, h4, w4, 2, 0u);

        run_memblock(enc_mb1_[0], s4_a, enc_mem_[0], enc_mem_valid_[0], s4_b, s4_t0, s4_t1, 64, h4, w4);
        run_memblock(enc_mb1_[1], s4_b, enc_mem_[1], enc_mem_valid_[1], s4_a, s4_t0, s4_t1, 64, h4, w4);
        run_memblock(enc_mb1_[2], s4_a, enc_mem_[2], enc_mem_valid_[2], g_pair[pair], s4_t0, s4_t1, 64, h4, w4);
    }

    Conv1x1Params tp2{64, 64, h4 * w4};
    backend_.dispatch("conv2d_1x1_cat2_f16", {h4 * w4, 64, 1}, {64, 4, 1},
                      {{&g_pair[0], 0}, {&g_pair[1], 1}, {&enc7_w_.buf, 2}, {&tpool2_out, 3}},
                      {{&tp2, sizeof(tp2), 4}});

    dispatch_conv3x3(backend_, scratch_, tpool2_out, enc8_w_.buf, enc8_w_.buf, tpool2_out, s8_a,
                     64, 64, h4, w4, h8, w8, 2, 0u);

    run_memblock(enc_mb2_[0], s8_a, enc_mem_[3], enc_mem_valid_[3], s8_b, s8_t0, s8_t1, 64, h8, w8);
    run_memblock(enc_mb2_[1], s8_b, enc_mem_[4], enc_mem_valid_[4], s8_a, s8_t0, s8_t1, 64, h8, w8);
    run_memblock(enc_mb2_[2], s8_a, enc_mem_[5], enc_mem_valid_[5], s8_b, s8_t0, s8_t1, 64, h8, w8);

    Conv1x1Params tp3{64, 64, h8 * w8};
    backend_.dispatch("conv2d_1x1_f16", {h8 * w8, 64, 1}, {64, 4, 1},
                      {{&s8_b, 0}, {&enc12_w_.buf, 1}, {&tpool3_out, 2}},
                      {{&tp3, sizeof(tp3), 3}});

    dispatch_conv3x3(backend_, scratch_, tpool3_out, enc13_w_.buf, enc13_w_.buf, tpool3_out, s16_a,
                     64, 64, h8, w8, h16, w16, 2, 0u);

    run_memblock(enc_mb3_[0], s16_a, enc_mem_[6], enc_mem_valid_[6], s16_b, s16_t0, s16_t1, 64, h16, w16);
    run_memblock(enc_mb3_[1], s16_b, enc_mem_[7], enc_mem_valid_[7], s16_a, s16_t0, s16_t1, 64, h16, w16);
    run_memblock(enc_mb3_[2], s16_a, enc_mem_[8], enc_mem_valid_[8], s16_b, s16_t0, s16_t1, 64, h16, w16);

    dispatch_conv3x3(backend_, scratch_, s16_b, enc17_w_.buf, enc17_b_.buf, s16_b, out_latent_f16,
                     64, 32, h16, w16, h16, w16, 1, 1u);

    backend_.end_batch_and_wait();
    last_encode_gpu_ms_ = backend_.last_gpu_ms();
}

void TAEHV::decode_one_latent_raw(const Buffer& latent_f16,
                                  bool emit_frames,
                                  bool bgra_mode,
                                  std::array<Buffer, 4>* out_frames) {
    ArenaScope scope(scratch_);
    uint32_t h16 = pixel_h_ / 16, w16 = pixel_w_ / 16;
    uint32_t h8  = pixel_h_ / 8,  w8  = pixel_w_ / 8;
    uint32_t h4  = pixel_h_ / 4,  w4  = pixel_w_ / 4;
    uint32_t h2  = pixel_h_ / 2,  w2  = pixel_w_ / 2;

    Buffer z_clamped = scratch_.alloc_bytes(32 * h16 * w16 * 2);
    Buffer s16_a  = scratch_.alloc_bytes(256 * h16 * w16 * 2);
    Buffer s16_b  = scratch_.alloc_bytes(256 * h16 * w16 * 2);
    Buffer s16_t0 = scratch_.alloc_bytes(256 * h16 * w16 * 2);
    Buffer s16_t1 = scratch_.alloc_bytes(256 * h16 * w16 * 2);

    Buffer up1    = scratch_.alloc_bytes(256 * h8 * w8 * 2);
    Buffer tg1    = scratch_.alloc_bytes(256 * h8 * w8 * 2);
    Buffer s8_a   = scratch_.alloc_bytes(128 * h8 * w8 * 2);
    Buffer s8_b   = scratch_.alloc_bytes(128 * h8 * w8 * 2);
    Buffer s8_t0  = scratch_.alloc_bytes(128 * h8 * w8 * 2);
    Buffer s8_t1  = scratch_.alloc_bytes(128 * h8 * w8 * 2);

    Buffer up2    = scratch_.alloc_bytes(128 * h4 * w4 * 2);
    Buffer tg2    = scratch_.alloc_bytes(256 * h4 * w4 * 2);
    Buffer s4_a   = scratch_.alloc_bytes(64 * h4 * w4 * 2);
    Buffer s4_b   = scratch_.alloc_bytes(64 * h4 * w4 * 2);
    Buffer s4_t0  = scratch_.alloc_bytes(64 * h4 * w4 * 2);
    Buffer s4_t1  = scratch_.alloc_bytes(64 * h4 * w4 * 2);

    Buffer up3    = scratch_.alloc_bytes(64 * h2 * w2 * 2);
    Buffer tg3    = scratch_.alloc_bytes(128 * h2 * w2 * 2);
    Buffer s2_a   = scratch_.alloc_bytes(64 * h2 * w2 * 2);
    Buffer s2_out = scratch_.alloc_bytes(12 * h2 * w2 * 2);

    uint32_t z_n = 32 * h16 * w16;
    backend_.dispatch("clamp_tanh3_f16", {z_n, 1, 1}, {256, 1, 1},
                      {{&latent_f16, 0}, {&z_clamped, 1}}, {{&z_n, sizeof(z_n), 2}});

    dispatch_conv3x3(backend_, scratch_, z_clamped, dec1_w_.buf, dec1_b_.buf, z_clamped, s16_a,
                     32, 256, h16, w16, h16, w16, 1, 1u | 2u);

    run_memblock(dec_mb1_[0], s16_a, dec_mem_[0], dec_mem_valid_[0], s16_b, s16_t0, s16_t1, 256, h16, w16);
    run_memblock(dec_mb1_[1], s16_b, dec_mem_[1], dec_mem_valid_[1], s16_a, s16_t0, s16_t1, 256, h16, w16);
    run_memblock(dec_mb1_[2], s16_a, dec_mem_[2], dec_mem_valid_[2], s16_b, s16_t0, s16_t1, 256, h16, w16);

    UpsampleParams u1{256, h16, w16};
    backend_.dispatch("upsample2x_nearest_f16", {w8, h8, 256}, {8, 8, 1},
                      {{&s16_b, 0}, {&up1, 1}}, {{&u1, sizeof(u1), 2}});

    Conv1x1Params c7{256, 256, h8 * w8};
    backend_.dispatch("conv2d_1x1_f16", {h8 * w8, 256, 1}, {64, 4, 1},
                      {{&up1, 0}, {&dec7_w_.buf, 1}, {&tg1, 2}}, {{&c7, sizeof(c7), 3}});

    dispatch_conv3x3(backend_, scratch_, tg1, dec8_w_.buf, dec8_w_.buf, tg1, s8_a,
                     256, 128, h8, w8, h8, w8, 1, 0u);

    run_memblock(dec_mb2_[0], s8_a, dec_mem_[3], dec_mem_valid_[3], s8_b, s8_t0, s8_t1, 128, h8, w8);
    run_memblock(dec_mb2_[1], s8_b, dec_mem_[4], dec_mem_valid_[4], s8_a, s8_t0, s8_t1, 128, h8, w8);
    run_memblock(dec_mb2_[2], s8_a, dec_mem_[5], dec_mem_valid_[5], s8_b, s8_t0, s8_t1, 128, h8, w8);

    UpsampleParams u2{128, h8, w8};
    backend_.dispatch("upsample2x_nearest_f16", {w4, h4, 128}, {8, 8, 1},
                      {{&s8_b, 0}, {&up2, 1}}, {{&u2, sizeof(u2), 2}});

    Conv1x1Params c13{128, 256, h4 * w4};
    backend_.dispatch("conv2d_1x1_f16", {h4 * w4, 256, 1}, {64, 4, 1},
                      {{&up2, 0}, {&dec13_w_.buf, 1}, {&tg2, 2}}, {{&c13, sizeof(c13), 3}});

    size_t half_tg2_bytes = static_cast<size_t>(128) * h4 * w4 * 2;
    size_t half_tg3_bytes = static_cast<size_t>(64) * h2 * w2 * 2;

    for (int f1 = 0; f1 < 2; ++f1) {
        Buffer tg2_f = tg2.sub(f1 * half_tg2_bytes, half_tg2_bytes);

        dispatch_conv3x3(backend_, scratch_, tg2_f, dec14_w_.buf, dec14_w_.buf, tg2_f, s4_a,
                         128, 64, h4, w4, h4, w4, 1, 0u);

        run_memblock(dec_mb3_[0], s4_a, dec_mem_[6], dec_mem_valid_[6], s4_b, s4_t0, s4_t1, 64, h4, w4);
        run_memblock(dec_mb3_[1], s4_b, dec_mem_[7], dec_mem_valid_[7], s4_a, s4_t0, s4_t1, 64, h4, w4);
        run_memblock(dec_mb3_[2], s4_a, dec_mem_[8], dec_mem_valid_[8], s4_b, s4_t0, s4_t1, 64, h4, w4);

        if (!emit_frames) {
            n_frames_decoded_ += 2;
            continue;
        }

        UpsampleParams u3{64, h4, w4};
        backend_.dispatch("upsample2x_nearest_f16", {w2, h2, 64}, {8, 8, 1},
                          {{&s4_b, 0}, {&up3, 1}}, {{&u3, sizeof(u3), 2}});

        Conv1x1Params c19{64, 128, h2 * w2};
        backend_.dispatch("conv2d_1x1_f16", {h2 * w2, 128, 1}, {64, 4, 1},
                          {{&up3, 0}, {&dec19_w_.buf, 1}, {&tg3, 2}}, {{&c19, sizeof(c19), 3}});

        for (int f2 = 0; f2 < 2; ++f2) {
            int out_idx = f1 * 2 + f2;
            Buffer tg3_f = tg3.sub(f2 * half_tg3_bytes, half_tg3_bytes);

            dispatch_conv3x3(backend_, scratch_, tg3_f, dec20_w_.buf, dec20_w_.buf, tg3_f, s2_a,
                             64, 64, h2, w2, h2, w2, 1, 2u);

            dispatch_conv3x3(backend_, scratch_, s2_a, dec22_w_.buf, dec22_b_.buf, s2_a, s2_out,
                             64, 12, h2, w2, h2, w2, 1, 1u);

            ImageParams ip{pixel_h_, pixel_w_};
            const char* kname = bgra_mode ? "shuffle_f16_to_bgra8" : "shuffle_f16_to_rgb_u8";
            backend_.dispatch(kname, {pixel_w_, pixel_h_, 1}, tg2d(pixel_w_, pixel_h_),
                              {{&s2_out, 0}, {&(*out_frames)[out_idx], 1}},
                              {{&ip, sizeof(ip), 2}});
            n_frames_decoded_ += 1;
        }
    }
}

void TAEHV::decode_chunk_rgb(const Buffer& latent_f16, std::vector<uint8_t>& out_rgb_4frames) {
    std::array<Buffer, 4> targets = {
        dec_rgb_staging_[0], dec_rgb_staging_[1], dec_rgb_staging_[2], dec_rgb_staging_[3]
    };

    backend_.begin_batch();
    if (n_frames_decoded_ == 0) {
        for (int w = 0; w < 3; ++w) {
            decode_one_latent_raw(latent_f16, false, false, nullptr);
        }
    }
    decode_one_latent_raw(latent_f16, true, false, &targets);
    backend_.end_batch_and_wait();
    last_decode_gpu_ms_ = backend_.last_gpu_ms();

    const size_t frame_bytes = static_cast<size_t>(pixel_h_) * pixel_w_ * 3;
    out_rgb_4frames.resize(4 * frame_bytes);
    for (int i = 0; i < 4; ++i) {
        std::memcpy(out_rgb_4frames.data() + i * frame_bytes,
                    dec_rgb_staging_[i].cpu_ptr, frame_bytes);
    }
}

void TAEHV::decode_chunk_bgra(const Buffer& latent_f16, std::array<Buffer, 4>& out_bgra_frames) {
    backend_.begin_batch();
    if (n_frames_decoded_ == 0) {
        for (int w = 0; w < 3; ++w) {
            decode_one_latent_raw(latent_f16, false, true, nullptr);
        }
    }
    decode_one_latent_raw(latent_f16, true, true, &out_bgra_frames);
    backend_.end_batch_and_wait();
    last_decode_gpu_ms_ = backend_.last_gpu_ms();
}

}  // namespace spatial
