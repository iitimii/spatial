#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "spatial/backend/metal_backend.hpp"
#include "spatial/core/spat_format.hpp"
#include "spatial/core/tensor.hpp"

namespace spatial {

// Clean-room C++20/Metal implementation of ChunkedStreamingTAEHV (TAEHV1.5).
// Temporal compression = 4 (4 RGB frames <-> 1 latent [32, H/16, W/16]).
class TAEHV {
public:
    TAEHV(MetalBackend& backend, const std::string& spat_path,
          uint32_t pixel_h = 256, uint32_t pixel_w = 512);

    void reset();

    // Encode 4 RGB uint8 frames [4, H, W, 3] -> 1 F16 latent [32, H/16, W/16]
    void encode_chunk(const uint8_t* rgb_4frames, Buffer& out_latent_f16);

    // Decode 1 F16 latent [32, H/16, W/16] -> 4 RGB uint8 frames [4 * H * W * 3]
    void decode_chunk_rgb(const Buffer& latent_f16, std::vector<uint8_t>& out_rgb_4frames);

    // Decode 1 F16 latent -> 4 BGRA8 frames directly in unified GPU buffers (for M4 zero-copy display!)
    void decode_chunk_bgra(const Buffer& latent_f16, std::array<Buffer, 4>& out_bgra_frames);

    uint32_t pixel_h()  const { return pixel_h_; }
    uint32_t pixel_w()  const { return pixel_w_; }
    uint32_t latent_h() const { return pixel_h_ / 16; }
    uint32_t latent_w() const { return pixel_w_ / 16; }
    uint32_t latent_c() const { return 32; }

    double last_encode_gpu_ms() const { return last_encode_gpu_ms_; }
    double last_decode_gpu_ms() const { return last_decode_gpu_ms_; }

private:
    struct MemBlockWeights {
        TensorView w0, b0;
        TensorView w2, b2;
        TensorView w4, b4;
    };

    void run_memblock(const MemBlockWeights& mb, const Buffer& x_in,
                      Buffer& past_buf, bool& past_valid,
                      Buffer& x_out, Buffer& tmp0, Buffer& tmp1,
                      uint32_t C, uint32_t H, uint32_t W);

    void decode_one_latent_raw(const Buffer& latent_f16,
                               bool emit_frames,
                               bool bgra_mode,
                               std::array<Buffer, 4>* out_frames);

    MetalBackend& backend_;
    SpatFile      file_;
    Buffer        weights_buf_{};
    uint32_t      pixel_h_ = 256;
    uint32_t      pixel_w_ = 512;

    // Encoder weights
    TensorView enc0_w_, enc0_b_;
    TensorView enc2_w_;
    TensorView enc3_w_;
    std::array<MemBlockWeights, 3> enc_mb1_;
    TensorView enc7_w_;
    TensorView enc8_w_;
    std::array<MemBlockWeights, 3> enc_mb2_;
    TensorView enc12_w_;
    TensorView enc13_w_;
    std::array<MemBlockWeights, 3> enc_mb3_;
    TensorView enc17_w_, enc17_b_;

    // Decoder weights
    TensorView dec1_w_, dec1_b_;
    std::array<MemBlockWeights, 3> dec_mb1_;
    TensorView dec7_w_;
    TensorView dec8_w_;
    std::array<MemBlockWeights, 3> dec_mb2_;
    TensorView dec13_w_;
    TensorView dec14_w_;
    std::array<MemBlockWeights, 3> dec_mb3_;
    TensorView dec19_w_;
    TensorView dec20_w_;
    TensorView dec22_w_, dec22_b_;

    // Persistent recurrent MemBlock & TPool buffers (GPU-resident!)
    std::array<Buffer, 9> enc_mem_;
    std::array<bool, 9>   enc_mem_valid_{};
    std::array<Buffer, 9> dec_mem_;
    std::array<bool, 9>   dec_mem_valid_{};
    uint64_t              n_frames_decoded_ = 0;

    // Pre-allocated scratch buffers (zero runtime allocations!)
    Buffer scratch_pool_{};
    Arena  scratch_;
    Buffer enc_rgb_staging_{};
    Buffer dec_rgb_staging_[4]{};

    double last_encode_gpu_ms_ = 0.0;
    double last_decode_gpu_ms_ = 0.0;
};

}  // namespace spatial
