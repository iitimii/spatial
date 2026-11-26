#pragma once

#include <cstdint>
#include <vector>

#include "spatial/backend/metal_backend.hpp"
#include "spatial/core/tensor.hpp"

namespace spatial {

// Hierarchical Spatial-Temporal Paged Frame KV Cache.
// Supports local sliding windows, dilated global pinned memory, and future
// pose/frustum-indexed spatial retrieval on the exact same slot layout!
struct SpatialKVConfig {
    uint32_t n_layers              = 24;
    uint32_t n_kv_heads            = 16;
    uint32_t d_head                = 64;
    uint32_t tokens_per_frame      = 128;
    uint32_t local_window          = 16;
    uint32_t global_window         = 128;
    uint32_t global_attn_period    = 4;
    int32_t  global_attn_offset    = -1;
    uint32_t global_pinned_dilation = 8;
};

struct LayerSlotPlan {
    uint32_t ring_slot  = 0;
    uint32_t tail_slot  = 0;
    bool     write_ring = false;
    std::vector<uint32_t> active_slots;
};

class SpatialKVCache {
public:
    SpatialKVCache(MetalBackend& backend, const SpatialKVConfig& cfg);

    void reset();
    void set_frozen(bool frozen) { frozen_ = frozen; }
    bool is_frozen() const { return frozen_; }

    // Plan the slot read/write layout for (layer, frame_idx), and update `written` metadata if !frozen
    LayerSlotPlan prepare_layer(uint32_t layer, uint64_t frame_idx);

    const Buffer& k_buffer(uint32_t layer) const { return layers_[layer].k_buf; }
    const Buffer& v_buffer(uint32_t layer) const { return layers_[layer].v_buf; }
    uint32_t max_slots(uint32_t layer) const { return layers_[layer].num_buckets + 1; }
    size_t total_bytes() const { return total_bytes_; }

private:
    struct LayerState {
        uint32_t num_buckets = 16;
        uint32_t dilation    = 1;
        Buffer   k_buf{};
        Buffer   v_buf{};
        std::vector<bool> written;
    };

    SpatialKVConfig cfg_;
    bool frozen_ = true;
    size_t total_bytes_ = 0;
    std::vector<LayerState> layers_;
};

}  // namespace spatial
