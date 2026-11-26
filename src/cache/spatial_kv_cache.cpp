#include "spatial/cache/spatial_kv_cache.hpp"

#include <cassert>

namespace spatial {

SpatialKVCache::SpatialKVCache(MetalBackend& backend, const SpatialKVConfig& cfg)
    : cfg_(cfg) {
    layers_.resize(cfg.n_layers);
    int32_t period = static_cast<int32_t>(cfg.global_attn_period);
    int32_t off_mod = ((cfg.global_attn_offset % period) + period) % period;

    for (uint32_t l = 0; l < cfg.n_layers; ++l) {
        int32_t diff = static_cast<int32_t>(l) - off_mod;
        bool is_global = (((diff % period) + period) % period) == 0;

        LayerState& ls = layers_[l];
        ls.dilation = is_global ? cfg.global_pinned_dilation : 1u;
        uint32_t win = is_global ? cfg.global_window : cfg.local_window;
        ls.num_buckets = win / ls.dilation;
        assert(ls.num_buckets > 0);

        uint32_t slots = ls.num_buckets + 1;  // history ring + 1 current tail slot
        size_t bytes = static_cast<size_t>(cfg.n_kv_heads) * slots *
                       cfg.tokens_per_frame * cfg.d_head * sizeof(uint16_t);
        ls.k_buf = backend.alloc_buffer(bytes);
        ls.v_buf = backend.alloc_buffer(bytes);
        ls.written.assign(ls.num_buckets, false);
        total_bytes_ += 2 * bytes;
    }
}

void SpatialKVCache::reset() {
    for (auto& ls : layers_) {
        std::fill(ls.written.begin(), ls.written.end(), false);
    }
    frozen_ = true;
}

LayerSlotPlan SpatialKVCache::prepare_layer(uint32_t layer, uint64_t frame_idx) {
    LayerState& ls = layers_[layer];
    uint64_t bucket = (frame_idx + (ls.dilation - 1)) / ls.dilation;
    uint32_t slot   = static_cast<uint32_t>(bucket % ls.num_buckets);
    bool write_step = (frame_idx % ls.dilation) == 0;

    LayerSlotPlan plan;
    plan.ring_slot  = slot;
    plan.tail_slot  = ls.num_buckets;
    plan.write_ring = (!frozen_) && write_step;
    plan.active_slots.reserve(ls.num_buckets + 1);

    for (uint32_t s = 0; s < ls.num_buckets; ++s) {
        if (ls.written[s] && !(write_step && s == slot)) {
            plan.active_slots.push_back(s);
        }
    }
    plan.active_slots.push_back(plan.tail_slot);

    if (plan.write_ring) {
        ls.written[slot] = true;
    }
    return plan;
}

}  // namespace spatial
