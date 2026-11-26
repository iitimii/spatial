#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "spatial/backend/metal_backend.hpp"
#include "spatial/cache/spatial_kv_cache.hpp"
#include "spatial/core/dtype.hpp"
#include "spatial/core/profiler.hpp"

using namespace spatial;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

struct GemmParams { uint32_t M, N, K, flags; };
struct AttnParams {
    uint32_t T, n_heads, n_kv_heads, d_head, max_slots, num_active_slots;
    float scale;
    uint32_t active_slots[32];
};

static std::vector<BlockQ8_0> quant_q8(const std::vector<float>& w) {
    size_t nb = w.size() / 32;
    std::vector<BlockQ8_0> out(nb);
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(w[b * 32 + i]));
        float sc = amax / 127.0f;
        out[b].scale_f16 = f32_to_f16(sc);
        float inv = sc > 0.0f ? 1.0f / f16_to_f32(out[b].scale_f16) : 0.0f;
        for (int i = 0; i < 32; ++i) {
            int q = static_cast<int>(std::lround(w[b * 32 + i] * inv));
            out[b].qs[i] = static_cast<int8_t>(std::clamp(q, -127, 127));
        }
    }
    return out;
}

static std::vector<BlockQ4_0> quant_q4(const std::vector<float>& w) {
    size_t nb = w.size() / 32;
    std::vector<BlockQ4_0> out(nb);
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(w[b * 32 + i]));
        float sc = amax / 7.0f;
        out[b].scale_f16 = f32_to_f16(sc);
        float inv = sc > 0.0f ? 1.0f / f16_to_f32(out[b].scale_f16) : 0.0f;
        for (int i = 0; i < 16; ++i) {
            int q0 = std::clamp(static_cast<int>(std::lround(w[b * 32 + i] * inv)), -8, 7) + 8;
            int q1 = std::clamp(static_cast<int>(std::lround(w[b * 32 + 16 + i] * inv)), -8, 7) + 8;
            out[b].qs[i] = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
    return out;
}

static int test_gemm_and_perf(MetalBackend& mb) {
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 0.25f);

    // 1. Correctness test: M=128, N=256, K=512 for F16, Q8_0, Q4_0
    uint32_t M = 128, N = 256, K = 512;
    std::vector<float> A_f(M * K), B_f(N * K), C_ref(M * N, 0.0f);
    for (auto& v : A_f) v = dist(rng);
    for (auto& v : B_f) v = dist(rng);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            double s = 0.0;
            for (uint32_t k = 0; k < K; ++k) s += double(A_f[m * K + k]) * double(B_f[n * K + k]);
            C_ref[m * N + n] = static_cast<float>(s);
        }

    Buffer bufA = mb.alloc_buffer(M * K * 2);
    Buffer bufB16 = mb.alloc_buffer(N * K * 2);
    Buffer bufC = mb.alloc_buffer(M * N * 2);
    for (size_t i = 0; i < A_f.size(); ++i) static_cast<uint16_t*>(bufA.cpu_ptr)[i] = f32_to_f16(A_f[i]);
    for (size_t i = 0; i < B_f.size(); ++i) static_cast<uint16_t*>(bufB16.cpu_ptr)[i] = f32_to_f16(B_f[i]);

    GemmParams gp{M, N, K, 0u};
    mb.begin_batch();
    mb.dispatch_threadgroups("gemm_f16_f16", {N / 64, M / 32, 1}, {128, 1, 1},
                             {{&bufA, 0}, {&bufB16, 1}, {&bufC, 2}}, {{&gp, sizeof(gp), 3}});
    mb.end_batch_and_wait();

    float max_e16 = 0.0f;
    for (size_t i = 0; i < C_ref.size(); ++i) {
        max_e16 = std::max(max_e16, std::fabs(f16_to_f32(static_cast<uint16_t*>(bufC.cpu_ptr)[i]) - C_ref[i]));
    }
    std::printf("[M2] simdgroup GEMM F16:  max_err=%.5f\n", max_e16);
    CHECK(max_e16 < 0.05f, "F16 GEMM error too large");

    // Q8_0
    auto bq8 = quant_q8(B_f);
    Buffer bufBq8 = mb.alloc_buffer(bq8.size() * sizeof(BlockQ8_0));
    std::memcpy(bufBq8.cpu_ptr, bq8.data(), bq8.size() * sizeof(BlockQ8_0));
    mb.begin_batch();
    mb.dispatch_threadgroups("gemm_q8_0_f16", {N / 64, M / 32, 1}, {128, 1, 1},
                             {{&bufA, 0}, {&bufBq8, 1}, {&bufC, 2}}, {{&gp, sizeof(gp), 3}});
    mb.end_batch_and_wait();
    double sum_e8 = 0.0;
    for (size_t i = 0; i < C_ref.size(); ++i)
        sum_e8 += std::fabs(f16_to_f32(static_cast<uint16_t*>(bufC.cpu_ptr)[i]) - C_ref[i]);
    std::printf("[M2] simdgroup GEMM Q8_0: mean_err=%.5f\n", sum_e8 / C_ref.size());
    CHECK(sum_e8 / C_ref.size() < 0.02, "Q8_0 GEMM error too large");

    // Q4_0
    auto bq4 = quant_q4(B_f);
    Buffer bufBq4 = mb.alloc_buffer(bq4.size() * sizeof(BlockQ4_0));
    std::memcpy(bufBq4.cpu_ptr, bq4.data(), bq4.size() * sizeof(BlockQ4_0));
    mb.begin_batch();
    mb.dispatch_threadgroups("gemm_q4_0_f16", {N / 64, M / 32, 1}, {128, 1, 1},
                             {{&bufA, 0}, {&bufBq4, 1}, {&bufC, 2}}, {{&gp, sizeof(gp), 3}});
    mb.end_batch_and_wait();
    double sum_e4 = 0.0;
    for (size_t i = 0; i < C_ref.size(); ++i)
        sum_e4 += std::fabs(f16_to_f32(static_cast<uint16_t*>(bufC.cpu_ptr)[i]) - C_ref[i]);
    std::printf("[M2] simdgroup GEMM Q4_0: mean_err=%.5f\n", sum_e4 / C_ref.size());
    CHECK(sum_e4 / C_ref.size() < 0.25, "Q4_0 GEMM error too large");

    // 2. Real DiT MLP shape microbenchmark: M=128, N=8192, K=2048!
    uint32_t bM = 128, bN = 8192, bK = 2048;
    Buffer bmA = mb.alloc_buffer(bM * bK * 2);
    Buffer bmBq4 = mb.alloc_buffer((bN * bK / 32) * sizeof(BlockQ4_0));
    Buffer bmBq8 = mb.alloc_buffer((bN * bK / 32) * sizeof(BlockQ8_0));
    Buffer bmC = mb.alloc_buffer(bM * bN * 2);
    GemmParams bgp{bM, bN, bK, 0u};

    // Warmup + timed 10x
    for (int it = 0; it < 2; ++it) {
        mb.begin_batch();
        for (int r = 0; r < 10; ++r)
            mb.dispatch_threadgroups("gemm_q4_0_f16", {bN / 64, bM / 32, 1}, {128, 1, 1},
                                     {{&bmA, 0}, {&bmBq4, 1}, {&bmC, 2}}, {{&bgp, sizeof(bgp), 3}});
        mb.end_batch_and_wait();
    }
    double q4_ms = mb.last_gpu_ms() / 10.0;
    double flops = 2.0 * bM * bN * bK;
    std::printf("[M2] M1 GEMM Bench [128x2048x8192] Q4_0: %.3f ms (%.2f TFLOPS)\n",
                q4_ms, flops / (q4_ms * 1e9));

    for (int it = 0; it < 2; ++it) {
        mb.begin_batch();
        for (int r = 0; r < 10; ++r)
            mb.dispatch_threadgroups("gemm_q8_0_f16", {bN / 64, bM / 32, 1}, {128, 1, 1},
                                     {{&bmA, 0}, {&bmBq8, 1}, {&bmC, 2}}, {{&bgp, sizeof(bgp), 3}});
        mb.end_batch_and_wait();
    }
    double q8_ms = mb.last_gpu_ms() / 10.0;
    std::printf("[M2] M1 GEMM Bench [128x2048x8192] Q8_0: %.3f ms (%.2f TFLOPS)\n",
                q8_ms, flops / (q8_ms * 1e9));
    return 0;
}

static int test_flash_attn_and_kv_cache(MetalBackend& mb) {
    // Verify SpatialKVCache slot indexing & eviction
    SpatialKVConfig kvc;
    SpatialKVCache cache(mb, kvc);
    std::printf("[M2] SpatialKVCache allocated: %.1f MB (24 layers, 512-frame span)\n",
                cache.total_bytes() / (1024.0 * 1024.0));
    CHECK(cache.total_bytes() < 230 * 1024 * 1024, "KV cache should be ~214 MB");

    cache.set_frozen(false);
    auto p0_loc = cache.prepare_layer(0, 0);   // local layer, frame 0
    auto p0_glb = cache.prepare_layer(3, 0);   // global layer (off=-1 -> l=3,7,11,15,19,23)
    CHECK(p0_loc.write_ring && p0_loc.ring_slot == 0 && p0_loc.active_slots.size() == 1, "frame 0 local");
    CHECK(p0_glb.write_ring && p0_glb.ring_slot == 0 && p0_glb.active_slots.size() == 1, "frame 0 global");

    auto p1_glb = cache.prepare_layer(3, 1);   // frame 1: NOT multiple of 8 -> shouldn't write global ring!
    CHECK(!p1_glb.write_ring && p1_glb.active_slots.size() == 2, "frame 1 global attends to slot 0 + tail");

    // Verify FlashAttention GQA numerical accuracy vs FP64 CPU reference
    uint32_t T = 128, n_heads = 32, n_kv = 16, Dh = 64, max_slots = 17;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    Buffer q_buf = mb.alloc_buffer(n_heads * T * Dh * 2);
    Buffer k_buf = mb.alloc_buffer(n_kv * max_slots * T * Dh * 2);
    Buffer v_buf = mb.alloc_buffer(n_kv * max_slots * T * Dh * 2);
    Buffer o_buf = mb.alloc_buffer(T * n_heads * Dh * 2);

    auto fill_rand = [&](Buffer& b) {
        auto* p = static_cast<uint16_t*>(b.cpu_ptr);
        for (size_t i = 0; i < b.size_bytes / 2; ++i) p[i] = f32_to_f16(dist(rng));
    };
    fill_rand(q_buf);
    fill_rand(k_buf);
    fill_rand(v_buf);

    AttnParams ap{};
    ap.T = T; ap.n_heads = n_heads; ap.n_kv_heads = n_kv; ap.d_head = Dh;
    ap.max_slots = max_slots; ap.num_active_slots = 3;
    ap.scale = 1.0f / std::sqrt(float(Dh));
    ap.active_slots[0] = 0; ap.active_slots[1] = 2; ap.active_slots[2] = 16;

    mb.begin_batch();
    mb.dispatch_threadgroups("flash_attn_gqa_f16", {T, n_heads, 1}, {32, 1, 1},
                             {{&q_buf, 0}, {&k_buf, 1}, {&v_buf, 2}, {&o_buf, 3}},
                             {{&ap, sizeof(ap), 4}});
    mb.end_batch_and_wait();
    double attn_ms = mb.last_gpu_ms();

    // Check head 5, token 17 against CPU FP64 reference
    const uint16_t* q16 = static_cast<const uint16_t*>(q_buf.cpu_ptr);
    const uint16_t* k16 = static_cast<const uint16_t*>(k_buf.cpu_ptr);
    const uint16_t* v16 = static_cast<const uint16_t*>(v_buf.cpu_ptr);
    const uint16_t* o16 = static_cast<const uint16_t*>(o_buf.cpu_ptr);

    float max_attn_err = 0.0f;
    for (uint32_t qh : {0u, 5u, 31u}) {
        uint32_t kvh = qh / 2;
        for (uint32_t qt : {0u, 17u, 127u}) {
            std::vector<double> scores;
            for (uint32_t si = 0; si < ap.num_active_slots; ++si) {
                uint32_t sl = ap.active_slots[si];
                for (uint32_t kt = 0; kt < T; ++kt) {
                    double dot = 0.0;
                    for (uint32_t d = 0; d < Dh; ++d) {
                        double qv = f16_to_f32(q16[(qh * T + qt) * Dh + d]);
                        double kv = f16_to_f32(k16[((kvh * max_slots + sl) * T + kt) * Dh + d]);
                        dot += qv * kv;
                    }
                    scores.push_back(dot * ap.scale);
                }
            }
            double mx = *std::max_element(scores.begin(), scores.end());
            double denom = 0.0;
            for (auto& s : scores) { s = std::exp(s - mx); denom += s; }
            for (uint32_t d = 0; d < Dh; ++d) {
                double acc = 0.0;
                size_t idx = 0;
                for (uint32_t si = 0; si < ap.num_active_slots; ++si) {
                    uint32_t sl = ap.active_slots[si];
                    for (uint32_t kt = 0; kt < T; ++kt, ++idx) {
                        double vv = f16_to_f32(v16[((kvh * max_slots + sl) * T + kt) * Dh + d]);
                        acc += (scores[idx] / denom) * vv;
                    }
                }
                float gpu_v = f16_to_f32(o16[qt * (n_heads * Dh) + qh * Dh + d]);
                max_attn_err = std::max(max_attn_err, std::fabs(gpu_v - float(acc)));
            }
        }
    }
    std::printf("[M2] GQA FlashAttention (3 slots): gpu=%.3f ms | max_err=%.6f\n",
                attn_ms, max_attn_err);
    CHECK(max_attn_err < 2e-3f, "FlashAttention error too large");

    // Full 17-slot (2,176 KV tokens!) latency benchmark
    ap.num_active_slots = 17;
    for (uint32_t i = 0; i < 17; ++i) ap.active_slots[i] = i;
    mb.begin_batch();
    mb.dispatch_threadgroups("flash_attn_gqa_f16", {T, n_heads, 1}, {32, 1, 1},
                             {{&q_buf, 0}, {&k_buf, 1}, {&v_buf, 2}, {&o_buf, 3}},
                             {{&ap, sizeof(ap), 4}});
    mb.end_batch_and_wait();
    std::printf("[M2] GQA FlashAttention (full 17 slots = 2176 KV): gpu=%.3f ms\n", mb.last_gpu_ms());
    return 0;
}

int main() {
    MetalBackend mb;
    mb.load_shader_file(std::string(SPATIAL_SHADER_DIR) + "/dit.metal");
    if (test_gemm_and_perf(mb) != 0) return 1;
    if (test_flash_attn_and_kv_cache(mb) != 0) return 1;
    std::printf("[M2] ALL DIT KERNELS & KV CACHE TESTS PASSED!\n");
    return 0;
}
