#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// Quantized block layouts (matching include/spatial/core/dtype.hpp)
// ============================================================================
struct BlockQ8_0 {
    half scale;
    char qs[32];
};

struct BlockQ4_0 {
    half  scale;
    uchar qs[16];  // low nibble = [0..15] + 8, high nibble = [16..31] + 8
};

// ============================================================================
// 1. Wide-Tile 32x64x32 simdgroup_matrix GEMM: C[M, N] = A[M, K] * B[N, K]^T
//    - 4 simdgroups (128 threads), each computes 16x32 = 2x4 = 8 float8x8 tiles!
//    - 2x arithmetic intensity on sA reuse vs 32x32!
// ============================================================================
struct GemmParams {
    uint M;
    uint N;
    uint K;
    uint flags;  // bit 0: apply SiLU to output
};

// Shared-memory tile union so sA+sB (6 KB) and sC (8 KB) NEVER coexist -> 8 KB total!
union GemmShared32x64 {
    struct {
        half A[32][32];
        half B[64][32];
    } ab;
    float C[32][64];
};

inline void mma_tile_32x64(
    threadgroup const half sA[32][32],
    threadgroup const half sB[64][32],
    uint sg_id,
    thread simdgroup_half8x8 (&acc)[2][4]
) {
    uint sg_m = (sg_id >> 1) * 16u;  // 0 or 16
    uint sg_n = (sg_id & 1u) * 32u;  // 0 or 32

    #pragma unroll
    for (uint kk = 0; kk < 32; kk += 8) {
        simdgroup_half8x8 a0, a1;
        simdgroup_load(a0, &sA[sg_m + 0][kk], 32, ulong2(0, 0), false);
        simdgroup_load(a1, &sA[sg_m + 8][kk], 32, ulong2(0, 0), false);

        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            simdgroup_half8x8 bj;
            simdgroup_load(bj, &sB[sg_n + j * 8][kk], 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][j], a0, bj, acc[0][j]);
            simdgroup_multiply_accumulate(acc[1][j], a1, bj, acc[1][j]);
        }
    }
}

inline void store_tile_32x64_h(
    thread simdgroup_half8x8 (&acc)[2][4],
    threadgroup half sCh[32][64],
    device half* C,
    uint m0, uint n0, uint M, uint N, uint flags,
    uint sg_id, uint tid
) {
    uint sg_m = (sg_id >> 1) * 16u;
    uint sg_n = (sg_id & 1u) * 32u;
    #pragma unroll
    for (uint i = 0; i < 2; ++i)
        #pragma unroll
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], &sCh[sg_m + i * 8][sg_n + j * 8], 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 2048u; i += 128u) {
        uint lm = i >> 6;
        uint ln = i & 63u;
        uint gm = m0 + lm;
        uint gn = n0 + ln;
        if (gm < M && gn < N) {
            half v = sCh[lm][ln];
            if (flags & 1u) {
                float fv = float(v);
                v = half(fv / (1.0f + exp(-fv)));
            }
            C[gm * N + gn] = v;
        }
    }
}

kernel void gemm_f16_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half*       C [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup union {
        struct { half A[32][32]; half B[64][32]; } ab;
        half Ch[32][64];
    } smem;

    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 64u;

    simdgroup_half8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0; i < 2; ++i)
        #pragma unroll
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));

    uint row = tid >> 2;
    uint col8 = (tid & 3u) * 8u;

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        uint gm = m0 + row;
        if (gm < p.M) {
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                uint gk = k0 + col8 + j;
                smem.ab.A[row][col8 + j] = (gk < p.K) ? A[gm * p.K + gk] : half(0.0h);
            }
        } else {
            #pragma unroll
            for (uint j = 0; j < 8; ++j) smem.ab.A[row][col8 + j] = half(0.0h);
        }

        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            uint lr = row * 2u + r;
            uint gn = n0 + lr;
            if (gn < p.N) {
                #pragma unroll
                for (uint j = 0; j < 8; ++j) {
                    uint gk = k0 + col8 + j;
                    smem.ab.B[lr][col8 + j] = (gk < p.K) ? B[gn * p.K + gk] : half(0.0h);
                }
            } else {
                #pragma unroll
                for (uint j = 0; j < 8; ++j) smem.ab.B[lr][col8 + j] = half(0.0h);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_tile_32x64(smem.ab.A, smem.ab.B, sg_id, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    store_tile_32x64_h(acc, smem.Ch, C, m0, n0, p.M, p.N, p.flags, sg_id, tid);
}

// Q8_0 wide-tile 32x64x32 simdgroup_half8x8 GEMM (6 KB shared mem -> max M1 occupancy!)
kernel void gemm_q8_0_f16(
    device const half*      A [[buffer(0)]],
    device const BlockQ8_0* B [[buffer(1)]],
    device half*            C [[buffer(2)]],
    constant GemmParams&    p [[buffer(3)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_index_in_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup union {
        struct { half A[32][32]; half B[64][32]; } ab;
        half Ch[32][64];
    } smem;

    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 64u;
    uint num_blocks_k = p.K >> 5;

    simdgroup_half8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0; i < 2; ++i)
        #pragma unroll
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));

    uint row  = tid >> 2;
    uint sub  = tid & 3u;
    uint col8 = sub * 8u;

    for (uint kb = 0; kb < num_blocks_k; ++kb) {
        uint k0 = kb << 5;
        uint gm = m0 + row;
        if (gm < p.M) {
            device const half4* a_ptr = reinterpret_cast<device const half4*>(A + gm * p.K + k0 + col8);
            *reinterpret_cast<threadgroup half4*>(&smem.ab.A[row][col8 + 0]) = a_ptr[0];
            *reinterpret_cast<threadgroup half4*>(&smem.ab.A[row][col8 + 4]) = a_ptr[1];
        } else {
            *reinterpret_cast<threadgroup half4*>(&smem.ab.A[row][col8 + 0]) = half4(0.0h);
            *reinterpret_cast<threadgroup half4*>(&smem.ab.A[row][col8 + 4]) = half4(0.0h);
        }

        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            uint lr = row * 2u + r;
            uint gn = n0 + lr;
            if (gn < p.N) {
                device const BlockQ8_0& blk = B[gn * num_blocks_k + kb];
                half4 sc = half4(blk.scale);
                half4 q0 = half4(blk.qs[col8 + 0], blk.qs[col8 + 1], blk.qs[col8 + 2], blk.qs[col8 + 3]);
                half4 q1 = half4(blk.qs[col8 + 4], blk.qs[col8 + 5], blk.qs[col8 + 6], blk.qs[col8 + 7]);
                *reinterpret_cast<threadgroup half4*>(&smem.ab.B[lr][col8 + 0]) = q0 * sc;
                *reinterpret_cast<threadgroup half4*>(&smem.ab.B[lr][col8 + 4]) = q1 * sc;
            } else {
                *reinterpret_cast<threadgroup half4*>(&smem.ab.B[lr][col8 + 0]) = half4(0.0h);
                *reinterpret_cast<threadgroup half4*>(&smem.ab.B[lr][col8 + 4]) = half4(0.0h);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_tile_32x64(smem.ab.A, smem.ab.B, sg_id, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    store_tile_32x64_h(acc, smem.Ch, C, m0, n0, p.M, p.N, p.flags, sg_id, tid);
}

// Q4_0 wide-tile 32x64x32 simdgroup_half8x8 GEMM (6 KB shared mem -> max M1 occupancy!)
kernel void gemm_q4_0_f16(
    device const half*      A [[buffer(0)]],
    device const BlockQ4_0* B [[buffer(1)]],
    device half*            C [[buffer(2)]],
    constant GemmParams&    p [[buffer(3)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_index_in_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup union {
        struct { half A[32][32]; half B[64][32]; } ab;
        half Ch[32][64];
    } smem;

    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 64u;
    uint num_blocks_k = p.K >> 5;

    simdgroup_half8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0; i < 2; ++i)
        #pragma unroll
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));

    uint row   = tid >> 2;
    uint sub   = tid & 3u;
    uint col8  = sub * 8u;
    uint b_off = sub * 4u;

    for (uint kb = 0; kb < num_blocks_k; ++kb) {
        uint k0 = kb << 5;
        uint gm = m0 + row;
        if (gm < p.M) {
            device const half4* a_ptr = reinterpret_cast<device const half4*>(A + gm * p.K + k0 + col8);
            *reinterpret_cast<threadgroup half4*>(&smem.ab.A[row][col8 + 0]) = a_ptr[0];
            *reinterpret_cast<threadgroup half4*>(&smem.ab.A[row][col8 + 4]) = a_ptr[1];
        } else {
            *reinterpret_cast<threadgroup half4*>(&smem.ab.A[row][col8 + 0]) = half4(0.0h);
            *reinterpret_cast<threadgroup half4*>(&smem.ab.A[row][col8 + 4]) = half4(0.0h);
        }

        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            uint lr = row * 2u + r;
            uint gn = n0 + lr;
            if (gn < p.N) {
                device const BlockQ4_0& blk = B[gn * num_blocks_k + kb];
                half4 sc = half4(blk.scale);
                uchar b0 = blk.qs[b_off + 0], b1 = blk.qs[b_off + 1];
                uchar b2 = blk.qs[b_off + 2], b3 = blk.qs[b_off + 3];
                half4 lo = half4(int(b0 & 0xFu) - 8, int(b1 & 0xFu) - 8,
                                 int(b2 & 0xFu) - 8, int(b3 & 0xFu) - 8) * sc;
                half4 hi = half4(int(b0 >> 4) - 8, int(b1 >> 4) - 8,
                                 int(b2 >> 4) - 8, int(b3 >> 4) - 8) * sc;
                *reinterpret_cast<threadgroup half4*>(&smem.ab.B[lr][b_off])       = lo;
                *reinterpret_cast<threadgroup half4*>(&smem.ab.B[lr][16u + b_off]) = hi;
            } else {
                *reinterpret_cast<threadgroup half4*>(&smem.ab.B[lr][b_off])       = half4(0.0h);
                *reinterpret_cast<threadgroup half4*>(&smem.ab.B[lr][16u + b_off]) = half4(0.0h);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_tile_32x64(smem.ab.A, smem.ab.B, sg_id, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    store_tile_32x64_h(acc, smem.Ch, C, m0, n0, p.M, p.N, p.flags, sg_id, tid);
}

// Fast GEMV (M == 1) for controller embeddings and fc1_c: 1 simdgroup per output row
kernel void gemv_f16_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half*       C [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid;
    if (row >= p.N) return;
    device const half* b_row = B + row * p.K;
    float sum = 0.0f;
    for (uint k = lane; k < p.K; k += 32u) {
        sum = fma(float(A[k]), float(b_row[k]), sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        if (p.flags & 1u) sum = sum / (1.0f + exp(-sum));
        C[row] = half(sum);
    }
}

kernel void gemv_q8_0_f16(
    device const half*      A [[buffer(0)]],
    device const BlockQ8_0* B [[buffer(1)]],
    device half*            C [[buffer(2)]],
    constant GemmParams&    p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid;
    if (row >= p.N) return;
    uint nb = p.K >> 5;
    device const BlockQ8_0* b_row = B + row * nb;
    float sum = 0.0f;
    for (uint kb = 0; kb < nb; ++kb) {
        float sc = float(b_row[kb].scale);
        float w = float(b_row[kb].qs[lane]) * sc;
        sum = fma(float(A[(kb << 5) + lane]), w, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        if (p.flags & 1u) sum = sum / (1.0f + exp(-sum));
        C[row] = half(sum);
    }
}

kernel void gemv_q4_0_f16(
    device const half*      A [[buffer(0)]],
    device const BlockQ4_0* B [[buffer(1)]],
    device half*            C [[buffer(2)]],
    constant GemmParams&    p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid;
    if (row >= p.N) return;
    uint nb = p.K >> 5;
    device const BlockQ4_0* b_row = B + row * nb;
    float sum = 0.0f;
    for (uint kb = 0; kb < nb; ++kb) {
        float sc = float(b_row[kb].scale);
        uchar bv = b_row[kb].qs[lane & 15u];
        int q = (lane < 16u) ? (int(bv & 0x0Fu) - 8) : (int(bv >> 4) - 8);
        sum = fma(float(A[(kb << 5) + lane]), float(q) * sc, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        if (p.flags & 1u) sum = sum / (1.0f + exp(-sum));
        C[row] = half(sum);
    }
}

// ============================================================================
// 2. Norms, Conditioning & Elementwise Fused Ops
// ============================================================================
struct NormParams {
    uint T;
    uint D;
    uint flags;  // bit 0: apply AdaLN (scale, bias), bit 1: apply SiLU after
};

// 1 threadgroup (256 threads) per token row
kernel void rmsnorm_ada_f16(
    device const half* x     [[buffer(0)]],
    device const half* scale [[buffer(1)]],
    device const half* bias  [[buffer(2)]],
    device half*       out   [[buffer(3)]],
    constant NormParams& p   [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (tgid >= p.T) return;
    threadgroup float partial[8];

    device const half* x_row = x + tgid * p.D;
    device half*       o_row = out + tgid * p.D;

    float sq = 0.0f;
    for (uint d = tid; d < p.D; d += 256u) {
        float v = float(x_row[d]);
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0) partial[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = 0.0f;
    if (sg == 0) {
        float v = (lane < 8u) ? partial[lane] : 0.0f;
        total_sq = simd_sum(v);
        if (lane == 0) partial[0] = rsqrt(total_sq / float(p.D) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = partial[0];

    for (uint d = tid; d < p.D; d += 256u) {
        float v = float(x_row[d]) * inv_rms;
        if (p.flags & 1u) {
            v = v * (1.0f + float(scale[d])) + float(bias[d]);
        }
        if (p.flags & 2u) {
            v = v / (1.0f + exp(-v));
        }
        o_row[d] = half(v);
    }
}

struct GateParams {
    uint T;
    uint D;
};

// out[t, d] = resid[t, d] + x[t, d] * gate[d]
kernel void ada_gate_add_f16(
    device const half* x     [[buffer(0)]],
    device const half* gate  [[buffer(1)]],
    device half*       resid [[buffer(2)]],
    constant GateParams& p   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint t = gid.y;
    if (d >= p.D || t >= p.T) return;
    uint idx = t * p.D + d;
    resid[idx] = half(float(resid[idx]) + float(x[idx]) * float(gate[d]));
}

// MLPFusion activation: out[t, d] = silu(x[t, d] + c[d])
kernel void add_vec_silu_f16(
    device const half* x   [[buffer(0)]],
    device const half* c   [[buffer(1)]],
    device half*       out [[buffer(2)]],
    constant GateParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint t = gid.y;
    if (d >= p.D || t >= p.T) return;
    uint idx = t * p.D + d;
    float v = float(x[idx]) + float(c[d]);
    out[idx] = half(v / (1.0f + exp(-v)));
}

// resid[i] += x[i]
kernel void add_inplace_f16(
    device const half* x     [[buffer(0)]],
    device half*       resid [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < n) resid[id] = half(float(resid[id]) + float(x[id]));
}

// ============================================================================
// 3. Fused QKV Unpack + Value Residual + QK-RMSNorm + OrthoRoPE + KV-Cache Upsert
//    Exact OrthoRoPE math:
//      x0 = x[2*i], x1 = x[2*i+1]  (i = 0..31)
//      y0 = x0*cos[i] - x1*sin[i]
//      y1 = x1*cos[i] + x0*sin[i]
//      out[i] = y0, out[32+i] = y1
// ============================================================================
struct QkvPostParams {
    uint T;            // tokens per frame (128)
    uint n_heads;      // 32
    uint n_kv_heads;   // 16
    uint d_head;       // 64
    uint layer_idx;
    uint tail_slot;    // 16
    uint ring_slot;    // 0..15
    uint write_ring;   // 1 if (!frozen && write_step), else 0
    uint max_slots;    // 17
    float v_lamb;
};

// Q kernel: 1 simdgroup (32 threads) per (head, token), 2 dims per lane
kernel void q_norm_rope_f16(
    device const half*  qkv      [[buffer(0)]],  // [T, (n_heads + 2*n_kv_heads)*d_head]
    device const float* rope_cos [[buffer(1)]],  // [T, d_head/2]
    device const float* rope_sin [[buffer(2)]],  // [T, d_head/2]
    device half*        q_out    [[buffer(3)]],  // [n_heads, T, d_head]
    constant QkvPostParams& p    [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint t = tgid.x;
    uint h = tgid.y;
    if (t >= p.T || h >= p.n_heads) return;

    uint qkv_stride = (p.n_heads + 2u * p.n_kv_heads) * p.d_head;
    device const half* src = qkv + t * qkv_stride + h * p.d_head;

    float x0 = float(src[2u * lane]);
    float x1 = float(src[2u * lane + 1u]);

    float sq = simd_sum(x0 * x0 + x1 * x1);
    float inv_rms = rsqrt(sq / float(p.d_head) + 1e-6f);
    x0 *= inv_rms;
    x1 *= inv_rms;

    float c = rope_cos[t * 32u + lane];
    float s = rope_sin[t * 32u + lane];
    float y0 = x0 * c - x1 * s;
    float y1 = x1 * c + x0 * s;

    device half* dst = q_out + (h * p.T + t) * p.d_head;
    dst[lane]       = half(y0);
    dst[32u + lane] = half(y1);
}

// KV kernel: 1 simdgroup (32 threads) per (kv_head, token)
// Handles K norm + OrthoRoPE, V value-residual lerp (and saves v1 on layer 0),
// and writes directly into tail_slot (and ring_slot if write_ring == 1)!
kernel void kv_post_upsert_f16(
    device const half*  qkv      [[buffer(0)]],
    device const float* rope_cos [[buffer(1)]],
    device const float* rope_sin [[buffer(2)]],
    device half*        v1_buf   [[buffer(3)]],  // [n_kv_heads, T, d_head]
    device half*        k_cache  [[buffer(4)]],  // [n_kv_heads, max_slots, T, d_head]
    device half*        v_cache  [[buffer(5)]],  // [n_kv_heads, max_slots, T, d_head]
    constant QkvPostParams& p    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint t = tgid.x;
    uint h = tgid.y;
    if (t >= p.T || h >= p.n_kv_heads) return;

    uint q_dim = p.n_heads * p.d_head;
    uint kv_dim = p.n_kv_heads * p.d_head;
    uint qkv_stride = q_dim + 2u * kv_dim;

    device const half* k_src = qkv + t * qkv_stride + q_dim + h * p.d_head;
    device const half* v_src = qkv + t * qkv_stride + q_dim + kv_dim + h * p.d_head;

    // 1) K RMSNorm + OrthoRoPE
    float k0 = float(k_src[2u * lane]);
    float k1 = float(k_src[2u * lane + 1u]);
    float sq = simd_sum(k0 * k0 + k1 * k1);
    float inv_rms = rsqrt(sq / float(p.d_head) + 1e-6f);
    k0 *= inv_rms;
    k1 *= inv_rms;

    float c = rope_cos[t * 32u + lane];
    float s = rope_sin[t * 32u + lane];
    half ky0 = half(k0 * c - k1 * s);
    half ky1 = half(k1 * c + k0 * s);

    // 2) V value residual (lerp(v, v1, v_lamb))
    uint v1_off = (h * p.T + t) * p.d_head;
    float vx0 = float(v_src[lane]);
    float vx1 = float(v_src[32u + lane]);
    if (p.layer_idx == 0u) {
        v1_buf[v1_off + lane]       = half(vx0);
        v1_buf[v1_off + 32u + lane] = half(vx1);
    } else {
        float r0 = float(v1_buf[v1_off + lane]);
        float r1 = float(v1_buf[v1_off + 32u + lane]);
        vx0 = vx0 + p.v_lamb * (r0 - vx0);
        vx1 = vx1 + p.v_lamb * (r1 - vx1);
    }
    half hv0 = half(vx0);
    half hv1 = half(vx1);

    // 3) Write K, V to tail_slot (and ring_slot if committing cache pass)
    uint tail_off = ((h * p.max_slots + p.tail_slot) * p.T + t) * p.d_head;
    k_cache[tail_off + lane]       = ky0;
    k_cache[tail_off + 32u + lane] = ky1;
    v_cache[tail_off + lane]       = hv0;
    v_cache[tail_off + 32u + lane] = hv1;

    if (p.write_ring != 0u) {
        uint ring_off = ((h * p.max_slots + p.ring_slot) * p.T + t) * p.d_head;
        k_cache[ring_off + lane]       = ky0;
        k_cache[ring_off + 32u + lane] = ky1;
        v_cache[ring_off + lane]       = hv0;
        v_cache[ring_off + 32u + lane] = hv1;
    }
}

// ============================================================================
// 4. Register-resident Online Softmax GQA FlashAttention over Paged Frame Cache
//    1 simdgroup (32 threads) per (q_head, q_token); Dh=64 -> 2 dims/lane!
//    Outputs directly in [T, n_heads * d_head] layout (zero transpose copy!)
// ============================================================================
struct AttnParams {
    uint  T;
    uint  n_heads;
    uint  n_kv_heads;
    uint  d_head;
    uint  max_slots;
    uint  num_active_slots;
    float scale;
    uint  active_slots[32];
};

kernel void flash_attn_gqa_f16(
    device const half* q       [[buffer(0)]],  // [n_heads, T, 64]
    device const half* k_cache [[buffer(1)]],  // [n_kv_heads, max_slots, T, 64]
    device const half* v_cache [[buffer(2)]],  // [n_kv_heads, max_slots, T, 64]
    device half*       out     [[buffer(3)]],  // [T, n_heads * 64]
    constant AttnParams& p     [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint qt = tgid.x;
    uint qh = tgid.y;
    if (qt >= p.T || qh >= p.n_heads) return;

    uint kvh = qh / (p.n_heads / p.n_kv_heads);

    device const half* q_ptr = q + (qh * p.T + qt) * 64u;
    float q0 = float(q_ptr[lane]);
    float q1 = float(q_ptr[32u + lane]);

    float m = -INFINITY;
    float l = 0.0f;
    float o0 = 0.0f;
    float o1 = 0.0f;

    uint head_base = kvh * p.max_slots * p.T * 64u;
    for (uint si = 0; si < p.num_active_slots; ++si) {
        uint slot = p.active_slots[si];
        device const half* k_slot = k_cache + head_base + slot * p.T * 64u;
        device const half* v_slot = v_cache + head_base + slot * p.T * 64u;

        // 4-token unrolled online softmax: 1 max + 1 rescaling per 4 KV tokens!
        for (uint kt = 0; kt < p.T; kt += 4u) {
            float sc[4];
            #pragma unroll
            for (uint u = 0; u < 4; ++u) {
                uint off = (kt + u) * 64u;
                float k0 = float(k_slot[off + lane]);
                float k1 = float(k_slot[off + 32u + lane]);
                sc[u] = simd_sum(fma(q0, k0, q1 * k1)) * p.scale;
            }
            float m4 = max(max(sc[0], sc[1]), max(sc[2], sc[3]));
            float m_new = max(m, m4);
            float alpha = exp(m - m_new);
            o0 *= alpha;
            o1 *= alpha;
            l  *= alpha;

            #pragma unroll
            for (uint u = 0; u < 4; ++u) {
                float w = exp(sc[u] - m_new);
                l += w;
                uint off = (kt + u) * 64u;
                o0 = fma(w, float(v_slot[off + lane]), o0);
                o1 = fma(w, float(v_slot[off + 32u + lane]), o1);
            }
            m = m_new;
        }
    }

    float inv_l = 1.0f / l;
    device half* o_ptr = out + qt * (p.n_heads * 64u) + qh * 64u;
    o_ptr[lane]       = half(o0 * inv_l);
    o_ptr[32u + lane] = half(o1 * inv_l);
}

// ============================================================================
// 5. Patchify / Unpatchify Rearrange & Rectified-Flow Euler Step
// ============================================================================
struct PatchParams {
    uint C;
    uint Hp;  // patch grid height (8)
    uint Wp;  // patch grid width  (16)
};

// [C, Hp*2, Wp*2] -> [Hp*Wp, C*4] (matches Conv2d weight [D, C, 2, 2] flattened [D, C*4])
kernel void patchify_rearrange_f16(
    device const half* inp [[buffer(0)]],
    device half*       out [[buffer(1)]],
    constant PatchParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint tok = gid.x;
    uint c   = gid.y;
    uint T = p.Hp * p.Wp;
    if (tok >= T || c >= p.C) return;

    uint py = tok / p.Wp;
    uint px = tok % p.Wp;
    uint H = p.Hp * 2u;
    uint W = p.Wp * 2u;
    device const half* c_in = inp + c * (H * W);
    device half* dst = out + tok * (p.C * 4u) + c * 4u;

    dst[0] = c_in[(py * 2u + 0u) * W + (px * 2u + 0u)];
    dst[1] = c_in[(py * 2u + 0u) * W + (px * 2u + 1u)];
    dst[2] = c_in[(py * 2u + 1u) * W + (px * 2u + 0u)];
    dst[3] = c_in[(py * 2u + 1u) * W + (px * 2u + 1u)];
}

// [Hp*Wp, C*4] + bias[C] -> [C, Hp*2, Wp*2]
kernel void unpatchify_rearrange_f16(
    device const half* inp  [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant PatchParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint tok = gid.x;
    uint c   = gid.y;
    uint T = p.Hp * p.Wp;
    if (tok >= T || c >= p.C) return;

    uint py = tok / p.Wp;
    uint px = tok % p.Wp;
    uint H = p.Hp * 2u;
    uint W = p.Wp * 2u;
    float b = float(bias[c]);
    device const half* src = inp + tok * (p.C * 4u) + c * 4u;
    device half* c_out = out + c * (H * W);

    c_out[(py * 2u + 0u) * W + (px * 2u + 0u)] = half(float(src[0]) + b);
    c_out[(py * 2u + 0u) * W + (px * 2u + 1u)] = half(float(src[1]) + b);
    c_out[(py * 2u + 1u) * W + (px * 2u + 0u)] = half(float(src[2]) + b);
    c_out[(py * 2u + 1u) * W + (px * 2u + 1u)] = half(float(src[3]) + b);
}

// Rectified flow Euler step: x[i] = x[i] + dsigma * v[i]
struct FlowStepParams {
    uint  n;
    float dsigma;
};

kernel void flow_euler_step_f16(
    device half*       x [[buffer(0)]],
    device const half* v [[buffer(1)]],
    constant FlowStepParams& p [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < p.n) {
        x[id] = half(fma(p.dsigma, float(v[id]), float(x[id])));
    }
}
