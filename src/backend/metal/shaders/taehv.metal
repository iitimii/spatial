#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct Conv3x3Params {
    uint C_in;
    uint C_out;
    uint H_in;
    uint W_in;
    uint H_out;
    uint W_out;
    uint stride;
    uint flags;  // bit 0: has_bias, bit 1: relu, bit 2: add_residual, bit 3: past_is_zero
};

// ============================================================================
// High-speed im2col + simdgroup_matrix Conv3x3 for Apple Silicon M1!
// Col layout: [HW_out, K] where K = C_in * 9 (matches Conv2d weight [C_out, C_in*9]!)
// Then Out[C_out, HW_out] = W[C_out, K] * Col[HW_out, K]^T (exact NCHW layout!)
// ============================================================================
kernel void im2col_3x3_f16(
    device const half* inp [[buffer(0)]],
    device half*       col [[buffer(1)]],
    constant Conv3x3Params& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint ic = gid.z;
    if (ow >= p.W_out || oh >= p.H_out || ic >= p.C_in) return;

    uint hw_idx = oh * p.W_out + ow;
    uint K = p.C_in * 9u;
    device half* dst = col + hw_idx * K + ic * 9u;

    int ih0 = int(oh * p.stride) - 1;
    int iw0 = int(ow * p.stride) - 1;
    device const half* inp_c = inp + ic * (p.H_in * p.W_in);

    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        int ih = ih0 + kh;
        bool valid_h = (ih >= 0 && ih < int(p.H_in));
        uint row_off = valid_h ? (uint(ih) * p.W_in) : 0u;
        #pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            int iw = iw0 + kw;
            dst[kh * 3 + kw] = (valid_h && iw >= 0 && iw < int(p.W_in))
                ? inp_c[row_off + uint(iw)]
                : half(0.0h);
        }
    }
}

// Fused cat([x, past], C) -> im2col [HW, 2*C_in*9]
kernel void im2col_3x3_cat2_f16(
    device const half* x    [[buffer(0)]],
    device const half* past [[buffer(1)]],
    device half*       col  [[buffer(2)]],
    constant Conv3x3Params& p [[buffer(3)]],  // C_in = channels of x (total = 2*C_in)
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint ic = gid.z;
    if (ow >= p.W_out || oh >= p.H_out || ic >= p.C_in) return;

    uint hw_idx = oh * p.W_out + ow;
    uint K = (p.C_in * 2u) * 9u;
    device half* dst_x = col + hw_idx * K + ic * 9u;
    device half* dst_p = col + hw_idx * K + (p.C_in + ic) * 9u;

    int ih0 = int(oh) - 1;
    int iw0 = int(ow) - 1;
    uint hw = p.H_in * p.W_in;
    device const half* x_c = x + ic * hw;
    device const half* p_c = past + ic * hw;
    bool zero_past = (p.flags & 8u) != 0u;

    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        int ih = ih0 + kh;
        bool valid_h = (ih >= 0 && ih < int(p.H_in));
        uint row_off = valid_h ? (uint(ih) * p.W_in) : 0u;
        #pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            int iw = iw0 + kw;
            bool valid = valid_h && iw >= 0 && iw < int(p.W_in);
            dst_x[kh * 3 + kw] = valid ? x_c[row_off + uint(iw)] : half(0.0h);
            dst_p[kh * 3 + kw] = (valid && !zero_past) ? p_c[row_off + uint(iw)] : half(0.0h);
        }
    }
}

struct ConvGemmParams {
    uint M;      // C_out
    uint N;      // HW_out
    uint K;      // C_in * 9 (or C_in for 1x1)
    uint flags;  // bit 0: has_bias, bit 1: relu, bit 2: add_residual
};

// Tiled 32x64x32 simdgroup_half8x8 GEMM for Conv2d (6 KB union shared mem -> max M1 occupancy!)
kernel void conv_gemm_f16(
    device const half* W     [[buffer(0)]],  // [M, K]
    device const half* Col   [[buffer(1)]],  // [N, K]
    device const half* bias  [[buffer(2)]],  // [M]
    device const half* resid [[buffer(3)]],  // [M, N]
    device half*       Out   [[buffer(4)]],  // [M, N]
    constant ConvGemmParams& p [[buffer(5)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_index_in_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup union {
        struct { half W[32][32]; half C[64][32]; } wc;
        half O[32][64];
    } smem;

    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 64u;

    simdgroup_half8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0; i < 2; ++i)
        #pragma unroll
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));

    uint sg_m = (sg_id >> 1) * 16u;
    uint sg_n = (sg_id & 1u) * 32u;

    uint w_row = tid >> 2;
    uint w_col = (tid & 3u) * 8u;
    uint c_row0 = (tid >> 2) * 2u;
    uint c_col  = (tid & 3u) * 8u;

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        uint gm = m0 + w_row;
        if (gm < p.M) {
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                uint gk = k0 + w_col + j;
                smem.wc.W[w_row][w_col + j] = (gk < p.K) ? W[gm * p.K + gk] : half(0.0h);
            }
        } else {
            #pragma unroll
            for (uint j = 0; j < 8; ++j) smem.wc.W[w_row][w_col + j] = half(0.0h);
        }

        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            uint lr = c_row0 + r;
            uint gn = n0 + lr;
            if (gn < p.N) {
                #pragma unroll
                for (uint j = 0; j < 8; ++j) {
                    uint gk = k0 + c_col + j;
                    smem.wc.C[lr][c_col + j] = (gk < p.K) ? Col[gn * p.K + gk] : half(0.0h);
                }
            } else {
                #pragma unroll
                for (uint j = 0; j < 8; ++j) smem.wc.C[lr][c_col + j] = half(0.0h);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a0, a1;
            simdgroup_load(a0, &smem.wc.W[sg_m + 0][kk], 32, ulong2(0, 0), false);
            simdgroup_load(a1, &smem.wc.W[sg_m + 8][kk], 32, ulong2(0, 0), false);
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                simdgroup_half8x8 bj;
                simdgroup_load(bj, &smem.wc.C[sg_n + j * 8][kk], 32, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[0][j], a0, bj, acc[0][j]);
                simdgroup_multiply_accumulate(acc[1][j], a1, bj, acc[1][j]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint i = 0; i < 2; ++i)
        #pragma unroll
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], &smem.O[sg_m + i * 8][sg_n + j * 8], 64);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool has_bias = (p.flags & 1u) != 0u;
    bool do_relu  = (p.flags & 2u) != 0u;
    bool do_resid = (p.flags & 4u) != 0u;

    for (uint i = tid; i < 2048u; i += 128u) {
        uint lm = i >> 6;
        uint ln = i & 63u;
        uint gm = m0 + lm;
        uint gn = n0 + ln;
        if (gm < p.M && gn < p.N) {
            float v = float(smem.O[lm][ln]);
            if (has_bias) v += float(bias[gm]);
            uint idx = gm * p.N + gn;
            if (do_resid) v += float(resid[idx]);
            if (do_relu)  v = max(v, 0.0f);
            Out[idx] = half(v);
        }
    }
}

struct Conv1x1Params {
    uint C_in;
    uint C_out;
    uint HW;
};

kernel void conv2d_1x1_f16(
    device const half* inp    [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half*       out    [[buffer(2)]],
    constant Conv1x1Params& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint hw_idx = gid.x;
    uint oc     = gid.y;
    if (hw_idx >= p.HW || oc >= p.C_out) return;

    device const half* w_oc = weight + oc * p.C_in;
    float acc = 0.0f;
    for (uint ic = 0; ic < p.C_in; ++ic) {
        acc = fma(float(inp[ic * p.HW + hw_idx]), float(w_oc[ic]), acc);
    }
    out[oc * p.HW + hw_idx] = half(acc);
}

kernel void conv2d_1x1_cat2_f16(
    device const half* f0     [[buffer(0)]],
    device const half* f1     [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half*       out    [[buffer(3)]],
    constant Conv1x1Params& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint hw_idx = gid.x;
    uint oc     = gid.y;
    if (hw_idx >= p.HW || oc >= p.C_out) return;

    device const half* w_oc = weight + oc * (2u * p.C_in);
    float acc = 0.0f;
    for (uint ic = 0; ic < p.C_in; ++ic) {
        acc = fma(float(f0[ic * p.HW + hw_idx]), float(w_oc[ic]), acc);
    }
    for (uint ic = 0; ic < p.C_in; ++ic) {
        acc = fma(float(f1[ic * p.HW + hw_idx]), float(w_oc[p.C_in + ic]), acc);
    }
    out[oc * p.HW + hw_idx] = half(acc);
}

kernel void clamp_tanh3_f16(
    device const half* inp [[buffer(0)]],
    device half*       out [[buffer(1)]],
    constant uint& n       [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;
    float x = float(inp[id]);
    out[id] = half(tanh(x / 3.0f) * 3.0f);
}

struct UpsampleParams {
    uint C;
    uint H_in;
    uint W_in;
};

kernel void upsample2x_nearest_f16(
    device const half* inp    [[buffer(0)]],
    device half*       out    [[buffer(1)]],
    constant UpsampleParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint c  = gid.z;
    uint H_out = p.H_in * 2u;
    uint W_out = p.W_in * 2u;
    if (ow >= W_out || oh >= H_out || c >= p.C) return;

    uint ih = oh >> 1;
    uint iw = ow >> 1;
    out[c * (H_out * W_out) + oh * W_out + ow] = inp[c * (p.H_in * p.W_in) + ih * p.W_in + iw];
}

struct ImageParams {
    uint H;
    uint W;
};

kernel void rgb_u8_to_unshuffle_f16(
    device const uchar* rgb [[buffer(0)]],
    device half*        out [[buffer(1)]],
    constant ImageParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint H2 = p.H >> 1;
    uint W2 = p.W >> 1;
    if (ow >= W2 || oh >= H2) return;

    uint hw2 = H2 * W2;
    uint dst_px = oh * W2 + ow;
    for (uint sy = 0; sy < 2; ++sy) {
        for (uint sx = 0; sx < 2; ++sx) {
            uint py = oh * 2u + sy;
            uint px = ow * 2u + sx;
            uint src_idx = (py * p.W + px) * 3u;
            uint sub = sy * 2u + sx;
            out[(0u * 4u + sub) * hw2 + dst_px] = half(float(rgb[src_idx + 0u]) / 255.0f);
            out[(1u * 4u + sub) * hw2 + dst_px] = half(float(rgb[src_idx + 1u]) / 255.0f);
            out[(2u * 4u + sub) * hw2 + dst_px] = half(float(rgb[src_idx + 2u]) / 255.0f);
        }
    }
}

kernel void shuffle_f16_to_rgb_u8(
    device const half* inp  [[buffer(0)]],
    device uchar*      rgb  [[buffer(1)]],
    constant ImageParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint px = gid.x;
    uint py = gid.y;
    if (px >= p.W || py >= p.H) return;

    uint H2 = p.H >> 1;
    uint W2 = p.W >> 1;
    uint hw2 = H2 * W2;
    uint oh = py >> 1;
    uint ow = px >> 1;
    uint sub = (py & 1u) * 2u + (px & 1u);
    uint src_px = oh * W2 + ow;

    float r = clamp(float(inp[(0u * 4u + sub) * hw2 + src_px]), 0.0f, 1.0f);
    float g = clamp(float(inp[(1u * 4u + sub) * hw2 + src_px]), 0.0f, 1.0f);
    float b = clamp(float(inp[(2u * 4u + sub) * hw2 + src_px]), 0.0f, 1.0f);

    uint dst = (py * p.W + px) * 3u;
    rgb[dst + 0u] = uchar(rint(r * 255.0f));
    rgb[dst + 1u] = uchar(rint(g * 255.0f));
    rgb[dst + 2u] = uchar(rint(b * 255.0f));
}

kernel void shuffle_f16_to_bgra8(
    device const half* inp  [[buffer(0)]],
    device uchar*      bgra [[buffer(1)]],
    constant ImageParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint px = gid.x;
    uint py = gid.y;
    if (px >= p.W || py >= p.H) return;

    uint H2 = p.H >> 1;
    uint W2 = p.W >> 1;
    uint hw2 = H2 * W2;
    uint oh = py >> 1;
    uint ow = px >> 1;
    uint sub = (py & 1u) * 2u + (px & 1u);
    uint src_px = oh * W2 + ow;

    float r = clamp(float(inp[(0u * 4u + sub) * hw2 + src_px]), 0.0f, 1.0f);
    float g = clamp(float(inp[(1u * 4u + sub) * hw2 + src_px]), 0.0f, 1.0f);
    float b = clamp(float(inp[(2u * 4u + sub) * hw2 + src_px]), 0.0f, 1.0f);

    uint dst = (py * p.W + px) * 4u;
    bgra[dst + 0u] = uchar(rint(b * 255.0f));
    bgra[dst + 1u] = uchar(rint(g * 255.0f));
    bgra[dst + 2u] = uchar(rint(r * 255.0f));
    bgra[dst + 3u] = 255u;
}

kernel void copy_f16(
    device const half* src [[buffer(0)]],
    device half*       dst [[buffer(1)]],
    constant uint& n       [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < n) dst[id] = src[id];
}
