#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace spatial {

enum class DType : uint32_t {
    F32  = 0,
    F16  = 1,
    BF16 = 2,
    Q8_0 = 3,  // block size 32: uint16_t f16 scale + int8_t qs[32] = 34 bytes
    Q4_0 = 4,  // block size 32: uint16_t f16 scale + uint8_t qs[16] = 18 bytes
    I32  = 5,
    U8   = 6,
};

constexpr size_t QK_BLOCK = 32;

struct alignas(2) BlockQ8_0 {
    uint16_t scale_f16;
    int8_t qs[QK_BLOCK];
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

struct alignas(2) BlockQ4_0 {
    uint16_t scale_f16;
    uint8_t qs[QK_BLOCK / 2];  // low nibble = [0..15], high nibble = [16..31], stored as (q + 8)
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

inline constexpr const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32:  return "F32";
        case DType::F16:  return "F16";
        case DType::BF16: return "BF16";
        case DType::Q8_0: return "Q8_0";
        case DType::Q4_0: return "Q4_0";
        case DType::I32:  return "I32";
        case DType::U8:   return "U8";
    }
    return "UNKNOWN";
}

inline DType dtype_from_name(std::string_view s) {
    if (s == "F32")  return DType::F32;
    if (s == "F16")  return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "Q8_0") return DType::Q8_0;
    if (s == "Q4_0") return DType::Q4_0;
    if (s == "I32")  return DType::I32;
    if (s == "U8")   return DType::U8;
    return DType::F32;
}

inline constexpr size_t dtype_nbytes(DType dt, size_t numel) {
    switch (dt) {
        case DType::F32:  return numel * 4;
        case DType::F16:  return numel * 2;
        case DType::BF16: return numel * 2;
        case DType::I32:  return numel * 4;
        case DType::U8:   return numel;
        case DType::Q8_0: return ((numel + QK_BLOCK - 1) / QK_BLOCK) * sizeof(BlockQ8_0);
        case DType::Q4_0: return ((numel + QK_BLOCK - 1) / QK_BLOCK) * sizeof(BlockQ4_0);
    }
    return 0;
}

// Fast IEEE-754 F16 <-> F32 conversion (ARM64 native __fp16 when available)
inline float f16_to_f32(uint16_t h) {
#if defined(__aarch64__)
    __fp16 v;
    std::memcpy(&v, &h, 2);
    return static_cast<float>(v);
#else
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            out = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
#endif
}

inline uint16_t f32_to_f16(float f) {
#if defined(__aarch64__)
    __fp16 v = static_cast<__fp16>(f);
    uint16_t h;
    std::memcpy(&h, &v, 2);
    return h;
#else
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = ((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
#endif
}

inline float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace spatial
