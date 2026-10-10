/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// CUDA's half precision: __half, __half2 and their intrinsics, as in CUDA's
// cuda_fp16.h. A __half holds the 16 bits of an IEEE binary16 number, so
// arrays of halves have CUDA's layout and copy to and from the host as they
// are.
//
// Operations convert to double, compute, and round to half once. That is
// exact before rounding for additions, subtractions and multiplications, and
// for divisions and square roots double's 53 bits make the double rounding
// harmless. So is __hfma's (see detail::fma). Conversions
// round as their suffix says: _rn (to nearest even), _rz (toward zero), _rd
// (down) and _ru (up). Transcendental functions (hexp, hsin, ...) round the
// double result, so they are at least as accurate as CUDA's.
//
// As in CUDA, __CUDA_NO_HALF_CONVERSIONS__, __CUDA_NO_HALF_OPERATORS__ and
// __CUDA_NO_HALF2_OPERATORS__ remove the implicit conversions and the
// operators.
//

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "cuda_runtime.h"

// The include guards of CUDA's own headers, which code tests
#ifndef __CUDA_FP16_H__
#define __CUDA_FP16_H__
#endif
#ifndef __CUDA_FP16_HPP__
#define __CUDA_FP16_HPP__
#endif
#ifndef __CUDA_FP16_TYPES_EXIST__
#define __CUDA_FP16_TYPES_EXIST__
#endif

namespace cuda4cpu {

namespace detail {

//! Rounding of a conversion or an operation to half
enum class half_rounding { nearest, toward_zero, down, up };

//! The bits of x rounded to half as rounding says
constexpr uint16_t to_half_bits(double x, half_rounding rounding = half_rounding::nearest)
{
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const uint16_t sign = uint16_t((bits >> 48) & 0x8000);
    const int biased = int((bits >> 52) & 0x7ff);
    const uint64_t fraction = bits & ((uint64_t(1) << 52) - 1);

    if (biased == 0x7ff)
        return fraction != 0 ? uint16_t(0x7fff) : uint16_t(sign | 0x7c00);     // NaN, infinity
    if (biased == 0 && fraction == 0)
        return sign;                                                          // zero, which is exact

    // Whether to round the magnitude up (away from zero) or down
    const bool negative = sign != 0;
    bool magnitude_up = false;
    if (rounding == half_rounding::up)
        magnitude_up = !negative;
    else if (rounding == half_rounding::down)
        magnitude_up = negative;

    // x is significand * 2^(exponent - 52)
    const uint64_t significand = biased != 0 ? fraction | (uint64_t(1) << 52) : fraction;
    const int exponent = biased != 0 ? biased - 1023 : -1022;
    if (exponent >= 16) {
        // Beyond the largest half (65504): infinity, unless the magnitude
        // rounds down
        const bool down = rounding != half_rounding::nearest && !magnitude_up;
        return uint16_t(sign | (down ? 0x7bff : 0x7c00));
    }

    // The half below |x| (base, as bits), and the remainder of |x| above it,
    // where half_unit is half a unit in the last place of the half
    uint32_t base;
    uint64_t remainder, half_unit;
    if (exponent >= -14) {                                  // normal: 11 significant bits
        base = uint32_t((exponent + 15) << 10) + uint32_t(significand >> 42) - 1024;
        remainder = significand & ((uint64_t(1) << 42) - 1);
        half_unit = uint64_t(1) << 41;
    } else {                                                // subnormal: units of 2^-24
        const int shift = std::min(28 - exponent, 63);
        base = uint32_t(significand >> shift);
        remainder = significand & ((uint64_t(1) << shift) - 1);
        half_unit = uint64_t(1) << (shift - 1);
    }

    uint32_t result = base;
    if (rounding == half_rounding::nearest) {
        if (remainder > half_unit || (remainder == half_unit && (base & 1)))
            result = base + 1;
    } else if (magnitude_up && remainder != 0) {
        result = base + 1;
    }
    // Rounding up the largest half gives the bits of infinity
    return uint16_t(sign | result);
}

//! The value of the half whose bits are h, which double holds exactly
constexpr double from_half_bits(uint16_t h)
{
    const uint64_t sign = uint64_t(h & 0x8000) << 48;
    const unsigned exponent = (h >> 10) & 0x1f;
    const uint64_t mantissa = h & 0x3ff;
    if (exponent == 0x1f)
        return std::bit_cast<double>(sign | 0x7ff0000000000000 | (mantissa << 42));
    if (exponent == 0) {
        const double magnitude = double(mantissa) * 0x1p-24;
        return sign != 0 ? -magnitude : magnitude;
    }
    return std::bit_cast<double>(sign | (uint64_t(exponent - 15 + 1023) << 52) | (mantissa << 42));
}

}

inline namespace cuda_api {

struct __half_raw {
    unsigned short x;
};

struct __half2_raw {
    unsigned short x;
    unsigned short y;
};

//! A binary16 number. Its conversions round to nearest, except those to
//! integers, which round toward zero, as in CUDA.
struct alignas(2) __half {
protected:
    unsigned short __x;

public:
    __half() = default;
    constexpr __half(const __half_raw &raw) : __x(raw.x) {}
    constexpr __half &operator=(const __half_raw &raw) { __x = raw.x; return *this; }
    constexpr operator __half_raw() const { return {__x}; }

#ifndef __CUDA_NO_HALF_CONVERSIONS__
    constexpr __half(float f) : __x(detail::to_half_bits(f)) {}
    constexpr __half(double d) : __x(detail::to_half_bits(d)) {}
    constexpr __half(short v) : __x(detail::to_half_bits(v)) {}
    constexpr __half(unsigned short v) : __x(detail::to_half_bits(v)) {}
    constexpr __half(int v) : __x(detail::to_half_bits(v)) {}
    constexpr __half(unsigned int v) : __x(detail::to_half_bits(v)) {}
    constexpr __half(long v) : __x(detail::to_half_bits(double(v))) {}
    constexpr __half(unsigned long v) : __x(detail::to_half_bits(double(v))) {}
    constexpr __half(long long v) : __x(detail::to_half_bits(double(v))) {}
    constexpr __half(unsigned long long v) : __x(detail::to_half_bits(double(v))) {}

    constexpr __half &operator=(float f) { __x = detail::to_half_bits(f); return *this; }
    constexpr __half &operator=(double d) { __x = detail::to_half_bits(d); return *this; }
    constexpr __half &operator=(short v) { __x = detail::to_half_bits(v); return *this; }
    constexpr __half &operator=(unsigned short v) { __x = detail::to_half_bits(v); return *this; }
    constexpr __half &operator=(int v) { __x = detail::to_half_bits(v); return *this; }
    constexpr __half &operator=(unsigned int v) { __x = detail::to_half_bits(v); return *this; }
    constexpr __half &operator=(long v) { __x = detail::to_half_bits(double(v)); return *this; }
    constexpr __half &operator=(unsigned long v) { __x = detail::to_half_bits(double(v)); return *this; }
    constexpr __half &operator=(long long v) { __x = detail::to_half_bits(double(v)); return *this; }
    constexpr __half &operator=(unsigned long long v) { __x = detail::to_half_bits(double(v)); return *this; }

    constexpr operator float() const { return float(detail::from_half_bits(__x)); }
    operator signed char() const { return to_integer<signed char>(); }
    operator unsigned char() const { return to_integer<unsigned char>(); }
    operator char() const { return to_integer<char>(); }
    operator short() const { return to_integer<short>(); }
    operator unsigned short() const { return to_integer<unsigned short>(); }
    operator int() const { return to_integer<int>(); }
    operator unsigned int() const { return to_integer<unsigned int>(); }
    operator long() const { return to_integer<long>(); }
    operator unsigned long() const { return to_integer<unsigned long>(); }
    operator long long() const { return to_integer<long long>(); }
    operator unsigned long long() const { return to_integer<unsigned long long>(); }
    constexpr operator bool() const { return (__x & 0x7fff) != 0; }

private:
    template <typename I>
    I to_integer() const
    {
        return detail::to_integer<I>(detail::from_half_bits(__x), [](double v) { return std::trunc(v); });
    }
#endif
};

struct alignas(4) __half2 {
    __half x;
    __half y;

    __half2() = default;
    constexpr __half2(const __half &a, const __half &b) : x(a), y(b) {}
    constexpr __half2(const __half2_raw &raw) : x(__half_raw{raw.x}), y(__half_raw{raw.y}) {}
    constexpr __half2 &operator=(const __half2_raw &raw)
    {
        x = __half_raw{raw.x};
        y = __half_raw{raw.y};
        return *this;
    }
    constexpr operator __half2_raw() const { return {__half_raw(x).x, __half_raw(y).x}; }
};

using half = __half;
using half2 = __half2;
using __nv_half = __half;
using __nv_half2 = __half2;
using __nv_half_raw = __half_raw;
using __nv_half2_raw = __half2_raw;

}

namespace detail {

constexpr double to_double(__half h)
{
    return from_half_bits(__half_raw(h).x);
}

constexpr __half to_half(double x, half_rounding rounding = half_rounding::nearest)
{
    return __half_raw{to_half_bits(x, rounding)};
}

constexpr bool is_nan(__half h)
{
    return (__half_raw(h).x & 0x7fff) > 0x7c00;
}

//! a * b + c, rounded once. The product is exact in double. The sum keeps
//! every bit of c (halves are multiples of 2^-24, and finite sums are below
//! 2^17), so it can only lose low bits of a product far smaller than c, whose
//! sum with c then lies strictly between c and the next tie: rounding the
//! double sum to nearest gives the correctly rounded half.
constexpr __half fma(__half a, __half b, __half c)
{
    return to_half(to_double(a) * to_double(b) + to_double(c));
}

template <typename F>
constexpr __half2 lanes(__half2 a, F f)
{
    return {f(a.x), f(a.y)};
}

template <typename F>
constexpr __half2 lanes(__half2 a, __half2 b, F f)
{
    return {f(a.x, b.x), f(a.y, b.y)};
}

constexpr __half half_bool(bool b)
{
    return __half_raw{uint16_t(b ? 0x3c00 : 0)};
}

constexpr __half saturate(__half h)
{
    const double v = to_double(h);
    if (is_nan(h) || v <= 0)
        return __half_raw{0};
    return v >= 1 ? __half(__half_raw{0x3c00}) : h;
}

constexpr __half relu(__half h)
{
    if (is_nan(h))
        return __half_raw{0x7fff};
    return to_double(h) < 0 ? __half(__half_raw{0}) : h;
}

}

inline namespace cuda_api {

//
// Constants, as in CUDA
//

#define CUDART_INF_FP16        __ushort_as_half((unsigned short)0x7C00U)
#define CUDART_NAN_FP16        __ushort_as_half((unsigned short)0x7FFFU)
#define CUDART_MIN_DENORM_FP16 __ushort_as_half((unsigned short)0x0001U)
#define CUDART_MAX_NORMAL_FP16 __ushort_as_half((unsigned short)0x7BFFU)
#define CUDART_NEG_ZERO_FP16   __ushort_as_half((unsigned short)0x8000U)
#define CUDART_ZERO_FP16       __ushort_as_half((unsigned short)0x0000U)
#define CUDART_ONE_FP16        __ushort_as_half((unsigned short)0x3C00U)

//
// Conversions between floating-point types, and bits
//

constexpr __half __float2half(float a) { return detail::to_half(a); }
constexpr __half __float2half_rn(float a) { return detail::to_half(a); }
constexpr __half __float2half_rz(float a) { return detail::to_half(a, detail::half_rounding::toward_zero); }
constexpr __half __float2half_rd(float a) { return detail::to_half(a, detail::half_rounding::down); }
constexpr __half __float2half_ru(float a) { return detail::to_half(a, detail::half_rounding::up); }
constexpr __half __double2half(double a) { return detail::to_half(a); }
constexpr float __half2float(__half a) { return float(detail::to_double(a)); }

constexpr __half2 __float2half2_rn(float a) { return {__float2half(a), __float2half(a)}; }
constexpr __half2 __floats2half2_rn(float a, float b) { return {__float2half(a), __float2half(b)}; }
constexpr __half2 __float22half2_rn(float2 a) { return {__float2half(a.x), __float2half(a.y)}; }
constexpr float2 __half22float2(__half2 a) { return {__half2float(a.x), __half2float(a.y)}; }
constexpr float __low2float(__half2 a) { return __half2float(a.x); }
constexpr float __high2float(__half2 a) { return __half2float(a.y); }

constexpr __half __low2half(__half2 a) { return a.x; }
constexpr __half __high2half(__half2 a) { return a.y; }
constexpr __half2 __low2half2(__half2 a) { return {a.x, a.x}; }
constexpr __half2 __high2half2(__half2 a) { return {a.y, a.y}; }
constexpr __half2 __lows2half2(__half2 a, __half2 b) { return {a.x, b.x}; }
constexpr __half2 __highs2half2(__half2 a, __half2 b) { return {a.y, b.y}; }
constexpr __half2 __lowhigh2highlow(__half2 a) { return {a.y, a.x}; }
constexpr __half2 __halves2half2(__half a, __half b) { return {a, b}; }
constexpr __half2 __half2half2(__half a) { return {a, a}; }
constexpr __half2 make_half2(__half x, __half y) { return {x, y}; }

constexpr short __half_as_short(__half h) { return std::bit_cast<short>(__half_raw(h).x); }
constexpr unsigned short __half_as_ushort(__half h) { return __half_raw(h).x; }
constexpr __half __short_as_half(short i) { return __half_raw{std::bit_cast<unsigned short>(i)}; }
constexpr __half __ushort_as_half(unsigned short i) { return __half_raw{i}; }

//
// Conversions to and from integers. Halves to integers saturate, and NaN
// becomes 0, as in CUDA.
//

#define CUDA4CPU_HALF_INTEGER(suffix, I)                                                                          \
    inline I __half2##suffix##_rn(__half h) { return detail::to_integer<I>(detail::to_double(h), [](double v) { return std::nearbyint(v); }); } \
    inline I __half2##suffix##_rz(__half h) { return detail::to_integer<I>(detail::to_double(h), [](double v) { return std::trunc(v); }); }     \
    inline I __half2##suffix##_rd(__half h) { return detail::to_integer<I>(detail::to_double(h), [](double v) { return std::floor(v); }); }    \
    inline I __half2##suffix##_ru(__half h) { return detail::to_integer<I>(detail::to_double(h), [](double v) { return std::ceil(v); }); }     \
    constexpr __half __##suffix##2half_rn(I i) { return detail::to_half(double(i)); }                                  \
    constexpr __half __##suffix##2half_rz(I i) { return detail::to_half(double(i), detail::half_rounding::toward_zero); } \
    constexpr __half __##suffix##2half_rd(I i) { return detail::to_half(double(i), detail::half_rounding::down); }       \
    constexpr __half __##suffix##2half_ru(I i) { return detail::to_half(double(i), detail::half_rounding::up); }

CUDA4CPU_HALF_INTEGER(int, int)
CUDA4CPU_HALF_INTEGER(uint, unsigned int)
CUDA4CPU_HALF_INTEGER(short, short)
CUDA4CPU_HALF_INTEGER(ushort, unsigned short)
CUDA4CPU_HALF_INTEGER(ll, long long)
CUDA4CPU_HALF_INTEGER(ull, unsigned long long)

#undef CUDA4CPU_HALF_INTEGER

inline signed char __half2char_rz(__half h)
{
    return detail::to_integer<signed char>(detail::to_double(h), [](double v) { return std::trunc(v); });
}

inline unsigned char __half2uchar_rz(__half h)
{
    return detail::to_integer<unsigned char>(detail::to_double(h), [](double v) { return std::trunc(v); });
}

//
// Arithmetic, rounded to nearest. The _rn forms, which on a GPU keep the
// compiler from fusing a multiplication and an addition, are the same.
//

constexpr __half __hadd(__half a, __half b) { return detail::to_half(detail::to_double(a) + detail::to_double(b)); }
constexpr __half __hsub(__half a, __half b) { return detail::to_half(detail::to_double(a) - detail::to_double(b)); }
constexpr __half __hmul(__half a, __half b) { return detail::to_half(detail::to_double(a) * detail::to_double(b)); }
constexpr __half __hdiv(__half a, __half b) { return detail::to_half(detail::to_double(a) / detail::to_double(b)); }
constexpr __half __hadd_rn(__half a, __half b) { return __hadd(a, b); }
constexpr __half __hsub_rn(__half a, __half b) { return __hsub(a, b); }
constexpr __half __hmul_rn(__half a, __half b) { return __hmul(a, b); }
constexpr __half __hfma(__half a, __half b, __half c) { return detail::fma(a, b, c); }
constexpr __half __hneg(__half a) { return __half_raw{uint16_t(__half_raw(a).x ^ 0x8000)}; }
constexpr __half __habs(__half a) { return __half_raw{uint16_t(__half_raw(a).x & 0x7fff)}; }

//! Results clamped to [0, 1], with NaN as 0
constexpr __half __hadd_sat(__half a, __half b) { return detail::saturate(__hadd(a, b)); }
constexpr __half __hsub_sat(__half a, __half b) { return detail::saturate(__hsub(a, b)); }
constexpr __half __hmul_sat(__half a, __half b) { return detail::saturate(__hmul(a, b)); }
inline __half __hfma_sat(__half a, __half b, __half c) { return detail::saturate(__hfma(a, b, c)); }

//! A fused multiply-add whose negative results become 0
inline __half __hfma_relu(__half a, __half b, __half c) { return detail::relu(__hfma(a, b, c)); }

constexpr __half2 __hadd2(__half2 a, __half2 b) { return detail::lanes(a, b, __hadd); }
constexpr __half2 __hsub2(__half2 a, __half2 b) { return detail::lanes(a, b, __hsub); }
constexpr __half2 __hmul2(__half2 a, __half2 b) { return detail::lanes(a, b, __hmul); }
constexpr __half2 __h2div(__half2 a, __half2 b) { return detail::lanes(a, b, __hdiv); }
constexpr __half2 __hadd2_rn(__half2 a, __half2 b) { return __hadd2(a, b); }
constexpr __half2 __hsub2_rn(__half2 a, __half2 b) { return __hsub2(a, b); }
constexpr __half2 __hmul2_rn(__half2 a, __half2 b) { return __hmul2(a, b); }
inline __half2 __hfma2(__half2 a, __half2 b, __half2 c) { return {__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)}; }
constexpr __half2 __hneg2(__half2 a) { return detail::lanes(a, __hneg); }
constexpr __half2 __habs2(__half2 a) { return detail::lanes(a, __habs); }
constexpr __half2 __hadd2_sat(__half2 a, __half2 b) { return detail::lanes(a, b, __hadd_sat); }
constexpr __half2 __hsub2_sat(__half2 a, __half2 b) { return detail::lanes(a, b, __hsub_sat); }
constexpr __half2 __hmul2_sat(__half2 a, __half2 b) { return detail::lanes(a, b, __hmul_sat); }
inline __half2 __hfma2_sat(__half2 a, __half2 b, __half2 c) { return {__hfma_sat(a.x, b.x, c.x), __hfma_sat(a.y, b.y, c.y)}; }
inline __half2 __hfma2_relu(__half2 a, __half2 b, __half2 c) { return {__hfma_relu(a.x, b.x, c.x), __hfma_relu(a.y, b.y, c.y)}; }

//
// Comparisons. The plain ones are false when an operand is NaN, the ones
// ending in u (unordered) true.
//

constexpr bool __heq(__half a, __half b) { return detail::to_double(a) == detail::to_double(b); }
constexpr bool __hne(__half a, __half b) { return !detail::is_nan(a) && !detail::is_nan(b) && !__heq(a, b); }
constexpr bool __hlt(__half a, __half b) { return detail::to_double(a) < detail::to_double(b); }
constexpr bool __hle(__half a, __half b) { return detail::to_double(a) <= detail::to_double(b); }
constexpr bool __hgt(__half a, __half b) { return detail::to_double(a) > detail::to_double(b); }
constexpr bool __hge(__half a, __half b) { return detail::to_double(a) >= detail::to_double(b); }
constexpr bool __hequ(__half a, __half b) { return detail::is_nan(a) || detail::is_nan(b) || __heq(a, b); }
constexpr bool __hneu(__half a, __half b) { return !__heq(a, b); }
constexpr bool __hltu(__half a, __half b) { return !__hge(a, b); }
constexpr bool __hleu(__half a, __half b) { return !__hgt(a, b); }
constexpr bool __hgtu(__half a, __half b) { return !__hle(a, b); }
constexpr bool __hgeu(__half a, __half b) { return !__hlt(a, b); }

constexpr bool __hisnan(__half a) { return detail::is_nan(a); }

//! -1 for negative infinity, 1 for positive infinity, 0 otherwise
constexpr int __hisinf(__half a)
{
    const unsigned short bits = __half_raw(a).x;
    return bits == 0x7c00 ? 1 : bits == 0xfc00 ? -1 : 0;
}

//! The larger or smaller operand: NaN only if both are NaN
constexpr __half __hmax(__half a, __half b)
{
    if (detail::is_nan(a))
        return detail::is_nan(b) ? __half(__half_raw{0x7fff}) : b;
    return detail::is_nan(b) || __hge(a, b) ? a : b;
}

constexpr __half __hmin(__half a, __half b)
{
    if (detail::is_nan(a))
        return detail::is_nan(b) ? __half(__half_raw{0x7fff}) : b;
    return detail::is_nan(b) || __hle(a, b) ? a : b;
}

//! The larger or smaller operand, or NaN if either is NaN
constexpr __half __hmax_nan(__half a, __half b)
{
    return detail::is_nan(a) || detail::is_nan(b) ? __half(__half_raw{0x7fff}) : __hmax(a, b);
}

constexpr __half __hmin_nan(__half a, __half b)
{
    return detail::is_nan(a) || detail::is_nan(b) ? __half(__half_raw{0x7fff}) : __hmin(a, b);
}

// Per lane, as 1.0 or 0.0
#define CUDA4CPU_HALF2_COMPARE(name)                                                                     \
    constexpr __half2 name##2(__half2 a, __half2 b)                                                      \
    {                                                                                                    \
        return {detail::half_bool(name(a.x, b.x)), detail::half_bool(name(a.y, b.y))};                   \
    }                                                                                                    \
    constexpr unsigned name##2_mask(__half2 a, __half2 b)                                                \
    {                                                                                                    \
        return (name(a.x, b.x) ? 0xffffu : 0u) | (name(a.y, b.y) ? 0xffff0000u : 0u);                    \
    }

CUDA4CPU_HALF2_COMPARE(__heq)
CUDA4CPU_HALF2_COMPARE(__hne)
CUDA4CPU_HALF2_COMPARE(__hlt)
CUDA4CPU_HALF2_COMPARE(__hle)
CUDA4CPU_HALF2_COMPARE(__hgt)
CUDA4CPU_HALF2_COMPARE(__hge)
CUDA4CPU_HALF2_COMPARE(__hequ)
CUDA4CPU_HALF2_COMPARE(__hneu)
CUDA4CPU_HALF2_COMPARE(__hltu)
CUDA4CPU_HALF2_COMPARE(__hleu)
CUDA4CPU_HALF2_COMPARE(__hgtu)
CUDA4CPU_HALF2_COMPARE(__hgeu)

#undef CUDA4CPU_HALF2_COMPARE

// Whether both lanes compare so
constexpr bool __hbeq2(__half2 a, __half2 b) { return __heq(a.x, b.x) && __heq(a.y, b.y); }
constexpr bool __hbne2(__half2 a, __half2 b) { return __hne(a.x, b.x) && __hne(a.y, b.y); }
constexpr bool __hblt2(__half2 a, __half2 b) { return __hlt(a.x, b.x) && __hlt(a.y, b.y); }
constexpr bool __hble2(__half2 a, __half2 b) { return __hle(a.x, b.x) && __hle(a.y, b.y); }
constexpr bool __hbgt2(__half2 a, __half2 b) { return __hgt(a.x, b.x) && __hgt(a.y, b.y); }
constexpr bool __hbge2(__half2 a, __half2 b) { return __hge(a.x, b.x) && __hge(a.y, b.y); }
constexpr bool __hbequ2(__half2 a, __half2 b) { return __hequ(a.x, b.x) && __hequ(a.y, b.y); }
constexpr bool __hbneu2(__half2 a, __half2 b) { return __hneu(a.x, b.x) && __hneu(a.y, b.y); }
constexpr bool __hbltu2(__half2 a, __half2 b) { return __hltu(a.x, b.x) && __hltu(a.y, b.y); }
constexpr bool __hbleu2(__half2 a, __half2 b) { return __hleu(a.x, b.x) && __hleu(a.y, b.y); }
constexpr bool __hbgtu2(__half2 a, __half2 b) { return __hgtu(a.x, b.x) && __hgtu(a.y, b.y); }
constexpr bool __hbgeu2(__half2 a, __half2 b) { return __hgeu(a.x, b.x) && __hgeu(a.y, b.y); }

constexpr __half2 __hisnan2(__half2 a)
{
    return {detail::half_bool(__hisnan(a.x)), detail::half_bool(__hisnan(a.y))};
}

constexpr __half2 __hmax2(__half2 a, __half2 b) { return detail::lanes(a, b, __hmax); }
constexpr __half2 __hmin2(__half2 a, __half2 b) { return detail::lanes(a, b, __hmin); }
constexpr __half2 __hmax2_nan(__half2 a, __half2 b) { return detail::lanes(a, b, __hmax_nan); }
constexpr __half2 __hmin2_nan(__half2 a, __half2 b) { return detail::lanes(a, b, __hmin_nan); }

//
// Math functions, computed in double and rounded to half
//

#define CUDA4CPU_HALF_MATH(name, h2name, expression)                                                   \
    inline __half name(__half a)                                                                       \
    {                                                                                                  \
        const double x = detail::to_double(a);                                                         \
        return detail::to_half(expression);                                                            \
    }                                                                                                  \
    inline __half2 h2name(__half2 a) { return {name(a.x), name(a.y)}; }

CUDA4CPU_HALF_MATH(hsqrt, h2sqrt, std::sqrt(x))
CUDA4CPU_HALF_MATH(hrsqrt, h2rsqrt, 1.0 / std::sqrt(x))
CUDA4CPU_HALF_MATH(hrcp, h2rcp, 1.0 / x)
CUDA4CPU_HALF_MATH(hexp, h2exp, std::exp(x))
CUDA4CPU_HALF_MATH(hexp2, h2exp2, std::exp2(x))
CUDA4CPU_HALF_MATH(hexp10, h2exp10, std::pow(10.0, x))
CUDA4CPU_HALF_MATH(hlog, h2log, std::log(x))
CUDA4CPU_HALF_MATH(hlog2, h2log2, std::log2(x))
CUDA4CPU_HALF_MATH(hlog10, h2log10, std::log10(x))
CUDA4CPU_HALF_MATH(hsin, h2sin, std::sin(x))
CUDA4CPU_HALF_MATH(hcos, h2cos, std::cos(x))
CUDA4CPU_HALF_MATH(htanh, h2tanh, std::tanh(x))
CUDA4CPU_HALF_MATH(hceil, h2ceil, std::ceil(x))
CUDA4CPU_HALF_MATH(hfloor, h2floor, std::floor(x))
CUDA4CPU_HALF_MATH(htrunc, h2trunc, std::trunc(x))
CUDA4CPU_HALF_MATH(hrint, h2rint, std::nearbyint(x))

#undef CUDA4CPU_HALF_MATH

//
// Operators
//

#ifndef __CUDA_NO_HALF_OPERATORS__

constexpr __half operator+(const __half &a, const __half &b) { return __hadd(a, b); }
constexpr __half operator-(const __half &a, const __half &b) { return __hsub(a, b); }
constexpr __half operator*(const __half &a, const __half &b) { return __hmul(a, b); }
constexpr __half operator/(const __half &a, const __half &b) { return __hdiv(a, b); }
constexpr __half &operator+=(__half &a, const __half &b) { return a = __hadd(a, b); }
constexpr __half &operator-=(__half &a, const __half &b) { return a = __hsub(a, b); }
constexpr __half &operator*=(__half &a, const __half &b) { return a = __hmul(a, b); }
constexpr __half &operator/=(__half &a, const __half &b) { return a = __hdiv(a, b); }
constexpr __half operator+(const __half &a) { return a; }
constexpr __half operator-(const __half &a) { return __hneg(a); }
constexpr __half &operator++(__half &a) { return a = __hadd(a, __half(__half_raw{0x3c00})); }
constexpr __half &operator--(__half &a) { return a = __hsub(a, __half(__half_raw{0x3c00})); }
constexpr __half operator++(__half &a, int) { const __half old = a; ++a; return old; }
constexpr __half operator--(__half &a, int) { const __half old = a; --a; return old; }
constexpr bool operator==(const __half &a, const __half &b) { return __heq(a, b); }
constexpr bool operator!=(const __half &a, const __half &b) { return __hneu(a, b); }
constexpr bool operator<(const __half &a, const __half &b) { return __hlt(a, b); }
constexpr bool operator<=(const __half &a, const __half &b) { return __hle(a, b); }
constexpr bool operator>(const __half &a, const __half &b) { return __hgt(a, b); }
constexpr bool operator>=(const __half &a, const __half &b) { return __hge(a, b); }

#endif

#ifndef __CUDA_NO_HALF2_OPERATORS__

constexpr __half2 operator+(const __half2 &a, const __half2 &b) { return __hadd2(a, b); }
constexpr __half2 operator-(const __half2 &a, const __half2 &b) { return __hsub2(a, b); }
constexpr __half2 operator*(const __half2 &a, const __half2 &b) { return __hmul2(a, b); }
constexpr __half2 operator/(const __half2 &a, const __half2 &b) { return __h2div(a, b); }
constexpr __half2 &operator+=(__half2 &a, const __half2 &b) { return a = __hadd2(a, b); }
constexpr __half2 &operator-=(__half2 &a, const __half2 &b) { return a = __hsub2(a, b); }
constexpr __half2 &operator*=(__half2 &a, const __half2 &b) { return a = __hmul2(a, b); }
constexpr __half2 &operator/=(__half2 &a, const __half2 &b) { return a = __h2div(a, b); }
constexpr __half2 operator+(const __half2 &a) { return a; }
constexpr __half2 operator-(const __half2 &a) { return __hneg2(a); }
constexpr __half2 &operator++(__half2 &a) { return a = __hadd2(a, __half2half2(__half_raw{0x3c00})); }
constexpr __half2 &operator--(__half2 &a) { return a = __hsub2(a, __half2half2(__half_raw{0x3c00})); }
constexpr __half2 operator++(__half2 &a, int) { const __half2 old = a; ++a; return old; }
constexpr __half2 operator--(__half2 &a, int) { const __half2 old = a; --a; return old; }
constexpr bool operator==(const __half2 &a, const __half2 &b) { return __hbeq2(a, b); }
constexpr bool operator!=(const __half2 &a, const __half2 &b) { return __hbneu2(a, b); }
constexpr bool operator<(const __half2 &a, const __half2 &b) { return __hblt2(a, b); }
constexpr bool operator<=(const __half2 &a, const __half2 &b) { return __hble2(a, b); }
constexpr bool operator>(const __half2 &a, const __half2 &b) { return __hbgt2(a, b); }
constexpr bool operator>=(const __half2 &a, const __half2 &b) { return __hbge2(a, b); }

#endif

//
// Atomics, as compare-and-swap loops
//

inline __half atomicAdd(__half *address, __half val)
{
    return detail::atomic_update(address, [val](__half old) { return __hadd(old, val); });
}

inline __half2 atomicAdd(__half2 *address, __half2 val)
{
    return detail::atomic_update(address, [val](__half2 old) { return __hadd2(old, val); });
}

}

}
