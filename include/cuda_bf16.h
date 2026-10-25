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
// CUDA's bfloat16: __nv_bfloat16, __nv_bfloat162 and their intrinsics, as in
// CUDA's cuda_bf16.h. A __nv_bfloat16 holds the top 16 bits of a float (8
// exponent bits, 7 mantissa bits), so arrays have CUDA's layout.
//
// As with half (cuda_fp16.h), operations convert to double, compute, and
// round once, which rounds correctly (see small_float.hpp), __hfma included.
// Conversions round as their suffix says, and transcendental functions round
// the double result. The arithmetic, comparison and math functions are
// overloads of half's (__hadd, __hlt, hsqrt, ...); conversions have names
// of their own (__float2bfloat16_rz, __bfloat162int_rn, ...).
//
// As in CUDA, __CUDA_NO_BFLOAT16_CONVERSIONS__, __CUDA_NO_BFLOAT16_OPERATORS__
// and __CUDA_NO_BFLOAT162_OPERATORS__ remove the implicit conversions and the
// operators.
//

#pragma once

#include <cmath>
#include <cstdint>

#include "cuda_fp16.h"

// The include guards of CUDA's own headers, which code tests
#ifndef __CUDA_BF16_H__
#define __CUDA_BF16_H__
#endif
#ifndef __CUDA_BF16_HPP__
#define __CUDA_BF16_HPP__
#endif
#ifndef __CUDA_BF16_TYPES_EXIST__
#define __CUDA_BF16_TYPES_EXIST__
#endif

namespace cuda4cpu {

namespace detail {

constexpr uint16_t to_bfloat16_bits(double x, float_rounding rounding = float_rounding::nearest)
{
    return uint16_t(to_bits<bfloat16_format>(x, rounding));
}

}

inline namespace cuda_api {

struct __nv_bfloat16_raw {
    unsigned short x;
};

struct __nv_bfloat162_raw {
    unsigned short x;
    unsigned short y;
};

//! A bfloat16 number. Its conversions round to nearest, except those to
//! integers, which round toward zero, as in CUDA.
struct alignas(2) __nv_bfloat16 {
protected:
    unsigned short __x;

public:
    __nv_bfloat16() = default;
    constexpr __nv_bfloat16(const __nv_bfloat16_raw &raw) : __x(raw.x) {}
    constexpr __nv_bfloat16 &operator=(const __nv_bfloat16_raw &raw) { __x = raw.x; return *this; }
    constexpr operator __nv_bfloat16_raw() const { return {__x}; }

#ifndef __CUDA_NO_BFLOAT16_CONVERSIONS__
    constexpr explicit __nv_bfloat16(const __half h) : __x(detail::to_bfloat16_bits(detail::to_double(h))) {}
    constexpr __nv_bfloat16(float f) : __x(detail::to_bfloat16_bits(f)) {}
    constexpr __nv_bfloat16(double d) : __x(detail::to_bfloat16_bits(d)) {}
    constexpr __nv_bfloat16(short v) : __x(detail::to_bfloat16_bits(v)) {}
    constexpr __nv_bfloat16(unsigned short v) : __x(detail::to_bfloat16_bits(v)) {}
    constexpr __nv_bfloat16(int v) : __x(detail::to_bfloat16_bits(v)) {}
    constexpr __nv_bfloat16(unsigned int v) : __x(detail::to_bfloat16_bits(v)) {}
    constexpr __nv_bfloat16(long v) : __x(detail::to_bfloat16_bits(double(v))) {}
    constexpr __nv_bfloat16(unsigned long v) : __x(detail::to_bfloat16_bits(double(v))) {}
    constexpr __nv_bfloat16(long long v) : __x(detail::to_bfloat16_bits(double(v))) {}
    constexpr __nv_bfloat16(unsigned long long v) : __x(detail::to_bfloat16_bits(double(v))) {}

    constexpr __nv_bfloat16 &operator=(float f) { __x = detail::to_bfloat16_bits(f); return *this; }
    constexpr __nv_bfloat16 &operator=(double d) { __x = detail::to_bfloat16_bits(d); return *this; }
    constexpr __nv_bfloat16 &operator=(short v) { __x = detail::to_bfloat16_bits(v); return *this; }
    constexpr __nv_bfloat16 &operator=(unsigned short v) { __x = detail::to_bfloat16_bits(v); return *this; }
    constexpr __nv_bfloat16 &operator=(int v) { __x = detail::to_bfloat16_bits(v); return *this; }
    constexpr __nv_bfloat16 &operator=(unsigned int v) { __x = detail::to_bfloat16_bits(v); return *this; }
    constexpr __nv_bfloat16 &operator=(long long v) { __x = detail::to_bfloat16_bits(double(v)); return *this; }
    constexpr __nv_bfloat16 &operator=(unsigned long long v) { __x = detail::to_bfloat16_bits(double(v)); return *this; }

    constexpr operator float() const { return float(detail::from_bits<detail::bfloat16_format>(__x)); }
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
        return detail::to_integer<I>(detail::from_bits<detail::bfloat16_format>(__x), [](double v) { return std::trunc(v); });
    }
#endif
};

struct alignas(4) __nv_bfloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;

    __nv_bfloat162() = default;
    constexpr __nv_bfloat162(const __nv_bfloat16 &a, const __nv_bfloat16 &b) : x(a), y(b) {}
    constexpr __nv_bfloat162(const __nv_bfloat162_raw &raw) : x(__nv_bfloat16_raw{raw.x}), y(__nv_bfloat16_raw{raw.y}) {}
    constexpr __nv_bfloat162 &operator=(const __nv_bfloat162_raw &raw)
    {
        x = __nv_bfloat16_raw{raw.x};
        y = __nv_bfloat16_raw{raw.y};
        return *this;
    }
    constexpr operator __nv_bfloat162_raw() const { return {__nv_bfloat16_raw(x).x, __nv_bfloat16_raw(y).x}; }
};

using nv_bfloat16 = __nv_bfloat16;
using nv_bfloat162 = __nv_bfloat162;

}

namespace detail {

constexpr double to_double(__nv_bfloat16 b)
{
    return from_bits<bfloat16_format>(__nv_bfloat16_raw(b).x);
}

constexpr __nv_bfloat16 to_bfloat16(double x, float_rounding rounding = float_rounding::nearest)
{
    return __nv_bfloat16_raw{to_bfloat16_bits(x, rounding)};
}

constexpr bool is_nan(__nv_bfloat16 b)
{
    return (__nv_bfloat16_raw(b).x & 0x7fff) > 0x7f80;
}

//! a * b + c, rounded once (see fma_bits)
constexpr __nv_bfloat16 fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c)
{
    return __nv_bfloat16_raw{uint16_t(fma_bits<bfloat16_format>(to_double(a), to_double(b), to_double(c)))};
}

template <typename F>
constexpr __nv_bfloat162 lanes(__nv_bfloat162 a, F f)
{
    return {f(a.x), f(a.y)};
}

template <typename F>
constexpr __nv_bfloat162 lanes(__nv_bfloat162 a, __nv_bfloat162 b, F f)
{
    return {f(a.x, b.x), f(a.y, b.y)};
}

constexpr __nv_bfloat16 bfloat16_bool(bool b)
{
    return __nv_bfloat16_raw{uint16_t(b ? 0x3f80 : 0)};
}

constexpr __nv_bfloat16 saturate(__nv_bfloat16 b)
{
    const double v = to_double(b);
    if (is_nan(b) || v <= 0)
        return __nv_bfloat16_raw{0};
    return v >= 1 ? __nv_bfloat16(__nv_bfloat16_raw{0x3f80}) : b;
}

constexpr __nv_bfloat16 relu(__nv_bfloat16 b)
{
    if (is_nan(b))
        return __nv_bfloat16_raw{0x7fff};
    return to_double(b) < 0 ? __nv_bfloat16(__nv_bfloat16_raw{0}) : b;
}

}

inline namespace cuda_api {

//
// Constants, as in CUDA
//

#define CUDART_INF_BF16        __ushort_as_bfloat16((unsigned short)0x7F80U)
#define CUDART_NAN_BF16        __ushort_as_bfloat16((unsigned short)0x7FFFU)
#define CUDART_MIN_DENORM_BF16 __ushort_as_bfloat16((unsigned short)0x0001U)
#define CUDART_MAX_NORMAL_BF16 __ushort_as_bfloat16((unsigned short)0x7F7FU)
#define CUDART_NEG_ZERO_BF16   __ushort_as_bfloat16((unsigned short)0x8000U)
#define CUDART_ZERO_BF16       __ushort_as_bfloat16((unsigned short)0x0000U)
#define CUDART_ONE_BF16        __ushort_as_bfloat16((unsigned short)0x3F80U)

//
// Conversions between floating-point types, and bits
//

constexpr __nv_bfloat16 __float2bfloat16(float a) { return detail::to_bfloat16(a); }
constexpr __nv_bfloat16 __float2bfloat16_rn(float a) { return detail::to_bfloat16(a); }
constexpr __nv_bfloat16 __float2bfloat16_rz(float a) { return detail::to_bfloat16(a, detail::float_rounding::toward_zero); }
constexpr __nv_bfloat16 __float2bfloat16_rd(float a) { return detail::to_bfloat16(a, detail::float_rounding::down); }
constexpr __nv_bfloat16 __float2bfloat16_ru(float a) { return detail::to_bfloat16(a, detail::float_rounding::up); }
constexpr __nv_bfloat16 __double2bfloat16(double a) { return detail::to_bfloat16(a); }
constexpr float __bfloat162float(__nv_bfloat16 a) { return float(detail::to_double(a)); }

constexpr __nv_bfloat162 __float2bfloat162_rn(float a) { return {__float2bfloat16(a), __float2bfloat16(a)}; }
constexpr __nv_bfloat162 __floats2bfloat162_rn(float a, float b) { return {__float2bfloat16(a), __float2bfloat16(b)}; }
constexpr __nv_bfloat162 __float22bfloat162_rn(float2 a) { return {__float2bfloat16(a.x), __float2bfloat16(a.y)}; }
constexpr float2 __bfloat1622float2(__nv_bfloat162 a) { return {__bfloat162float(a.x), __bfloat162float(a.y)}; }
constexpr float __low2float(__nv_bfloat162 a) { return __bfloat162float(a.x); }
constexpr float __high2float(__nv_bfloat162 a) { return __bfloat162float(a.y); }

constexpr __nv_bfloat16 __low2bfloat16(__nv_bfloat162 a) { return a.x; }
constexpr __nv_bfloat16 __high2bfloat16(__nv_bfloat162 a) { return a.y; }
constexpr __nv_bfloat162 __low2bfloat162(__nv_bfloat162 a) { return {a.x, a.x}; }
constexpr __nv_bfloat162 __high2bfloat162(__nv_bfloat162 a) { return {a.y, a.y}; }
constexpr __nv_bfloat162 __lows2bfloat162(__nv_bfloat162 a, __nv_bfloat162 b) { return {a.x, b.x}; }
constexpr __nv_bfloat162 __highs2bfloat162(__nv_bfloat162 a, __nv_bfloat162 b) { return {a.y, b.y}; }
constexpr __nv_bfloat162 __lowhigh2highlow(__nv_bfloat162 a) { return {a.y, a.x}; }
constexpr __nv_bfloat162 __halves2bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) { return {a, b}; }
constexpr __nv_bfloat162 __bfloat162bfloat162(__nv_bfloat16 a) { return {a, a}; }
constexpr __nv_bfloat162 make_bfloat162(__nv_bfloat16 x, __nv_bfloat16 y) { return {x, y}; }

constexpr short __bfloat16_as_short(__nv_bfloat16 h) { return std::bit_cast<short>(__nv_bfloat16_raw(h).x); }
constexpr unsigned short __bfloat16_as_ushort(__nv_bfloat16 h) { return __nv_bfloat16_raw(h).x; }
constexpr __nv_bfloat16 __short_as_bfloat16(short i) { return __nv_bfloat16_raw{std::bit_cast<unsigned short>(i)}; }
constexpr __nv_bfloat16 __ushort_as_bfloat16(unsigned short i) { return __nv_bfloat16_raw{i}; }

//
// Conversions to and from integers. bfloat16s to integers saturate, and NaN
// becomes 0, as in CUDA.
//

#define CUDA4CPU_BFLOAT16_INTEGER(suffix, I)                                                                         \
    inline I __bfloat162##suffix##_rn(__nv_bfloat16 h) { return detail::to_integer<I>(detail::to_double(h), [](double v) { return std::nearbyint(v); }); } \
    inline I __bfloat162##suffix##_rz(__nv_bfloat16 h) { return detail::to_integer<I>(detail::to_double(h), [](double v) { return std::trunc(v); }); }     \
    inline I __bfloat162##suffix##_rd(__nv_bfloat16 h) { return detail::to_integer<I>(detail::to_double(h), [](double v) { return std::floor(v); }); }    \
    inline I __bfloat162##suffix##_ru(__nv_bfloat16 h) { return detail::to_integer<I>(detail::to_double(h), [](double v) { return std::ceil(v); }); }     \
    constexpr __nv_bfloat16 __##suffix##2bfloat16_rn(I i) { return detail::to_bfloat16(double(i)); }                                  \
    constexpr __nv_bfloat16 __##suffix##2bfloat16_rz(I i) { return detail::to_bfloat16(double(i), detail::float_rounding::toward_zero); } \
    constexpr __nv_bfloat16 __##suffix##2bfloat16_rd(I i) { return detail::to_bfloat16(double(i), detail::float_rounding::down); }       \
    constexpr __nv_bfloat16 __##suffix##2bfloat16_ru(I i) { return detail::to_bfloat16(double(i), detail::float_rounding::up); }

CUDA4CPU_BFLOAT16_INTEGER(int, int)
CUDA4CPU_BFLOAT16_INTEGER(uint, unsigned int)
CUDA4CPU_BFLOAT16_INTEGER(short, short)
CUDA4CPU_BFLOAT16_INTEGER(ushort, unsigned short)
CUDA4CPU_BFLOAT16_INTEGER(ll, long long)
CUDA4CPU_BFLOAT16_INTEGER(ull, unsigned long long)

#undef CUDA4CPU_BFLOAT16_INTEGER

inline signed char __bfloat162char_rz(__nv_bfloat16 h)
{
    return detail::to_integer<signed char>(detail::to_double(h), [](double v) { return std::trunc(v); });
}

inline unsigned char __bfloat162uchar_rz(__nv_bfloat16 h)
{
    return detail::to_integer<unsigned char>(detail::to_double(h), [](double v) { return std::trunc(v); });
}

//
// Arithmetic, rounded to nearest
//

constexpr __nv_bfloat16 __hadd(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::to_bfloat16(detail::to_double(a) + detail::to_double(b)); }
constexpr __nv_bfloat16 __hsub(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::to_bfloat16(detail::to_double(a) - detail::to_double(b)); }
constexpr __nv_bfloat16 __hmul(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::to_bfloat16(detail::to_double(a) * detail::to_double(b)); }
constexpr __nv_bfloat16 __hdiv(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::to_bfloat16(detail::to_double(a) / detail::to_double(b)); }
constexpr __nv_bfloat16 __hadd_rn(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
constexpr __nv_bfloat16 __hsub_rn(__nv_bfloat16 a, __nv_bfloat16 b) { return __hsub(a, b); }
constexpr __nv_bfloat16 __hmul_rn(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
constexpr __nv_bfloat16 __hfma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) { return detail::fma(a, b, c); }
constexpr __nv_bfloat16 __hneg(__nv_bfloat16 a) { return __nv_bfloat16_raw{uint16_t(__nv_bfloat16_raw(a).x ^ 0x8000)}; }
constexpr __nv_bfloat16 __habs(__nv_bfloat16 a) { return __nv_bfloat16_raw{uint16_t(__nv_bfloat16_raw(a).x & 0x7fff)}; }

//! Results clamped to [0, 1], with NaN as 0
constexpr __nv_bfloat16 __hadd_sat(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::saturate(__hadd(a, b)); }
constexpr __nv_bfloat16 __hsub_sat(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::saturate(__hsub(a, b)); }
constexpr __nv_bfloat16 __hmul_sat(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::saturate(__hmul(a, b)); }
constexpr __nv_bfloat16 __hfma_sat(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) { return detail::saturate(__hfma(a, b, c)); }

//! A fused multiply-add whose negative results become 0
constexpr __nv_bfloat16 __hfma_relu(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) { return detail::relu(__hfma(a, b, c)); }

constexpr __nv_bfloat162 __hadd2(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hadd(x, y); }); }
constexpr __nv_bfloat162 __hsub2(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hsub(x, y); }); }
constexpr __nv_bfloat162 __hmul2(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hmul(x, y); }); }
constexpr __nv_bfloat162 __h2div(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hdiv(x, y); }); }
constexpr __nv_bfloat162 __hadd2_rn(__nv_bfloat162 a, __nv_bfloat162 b) { return __hadd2(a, b); }
constexpr __nv_bfloat162 __hsub2_rn(__nv_bfloat162 a, __nv_bfloat162 b) { return __hsub2(a, b); }
constexpr __nv_bfloat162 __hmul2_rn(__nv_bfloat162 a, __nv_bfloat162 b) { return __hmul2(a, b); }
constexpr __nv_bfloat162 __hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) { return {__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)}; }
constexpr __nv_bfloat162 __hneg2(__nv_bfloat162 a) { return detail::lanes(a, [](__nv_bfloat16 x) { return __hneg(x); }); }
constexpr __nv_bfloat162 __habs2(__nv_bfloat162 a) { return detail::lanes(a, [](__nv_bfloat16 x) { return __habs(x); }); }
constexpr __nv_bfloat162 __hadd2_sat(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hadd_sat(x, y); }); }
constexpr __nv_bfloat162 __hsub2_sat(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hsub_sat(x, y); }); }
constexpr __nv_bfloat162 __hmul2_sat(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hmul_sat(x, y); }); }
constexpr __nv_bfloat162 __hfma2_sat(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) { return {__hfma_sat(a.x, b.x, c.x), __hfma_sat(a.y, b.y, c.y)}; }
constexpr __nv_bfloat162 __hfma2_relu(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) { return {__hfma_relu(a.x, b.x, c.x), __hfma_relu(a.y, b.y, c.y)}; }

//! a * b + c for complex numbers, held as (real, imaginary) pairs, computed
//! in double and rounded to the type of the parts
constexpr __nv_bfloat162 __hcmadd(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c)
{
    using detail::to_double;
    return {detail::to_bfloat16(to_double(a.x) * to_double(b.x) - to_double(a.y) * to_double(b.y) + to_double(c.x)),
            detail::to_bfloat16(to_double(a.x) * to_double(b.y) + to_double(a.y) * to_double(b.x) + to_double(c.y))};
}

//
// Comparisons. The plain ones are false when an operand is NaN, the ones
// ending in u (unordered) true.
//

constexpr bool __heq(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::to_double(a) == detail::to_double(b); }
constexpr bool __hne(__nv_bfloat16 a, __nv_bfloat16 b) { return !detail::is_nan(a) && !detail::is_nan(b) && !__heq(a, b); }
constexpr bool __hlt(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::to_double(a) < detail::to_double(b); }
constexpr bool __hle(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::to_double(a) <= detail::to_double(b); }
constexpr bool __hgt(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::to_double(a) > detail::to_double(b); }
constexpr bool __hge(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::to_double(a) >= detail::to_double(b); }
constexpr bool __hequ(__nv_bfloat16 a, __nv_bfloat16 b) { return detail::is_nan(a) || detail::is_nan(b) || __heq(a, b); }
constexpr bool __hneu(__nv_bfloat16 a, __nv_bfloat16 b) { return !__heq(a, b); }
constexpr bool __hltu(__nv_bfloat16 a, __nv_bfloat16 b) { return !__hge(a, b); }
constexpr bool __hleu(__nv_bfloat16 a, __nv_bfloat16 b) { return !__hgt(a, b); }
constexpr bool __hgtu(__nv_bfloat16 a, __nv_bfloat16 b) { return !__hle(a, b); }
constexpr bool __hgeu(__nv_bfloat16 a, __nv_bfloat16 b) { return !__hlt(a, b); }

constexpr bool __hisnan(__nv_bfloat16 a) { return detail::is_nan(a); }

//! -1 for negative infinity, 1 for positive infinity, 0 otherwise
constexpr int __hisinf(__nv_bfloat16 a)
{
    const unsigned short bits = __nv_bfloat16_raw(a).x;
    return bits == 0x7f80 ? 1 : bits == 0xff80 ? -1 : 0;
}

//! The larger or smaller operand: NaN only if both are NaN
constexpr __nv_bfloat16 __hmax(__nv_bfloat16 a, __nv_bfloat16 b)
{
    if (detail::is_nan(a))
        return detail::is_nan(b) ? __nv_bfloat16(__nv_bfloat16_raw{0x7fff}) : b;
    return detail::is_nan(b) || __hge(a, b) ? a : b;
}

constexpr __nv_bfloat16 __hmin(__nv_bfloat16 a, __nv_bfloat16 b)
{
    if (detail::is_nan(a))
        return detail::is_nan(b) ? __nv_bfloat16(__nv_bfloat16_raw{0x7fff}) : b;
    return detail::is_nan(b) || __hle(a, b) ? a : b;
}

//! The larger or smaller operand, or NaN if either is NaN
constexpr __nv_bfloat16 __hmax_nan(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return detail::is_nan(a) || detail::is_nan(b) ? __nv_bfloat16(__nv_bfloat16_raw{0x7fff}) : __hmax(a, b);
}

constexpr __nv_bfloat16 __hmin_nan(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return detail::is_nan(a) || detail::is_nan(b) ? __nv_bfloat16(__nv_bfloat16_raw{0x7fff}) : __hmin(a, b);
}

// Per lane, as 1.0 or 0.0
#define CUDA4CPU_BFLOAT162_COMPARE(name)                                                                 \
    constexpr __nv_bfloat162 name##2(__nv_bfloat162 a, __nv_bfloat162 b)                                 \
    {                                                                                                    \
        return {detail::bfloat16_bool(name(a.x, b.x)), detail::bfloat16_bool(name(a.y, b.y))};           \
    }                                                                                                    \
    constexpr unsigned name##2_mask(__nv_bfloat162 a, __nv_bfloat162 b)                                  \
    {                                                                                                    \
        return (name(a.x, b.x) ? 0xffffu : 0u) | (name(a.y, b.y) ? 0xffff0000u : 0u);                    \
    }

CUDA4CPU_BFLOAT162_COMPARE(__heq)
CUDA4CPU_BFLOAT162_COMPARE(__hne)
CUDA4CPU_BFLOAT162_COMPARE(__hlt)
CUDA4CPU_BFLOAT162_COMPARE(__hle)
CUDA4CPU_BFLOAT162_COMPARE(__hgt)
CUDA4CPU_BFLOAT162_COMPARE(__hge)
CUDA4CPU_BFLOAT162_COMPARE(__hequ)
CUDA4CPU_BFLOAT162_COMPARE(__hneu)
CUDA4CPU_BFLOAT162_COMPARE(__hltu)
CUDA4CPU_BFLOAT162_COMPARE(__hleu)
CUDA4CPU_BFLOAT162_COMPARE(__hgtu)
CUDA4CPU_BFLOAT162_COMPARE(__hgeu)

#undef CUDA4CPU_BFLOAT162_COMPARE

// Whether both lanes compare so
constexpr bool __hbeq2(__nv_bfloat162 a, __nv_bfloat162 b) { return __heq(a.x, b.x) && __heq(a.y, b.y); }
constexpr bool __hbne2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hne(a.x, b.x) && __hne(a.y, b.y); }
constexpr bool __hblt2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hlt(a.x, b.x) && __hlt(a.y, b.y); }
constexpr bool __hble2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hle(a.x, b.x) && __hle(a.y, b.y); }
constexpr bool __hbgt2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hgt(a.x, b.x) && __hgt(a.y, b.y); }
constexpr bool __hbge2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hge(a.x, b.x) && __hge(a.y, b.y); }
constexpr bool __hbequ2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hequ(a.x, b.x) && __hequ(a.y, b.y); }
constexpr bool __hbneu2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hneu(a.x, b.x) && __hneu(a.y, b.y); }
constexpr bool __hbltu2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hltu(a.x, b.x) && __hltu(a.y, b.y); }
constexpr bool __hbleu2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hleu(a.x, b.x) && __hleu(a.y, b.y); }
constexpr bool __hbgtu2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hgtu(a.x, b.x) && __hgtu(a.y, b.y); }
constexpr bool __hbgeu2(__nv_bfloat162 a, __nv_bfloat162 b) { return __hgeu(a.x, b.x) && __hgeu(a.y, b.y); }

constexpr __nv_bfloat162 __hisnan2(__nv_bfloat162 a)
{
    return {detail::bfloat16_bool(__hisnan(a.x)), detail::bfloat16_bool(__hisnan(a.y))};
}

constexpr __nv_bfloat162 __hmax2(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hmax(x, y); }); }
constexpr __nv_bfloat162 __hmin2(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hmin(x, y); }); }
constexpr __nv_bfloat162 __hmax2_nan(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hmax_nan(x, y); }); }
constexpr __nv_bfloat162 __hmin2_nan(__nv_bfloat162 a, __nv_bfloat162 b) { return detail::lanes(a, b, [](__nv_bfloat16 x, __nv_bfloat16 y) { return __hmin_nan(x, y); }); }

//
// Math functions, computed in double and rounded to bfloat16
//

#define CUDA4CPU_BFLOAT16_MATH(name, h2name, expression)                                               \
    inline __nv_bfloat16 name(__nv_bfloat16 a)                                                         \
    {                                                                                                  \
        const double x = detail::to_double(a);                                                         \
        return detail::to_bfloat16(expression);                                                        \
    }                                                                                                  \
    inline __nv_bfloat162 h2name(__nv_bfloat162 a) { return {name(a.x), name(a.y)}; }

CUDA4CPU_BFLOAT16_MATH(hsqrt, h2sqrt, std::sqrt(x))
CUDA4CPU_BFLOAT16_MATH(hrsqrt, h2rsqrt, 1.0 / std::sqrt(x))
CUDA4CPU_BFLOAT16_MATH(hrcp, h2rcp, 1.0 / x)
CUDA4CPU_BFLOAT16_MATH(hexp, h2exp, std::exp(x))
CUDA4CPU_BFLOAT16_MATH(hexp2, h2exp2, std::exp2(x))
CUDA4CPU_BFLOAT16_MATH(hexp10, h2exp10, std::pow(10.0, x))
CUDA4CPU_BFLOAT16_MATH(hlog, h2log, std::log(x))
CUDA4CPU_BFLOAT16_MATH(hlog2, h2log2, std::log2(x))
CUDA4CPU_BFLOAT16_MATH(hlog10, h2log10, std::log10(x))
CUDA4CPU_BFLOAT16_MATH(hsin, h2sin, std::sin(x))
CUDA4CPU_BFLOAT16_MATH(hcos, h2cos, std::cos(x))
CUDA4CPU_BFLOAT16_MATH(htanh, h2tanh, std::tanh(x))
CUDA4CPU_BFLOAT16_MATH(hceil, h2ceil, std::ceil(x))
CUDA4CPU_BFLOAT16_MATH(hfloor, h2floor, std::floor(x))
CUDA4CPU_BFLOAT16_MATH(htrunc, h2trunc, std::trunc(x))
CUDA4CPU_BFLOAT16_MATH(hrint, h2rint, std::nearbyint(x))

#undef CUDA4CPU_BFLOAT16_MATH

//
// Operators
//

#ifndef __CUDA_NO_BFLOAT16_OPERATORS__

constexpr __nv_bfloat16 operator+(const __nv_bfloat16 &a, const __nv_bfloat16 &b) { return __hadd(a, b); }
constexpr __nv_bfloat16 operator-(const __nv_bfloat16 &a, const __nv_bfloat16 &b) { return __hsub(a, b); }
constexpr __nv_bfloat16 operator*(const __nv_bfloat16 &a, const __nv_bfloat16 &b) { return __hmul(a, b); }
constexpr __nv_bfloat16 operator/(const __nv_bfloat16 &a, const __nv_bfloat16 &b) { return __hdiv(a, b); }
constexpr __nv_bfloat16 &operator+=(__nv_bfloat16 &a, const __nv_bfloat16 &b) { return a = __hadd(a, b); }
constexpr __nv_bfloat16 &operator-=(__nv_bfloat16 &a, const __nv_bfloat16 &b) { return a = __hsub(a, b); }
constexpr __nv_bfloat16 &operator*=(__nv_bfloat16 &a, const __nv_bfloat16 &b) { return a = __hmul(a, b); }
constexpr __nv_bfloat16 &operator/=(__nv_bfloat16 &a, const __nv_bfloat16 &b) { return a = __hdiv(a, b); }
constexpr __nv_bfloat16 operator+(const __nv_bfloat16 &a) { return a; }
constexpr __nv_bfloat16 operator-(const __nv_bfloat16 &a) { return __hneg(a); }
constexpr __nv_bfloat16 &operator++(__nv_bfloat16 &a) { return a = __hadd(a, __nv_bfloat16(__nv_bfloat16_raw{0x3f80})); }
constexpr __nv_bfloat16 &operator--(__nv_bfloat16 &a) { return a = __hsub(a, __nv_bfloat16(__nv_bfloat16_raw{0x3f80})); }
constexpr __nv_bfloat16 operator++(__nv_bfloat16 &a, int) { const __nv_bfloat16 old = a; ++a; return old; }
constexpr __nv_bfloat16 operator--(__nv_bfloat16 &a, int) { const __nv_bfloat16 old = a; --a; return old; }
constexpr bool operator==(const __nv_bfloat16 &a, const __nv_bfloat16 &b) { return __heq(a, b); }
constexpr bool operator!=(const __nv_bfloat16 &a, const __nv_bfloat16 &b) { return __hneu(a, b); }
constexpr bool operator<(const __nv_bfloat16 &a, const __nv_bfloat16 &b) { return __hlt(a, b); }
constexpr bool operator<=(const __nv_bfloat16 &a, const __nv_bfloat16 &b) { return __hle(a, b); }
constexpr bool operator>(const __nv_bfloat16 &a, const __nv_bfloat16 &b) { return __hgt(a, b); }
constexpr bool operator>=(const __nv_bfloat16 &a, const __nv_bfloat16 &b) { return __hge(a, b); }

#endif

#ifndef __CUDA_NO_BFLOAT162_OPERATORS__

constexpr __nv_bfloat162 operator+(const __nv_bfloat162 &a, const __nv_bfloat162 &b) { return __hadd2(a, b); }
constexpr __nv_bfloat162 operator-(const __nv_bfloat162 &a, const __nv_bfloat162 &b) { return __hsub2(a, b); }
constexpr __nv_bfloat162 operator*(const __nv_bfloat162 &a, const __nv_bfloat162 &b) { return __hmul2(a, b); }
constexpr __nv_bfloat162 operator/(const __nv_bfloat162 &a, const __nv_bfloat162 &b) { return __h2div(a, b); }
constexpr __nv_bfloat162 &operator+=(__nv_bfloat162 &a, const __nv_bfloat162 &b) { return a = __hadd2(a, b); }
constexpr __nv_bfloat162 &operator-=(__nv_bfloat162 &a, const __nv_bfloat162 &b) { return a = __hsub2(a, b); }
constexpr __nv_bfloat162 &operator*=(__nv_bfloat162 &a, const __nv_bfloat162 &b) { return a = __hmul2(a, b); }
constexpr __nv_bfloat162 &operator/=(__nv_bfloat162 &a, const __nv_bfloat162 &b) { return a = __h2div(a, b); }
constexpr __nv_bfloat162 operator+(const __nv_bfloat162 &a) { return a; }
constexpr __nv_bfloat162 operator-(const __nv_bfloat162 &a) { return __hneg2(a); }
constexpr __nv_bfloat162 &operator++(__nv_bfloat162 &a) { return a = __hadd2(a, __bfloat162bfloat162(__nv_bfloat16_raw{0x3f80})); }
constexpr __nv_bfloat162 &operator--(__nv_bfloat162 &a) { return a = __hsub2(a, __bfloat162bfloat162(__nv_bfloat16_raw{0x3f80})); }
constexpr __nv_bfloat162 operator++(__nv_bfloat162 &a, int) { const __nv_bfloat162 old = a; ++a; return old; }
constexpr __nv_bfloat162 operator--(__nv_bfloat162 &a, int) { const __nv_bfloat162 old = a; --a; return old; }
constexpr bool operator==(const __nv_bfloat162 &a, const __nv_bfloat162 &b) { return __hbeq2(a, b); }
constexpr bool operator!=(const __nv_bfloat162 &a, const __nv_bfloat162 &b) { return __hbneu2(a, b); }
constexpr bool operator<(const __nv_bfloat162 &a, const __nv_bfloat162 &b) { return __hblt2(a, b); }
constexpr bool operator<=(const __nv_bfloat162 &a, const __nv_bfloat162 &b) { return __hble2(a, b); }
constexpr bool operator>(const __nv_bfloat162 &a, const __nv_bfloat162 &b) { return __hbgt2(a, b); }
constexpr bool operator>=(const __nv_bfloat162 &a, const __nv_bfloat162 &b) { return __hbge2(a, b); }

#endif

//
// Atomics, as compare-and-swap loops
//

inline __nv_bfloat16 atomicAdd(__nv_bfloat16 *address, __nv_bfloat16 val)
{
    return detail::atomic_update(address, [val](__nv_bfloat16 old) { return __hadd(old, val); });
}

inline __nv_bfloat162 atomicAdd(__nv_bfloat162 *address, __nv_bfloat162 val)
{
    return detail::atomic_update(address, [val](__nv_bfloat162 old) { return __hadd2(old, val); });
}

}

}
