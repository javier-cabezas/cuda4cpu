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
// CUDA's 8-bit floats, as in CUDA's cuda_fp8.h: E4M3 (4 exponent bits, 3
// mantissa bits, NaN but no infinity, up to 448), E5M2 (IEEE-like, up to
// 57344) and E8M0 (an unsigned power of two, for scale factors), the
// conversions between them and wider types, and the classes __nv_fp8_e4m3,
// __nv_fp8x2_e4m3, __nv_fp8x4_e4m3 and their E5M2 and E8M0 counterparts.
// As in CUDA, there is no arithmetic on them: code converts to a wider type.
//
// Conversions to E4M3 and E5M2 round to nearest even (small_float.hpp), and
// follow CUDA's rules for what doesn't fit: with __NV_SATFINITE, values
// beyond the largest finite one, infinities included, saturate to it;
// without, they become NaN in E4M3 and infinity in E5M2. Conversions to E8M0
// round toward zero or up (cudaRoundZero, cudaRoundPosInf).
//
// hip/hip_fp8.h has AMD's API, with AMD's rules and its FNUZ formats, on the
// same conversions.
//
// As in CUDA, __CUDA_NO_FP8_CONVERSIONS__ and
// __CUDA_NO_FP8_CONVERSION_OPERATORS__ remove the classes' conversions.
//

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

#include "cuda_bf16.h"
#include "cuda_fp16.h"

// The include guards of CUDA's own headers, which code tests
#ifndef __CUDA_FP8_H__
#define __CUDA_FP8_H__
#endif
#ifndef __CUDA_FP8_HPP__
#define __CUDA_FP8_HPP__
#endif
#ifndef __CUDA_FP8_TYPES_EXIST__
#define __CUDA_FP8_TYPES_EXIST__
#endif

namespace cuda4cpu {

inline namespace cuda_api {

typedef unsigned char __nv_fp8_storage_t;
typedef unsigned short int __nv_fp8x2_storage_t;
typedef unsigned int __nv_fp8x4_storage_t;

typedef enum __nv_saturation_t {
    __NV_NOSAT,          //!< Values that don't fit become NaN or infinity
    __NV_SATFINITE,      //!< They become the largest finite value of their sign
} __nv_saturation_t;

typedef enum __nv_fp8_interpretation_t {
    __NV_E4M3,
    __NV_E5M2,
} __nv_fp8_interpretation_t;

}

namespace detail {

//! CUDA's conversion to E4M3 or E5M2 (F): rounded to nearest even. Beyond
//! the largest finite value (after rounding), with saturation, the largest
//! finite value of x's sign, infinity included; without, NaN in E4M3 and
//! infinity in E5M2. NaN stays NaN with its sign, and in E5M2 with the top
//! two bits of its payload.
template <float_format F>
constexpr uint8_t nv_fp8_bits(double x, bool saturate)
{
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const uint8_t sign = uint8_t((bits >> 56) & 0x80);
    if (x != x) {
        if constexpr (F.specials == float_specials::one_nan)
            return sign | 0x7f;
        else
            return uint8_t(sign | 0x7e | ((bits >> 50) & 3));
    }
    uint8_t result = uint8_t(to_bits<F>(x, float_rounding::nearest, saturate));
    if (F.is_nan(result))
        result |= sign;                                 // E4M3's NaN for a value too large keeps its sign
    return result;
}

constexpr uint8_t nv_fp8_bits(double x, __nv_saturation_t saturate, __nv_fp8_interpretation_t interpretation)
{
    if (interpretation == __NV_E4M3)
        return nv_fp8_bits<e4m3_format>(x, saturate == __NV_SATFINITE);
    return nv_fp8_bits<e5m2_format>(x, saturate == __NV_SATFINITE);
}

//! The bits of the half that holds the F number b, which all of them can
//! exactly. NaN gives half's canonical NaN, without its sign, as in CUDA.
template <float_format F>
constexpr uint16_t fp8_to_half_bits(uint8_t b)
{
    if (F.is_nan(b))
        return 0x7fff;
    return to_half_bits(from_bits<F>(b));
}

//! The E8M0 code of x, a power of two 2^(code - 127), or 0xff for NaN. The
//! sign is ignored. cudaRoundZero takes the power of two at or below |x|,
//! and other modes the one at or above it, as CUDA's conversions from float
//! do (its own only take cudaRoundZero and cudaRoundPosInf). Below 2^-127,
//! the code is 0. Beyond 2^127, rounding up gives NaN, or 2^127 with
//! saturation; toward zero, 2^127.
constexpr uint8_t e8m0_bits(double x, bool saturate, cudaRoundMode rounding)
{
    if (x != x)
        return 0xff;
    const uint64_t magnitude = std::bit_cast<uint64_t>(x) & 0x7fffffffffffffff;
    if (magnitude == 0x7ff0000000000000)
        return saturate ? 0xfe : 0xff;
    if (magnitude == 0)
        return 0;
    const int biased = int(magnitude >> 52);
    const uint64_t fraction = magnitude & ((uint64_t(1) << 52) - 1);
    // |x| = 2^exponent * (1 + something), exactly a power of two or not
    const int exponent = biased != 0 ? biased - 1023 : int(std::bit_width(fraction)) - 1 - 1074;
    const bool exact = biased != 0 ? fraction == 0 : std::has_single_bit(fraction);
    const int code = exponent + 127 + (rounding != cudaRoundZero && !exact ? 1 : 0);
    if (code < 0)
        return 0;
    if (code >= 255)
        return rounding == cudaRoundZero || saturate ? 0xfe : 0xff;
    return uint8_t(code);
}

//! The E8M0 code of the power of two at or above |v|, for any integer
template <typename I>
constexpr uint8_t e8m0_bits_of_integer(I v)
{
    const unsigned long long magnitude = v < 0 ? 0ull - (unsigned long long)(v) : (unsigned long long)(v);
    if (magnitude == 0)
        return 0;
    return uint8_t(int(std::bit_width(magnitude)) - 1 + (std::has_single_bit(magnitude) ? 0 : 1) + 127);
}

constexpr double from_e8m0_bits(uint8_t b)
{
    if (b == 0xff)
        return std::bit_cast<double>(uint64_t(0x7ff8000000000000));
    return std::bit_cast<double>(uint64_t(int(b) - 127 + 1023) << 52);
}

constexpr uint16_t e8m0_to_bfloat16_bits(uint8_t b)
{
    return b == 0xff ? 0x7fff : to_bfloat16_bits(from_e8m0_bits(b));
}

}

inline namespace cuda_api {

//
// Conversions to E4M3 and E5M2. The pairs keep x (the low lane) in the low
// byte.
//

constexpr __nv_fp8_storage_t __nv_cvt_double_to_fp8(double x, __nv_saturation_t saturate,
                                                    __nv_fp8_interpretation_t interpretation)
{
    return detail::nv_fp8_bits(x, saturate, interpretation);
}

constexpr __nv_fp8x2_storage_t __nv_cvt_double2_to_fp8x2(double2 x, __nv_saturation_t saturate,
                                                         __nv_fp8_interpretation_t interpretation)
{
    return __nv_fp8x2_storage_t(detail::nv_fp8_bits(x.x, saturate, interpretation) |
                                detail::nv_fp8_bits(x.y, saturate, interpretation) << 8);
}

constexpr __nv_fp8_storage_t __nv_cvt_float_to_fp8(float x, __nv_saturation_t saturate,
                                                   __nv_fp8_interpretation_t interpretation)
{
    return detail::nv_fp8_bits(x, saturate, interpretation);
}

constexpr __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2 x, __nv_saturation_t saturate,
                                                        __nv_fp8_interpretation_t interpretation)
{
    return __nv_cvt_double2_to_fp8x2({x.x, x.y}, saturate, interpretation);
}

constexpr __nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(__half_raw x, __nv_saturation_t saturate,
                                                     __nv_fp8_interpretation_t interpretation)
{
    return detail::nv_fp8_bits(detail::from_half_bits(x.x), saturate, interpretation);
}

constexpr __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(__half2_raw x, __nv_saturation_t saturate,
                                                          __nv_fp8_interpretation_t interpretation)
{
    return __nv_fp8x2_storage_t(__nv_cvt_halfraw_to_fp8({x.x}, saturate, interpretation) |
                                __nv_cvt_halfraw_to_fp8({x.y}, saturate, interpretation) << 8);
}

constexpr __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw x, __nv_saturation_t saturate,
                                                         __nv_fp8_interpretation_t interpretation)
{
    return detail::nv_fp8_bits(detail::from_bits<detail::bfloat16_format>(x.x), saturate, interpretation);
}

constexpr __nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(__nv_bfloat162_raw x, __nv_saturation_t saturate,
                                                              __nv_fp8_interpretation_t interpretation)
{
    return __nv_fp8x2_storage_t(__nv_cvt_bfloat16raw_to_fp8({x.x}, saturate, interpretation) |
                                __nv_cvt_bfloat16raw_to_fp8({x.y}, saturate, interpretation) << 8);
}

//
// Conversions from E4M3 and E5M2, which half holds exactly
//

constexpr __half_raw __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t x, __nv_fp8_interpretation_t interpretation)
{
    if (interpretation == __NV_E4M3)
        return {detail::fp8_to_half_bits<detail::e4m3_format>(x)};
    return {detail::fp8_to_half_bits<detail::e5m2_format>(x)};
}

constexpr __half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t x, __nv_fp8_interpretation_t interpretation)
{
    return {__nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t(x), interpretation).x,
            __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t(x >> 8), interpretation).x};
}

//
// Conversions to and from E8M0
//

constexpr __nv_fp8_storage_t __nv_cvt_double_to_e8m0(double x, __nv_saturation_t saturate, cudaRoundMode rounding)
{
    return detail::e8m0_bits(x, saturate == __NV_SATFINITE, rounding);
}

constexpr __nv_fp8x2_storage_t __nv_cvt_double2_to_e8m0x2(double2 x, __nv_saturation_t saturate, cudaRoundMode rounding)
{
    return __nv_fp8x2_storage_t(__nv_cvt_double_to_e8m0(x.x, saturate, rounding) |
                                __nv_cvt_double_to_e8m0(x.y, saturate, rounding) << 8);
}

constexpr __nv_fp8_storage_t __nv_cvt_float_to_e8m0(float x, __nv_saturation_t saturate, cudaRoundMode rounding)
{
    return __nv_cvt_double_to_e8m0(x, saturate, rounding);
}

constexpr __nv_fp8x2_storage_t __nv_cvt_float2_to_e8m0x2(float2 x, __nv_saturation_t saturate, cudaRoundMode rounding)
{
    return __nv_cvt_double2_to_e8m0x2({x.x, x.y}, saturate, rounding);
}

constexpr __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_e8m0(__nv_bfloat16_raw x, __nv_saturation_t saturate,
                                                          cudaRoundMode rounding)
{
    return __nv_cvt_double_to_e8m0(detail::from_bits<detail::bfloat16_format>(x.x), saturate, rounding);
}

constexpr __nv_fp8x2_storage_t __nv_cvt_bfloat162raw_to_e8m0x2(__nv_bfloat162_raw x, __nv_saturation_t saturate,
                                                               cudaRoundMode rounding)
{
    return __nv_fp8x2_storage_t(__nv_cvt_bfloat16raw_to_e8m0({x.x}, saturate, rounding) |
                                __nv_cvt_bfloat16raw_to_e8m0({x.y}, saturate, rounding) << 8);
}

//! 2^(x - 127), which bfloat16 holds exactly (0 is the subnormal 2^-127), or
//! NaN for 0xff
constexpr __nv_bfloat16_raw __nv_cvt_e8m0_to_bf16raw(__nv_fp8_storage_t x)
{
    return {detail::e8m0_to_bfloat16_bits(x)};
}

constexpr __nv_bfloat162_raw __nv_cvt_e8m0x2_to_bf162raw(__nv_fp8x2_storage_t x)
{
    return {detail::e8m0_to_bfloat16_bits(__nv_fp8_storage_t(x)), detail::e8m0_to_bfloat16_bits(__nv_fp8_storage_t(x >> 8))};
}

}

namespace detail {

//! __nv_fp8_e4m3 and __nv_fp8_e5m2. Conversions to them saturate, and those
//! to integers truncate and saturate, with NaN as 0, as in CUDA.
template <__nv_fp8_interpretation_t I>
struct nv_fp8 {
    __nv_fp8_storage_t __x;

    nv_fp8() = default;

#ifndef __CUDA_NO_FP8_CONVERSIONS__
    constexpr explicit nv_fp8(__half f) : __x(__nv_cvt_halfraw_to_fp8(f, __NV_SATFINITE, I)) {}
    constexpr explicit nv_fp8(__nv_bfloat16 f) : __x(__nv_cvt_bfloat16raw_to_fp8(f, __NV_SATFINITE, I)) {}
    constexpr explicit nv_fp8(float f) : __x(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, I)) {}
    constexpr explicit nv_fp8(double f) : __x(__nv_cvt_double_to_fp8(f, __NV_SATFINITE, I)) {}
    constexpr explicit nv_fp8(unsigned short v) : nv_fp8(double(v)) {}
    constexpr explicit nv_fp8(unsigned int v) : nv_fp8(double(v)) {}
    constexpr explicit nv_fp8(unsigned long v) : nv_fp8(double(v)) {}
    constexpr explicit nv_fp8(unsigned long long v) : nv_fp8(double(v)) {}
    constexpr explicit nv_fp8(short v) : nv_fp8(double(v)) {}
    constexpr explicit nv_fp8(int v) : nv_fp8(double(v)) {}
    constexpr explicit nv_fp8(long v) : nv_fp8(double(v)) {}
    constexpr explicit nv_fp8(long long v) : nv_fp8(double(v)) {}
#endif

#ifndef __CUDA_NO_FP8_CONVERSION_OPERATORS__
    constexpr explicit operator __half() const { return __half_raw(__nv_cvt_fp8_to_halfraw(__x, I)); }
    constexpr explicit operator float() const { return float(value()); }
    constexpr explicit operator double() const { return value(); }
    constexpr explicit operator __nv_bfloat16() const { return detail::to_bfloat16(value()); }
    explicit operator unsigned char() const { return to_integer<unsigned char>(); }
    explicit operator unsigned short() const { return to_integer<unsigned short>(); }
    explicit operator unsigned int() const { return to_integer<unsigned int>(); }
    explicit operator unsigned long() const { return to_integer<unsigned long>(); }
    explicit operator unsigned long long() const { return to_integer<unsigned long long>(); }
    explicit operator signed char() const { return to_integer<signed char>(); }
    explicit operator char() const { return to_integer<char>(); }
    explicit operator short() const { return to_integer<short>(); }
    explicit operator int() const { return to_integer<int>(); }
    explicit operator long() const { return to_integer<long>(); }
    explicit operator long long() const { return to_integer<long long>(); }
    constexpr explicit operator bool() const { return (__x & 0x7f) != 0; }

private:
    constexpr double value() const { return from_half_bits(__nv_cvt_fp8_to_halfraw(__x, I).x); }

    template <typename T>
    T to_integer() const
    {
        return detail::to_integer<T>(value(), [](double v) { return std::trunc(v); });
    }
#endif
};

template <__nv_fp8_interpretation_t I>
struct nv_fp8x2 {
    __nv_fp8x2_storage_t __x;

    nv_fp8x2() = default;

#ifndef __CUDA_NO_FP8_CONVERSIONS__
    constexpr explicit nv_fp8x2(__half2 f) : __x(__nv_cvt_halfraw2_to_fp8x2(f, __NV_SATFINITE, I)) {}
    constexpr explicit nv_fp8x2(__nv_bfloat162 f) : __x(__nv_cvt_bfloat16raw2_to_fp8x2(f, __NV_SATFINITE, I)) {}
    constexpr explicit nv_fp8x2(float2 f) : __x(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, I)) {}
    constexpr explicit nv_fp8x2(double2 f) : __x(__nv_cvt_double2_to_fp8x2(f, __NV_SATFINITE, I)) {}
#endif

#ifndef __CUDA_NO_FP8_CONVERSION_OPERATORS__
    constexpr explicit operator __half2() const { return __half2_raw(__nv_cvt_fp8x2_to_halfraw2(__x, I)); }
    constexpr explicit operator float2() const
    {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__x, I);
        return {float(from_half_bits(h.x)), float(from_half_bits(h.y))};
    }
#endif
};

template <__nv_fp8_interpretation_t I>
struct nv_fp8x4 {
    __nv_fp8x4_storage_t __x;

    nv_fp8x4() = default;

#ifndef __CUDA_NO_FP8_CONVERSIONS__
    constexpr explicit nv_fp8x4(__half2 lo, __half2 hi) :
        __x(pack(__nv_cvt_halfraw2_to_fp8x2(lo, __NV_SATFINITE, I), __nv_cvt_halfraw2_to_fp8x2(hi, __NV_SATFINITE, I)))
    {
    }
    constexpr explicit nv_fp8x4(__nv_bfloat162 lo, __nv_bfloat162 hi) :
        __x(pack(__nv_cvt_bfloat16raw2_to_fp8x2(lo, __NV_SATFINITE, I),
                 __nv_cvt_bfloat16raw2_to_fp8x2(hi, __NV_SATFINITE, I)))
    {
    }
    constexpr explicit nv_fp8x4(float4 f) :
        __x(pack(__nv_cvt_float2_to_fp8x2({f.x, f.y}, __NV_SATFINITE, I),
                 __nv_cvt_float2_to_fp8x2({f.z, f.w}, __NV_SATFINITE, I)))
    {
    }
    constexpr explicit nv_fp8x4(double4 f) :
        __x(pack(__nv_cvt_double2_to_fp8x2({f.x, f.y}, __NV_SATFINITE, I),
                 __nv_cvt_double2_to_fp8x2({f.z, f.w}, __NV_SATFINITE, I)))
    {
    }
#endif

#ifndef __CUDA_NO_FP8_CONVERSION_OPERATORS__
    constexpr explicit operator float4() const
    {
        const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(__x), I);
        const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(__x >> 16), I);
        return {float(from_half_bits(lo.x)), float(from_half_bits(lo.y)), float(from_half_bits(hi.x)),
                float(from_half_bits(hi.y))};
    }
#endif

private:
    static constexpr __nv_fp8x4_storage_t pack(__nv_fp8x2_storage_t lo, __nv_fp8x2_storage_t hi)
    {
        return __nv_fp8x4_storage_t(lo) | __nv_fp8x4_storage_t(hi) << 16;
    }
};

}

inline namespace cuda_api {

struct alignas(1) __nv_fp8_e4m3 : detail::nv_fp8<__NV_E4M3> {
    using nv_fp8::nv_fp8;
};

struct alignas(2) __nv_fp8x2_e4m3 : detail::nv_fp8x2<__NV_E4M3> {
    using nv_fp8x2::nv_fp8x2;
};

struct alignas(4) __nv_fp8x4_e4m3 : detail::nv_fp8x4<__NV_E4M3> {
    using nv_fp8x4::nv_fp8x4;
};

struct alignas(1) __nv_fp8_e5m2 : detail::nv_fp8<__NV_E5M2> {
    using nv_fp8::nv_fp8;
};

struct alignas(2) __nv_fp8x2_e5m2 : detail::nv_fp8x2<__NV_E5M2> {
    using nv_fp8x2::nv_fp8x2;
};

struct alignas(4) __nv_fp8x4_e5m2 : detail::nv_fp8x4<__NV_E5M2> {
    using nv_fp8x4::nv_fp8x4;
};

//! A power of two, 2^(__x - 127), or NaN (0xff). Conversions to it round up
//! and saturate; those to integers truncate and saturate, with NaN as 0.
struct alignas(1) __nv_fp8_e8m0 {
    __nv_fp8_storage_t __x;

    __nv_fp8_e8m0() = default;

#ifndef __CUDA_NO_FP8_CONVERSIONS__
    constexpr explicit __nv_fp8_e8m0(__half f) : __x(detail::e8m0_bits(detail::to_double(f), true, cudaRoundPosInf)) {}
    constexpr explicit __nv_fp8_e8m0(__nv_bfloat16 f) : __x(detail::e8m0_bits(detail::to_double(f), true, cudaRoundPosInf)) {}
    constexpr explicit __nv_fp8_e8m0(float f) : __x(detail::e8m0_bits(f, true, cudaRoundPosInf)) {}
    constexpr explicit __nv_fp8_e8m0(double f) : __x(detail::e8m0_bits(f, true, cudaRoundPosInf)) {}
    constexpr explicit __nv_fp8_e8m0(unsigned short v) : __x(detail::e8m0_bits_of_integer(v)) {}
    constexpr explicit __nv_fp8_e8m0(unsigned int v) : __x(detail::e8m0_bits_of_integer(v)) {}
    constexpr explicit __nv_fp8_e8m0(unsigned long v) : __x(detail::e8m0_bits_of_integer(v)) {}
    constexpr explicit __nv_fp8_e8m0(unsigned long long v) : __x(detail::e8m0_bits_of_integer(v)) {}
    constexpr explicit __nv_fp8_e8m0(short v) : __x(detail::e8m0_bits_of_integer(v)) {}
    constexpr explicit __nv_fp8_e8m0(int v) : __x(detail::e8m0_bits_of_integer(v)) {}
    constexpr explicit __nv_fp8_e8m0(long v) : __x(detail::e8m0_bits_of_integer(v)) {}
    constexpr explicit __nv_fp8_e8m0(long long v) : __x(detail::e8m0_bits_of_integer(v)) {}
#endif

#ifndef __CUDA_NO_FP8_CONVERSION_OPERATORS__
    constexpr explicit operator float() const { return float(detail::from_e8m0_bits(__x)); }
    constexpr explicit operator double() const { return detail::from_e8m0_bits(__x); }
    constexpr explicit operator __half() const { return detail::to_half(detail::from_e8m0_bits(__x)); }
    constexpr explicit operator __nv_bfloat16() const { return __nv_cvt_e8m0_to_bf16raw(__x); }
    explicit operator unsigned char() const { return to_integer<unsigned char>(); }
    explicit operator unsigned short() const { return to_integer<unsigned short>(); }
    explicit operator unsigned int() const { return to_integer<unsigned int>(); }
    explicit operator unsigned long() const { return to_integer<unsigned long>(); }
    explicit operator unsigned long long() const { return to_integer<unsigned long long>(); }
    explicit operator signed char() const { return to_integer<signed char>(); }
    explicit operator char() const { return to_integer<char>(); }
    explicit operator short() const { return to_integer<short>(); }
    explicit operator int() const { return to_integer<int>(); }
    explicit operator long() const { return to_integer<long>(); }
    explicit operator long long() const { return to_integer<long long>(); }
    //! No E8M0 is zero
    constexpr explicit operator bool() const { return true; }

private:
    template <typename T>
    T to_integer() const
    {
        return detail::to_integer<T>(detail::from_e8m0_bits(__x), [](double v) { return std::trunc(v); });
    }
#endif
};

struct alignas(2) __nv_fp8x2_e8m0 {
    __nv_fp8x2_storage_t __x;

    __nv_fp8x2_e8m0() = default;

#ifndef __CUDA_NO_FP8_CONVERSIONS__
    constexpr explicit __nv_fp8x2_e8m0(__half2 f) :
        __x(__nv_cvt_double2_to_e8m0x2({detail::to_double(f.x), detail::to_double(f.y)}, __NV_SATFINITE, cudaRoundPosInf))
    {
    }
    constexpr explicit __nv_fp8x2_e8m0(__nv_bfloat162 f) :
        __x(__nv_cvt_bfloat162raw_to_e8m0x2(f, __NV_SATFINITE, cudaRoundPosInf))
    {
    }
    constexpr explicit __nv_fp8x2_e8m0(float2 f) : __x(__nv_cvt_float2_to_e8m0x2(f, __NV_SATFINITE, cudaRoundPosInf)) {}
    constexpr explicit __nv_fp8x2_e8m0(double2 f) : __x(__nv_cvt_double2_to_e8m0x2(f, __NV_SATFINITE, cudaRoundPosInf)) {}
#endif

#ifndef __CUDA_NO_FP8_CONVERSION_OPERATORS__
    constexpr explicit operator __nv_bfloat162() const { return __nv_cvt_e8m0x2_to_bf162raw(__x); }
    constexpr explicit operator float2() const
    {
        return {float(detail::from_e8m0_bits(__nv_fp8_storage_t(__x))), float(detail::from_e8m0_bits(__nv_fp8_storage_t(__x >> 8)))};
    }
    constexpr explicit operator __half2() const
    {
        return {detail::to_half(detail::from_e8m0_bits(__nv_fp8_storage_t(__x))),
                detail::to_half(detail::from_e8m0_bits(__nv_fp8_storage_t(__x >> 8)))};
    }
#endif
};

struct alignas(4) __nv_fp8x4_e8m0 {
    __nv_fp8x4_storage_t __x;

    __nv_fp8x4_e8m0() = default;

#ifndef __CUDA_NO_FP8_CONVERSIONS__
    constexpr explicit __nv_fp8x4_e8m0(__half2 lo, __half2 hi) :
        __x(pack(__nv_fp8x2_e8m0(lo).__x, __nv_fp8x2_e8m0(hi).__x))
    {
    }
    constexpr explicit __nv_fp8x4_e8m0(__nv_bfloat162 lo, __nv_bfloat162 hi) :
        __x(pack(__nv_fp8x2_e8m0(lo).__x, __nv_fp8x2_e8m0(hi).__x))
    {
    }
    constexpr explicit __nv_fp8x4_e8m0(float4 f) :
        __x(pack(__nv_cvt_float2_to_e8m0x2({f.x, f.y}, __NV_SATFINITE, cudaRoundPosInf),
                 __nv_cvt_float2_to_e8m0x2({f.z, f.w}, __NV_SATFINITE, cudaRoundPosInf)))
    {
    }
    constexpr explicit __nv_fp8x4_e8m0(double4 f) :
        __x(pack(__nv_cvt_double2_to_e8m0x2({f.x, f.y}, __NV_SATFINITE, cudaRoundPosInf),
                 __nv_cvt_double2_to_e8m0x2({f.z, f.w}, __NV_SATFINITE, cudaRoundPosInf)))
    {
    }
#endif

#ifndef __CUDA_NO_FP8_CONVERSION_OPERATORS__
    constexpr explicit operator float4() const
    {
        return {float(detail::from_e8m0_bits(__nv_fp8_storage_t(__x))), float(detail::from_e8m0_bits(__nv_fp8_storage_t(__x >> 8))),
                float(detail::from_e8m0_bits(__nv_fp8_storage_t(__x >> 16))), float(detail::from_e8m0_bits(__nv_fp8_storage_t(__x >> 24)))};
    }
#endif

private:
    static constexpr __nv_fp8x4_storage_t pack(__nv_fp8x2_storage_t lo, __nv_fp8x2_storage_t hi)
    {
        return __nv_fp8x4_storage_t(lo) | __nv_fp8x4_storage_t(hi) << 16;
    }
};

}

}
