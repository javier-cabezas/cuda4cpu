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

// HIP's 8-bit floats. HIP has them only on AMD GPUs, so this is AMD's API
// (amd_hip_fp8.h): the OCP formats E4M3 and E5M2, which are CUDA's, and AMD's
// FNUZ formats E4M3_FNUZ and E5M2_FNUZ (no infinity and no negative zero,
// with NaN at 0x80, and exponent biases one larger), the conversions
// __hip_cvt_*, and the classes __hip_fp8_e4m3, __hip_fp8x2_e4m3,
// __hip_fp8x4_e4m3 and their E5M2 and FNUZ counterparts.
//
// The conversions round to nearest even, on cuda_fp8.h's, with AMD's rules
// for what doesn't fit, which differ from CUDA's: a value beyond the largest
// finite one overflows even if it would round down to it, and infinity stays
// infinity in E5M2, or becomes NaN in the other formats, even with
// saturation. NaN is 0x7f with the input's sign (0x80 in FNUZ). Unlike
// CUDA's, the classes convert implicitly, saturating.

#pragma once

#include "hip_runtime_api.h"
#include "hip_bf16.h"
#include "hip_fp16.h"
#include "../cuda_fp8.h"

#define HIP_FP8_TYPE_OCP 1
#define HIP_FP8_TYPE_FNUZ 1
#define HIP_FP8_CVT_FAST_PATH 0

namespace cuda4cpu {

inline namespace hip_api {

enum __hip_fp8_interpretation_t {
    __HIP_E4M3 = 0,
    __HIP_E5M2 = 1,
    __HIP_E4M3_FNUZ = 2,
    __HIP_E5M2_FNUZ = 3,
};

enum __hip_saturation_t {
    __HIP_NOSAT = 0,
    __HIP_SATFINITE = 1,
};

typedef unsigned char __hip_fp8_storage_t;
typedef unsigned short int __hip_fp8x2_storage_t;
typedef unsigned int __hip_fp8x4_storage_t;

}

namespace detail {

//! AMD's conversion to F: rounded to nearest even, but magnitudes above the
//! largest finite value overflow first: to it with saturation, or else to
//! infinity (E5M2) or NaN. Infinity is infinity in E5M2 and NaN in the other
//! formats, even with saturation. NaN is 0x7f with x's sign, or 0x80 in
//! FNUZ.
template <float_format F>
constexpr uint8_t amd_fp8_bits(double x, bool saturate)
{
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const uint8_t sign = uint8_t((bits >> 56) & 0x80);
    const uint8_t nan = F.specials == float_specials::fnuz ? uint8_t(0x80) : uint8_t(sign | 0x7f);
    if (x != x)
        return nan;
    const double magnitude = std::bit_cast<double>(bits & 0x7fffffffffffffff);
    if (magnitude == std::bit_cast<double>(uint64_t(0x7ff0000000000000)))
        return F.specials == float_specials::ieee ? uint8_t(sign | F.infinity()) : nan;
    if (magnitude > from_bits<F>(F.max_finite())) {
        if (saturate)
            return uint8_t(sign | F.max_finite());
        return F.specials == float_specials::ieee ? uint8_t(sign | F.infinity()) : nan;
    }
    return uint8_t(to_bits<F>(x));
}

constexpr uint8_t amd_fp8_bits(double x, __hip_saturation_t saturate, __hip_fp8_interpretation_t interpretation)
{
    const bool clip = saturate == __HIP_SATFINITE;
    switch (interpretation) {
    case __HIP_E4M3:
        return amd_fp8_bits<e4m3_format>(x, clip);
    case __HIP_E5M2:
        return amd_fp8_bits<e5m2_format>(x, clip);
    case __HIP_E4M3_FNUZ:
        return amd_fp8_bits<e4m3_fnuz_format>(x, clip);
    default:
        return amd_fp8_bits<e5m2_fnuz_format>(x, clip);
    }
}

constexpr uint16_t amd_fp8_to_half_bits(uint8_t b, __hip_fp8_interpretation_t interpretation)
{
    switch (interpretation) {
    case __HIP_E4M3:
        return fp8_to_half_bits<e4m3_format>(b);
    case __HIP_E5M2:
        return fp8_to_half_bits<e5m2_format>(b);
    case __HIP_E4M3_FNUZ:
        return fp8_to_half_bits<e4m3_fnuz_format>(b);
    default:
        return fp8_to_half_bits<e5m2_fnuz_format>(b);
    }
}

}

inline namespace hip_api {

//
// Conversions. The pairs keep x (the low lane) in the low byte.
//

constexpr __hip_fp8_storage_t __hip_cvt_double_to_fp8(double d, __hip_saturation_t sat,
                                                      __hip_fp8_interpretation_t interp)
{
    return detail::amd_fp8_bits(d, sat, interp);
}

constexpr __hip_fp8x2_storage_t __hip_cvt_double2_to_fp8x2(double2 d2, __hip_saturation_t sat,
                                                           __hip_fp8_interpretation_t interp)
{
    return __hip_fp8x2_storage_t(detail::amd_fp8_bits(d2.x, sat, interp) | detail::amd_fp8_bits(d2.y, sat, interp) << 8);
}

constexpr __hip_fp8_storage_t __hip_cvt_float_to_fp8(float f, __hip_saturation_t sat,
                                                     __hip_fp8_interpretation_t interp)
{
    return detail::amd_fp8_bits(f, sat, interp);
}

constexpr __hip_fp8x2_storage_t __hip_cvt_float2_to_fp8x2(float2 f2, __hip_saturation_t sat,
                                                          __hip_fp8_interpretation_t interp)
{
    return __hip_cvt_double2_to_fp8x2({f2.x, f2.y}, sat, interp);
}

constexpr __hip_fp8_storage_t __hip_cvt_halfraw_to_fp8(__half_raw x, __hip_saturation_t sat,
                                                       __hip_fp8_interpretation_t interp)
{
    return detail::amd_fp8_bits(detail::from_half_bits(x.x), sat, interp);
}

constexpr __hip_fp8x2_storage_t __hip_cvt_halfraw2_to_fp8x2(__half2_raw x, __hip_saturation_t sat,
                                                            __hip_fp8_interpretation_t interp)
{
    return __hip_fp8x2_storage_t(__hip_cvt_halfraw_to_fp8({x.x}, sat, interp) | __hip_cvt_halfraw_to_fp8({x.y}, sat, interp) << 8);
}

constexpr __hip_fp8_storage_t __hip_cvt_bfloat16raw_to_fp8(__hip_bfloat16_raw hr, __hip_saturation_t sat,
                                                           __hip_fp8_interpretation_t interp)
{
    return detail::amd_fp8_bits(detail::from_bits<detail::bfloat16_format>(hr.x), sat, interp);
}

constexpr __hip_fp8x2_storage_t __hip_cvt_bfloat16raw2_to_fp8x2(__hip_bfloat162_raw hr, __hip_saturation_t sat,
                                                                __hip_fp8_interpretation_t interp)
{
    return __hip_fp8x2_storage_t(__hip_cvt_bfloat16raw_to_fp8({hr.x}, sat, interp) |
                                 __hip_cvt_bfloat16raw_to_fp8({hr.y}, sat, interp) << 8);
}

//! To half, which holds every fp8 value exactly (NaN as half's canonical NaN)
constexpr __half_raw __hip_cvt_fp8_to_halfraw(__hip_fp8_storage_t x, __hip_fp8_interpretation_t interp)
{
    return {detail::amd_fp8_to_half_bits(x, interp)};
}

constexpr __half2_raw __hip_cvt_fp8x2_to_halfraw2(__hip_fp8x2_storage_t x, __hip_fp8_interpretation_t interp)
{
    return {detail::amd_fp8_to_half_bits(__hip_fp8_storage_t(x), interp),
            detail::amd_fp8_to_half_bits(__hip_fp8_storage_t(x >> 8), interp)};
}

// E8M0, as in cuda_fp8.h
constexpr __hip_fp8_storage_t __hip_cvt_double_to_e8m0(double val, __hip_saturation_t saturate, hipRoundMode rounding)
{
    return detail::e8m0_bits(val, saturate == __HIP_SATFINITE, rounding);
}

constexpr __hip_fp8_storage_t __hip_cvt_float_to_e8m0(float val, __hip_saturation_t saturate, hipRoundMode rounding)
{
    return detail::e8m0_bits(val, saturate == __HIP_SATFINITE, rounding);
}

constexpr __hip_fp8_storage_t __hip_cvt_bfloat16raw_to_e8m0(__hip_bfloat16_raw hr, __hip_saturation_t saturate,
                                                            hipRoundMode rounding)
{
    return detail::e8m0_bits(detail::from_bits<detail::bfloat16_format>(hr.x), saturate == __HIP_SATFINITE, rounding);
}

constexpr __hip_bfloat16_raw __hip_cvt_e8m0_to_bf16raw(__hip_fp8_storage_t x)
{
    return __nv_cvt_e8m0_to_bf16raw(x);
}

}

namespace detail {

//! AMD's fp8 classes, in format I: implicit conversions, saturating by
//! default; to integers, truncated and saturated, with NaN as 0
template <__hip_fp8_interpretation_t I, unsigned We, unsigned Wm>
struct amd_fp8 {
    __hip_fp8_storage_t __x;

    static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
    static constexpr __hip_fp8_interpretation_t __default_interpret = I;
    static constexpr unsigned int __we = We;
    static constexpr unsigned int __wm = Wm;

    amd_fp8() = default;
    constexpr amd_fp8(long v) : amd_fp8(double(v)) {}
    constexpr amd_fp8(int v) : amd_fp8(double(v)) {}
    constexpr amd_fp8(short v) : amd_fp8(double(v)) {}
    constexpr amd_fp8(unsigned long v) : amd_fp8(double(v)) {}
    constexpr amd_fp8(unsigned int v) : amd_fp8(double(v)) {}
    constexpr amd_fp8(unsigned short v) : amd_fp8(double(v)) {}
    constexpr amd_fp8(double f) : __x(__hip_cvt_double_to_fp8(f, __default_saturation, I)) {}
    constexpr amd_fp8(float f) : __x(__hip_cvt_float_to_fp8(f, __default_saturation, I)) {}
    constexpr amd_fp8(__hip_bfloat16 f) : __x(__hip_cvt_bfloat16raw_to_fp8(f, __default_saturation, I)) {}
    constexpr amd_fp8(__half f) : __x(__hip_cvt_halfraw_to_fp8(f, __default_saturation, I)) {}

    constexpr operator __half() const { return __half_raw(__hip_cvt_fp8_to_halfraw(__x, I)); }
    constexpr operator __hip_bfloat16() const { return detail::to_bfloat16(value()); }
    constexpr operator float() const { return float(value()); }
    constexpr operator double() const { return value(); }
    //! False for zero, and in FNUZ formats, for NaN (0x80)
    constexpr operator bool() const
    {
        if (I == __HIP_E4M3_FNUZ || I == __HIP_E5M2_FNUZ)
            return __x != 0 && __x != 0x80;
        return (__x & 0x7f) != 0;
    }
    operator char() const { return to_integer<char>(); }
    operator signed char() const { return to_integer<signed char>(); }
    operator unsigned char() const { return to_integer<unsigned char>(); }
    operator short() const { return to_integer<short>(); }
    operator unsigned short() const { return to_integer<unsigned short>(); }
    operator int() const { return to_integer<int>(); }
    operator unsigned int() const { return to_integer<unsigned int>(); }
    operator long() const { return to_integer<long>(); }
    operator unsigned long() const { return to_integer<unsigned long>(); }
    operator long long() const { return to_integer<long long>(); }
    operator unsigned long long() const { return to_integer<unsigned long long>(); }

private:
    constexpr double value() const { return from_half_bits(amd_fp8_to_half_bits(__x, I)); }

    template <typename T>
    T to_integer() const
    {
        return detail::to_integer<T>(value(), [](double v) { return std::trunc(v); });
    }
};

template <__hip_fp8_interpretation_t I, unsigned We, unsigned Wm>
struct amd_fp8x2 {
    __hip_fp8x2_storage_t __x;

    static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
    static constexpr __hip_fp8_interpretation_t __default_interpret = I;
    static constexpr unsigned int __we = We;
    static constexpr unsigned int __wm = Wm;

    amd_fp8x2() = default;
    constexpr amd_fp8x2(double2 val) : __x(__hip_cvt_double2_to_fp8x2(val, __default_saturation, I)) {}
    constexpr amd_fp8x2(float2 val) : __x(__hip_cvt_float2_to_fp8x2(val, __default_saturation, I)) {}
    constexpr amd_fp8x2(__hip_bfloat162 val) : __x(__hip_cvt_bfloat16raw2_to_fp8x2(val, __default_saturation, I)) {}
    constexpr amd_fp8x2(__half2 val) : __x(__hip_cvt_halfraw2_to_fp8x2(val, __default_saturation, I)) {}

    constexpr operator __half2() const { return __half2_raw(__hip_cvt_fp8x2_to_halfraw2(__x, I)); }
    constexpr operator float2() const
    {
        const __half2_raw h = __hip_cvt_fp8x2_to_halfraw2(__x, I);
        return {float(from_half_bits(h.x)), float(from_half_bits(h.y))};
    }
};

template <__hip_fp8_interpretation_t I, unsigned We, unsigned Wm>
struct amd_fp8x4 {
    __hip_fp8x4_storage_t __x;

    static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
    static constexpr __hip_fp8_interpretation_t __default_interpret = I;
    static constexpr unsigned int __we = We;
    static constexpr unsigned int __wm = Wm;

    amd_fp8x4() = default;
    constexpr amd_fp8x4(double4 val) :
        __x(pack(__hip_cvt_double2_to_fp8x2({val.x, val.y}, __default_saturation, I),
                 __hip_cvt_double2_to_fp8x2({val.z, val.w}, __default_saturation, I)))
    {
    }
    constexpr amd_fp8x4(float4 val) :
        __x(pack(__hip_cvt_float2_to_fp8x2({val.x, val.y}, __default_saturation, I),
                 __hip_cvt_float2_to_fp8x2({val.z, val.w}, __default_saturation, I)))
    {
    }
    constexpr amd_fp8x4(__hip_bfloat162 low, __hip_bfloat162 high) :
        __x(pack(__hip_cvt_bfloat16raw2_to_fp8x2(low, __default_saturation, I),
                 __hip_cvt_bfloat16raw2_to_fp8x2(high, __default_saturation, I)))
    {
    }
    constexpr amd_fp8x4(__half2 low, __half2 high) :
        __x(pack(__hip_cvt_halfraw2_to_fp8x2(low, __default_saturation, I),
                 __hip_cvt_halfraw2_to_fp8x2(high, __default_saturation, I)))
    {
    }

    constexpr operator float4() const
    {
        const __half2_raw lo = __hip_cvt_fp8x2_to_halfraw2(__hip_fp8x2_storage_t(__x), I);
        const __half2_raw hi = __hip_cvt_fp8x2_to_halfraw2(__hip_fp8x2_storage_t(__x >> 16), I);
        return {float(from_half_bits(lo.x)), float(from_half_bits(lo.y)), float(from_half_bits(hi.x)),
                float(from_half_bits(hi.y))};
    }

private:
    static constexpr __hip_fp8x4_storage_t pack(__hip_fp8x2_storage_t lo, __hip_fp8x2_storage_t hi)
    {
        return __hip_fp8x4_storage_t(lo) | __hip_fp8x4_storage_t(hi) << 16;
    }
};

}

inline namespace hip_api {

struct __hip_fp8_e4m3 : detail::amd_fp8<__HIP_E4M3, 4, 3> {
    using amd_fp8::amd_fp8;
};

struct __hip_fp8x2_e4m3 : detail::amd_fp8x2<__HIP_E4M3, 4, 3> {
    using amd_fp8x2::amd_fp8x2;
};

struct __hip_fp8x4_e4m3 : detail::amd_fp8x4<__HIP_E4M3, 4, 3> {
    using amd_fp8x4::amd_fp8x4;
};

struct __hip_fp8_e5m2 : detail::amd_fp8<__HIP_E5M2, 5, 2> {
    using amd_fp8::amd_fp8;
};

struct __hip_fp8x2_e5m2 : detail::amd_fp8x2<__HIP_E5M2, 5, 2> {
    using amd_fp8x2::amd_fp8x2;
};

struct __hip_fp8x4_e5m2 : detail::amd_fp8x4<__HIP_E5M2, 5, 2> {
    using amd_fp8x4::amd_fp8x4;
};

struct __hip_fp8_e4m3_fnuz : detail::amd_fp8<__HIP_E4M3_FNUZ, 4, 3> {
    using amd_fp8::amd_fp8;
};

struct __hip_fp8x2_e4m3_fnuz : detail::amd_fp8x2<__HIP_E4M3_FNUZ, 4, 3> {
    using amd_fp8x2::amd_fp8x2;
};

struct __hip_fp8x4_e4m3_fnuz : detail::amd_fp8x4<__HIP_E4M3_FNUZ, 4, 3> {
    using amd_fp8x4::amd_fp8x4;
};

struct __hip_fp8_e5m2_fnuz : detail::amd_fp8<__HIP_E5M2_FNUZ, 5, 2> {
    using amd_fp8::amd_fp8;
};

struct __hip_fp8x2_e5m2_fnuz : detail::amd_fp8x2<__HIP_E5M2_FNUZ, 5, 2> {
    using amd_fp8x2::amd_fp8x2;
};

struct __hip_fp8x4_e5m2_fnuz : detail::amd_fp8x4<__HIP_E5M2_FNUZ, 5, 2> {
    using amd_fp8x4::amd_fp8x4;
};

}

}
