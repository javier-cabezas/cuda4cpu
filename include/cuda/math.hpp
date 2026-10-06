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

#pragma once

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>

// glibc 2.41 and later declare the C23 rsqrt functions in the global namespace
#if defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 41) && __GLIBC_USE(IEC_60559_FUNCS_EXT_C23)
#define CUDA4CPU_LIBC_HAS_RSQRT 1
#endif
#endif

//
// CUDA's fast single-precision intrinsics, computed with the full-precision
// functions. glibc's <math.h> declares functions with several of these names
// (__expf, __sinf, ...) without exporting them, so calls would fail to link.
// Defining all of them here as inline C functions completes those
// declarations, and provides them where the C library doesn't declare them.
//
extern "C" {

inline float __expf(float x) noexcept { return std::exp(x); }
inline float __exp10f(float x) noexcept { return std::pow(10.0f, x); }
inline float __logf(float x) noexcept { return std::log(x); }
inline float __log2f(float x) noexcept { return std::log2(x); }
inline float __log10f(float x) noexcept { return std::log10(x); }
inline float __sinf(float x) noexcept { return std::sin(x); }
inline float __cosf(float x) noexcept { return std::cos(x); }
inline float __tanf(float x) noexcept { return std::tan(x); }
inline float __powf(float x, float y) noexcept { return std::pow(x, y); }

inline void __sincosf(float x, float *sptr, float *cptr) noexcept
{
    *sptr = std::sin(x);
    *cptr = std::cos(x);
}

}

namespace cuda4cpu {

inline namespace cuda_api {

//
// CUDA math functions that have no equivalent in the C++ standard library
//

#ifndef CUDA4CPU_LIBC_HAS_RSQRT
//! Reciprocal of the square root
inline float rsqrtf(float x)
{
    return 1.0f / std::sqrt(x);
}

//! Reciprocal of the square root
inline double rsqrt(double x)
{
    return 1.0 / std::sqrt(x);
}
#endif

inline float __fdividef(float x, float y)
{
    return x / y;
}

//! Clamps x to [0, 1]; NaN becomes 0
inline float __saturatef(float x)
{
    return x >= 1.0f ? 1.0f : (x > 0.0f ? x : 0.0f);
}

//
// Integer intrinsics
//

inline int __popc(unsigned int x) { return std::popcount(x); }
inline int __popcll(unsigned long long x) { return std::popcount(x); }

//! Number of leading zero bits
inline int __clz(int x) { return std::countl_zero(static_cast<unsigned int>(x)); }
inline int __clzll(long long x) { return std::countl_zero(static_cast<unsigned long long>(x)); }

//! Position of the least significant set bit, starting at 1, or 0 if x is 0
inline int __ffs(int x) { return x == 0 ? 0 : std::countr_zero(static_cast<unsigned int>(x)) + 1; }
inline int __ffsll(long long x) { return x == 0 ? 0 : std::countr_zero(static_cast<unsigned long long>(x)) + 1; }

//! Reverses the bits
inline unsigned int __brev(unsigned int x)
{
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    return std::byteswap(x);
}

inline unsigned long long __brevll(unsigned long long x)
{
    return (static_cast<unsigned long long>(__brev(static_cast<unsigned int>(x))) << 32) |
           __brev(static_cast<unsigned int>(x >> 32));
}

//! Product of the low 24 bits of x and y (sign-extended, or unsigned)
inline int __mul24(int x, int y)
{
    return int(int64_t(int32_t(uint32_t(x) << 8) >> 8) * int64_t(int32_t(uint32_t(y) << 8) >> 8));
}

inline unsigned int __umul24(unsigned int x, unsigned int y)
{
    return (x & 0xffffffu) * (y & 0xffffffu);
}

}

namespace detail {

//! Evaluates op(args...) with the floating-point rounding mode set to Mode.
//! The operands and the result go through volatile variables, so that the
//! compiler doesn't move the operation outside the change of mode.
template <int Mode, typename T, typename Op, typename... Args>
inline T rounded(Op op, Args... args)
{
    const int previous = std::fegetround();
    std::fesetround(Mode);
    volatile T result = op(static_cast<const volatile Args &>(args)...);
    std::fesetround(previous);
    return result;
}

}

inline namespace cuda_api {

//
// Arithmetic with an explicit rounding mode: _rn (to nearest even), _rz
// (toward zero), _ru (up) and _rd (down), as in CUDA
//

#define CUDA4CPU_ROUNDED(name, T, op, PARAMS, ARGS)                                         \
    inline T name##_rn PARAMS { return detail::rounded<FE_TONEAREST, T>(op, ARGS); }        \
    inline T name##_rz PARAMS { return detail::rounded<FE_TOWARDZERO, T>(op, ARGS); }       \
    inline T name##_ru PARAMS { return detail::rounded<FE_UPWARD, T>(op, ARGS); }           \
    inline T name##_rd PARAMS { return detail::rounded<FE_DOWNWARD, T>(op, ARGS); }

#define CUDA4CPU_X x
#define CUDA4CPU_XY x, y
#define CUDA4CPU_XYZ x, y, z
CUDA4CPU_ROUNDED(__fadd, float, ([](float a, float b) { return a + b; }), (float x, float y), CUDA4CPU_XY)
CUDA4CPU_ROUNDED(__fsub, float, ([](float a, float b) { return a - b; }), (float x, float y), CUDA4CPU_XY)
CUDA4CPU_ROUNDED(__fmul, float, ([](float a, float b) { return a * b; }), (float x, float y), CUDA4CPU_XY)
CUDA4CPU_ROUNDED(__fdiv, float, ([](float a, float b) { return a / b; }), (float x, float y), CUDA4CPU_XY)
CUDA4CPU_ROUNDED(__dadd, double, ([](double a, double b) { return a + b; }), (double x, double y), CUDA4CPU_XY)
CUDA4CPU_ROUNDED(__dsub, double, ([](double a, double b) { return a - b; }), (double x, double y), CUDA4CPU_XY)
CUDA4CPU_ROUNDED(__dmul, double, ([](double a, double b) { return a * b; }), (double x, double y), CUDA4CPU_XY)
CUDA4CPU_ROUNDED(__ddiv, double, ([](double a, double b) { return a / b; }), (double x, double y), CUDA4CPU_XY)
CUDA4CPU_ROUNDED(__frcp, float, ([](float a) { return 1.0f / a; }), (float x), CUDA4CPU_X)
CUDA4CPU_ROUNDED(__fsqrt, float, ([](float a) { return std::sqrt(a); }), (float x), CUDA4CPU_X)
CUDA4CPU_ROUNDED(__drcp, double, ([](double a) { return 1.0 / a; }), (double x), CUDA4CPU_X)
CUDA4CPU_ROUNDED(__dsqrt, double, ([](double a) { return std::sqrt(a); }), (double x), CUDA4CPU_X)
CUDA4CPU_ROUNDED(__fmaf, float, ([](float a, float b, float c) { return std::fma(a, b, c); }), (float x, float y, float z), CUDA4CPU_XYZ)
CUDA4CPU_ROUNDED(__fma, double, ([](double a, double b, double c) { return std::fma(a, b, c); }), (double x, double y, double z), CUDA4CPU_XYZ)
#undef CUDA4CPU_X
#undef CUDA4CPU_XY
#undef CUDA4CPU_XYZ
#undef CUDA4CPU_ROUNDED

//
// min and max, as CUDA declares them for host and device code. The names are
// parenthesized so that min/max macros defined by the program don't break the
// declarations.
//

#define CUDA4CPU_MINMAX(R, A, B)                                                   \
    inline R (min)(A a, B b) { R x = static_cast<R>(a), y = static_cast<R>(b); return x < y ? x : y; } \
    inline R (max)(A a, B b) { R x = static_cast<R>(a), y = static_cast<R>(b); return x > y ? x : y; }

CUDA4CPU_MINMAX(int, int, int)
CUDA4CPU_MINMAX(unsigned int, unsigned int, unsigned int)
CUDA4CPU_MINMAX(unsigned int, int, unsigned int)
CUDA4CPU_MINMAX(unsigned int, unsigned int, int)
CUDA4CPU_MINMAX(long, long, long)
CUDA4CPU_MINMAX(unsigned long, unsigned long, unsigned long)
CUDA4CPU_MINMAX(long long, long long, long long)
CUDA4CPU_MINMAX(unsigned long long, unsigned long long, unsigned long long)
CUDA4CPU_MINMAX(unsigned long long, long long, unsigned long long)
CUDA4CPU_MINMAX(unsigned long long, unsigned long long, long long)
CUDA4CPU_MINMAX(float, float, float)
CUDA4CPU_MINMAX(double, double, double)
CUDA4CPU_MINMAX(double, float, double)
CUDA4CPU_MINMAX(double, double, float)

#undef CUDA4CPU_MINMAX

//
// Reinterpreting bits
//

inline float __int_as_float(int x) { return std::bit_cast<float>(x); }
inline int __float_as_int(float x) { return std::bit_cast<int>(x); }
inline float __uint_as_float(unsigned int x) { return std::bit_cast<float>(x); }
inline unsigned int __float_as_uint(float x) { return std::bit_cast<unsigned int>(x); }
inline double __longlong_as_double(long long x) { return std::bit_cast<double>(x); }
inline long long __double_as_longlong(double x) { return std::bit_cast<long long>(x); }

inline double __hiloint2double(int hi, int lo)
{
    return std::bit_cast<double>((uint64_t(uint32_t(hi)) << 32) | uint32_t(lo));
}

inline int __double2hiint(double x) { return int(std::bit_cast<uint64_t>(x) >> 32); }
inline int __double2loint(double x) { return int(uint32_t(std::bit_cast<uint64_t>(x))); }

}

namespace detail {

//! Converts x to the integer type I, rounded as round says, and saturated to
//! the range of I as CUDA does (NaN becomes 0)
template <typename I, typename F, typename Round>
inline I to_integer(F x, Round round)
{
    if (std::isnan(x))
        return 0;
    F r = round(x);
    if (r <= F(std::numeric_limits<I>::min()))
        return std::numeric_limits<I>::min();
    if (r >= F(std::numeric_limits<I>::max()))
        return std::numeric_limits<I>::max();
    return I(r);
}

}

inline namespace cuda_api {

//
// Conversions with an explicit rounding mode
//

#define CUDA4CPU_TO_INTEGER(name, F, I)                                                              \
    inline I name##_rn(F x) { return detail::to_integer<I>(x, [](F v) { return std::nearbyint(v); }); } \
    inline I name##_rz(F x) { return detail::to_integer<I>(x, [](F v) { return std::trunc(v); }); }     \
    inline I name##_ru(F x) { return detail::to_integer<I>(x, [](F v) { return std::ceil(v); }); }      \
    inline I name##_rd(F x) { return detail::to_integer<I>(x, [](F v) { return std::floor(v); }); }

CUDA4CPU_TO_INTEGER(__float2int, float, int)
CUDA4CPU_TO_INTEGER(__float2uint, float, unsigned int)
CUDA4CPU_TO_INTEGER(__float2ll, float, long long)
CUDA4CPU_TO_INTEGER(__float2ull, float, unsigned long long)
CUDA4CPU_TO_INTEGER(__double2int, double, int)
CUDA4CPU_TO_INTEGER(__double2uint, double, unsigned int)
CUDA4CPU_TO_INTEGER(__double2ll, double, long long)
CUDA4CPU_TO_INTEGER(__double2ull, double, unsigned long long)

#undef CUDA4CPU_TO_INTEGER

#define CUDA4CPU_TO_FLOAT(name, I, F)                                                               \
    inline F name##_rn(I x) { return detail::rounded<FE_TONEAREST, F>([](I v) { return F(v); }, x); }  \
    inline F name##_rz(I x) { return detail::rounded<FE_TOWARDZERO, F>([](I v) { return F(v); }, x); } \
    inline F name##_ru(I x) { return detail::rounded<FE_UPWARD, F>([](I v) { return F(v); }, x); }     \
    inline F name##_rd(I x) { return detail::rounded<FE_DOWNWARD, F>([](I v) { return F(v); }, x); }

CUDA4CPU_TO_FLOAT(__int2float, int, float)
CUDA4CPU_TO_FLOAT(__uint2float, unsigned int, float)
CUDA4CPU_TO_FLOAT(__ll2float, long long, float)
CUDA4CPU_TO_FLOAT(__ull2float, unsigned long long, float)
CUDA4CPU_TO_FLOAT(__ll2double, long long, double)
CUDA4CPU_TO_FLOAT(__ull2double, unsigned long long, double)
CUDA4CPU_TO_FLOAT(__double2float, double, float)

#undef CUDA4CPU_TO_FLOAT

//! Conversions that are always exact
inline double __int2double_rn(int x) { return double(x); }
inline double __uint2double_rn(unsigned int x) { return double(x); }

//! Reciprocal square root, rounded to nearest
inline float __frsqrt_rn(float x) { return detail::rounded<FE_TONEAREST, float>([](float v) { return 1.0f / std::sqrt(v); }, x); }

//! Read-only data cache load; an ordinary load on the CPU
template <typename T>
inline T __ldg(const T *ptr)
{
    return *ptr;
}

}

}
