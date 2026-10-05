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
#include <cmath>
#include <cstdint>

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

//! Read-only data cache load; an ordinary load on the CPU
template <typename T>
inline T __ldg(const T *ptr)
{
    return *ptr;
}

}

}
