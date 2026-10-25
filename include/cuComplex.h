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
// CUDA's complex numbers, as in CUDA's cuComplex.h: cuFloatComplex
// (cuComplex) and cuDoubleComplex are float2 and double2, x the real part and
// y the imaginary one, with functions instead of operators. Each function
// computes as CUDA's does: division and the absolute value scale their
// operands, which avoids overflow and underflow in intermediate results, and
// multiplication doesn't.
//

#pragma once

#include <cmath>
#include <limits>

#include "cuda_runtime.h"

// The include guard of CUDA's own header, which code tests
#ifndef CU_COMPLEX_H_
#define CU_COMPLEX_H_
#endif

namespace cuda4cpu {

namespace detail {

//! x / y, scaled by |y.x| + |y.y| so that neither y's squared magnitude nor
//! the products overflow or underflow
template <typename C, typename T>
constexpr C complex_divide(C x, C y)
{
    T s = std::fabs(y.x) + std::fabs(y.y);
    T oos = T(1) / s;
    const T ars = x.x * oos, ais = x.y * oos, brs = y.x * oos, bis = y.y * oos;
    s = brs * brs + bis * bis;
    oos = T(1) / s;
    return {(ars * brs + ais * bis) * oos, (ais * brs - ars * bis) * oos};
}

//! |x|, as v * sqrt(1 + (w / v)^2) for the larger and smaller magnitudes of
//! the parts, which can't overflow unless the result does
template <typename C, typename T>
inline T complex_abs(C x)
{
    const T a = std::fabs(x.x), b = std::fabs(x.y);
    const T v = a > b ? a : b, w = a > b ? b : a;
    if (v == 0 || v > std::numeric_limits<T>::max() || w > std::numeric_limits<T>::max())
        return v + w;
    const T t = w / v;
    return v * std::sqrt(T(1) + t * t);
}

//! x * y + d, with the real part's products added in CUDA's order
template <typename C, typename T>
constexpr C complex_fma(C x, C y, C d)
{
    T real = x.x * y.x + d.x;
    T imag = x.x * y.y + d.y;
    real = -(x.y * y.y) + real;
    imag = x.y * y.x + imag;
    return {real, imag};
}

}

inline namespace cuda_api {

typedef float2 cuFloatComplex;
typedef double2 cuDoubleComplex;
typedef cuFloatComplex cuComplex;

constexpr float cuCrealf(cuFloatComplex x) { return x.x; }
constexpr float cuCimagf(cuFloatComplex x) { return x.y; }
constexpr cuFloatComplex make_cuFloatComplex(float r, float i) { return {r, i}; }
constexpr cuComplex make_cuComplex(float x, float y) { return {x, y}; }
constexpr cuFloatComplex cuConjf(cuFloatComplex x) { return {x.x, -x.y}; }
constexpr cuFloatComplex cuCaddf(cuFloatComplex x, cuFloatComplex y) { return {x.x + y.x, x.y + y.y}; }
constexpr cuFloatComplex cuCsubf(cuFloatComplex x, cuFloatComplex y) { return {x.x - y.x, x.y - y.y}; }
constexpr cuFloatComplex cuCmulf(cuFloatComplex x, cuFloatComplex y) { return {x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x}; }
constexpr cuFloatComplex cuCdivf(cuFloatComplex x, cuFloatComplex y) { return detail::complex_divide<cuFloatComplex, float>(x, y); }
inline float cuCabsf(cuFloatComplex x) { return detail::complex_abs<cuFloatComplex, float>(x); }
constexpr cuComplex cuCfmaf(cuComplex x, cuComplex y, cuComplex d) { return detail::complex_fma<cuComplex, float>(x, y, d); }

constexpr double cuCreal(cuDoubleComplex x) { return x.x; }
constexpr double cuCimag(cuDoubleComplex x) { return x.y; }
constexpr cuDoubleComplex make_cuDoubleComplex(double r, double i) { return {r, i}; }
constexpr cuDoubleComplex cuConj(cuDoubleComplex x) { return {x.x, -x.y}; }
constexpr cuDoubleComplex cuCadd(cuDoubleComplex x, cuDoubleComplex y) { return {x.x + y.x, x.y + y.y}; }
constexpr cuDoubleComplex cuCsub(cuDoubleComplex x, cuDoubleComplex y) { return {x.x - y.x, x.y - y.y}; }
constexpr cuDoubleComplex cuCmul(cuDoubleComplex x, cuDoubleComplex y) { return {x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x}; }
constexpr cuDoubleComplex cuCdiv(cuDoubleComplex x, cuDoubleComplex y) { return detail::complex_divide<cuDoubleComplex, double>(x, y); }
inline double cuCabs(cuDoubleComplex x) { return detail::complex_abs<cuDoubleComplex, double>(x); }
constexpr cuDoubleComplex cuCfma(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex d) { return detail::complex_fma<cuDoubleComplex, double>(x, y, d); }

constexpr cuDoubleComplex cuComplexFloatToDouble(cuFloatComplex c) { return {double(c.x), double(c.y)}; }
constexpr cuFloatComplex cuComplexDoubleToFloat(cuDoubleComplex c) { return {float(c.x), float(c.y)}; }

}

}
