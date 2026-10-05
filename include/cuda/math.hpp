/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
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

#include <cmath>

// glibc 2.41 and later declare the C23 rsqrt functions in the global namespace
#if defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 41) && __GLIBC_USE(IEC_60559_FUNCS_EXT_C23)
#define CUDA4CPU_LIBC_HAS_RSQRT 1
#endif
#endif

namespace cuda4cpu {

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

}
