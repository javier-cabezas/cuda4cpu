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

#include <chrono>

namespace cuda4cpu {

// The CUDA API lives in the inline namespace cuda_api: code that includes
// cuda4cpu.hpp sees it as cuda4cpu::name, and cuda_runtime.h brings exactly
// these names into the global namespace, like the real header.
inline namespace cuda_api {

//! Error codes, with the same values as CUDA's
enum cudaError {
    cudaSuccess                     = 0,
    cudaErrorInvalidValue           = 1,
    cudaErrorMemoryAllocation       = 2,
    cudaErrorInitializationError    = 3,
    cudaErrorInvalidConfiguration   = 9,
    cudaErrorInvalidSymbol          = 13,
    cudaErrorInvalidDevicePointer   = 17,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorNoDevice               = 100,
    cudaErrorInvalidDevice          = 101,
    cudaErrorInvalidResourceHandle  = 400,
    cudaErrorNotReady               = 600,
    cudaErrorLaunchFailure          = 719,
    cudaErrorNotSupported           = 801,
    cudaErrorUnknown                = 999
};

using cudaError_t = cudaError;

struct dim3 {
    unsigned x, y ,z;

    constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) :
        x(x_), y(y_), z(z_)
    {}
};

#define VECTOR_TYPE(n,t) \
    typedef struct {     \
        t x;             \
    } n##1;              \
                         \
    typedef struct {     \
        t x;             \
        t y;             \
    } n##2;              \
                         \
    typedef struct {     \
        t x;             \
        t y;             \
        t z;             \
    } n##3;              \
                         \
    typedef struct {     \
        t x;             \
        t y;             \
        t z;             \
        t w;             \
    } n##4;              \
                         \
    inline n##1 make_##n##1(t x) { return {x}; }                    \
    inline n##2 make_##n##2(t x, t y) { return {x, y}; }            \
    inline n##3 make_##n##3(t x, t y, t z) { return {x, y, z}; }    \
    inline n##4 make_##n##4(t x, t y, t z, t w) { return {x, y, z, w}; }

VECTOR_TYPE(char, char)
VECTOR_TYPE(uchar, unsigned char)
VECTOR_TYPE(short, short)
VECTOR_TYPE(ushort, unsigned short)
VECTOR_TYPE(int, int)
VECTOR_TYPE(uint, unsigned)
VECTOR_TYPE(long, int long)
VECTOR_TYPE(ulong, unsigned long)
VECTOR_TYPE(longlong, int long long)
VECTOR_TYPE(ulonglong, unsigned long long)
VECTOR_TYPE(float, float)
VECTOR_TYPE(double, double)

struct cudaStream__ {
    int flags;
    int priority;
};

using cudaStream_t = cudaStream__ *;

struct cudaEvent__ {
    std::chrono::time_point<std::chrono::system_clock> tstamp;
    cudaStream_t stream;
};

using cudaEvent_t = cudaEvent__ *;

using cudaStreamCallback_t = void(*)(cudaStream_t stream, cudaError_t status, void *userData);

}

}
