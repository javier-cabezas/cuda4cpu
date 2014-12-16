/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
 */

#pragma once

#include <chrono>

namespace cuda4cpu {

using cudaError_t = int;

struct dim3 {
    unsigned x, y ,z;

    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) :
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
    } n##4;

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
