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

#ifndef CUDA4CPU_DEFINES_HPP_
#define CUDA4CPU_DEFINES_HPP_

#include <cstddef>
#include <cstdlib>
#include <cstring>

//
// CUDA keyword overrides
//
#ifdef __global__
#undef __global__
#endif

#ifdef __device__
#undef __device__
#endif

#ifdef __host__
#undef __host__
#endif

#ifdef __shared__
#undef __shared__
#endif

#ifdef __constant__
#undef __constant__
#endif

#define __global__
#define __shared__ static thread_local
#define __syncthreads cuda4cpu::thread_block::syncthreads

#define threadIdx cuda4cpu::thread_block::get_thread()
#define blockIdx  cuda4cpu::thread_block::get_block()

#define blockDim cuda4cpu::thread_block::get_block_dim()
#define gridDim  cuda4cpu::thread_block::get_grid_dim()

namespace cuda4cpu {
using cudaError_t = int;

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault
};

static inline
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind)
{
    std::memcpy(dst, src, count);

    return 0;
}

template <typename T>
static inline
cudaError_t cudaMallocHost(T **ptr, size_t count)
{
    T *tmp = (T *) std::malloc(count);
    if (tmp != nullptr)
        *ptr = tmp;

    return 0;
}

template <typename T>
static inline
cudaError_t cudaMalloc(T **ptr, size_t count)
{
    return cudaMallocHost(ptr, count);
}

template <typename T>
static inline
cudaError_t cudaFreeHost(T *ptr)
{
    std::free(ptr);

    return 0;
}

template <typename T>
static inline
cudaError_t cudaFree(T *ptr)
{
    return cudaFreeHost(ptr);
}
}

#endif // CUDA4CPU_DEFINES_HPP_
