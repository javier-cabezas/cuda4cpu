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

#include <cstring>

#include "types.hpp"

namespace cuda4cpu {

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

static inline
cudaError_t cudaMallocHost(void **ptr, size_t count)
{
    void *tmp = std::malloc(count);
    if (tmp != nullptr)
        *ptr = tmp;

    return 0;
}

static inline
cudaError_t cudaHostAlloc(void **ptr, size_t count, unsigned int /*flags*/)
{
    return cudaMallocHost(ptr, count);
}

static inline
cudaError_t cudaMalloc(void **ptr, size_t count)
{
    return cudaMallocHost(ptr, count);
}

static inline
cudaError_t cudaFreeHost(void *ptr)
{
    std::free(ptr);

    return 0;
}

static inline
cudaError_t cudaFree(void *ptr)
{
    return cudaFreeHost(ptr);
}

}
