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
cudaError_t cudaMemcpyToSymbol(void *dst, const void *src, size_t count, size_t offset = 0, cudaMemcpyKind /* kind */ = cudaMemcpyHostToDevice)
{
    std::memcpy(static_cast<char *>(dst) + offset, src, count);

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
