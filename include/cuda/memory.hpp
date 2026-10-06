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

#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>

#include "error.hpp"
#include "types.hpp"

namespace cuda4cpu {

//
// Device memory is host memory: every allocation is ordinary memory that both
// host code and kernels can access, and copies are memcpy. Asynchronous
// operations complete before they return.
//

inline namespace cuda_api {

enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
};

enum : unsigned int {
    cudaHostAllocDefault       = 0x00,
    cudaHostAllocPortable      = 0x01,
    cudaHostAllocMapped        = 0x02,
    cudaHostAllocWriteCombined = 0x04,
    cudaMemAttachGlobal        = 0x01,
    cudaMemAttachHost          = 0x02,
    cudaMemAttachSingle        = 0x04,
    cudaHostRegisterDefault    = 0x00,
    cudaHostRegisterPortable   = 0x01,
    cudaHostRegisterMapped     = 0x02,
    cudaHostRegisterIoMemory   = 0x04,
    cudaHostRegisterReadOnly   = 0x08
};

//! Where memory lives, for the unified memory hints
enum cudaMemLocationType {
    cudaMemLocationTypeInvalid         = 0,
    cudaMemLocationTypeDevice          = 1,
    cudaMemLocationTypeHost            = 2,
    cudaMemLocationTypeHostNuma        = 3,
    cudaMemLocationTypeHostNumaCurrent = 4
};

struct cudaMemLocation {
    cudaMemLocationType type;
    int id;
};

enum cudaMemoryAdvise {
    cudaMemAdviseSetReadMostly          = 1,
    cudaMemAdviseUnsetReadMostly        = 2,
    cudaMemAdviseSetPreferredLocation   = 3,
    cudaMemAdviseUnsetPreferredLocation = 4,
    cudaMemAdviseSetAccessedBy          = 5,
    cudaMemAdviseUnsetAccessedBy        = 6
};

//! cudaCpuDeviceId names the host as a location
inline constexpr int cudaCpuDeviceId = -1;

//! Memory pools, for stream-ordered allocation. Allocations come from the
//! process heap, so there is one pool and its attributes have no effect.
struct cudaMemPool__ {};
using cudaMemPool_t = cudaMemPool__ *;

enum cudaMemPoolAttr {
    cudaMemPoolReuseFollowEventDependencies   = 1,
    cudaMemPoolReuseAllowOpportunistic        = 2,
    cudaMemPoolReuseAllowInternalDependencies = 3,
    cudaMemPoolAttrReleaseThreshold           = 4,
    cudaMemPoolAttrReservedMemCurrent         = 5,
    cudaMemPoolAttrReservedMemHigh            = 6,
    cudaMemPoolAttrUsedMemCurrent             = 7,
    cudaMemPoolAttrUsedMemHigh                = 8
};

}

namespace detail {

//! Alignment of every allocation, as guaranteed by cudaMalloc
inline constexpr size_t allocation_alignment = 256;

//! Allocates size bytes aligned to allocation_alignment. Defined in the
//! library, where the compiler can't elide the allocation: it may assume that
//! an inline allocation whose result is never dereferenced succeeds.
void *allocate_aligned(size_t size);

inline cudaError_t allocate(void **ptr, size_t size)
{
    if (ptr == nullptr)
        return record_error(cudaErrorInvalidValue);
    if (size == 0) {
        *ptr = nullptr;
        return cudaSuccess;
    }

    void *p = allocate_aligned(size);
    if (p == nullptr)
        return record_error(cudaErrorMemoryAllocation);

    *ptr = p;
    return cudaSuccess;
}

inline cudaError_t copy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault)
        return record_error(cudaErrorInvalidMemcpyDirection);
    if (count > 0 && (dst == nullptr || src == nullptr))
        return record_error(cudaErrorInvalidValue);

    std::memmove(dst, src, count);
    return cudaSuccess;
}

}

inline namespace cuda_api {

static inline
cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    return detail::allocate(devPtr, size);
}

static inline
cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    return detail::allocate(ptr, size);
}

static inline
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int /* flags */)
{
    return detail::allocate(pHost, size);
}

static inline
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int /* flags */ = cudaMemAttachGlobal)
{
    return detail::allocate(devPtr, size);
}

// Typed overloads, so that cudaMalloc(&ptr, size) works without a cast, as in CUDA

template <typename T>
static inline
cudaError_t cudaMalloc(T **devPtr, size_t size)
{
    return detail::allocate(reinterpret_cast<void **>(devPtr), size);
}

template <typename T>
static inline
cudaError_t cudaMallocHost(T **ptr, size_t size, unsigned int /* flags */ = 0)
{
    return detail::allocate(reinterpret_cast<void **>(ptr), size);
}

template <typename T>
static inline
cudaError_t cudaHostAlloc(T **pHost, size_t size, unsigned int /* flags */)
{
    return detail::allocate(reinterpret_cast<void **>(pHost), size);
}

template <typename T>
static inline
cudaError_t cudaMallocManaged(T **devPtr, size_t size, unsigned int /* flags */ = cudaMemAttachGlobal)
{
    return detail::allocate(reinterpret_cast<void **>(devPtr), size);
}

static inline
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t /* stream */)
{
    return detail::allocate(devPtr, size);
}

template <typename T>
static inline
cudaError_t cudaMallocAsync(T **devPtr, size_t size, cudaStream_t /* stream */)
{
    return detail::allocate(reinterpret_cast<void **>(devPtr), size);
}

static inline
cudaError_t cudaMallocFromPoolAsync(void **devPtr, size_t size, cudaMemPool_t /* pool */,
                                    cudaStream_t /* stream */)
{
    return detail::allocate(devPtr, size);
}

static inline
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t /* stream */)
{
    std::free(devPtr);
    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device)
{
    static cudaMemPool__ pool;
    if (device != 0)
        return detail::record_error(cudaErrorInvalidDevice);
    *memPool = &pool;
    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device)
{
    return cudaDeviceGetDefaultMemPool(memPool, device);
}

static inline
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t /* memPool */, cudaMemPoolAttr /* attr */,
                                    void * /* value */)
{
    return cudaSuccess;
}

static inline
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t /* memPool */, cudaMemPoolAttr attr, void *value)
{
    // The reuse attributes are ints; the memory sizes are 64-bit
    if (attr <= cudaMemPoolReuseAllowInternalDependencies)
        *static_cast<int *>(value) = 1;
    else
        *static_cast<unsigned long long *>(value) = 0;
    return cudaSuccess;
}

static inline
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t /* memPool */, size_t /* minBytesToKeep */)
{
    return cudaSuccess;
}

//
// Unified memory hints. All memory is host memory, so prefetching and advice
// have nothing to do. Both CUDA's older signatures (with a device number) and
// the ones with a cudaMemLocation are accepted.
//

static inline
cudaError_t cudaMemPrefetchAsync(const void * /* devPtr */, size_t /* count */, int /* dstDevice */,
                                 cudaStream_t /* stream */ = nullptr)
{
    return cudaSuccess;
}

static inline
cudaError_t cudaMemPrefetchAsync(const void * /* devPtr */, size_t /* count */, cudaMemLocation /* location */,
                                 unsigned int /* flags */, cudaStream_t /* stream */ = nullptr)
{
    return cudaSuccess;
}

static inline
cudaError_t cudaStreamAttachMemAsync(cudaStream_t /* stream */, void * /* devPtr */, size_t /* length */ = 0,
                                     unsigned int /* flags */ = cudaMemAttachSingle)
{
    return cudaSuccess;
}

static inline
cudaError_t cudaMemAdvise(const void * /* devPtr */, size_t /* count */, cudaMemoryAdvise /* advice */,
                          int /* device */)
{
    return cudaSuccess;
}

static inline
cudaError_t cudaMemAdvise(const void * /* devPtr */, size_t /* count */, cudaMemoryAdvise /* advice */,
                          cudaMemLocation /* location */)
{
    return cudaSuccess;
}

//! Host memory is device memory: registering it changes nothing
static inline
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int /* flags */)
{
    return size > 0 && ptr == nullptr ? detail::record_error(cudaErrorInvalidValue) : cudaSuccess;
}

static inline
cudaError_t cudaHostUnregister(void * /* ptr */)
{
    return cudaSuccess;
}

//! The device address of mapped host memory is its host address
static inline
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int /* flags */)
{
    *pDevice = pHost;
    return cudaSuccess;
}

template <typename T>
static inline
cudaError_t cudaHostGetDevicePointer(T **pDevice, void *pHost, unsigned int /* flags */)
{
    *pDevice = static_cast<T *>(pHost);
    return cudaSuccess;
}

static inline
cudaError_t cudaFree(void *devPtr)
{
    std::free(devPtr);
    return cudaSuccess;
}

static inline
cudaError_t cudaFreeHost(void *ptr)
{
    std::free(ptr);
    return cudaSuccess;
}

static inline
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    return detail::copy(dst, src, count, kind);
}

static inline
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                            cudaStream_t /* stream */ = nullptr)
{
    return detail::copy(dst, src, count, kind);
}

static inline
cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
    if (count > 0 && devPtr == nullptr)
        return detail::record_error(cudaErrorInvalidValue);

    std::memset(devPtr, value, count);
    return cudaSuccess;
}

static inline
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t /* stream */ = nullptr)
{
    return cudaMemset(devPtr, value, count);
}

//
// Symbols (__constant__ and __device__ variables) are passed by name, as in
// CUDA: cudaMemcpyToSymbol(coeffs, host_coeffs, sizeof(coeffs)). Passing the
// address of the symbol is rejected at compile time.
//

template <typename T>
static inline
cudaError_t cudaMemcpyToSymbol(T &&symbol, const void *src, size_t count, size_t offset = 0,
                               cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
    static_assert(std::is_lvalue_reference_v<T>,
                  "cudaMemcpyToSymbol takes the symbol itself, as in CUDA, not its address");
    if (offset + count > sizeof(symbol))
        return detail::record_error(cudaErrorInvalidValue);

    return detail::copy(reinterpret_cast<char *>(std::addressof(symbol)) + offset, src, count, kind);
}

template <typename T>
static inline
cudaError_t cudaMemcpyFromSymbol(void *dst, T &&symbol, size_t count, size_t offset = 0,
                                 cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
{
    static_assert(std::is_lvalue_reference_v<T>,
                  "cudaMemcpyFromSymbol takes the symbol itself, as in CUDA, not its address");
    if (offset + count > sizeof(symbol))
        return detail::record_error(cudaErrorInvalidValue);

    return detail::copy(dst, reinterpret_cast<const char *>(std::addressof(symbol)) + offset, count, kind);
}

template <typename T>
static inline
cudaError_t cudaGetSymbolAddress(void **devPtr, T &&symbol)
{
    static_assert(std::is_lvalue_reference_v<T>,
                  "cudaGetSymbolAddress takes the symbol itself, as in CUDA, not its address");
    *devPtr = const_cast<void *>(static_cast<const void *>(std::addressof(symbol)));
    return cudaSuccess;
}

template <typename T>
static inline
cudaError_t cudaGetSymbolSize(size_t *size, T &&symbol)
{
    static_assert(std::is_lvalue_reference_v<T>,
                  "cudaGetSymbolSize takes the symbol itself, as in CUDA, not its address");
    *size = sizeof(symbol);
    return cudaSuccess;
}

}

}
