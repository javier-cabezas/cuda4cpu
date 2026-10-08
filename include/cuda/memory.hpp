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

inline namespace cuda_api {

//
// 2D and 3D memory, and the parameters of copy and memset graph nodes
//

//! CUDA arrays (texture memory) aren't implemented: copies that use them fail
struct cudaArray;
using cudaArray_t       = cudaArray *;
using cudaArray_const_t = const cudaArray *;

//! Memory of rows of xsize bytes (pitch apart) and ysize rows per slice
struct cudaPitchedPtr {
    void *ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
};

//! In bytes for linear memory (width) and elements for rows and slices
struct cudaExtent {
    size_t width;
    size_t height;
    size_t depth;
};

struct cudaPos {
    size_t x;
    size_t y;
    size_t z;
};

struct cudaMemcpy3DParms {
    cudaArray_t srcArray;
    cudaPos srcPos;
    cudaPitchedPtr srcPtr;
    cudaArray_t dstArray;
    cudaPos dstPos;
    cudaPitchedPtr dstPtr;
    cudaExtent extent;
    cudaMemcpyKind kind;
};

//! A memset of height rows of width elements of elementSize (1, 2 or 4) bytes
struct cudaMemsetParams {
    void *dst;
    size_t pitch;
    unsigned int value;
    unsigned int elementSize;
    size_t width;
    size_t height;
};

inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz)
{
    return {d, p, xsz, ysz};
}

inline cudaPos make_cudaPos(size_t x, size_t y, size_t z)
{
    return {x, y, z};
}

inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
{
    return {w, h, d};
}

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

//! A copy between pitched memory, as described by a cudaMemcpy3DParms
inline cudaError_t copy_3d(const cudaMemcpy3DParms &p)
{
    if (p.srcArray != nullptr || p.dstArray != nullptr)
        return record_error(cudaErrorNotSupported);
    if (p.kind < cudaMemcpyHostToHost || p.kind > cudaMemcpyDefault)
        return record_error(cudaErrorInvalidMemcpyDirection);
    const cudaExtent &e = p.extent;
    if (e.width == 0 || e.height == 0 || e.depth == 0)
        return cudaSuccess;
    if (p.srcPtr.ptr == nullptr || p.dstPtr.ptr == nullptr || e.width > p.srcPtr.pitch ||
        e.width > p.dstPtr.pitch)
        return record_error(cudaErrorInvalidValue);

    // Bytes from one slice to the next: a pitched pointer has ysize rows per slice
    const size_t src_slice = p.srcPtr.pitch * (p.srcPtr.ysize ? p.srcPtr.ysize : e.height);
    const size_t dst_slice = p.dstPtr.pitch * (p.dstPtr.ysize ? p.dstPtr.ysize : e.height);
    const auto *src = static_cast<const char *>(p.srcPtr.ptr) + p.srcPos.z * src_slice +
                      p.srcPos.y * p.srcPtr.pitch + p.srcPos.x;
    auto *dst = static_cast<char *>(p.dstPtr.ptr) + p.dstPos.z * dst_slice + p.dstPos.y * p.dstPtr.pitch +
                p.dstPos.x;
    for (size_t z = 0; z < e.depth; ++z) {
        for (size_t y = 0; y < e.height; ++y)
            std::memmove(dst + z * dst_slice + y * p.dstPtr.pitch, src + z * src_slice + y * p.srcPtr.pitch,
                         e.width);
    }
    return cudaSuccess;
}

//! A memset of rows of elements of 1, 2 or 4 bytes, as described by a cudaMemsetParams
inline cudaError_t fill(const cudaMemsetParams &p)
{
    if (p.elementSize != 1 && p.elementSize != 2 && p.elementSize != 4)
        return record_error(cudaErrorInvalidValue);
    if (p.width == 0 || p.height == 0)
        return cudaSuccess;
    if (p.dst == nullptr || (p.height > 1 && p.pitch < p.width * p.elementSize))
        return record_error(cudaErrorInvalidValue);

    for (size_t y = 0; y < p.height; ++y) {
        auto *row = static_cast<char *>(p.dst) + y * p.pitch;
        for (size_t x = 0; x < p.width; ++x)
            std::memcpy(row + x * p.elementSize, &p.value, p.elementSize);   // little-endian: the low bytes
    }
    return cudaSuccess;
}

// Work issued to a stream that is being captured becomes a graph node instead
// of running. Defined in the library, with the graphs.
cudaError_t capture_copy(cudaStream_t stream, const cudaMemcpy3DParms &params);
cudaError_t capture_fill(cudaStream_t stream, const cudaMemsetParams &params);
cudaError_t capture_alloc(cudaStream_t stream, void **ptr, size_t size);
cudaError_t capture_free(cudaStream_t stream, void *ptr);

//! Frees memory from cudaMalloc and the like, or from a graph allocation node
cudaError_t free_memory(void *ptr);

//! A 1D copy as 3D parameters
inline cudaMemcpy3DParms linear_copy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    cudaMemcpy3DParms p{};
    p.srcPtr = make_cudaPitchedPtr(const_cast<void *>(src), count, count, 1);
    p.dstPtr = make_cudaPitchedPtr(dst, count, count, 1);
    p.extent = make_cudaExtent(count, 1, 1);
    p.kind   = kind;
    return p;
}

//! A 2D copy as 3D parameters
inline cudaMemcpy3DParms pitched_copy(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                                      size_t height, cudaMemcpyKind kind)
{
    cudaMemcpy3DParms p{};
    p.srcPtr = make_cudaPitchedPtr(const_cast<void *>(src), spitch, width, height);
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, width, height);
    p.extent = make_cudaExtent(width, height, 1);
    p.kind   = kind;
    return p;
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
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream)
{
    if (detail::capturing(stream))
        return detail::capture_alloc(stream, devPtr, size);
    return detail::allocate(devPtr, size);
}

template <typename T>
static inline
cudaError_t cudaMallocAsync(T **devPtr, size_t size, cudaStream_t stream)
{
    return cudaMallocAsync(reinterpret_cast<void **>(devPtr), size, stream);
}

static inline
cudaError_t cudaMallocFromPoolAsync(void **devPtr, size_t size, cudaMemPool_t /* pool */,
                                    cudaStream_t stream)
{
    return cudaMallocAsync(devPtr, size, stream);
}

static inline
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream)
{
    if (detail::capturing(stream))
        return detail::capture_free(stream, devPtr);
    return detail::free_memory(devPtr);
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
    return detail::free_memory(devPtr);
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
                            cudaStream_t stream = nullptr)
{
    if (detail::capturing(stream))
        return detail::capture_copy(stream, detail::linear_copy(dst, src, count, kind));
    return detail::copy(dst, src, count, kind);
}

static inline
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                         cudaMemcpyKind kind)
{
    return detail::copy_3d(detail::pitched_copy(dst, dpitch, src, spitch, width, height, kind));
}

static inline
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                              size_t height, cudaMemcpyKind kind, cudaStream_t stream = nullptr)
{
    const cudaMemcpy3DParms p = detail::pitched_copy(dst, dpitch, src, spitch, width, height, kind);
    return detail::capturing(stream) ? detail::capture_copy(stream, p) : detail::copy_3d(p);
}

static inline
cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *p)
{
    return p == nullptr ? detail::record_error(cudaErrorInvalidValue) : detail::copy_3d(*p);
}

static inline
cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *p, cudaStream_t stream = nullptr)
{
    if (p == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    return detail::capturing(stream) ? detail::capture_copy(stream, *p) : detail::copy_3d(*p);
}

//! Rows of width bytes, pitch bytes apart
static inline
cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
    if (pitch == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *pitch = (width + detail::allocation_alignment - 1) / detail::allocation_alignment *
             detail::allocation_alignment;
    return detail::allocate(devPtr, *pitch * height);
}

template <typename T>
static inline
cudaError_t cudaMallocPitch(T **devPtr, size_t *pitch, size_t width, size_t height)
{
    return cudaMallocPitch(reinterpret_cast<void **>(devPtr), pitch, width, height);
}

static inline
cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr, cudaExtent extent)
{
    if (pitchedDevPtr == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    size_t pitch = 0;
    void *ptr = nullptr;
    if (cudaError_t err = cudaMallocPitch(&ptr, &pitch, extent.width, extent.height * extent.depth))
        return err;
    *pitchedDevPtr = make_cudaPitchedPtr(ptr, pitch, extent.width, extent.height);
    return cudaSuccess;
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
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream = nullptr)
{
    if (detail::capturing(stream))
        return detail::capture_fill(stream, {devPtr, count, unsigned(value) & 0xff, 1, count, 1});
    return cudaMemset(devPtr, value, count);
}

static inline
cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    return detail::fill({devPtr, pitch, unsigned(value) & 0xff, 1, width, height});
}

static inline
cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height,
                              cudaStream_t stream = nullptr)
{
    const cudaMemsetParams p{devPtr, pitch, unsigned(value) & 0xff, 1, width, height};
    return detail::capturing(stream) ? detail::capture_fill(stream, p) : detail::fill(p);
}

static inline
cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent)
{
    for (size_t z = 0; z < extent.depth; ++z) {
        void *slice = static_cast<char *>(pitchedDevPtr.ptr) + z * pitchedDevPtr.pitch * pitchedDevPtr.ysize;
        if (cudaError_t err = cudaMemset2D(slice, pitchedDevPtr.pitch, value, extent.width, extent.height))
            return err;
    }
    return cudaSuccess;
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
cudaError_t cudaMemcpyToSymbolAsync(T &&symbol, const void *src, size_t count, size_t offset = 0,
                                    cudaMemcpyKind kind = cudaMemcpyHostToDevice, cudaStream_t stream = nullptr)
{
    static_assert(std::is_lvalue_reference_v<T>,
                  "cudaMemcpyToSymbolAsync takes the symbol itself, as in CUDA, not its address");
    if (offset + count > sizeof(symbol))
        return detail::record_error(cudaErrorInvalidValue);
    return cudaMemcpyAsync(reinterpret_cast<char *>(std::addressof(symbol)) + offset, src, count, kind, stream);
}

template <typename T>
static inline
cudaError_t cudaMemcpyFromSymbolAsync(void *dst, T &&symbol, size_t count, size_t offset = 0,
                                      cudaMemcpyKind kind = cudaMemcpyDeviceToHost, cudaStream_t stream = nullptr)
{
    static_assert(std::is_lvalue_reference_v<T>,
                  "cudaMemcpyFromSymbolAsync takes the symbol itself, as in CUDA, not its address");
    if (offset + count > sizeof(symbol))
        return detail::record_error(cudaErrorInvalidValue);
    return cudaMemcpyAsync(dst, reinterpret_cast<const char *>(std::addressof(symbol)) + offset, count, kind,
                           stream);
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
