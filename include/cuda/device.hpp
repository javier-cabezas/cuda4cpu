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

#include <algorithm>
#include <atomic>
#include <cstddef>

#include "error.hpp"
#include "types.hpp"

//! The CUDA runtime API version that cuda4cpu follows
#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
#endif

namespace cuda4cpu {

inline namespace cuda_api {

enum cudaLimit {
    cudaLimitStackSize      = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02
};

struct cudaUUID_t {
    char bytes[16];
};

//! Properties of the device. cuda4cpu describes the CPU it runs on as one
//! device, with one multiprocessor per OS thread and CUDA's limits for
//! threads, blocks and grids. Fields that have no CPU equivalent are zero.
struct cudaDeviceProp {
    char name[256];
    cudaUUID_t uuid;
    char luid[8];
    unsigned int luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
};

//! Device attributes, with the same values as CUDA's
enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock             = 1,
    cudaDevAttrMaxBlockDimX                   = 2,
    cudaDevAttrMaxBlockDimY                   = 3,
    cudaDevAttrMaxBlockDimZ                   = 4,
    cudaDevAttrMaxGridDimX                    = 5,
    cudaDevAttrMaxGridDimY                    = 6,
    cudaDevAttrMaxGridDimZ                    = 7,
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,
    cudaDevAttrTotalConstantMemory            = 9,
    cudaDevAttrWarpSize                       = 10,
    cudaDevAttrMaxRegistersPerBlock           = 12,
    cudaDevAttrClockRate                      = 13,
    cudaDevAttrMultiProcessorCount            = 16,
    cudaDevAttrIntegrated                     = 18,
    cudaDevAttrCanMapHostMemory               = 19,
    cudaDevAttrComputeMode                    = 20,
    cudaDevAttrConcurrentKernels              = 31,
    cudaDevAttrEccEnabled                     = 32,
    cudaDevAttrL2CacheSize                    = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39,
    cudaDevAttrAsyncEngineCount               = 40,
    cudaDevAttrUnifiedAddressing              = 41,
    cudaDevAttrComputeCapabilityMajor         = 75,
    cudaDevAttrComputeCapabilityMinor         = 76,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82,
    cudaDevAttrManagedMemory                  = 83,
    cudaDevAttrConcurrentManagedAccess        = 89,
    cudaDevAttrMaxSharedMemoryPerBlockOptin   = 97,
    cudaDevAttrMaxBlocksPerMultiprocessor     = 106
};

}

namespace detail {

//! Smallest stack given to each CUDA thread. CPU code (libc calls, the
//! launcher) needs much more stack than GPU code, so smaller requests are
//! rounded up to this size.
inline constexpr size_t min_stack_size = 64 * 1024;

//! Stack size of each CUDA thread for subsequent launches
inline std::atomic<size_t> stack_size{min_stack_size};

//! Describes the CPU as a CUDA device (defined in the library)
void get_device_properties(cudaDeviceProp &prop);

}

inline namespace cuda_api {

static inline
cudaError_t cudaDeviceSynchronize()
{
    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceReset()
{
    return cudaSuccess;
}

static inline
cudaError_t cudaGetDeviceCount(int *count)
{
    *count = 1;
    return cudaSuccess;
}

static inline
cudaError_t cudaSetDevice(int device)
{
    return device == 0 ? cudaSuccess : detail::record_error(cudaErrorInvalidDevice);
}

static inline
cudaError_t cudaGetDevice(int *device)
{
    *device = 0;
    return cudaSuccess;
}

static inline
cudaError_t cudaDriverGetVersion(int *version)
{
    *version = CUDART_VERSION;
    return cudaSuccess;
}

static inline
cudaError_t cudaRuntimeGetVersion(int *version)
{
    *version = CUDART_VERSION;
    return cudaSuccess;
}

static inline
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
{
    if (prop == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    if (device != 0)
        return detail::record_error(cudaErrorInvalidDevice);

    detail::get_device_properties(*prop);
    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device)
{
    if (value == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    if (device != 0)
        return detail::record_error(cudaErrorInvalidDevice);

    cudaDeviceProp p;
    detail::get_device_properties(p);

    switch (attr) {
    case cudaDevAttrMaxThreadsPerBlock:           *value = p.maxThreadsPerBlock; break;
    case cudaDevAttrMaxBlockDimX:                 *value = p.maxThreadsDim[0]; break;
    case cudaDevAttrMaxBlockDimY:                 *value = p.maxThreadsDim[1]; break;
    case cudaDevAttrMaxBlockDimZ:                 *value = p.maxThreadsDim[2]; break;
    case cudaDevAttrMaxGridDimX:                  *value = p.maxGridSize[0]; break;
    case cudaDevAttrMaxGridDimY:                  *value = p.maxGridSize[1]; break;
    case cudaDevAttrMaxGridDimZ:                  *value = p.maxGridSize[2]; break;
    case cudaDevAttrMaxSharedMemoryPerBlock:      *value = int(p.sharedMemPerBlock); break;
    case cudaDevAttrTotalConstantMemory:          *value = int(p.totalConstMem); break;
    case cudaDevAttrWarpSize:                     *value = p.warpSize; break;
    case cudaDevAttrMaxRegistersPerBlock:         *value = p.regsPerBlock; break;
    case cudaDevAttrClockRate:                    *value = p.clockRate; break;
    case cudaDevAttrMultiProcessorCount:          *value = p.multiProcessorCount; break;
    case cudaDevAttrIntegrated:                   *value = p.integrated; break;
    case cudaDevAttrCanMapHostMemory:             *value = p.canMapHostMemory; break;
    case cudaDevAttrComputeMode:                  *value = p.computeMode; break;
    case cudaDevAttrConcurrentKernels:            *value = p.concurrentKernels; break;
    case cudaDevAttrEccEnabled:                   *value = p.ECCEnabled; break;
    case cudaDevAttrL2CacheSize:                  *value = p.l2CacheSize; break;
    case cudaDevAttrMaxThreadsPerMultiProcessor:  *value = p.maxThreadsPerMultiProcessor; break;
    case cudaDevAttrAsyncEngineCount:             *value = p.asyncEngineCount; break;
    case cudaDevAttrUnifiedAddressing:            *value = p.unifiedAddressing; break;
    case cudaDevAttrComputeCapabilityMajor:       *value = p.major; break;
    case cudaDevAttrComputeCapabilityMinor:       *value = p.minor; break;
    case cudaDevAttrMaxSharedMemoryPerMultiprocessor: *value = int(p.sharedMemPerMultiprocessor); break;
    case cudaDevAttrMaxRegistersPerMultiprocessor: *value = p.regsPerMultiprocessor; break;
    case cudaDevAttrManagedMemory:                *value = p.managedMemory; break;
    case cudaDevAttrConcurrentManagedAccess:      *value = p.concurrentManagedAccess; break;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = int(p.sharedMemPerBlockOptin); break;
    case cudaDevAttrMaxBlocksPerMultiprocessor:   *value = p.maxBlocksPerMultiProcessor; break;
    default:
        return detail::record_error(cudaErrorInvalidValue);
    }
    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value)
{
    if (limit == cudaLimitStackSize)
        detail::stack_size = std::max(value, detail::min_stack_size);

    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceGetLimit(size_t *value, cudaLimit limit)
{
    *value = (limit == cudaLimitStackSize) ? detail::stack_size.load() : 0;

    return cudaSuccess;
}

}

}
