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

enum : unsigned int {
    cudaDeviceScheduleAuto         = 0x00,
    cudaDeviceScheduleSpin         = 0x01,
    cudaDeviceScheduleYield        = 0x02,
    cudaDeviceScheduleBlockingSync = 0x04,
    cudaDeviceMapHost              = 0x08,
    cudaDeviceLmemResizeToMax      = 0x10
};

enum cudaComputeMode {
    cudaComputeModeDefault          = 0,
    cudaComputeModeExclusive        = 1,
    cudaComputeModeProhibited       = 2,
    cudaComputeModeExclusiveProcess = 3
};

enum cudaLimit {
    cudaLimitStackSize                    = 0x00,
    cudaLimitPrintfFifoSize               = 0x01,
    cudaLimitMallocHeapSize               = 0x02,
    cudaLimitDevRuntimeSyncDepth          = 0x03,
    cudaLimitDevRuntimePendingLaunchCount = 0x04,
    cudaLimitMaxL2FetchGranularity        = 0x05,
    cudaLimitPersistingL2CacheSize        = 0x06
};

//! Cache and shared memory configuration: hints for GPU caches, accepted and
//! ignored
enum cudaFuncCache {
    cudaFuncCachePreferNone   = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1     = 2,
    cudaFuncCachePreferEqual  = 3
};

enum cudaSharedMemConfig {
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};

enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize    = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9
};

struct cudaFuncAttributes {
    size_t sharedSizeBytes;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int ptxVersion;
    int binaryVersion;
    int cacheModeCA;
    int maxDynamicSharedSizeBytes;
    int preferredShmemCarveout;
};

//! The L2 access policy of a stream: hints for the GPU's L2 cache, accepted
//! and ignored
enum cudaAccessProperty {
    cudaAccessPropertyNormal     = 0,
    cudaAccessPropertyStreaming  = 1,
    cudaAccessPropertyPersisting = 2
};

struct cudaAccessPolicyWindow {
    void *base_ptr;
    size_t num_bytes;
    float hitRatio;
    cudaAccessProperty hitProp;
    cudaAccessProperty missProp;
};

enum cudaLaunchAttributeID {
    cudaLaunchAttributeAccessPolicyWindow   = 1,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributePriority             = 8
};

union cudaLaunchAttributeValue {
    char pad[64];
    cudaAccessPolicyWindow accessPolicyWindow;
    int priority;
};

using cudaStreamAttrID    = cudaLaunchAttributeID;
using cudaStreamAttrValue = cudaLaunchAttributeValue;
inline constexpr cudaStreamAttrID cudaStreamAttributeAccessPolicyWindow = cudaLaunchAttributeAccessPolicyWindow;
inline constexpr cudaStreamAttrID cudaStreamAttributePriority = cudaLaunchAttributePriority;

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
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
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
    int hostRegisterSupported;
    int sparseCudaArraySupported;
    int hostRegisterReadOnlySupported;
    int timelineSemaphoreInteropSupported;
    int memoryPoolsSupported;
    int gpuDirectRDMASupported;
    unsigned int gpuDirectRDMAFlushWritesOptions;
    int gpuDirectRDMAWritesOrdering;
    unsigned int memoryPoolSupportedHandleTypes;
    int deferredMappingCudaArraySupported;
    int ipcEventSupported;
    int clusterLaunch;
    int unifiedFunctionPointers;
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
    cudaDevAttrMaxPitch                       = 11,
    cudaDevAttrMaxRegistersPerBlock           = 12,
    cudaDevAttrClockRate                      = 13,
    cudaDevAttrTextureAlignment               = 14,
    cudaDevAttrGpuOverlap                     = 15,
    cudaDevAttrMultiProcessorCount            = 16,
    cudaDevAttrKernelExecTimeout              = 17,
    cudaDevAttrIntegrated                     = 18,
    cudaDevAttrCanMapHostMemory               = 19,
    cudaDevAttrComputeMode                    = 20,
    cudaDevAttrConcurrentKernels              = 31,
    cudaDevAttrEccEnabled                     = 32,
    cudaDevAttrPciBusId                       = 33,
    cudaDevAttrPciDeviceId                    = 34,
    cudaDevAttrMemoryClockRate                = 36,
    cudaDevAttrGlobalMemoryBusWidth           = 37,
    cudaDevAttrL2CacheSize                    = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39,
    cudaDevAttrAsyncEngineCount               = 40,
    cudaDevAttrUnifiedAddressing              = 41,
    cudaDevAttrPciDomainId                    = 50,
    cudaDevAttrComputeCapabilityMajor         = 75,
    cudaDevAttrComputeCapabilityMinor         = 76,
    cudaDevAttrStreamPrioritiesSupported      = 78,
    cudaDevAttrGlobalL1CacheSupported         = 79,
    cudaDevAttrLocalL1CacheSupported          = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82,
    cudaDevAttrManagedMemory                  = 83,
    cudaDevAttrIsMultiGpuBoard                = 84,
    cudaDevAttrHostNativeAtomicSupported      = 86,
    cudaDevAttrPageableMemoryAccess           = 88,
    cudaDevAttrConcurrentManagedAccess        = 89,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrCooperativeLaunch              = 95,
    cudaDevAttrHostRegisterSupported          = 99,
    cudaDevAttrMaxSharedMemoryPerBlockOptin   = 97,
    cudaDevAttrMaxBlocksPerMultiprocessor     = 106,
    cudaDevAttrMemoryPoolsSupported           = 115
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

//! Flags given to cudaSetDeviceFlags
inline std::atomic<unsigned int> device_flags{0};

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

//! Deprecated names of cudaDeviceSynchronize and cudaDeviceReset
static inline
cudaError_t cudaThreadSynchronize()
{
    return cudaSuccess;
}

static inline
cudaError_t cudaThreadExit()
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
cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
    detail::device_flags = flags;
    return cudaSuccess;
}

static inline
cudaError_t cudaGetDeviceFlags(unsigned int *flags)
{
    *flags = detail::device_flags;
    return cudaSuccess;
}

//! Stream priorities have no effect: streams are synchronous. The range is
//! [0, 0], which is how CUDA reports that priorities are not supported.
static inline
cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority)
{
    if (leastPriority)
        *leastPriority = 0;
    if (greatestPriority)
        *greatestPriority = 0;
    return cudaSuccess;
}

//! There is one device, so no peer to access
static inline
cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int /* device */, int /* peerDevice */)
{
    *canAccessPeer = 0;
    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceEnablePeerAccess(int /* peerDevice */, unsigned int /* flags */)
{
    return detail::record_error(cudaErrorInvalidDevice);
}

static inline
cudaError_t cudaDeviceDisablePeerAccess(int /* peerDevice */)
{
    return detail::record_error(cudaErrorInvalidDevice);
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
    case cudaDevAttrMemoryPoolsSupported:         *value = p.memoryPoolsSupported; break;
    case cudaDevAttrMaxPitch:                     *value = int(p.memPitch); break;
    case cudaDevAttrTextureAlignment:             *value = int(p.textureAlignment); break;
    case cudaDevAttrGpuOverlap:                   *value = p.deviceOverlap; break;
    case cudaDevAttrKernelExecTimeout:            *value = p.kernelExecTimeoutEnabled; break;
    case cudaDevAttrPciBusId:                     *value = p.pciBusID; break;
    case cudaDevAttrPciDeviceId:                  *value = p.pciDeviceID; break;
    case cudaDevAttrPciDomainId:                  *value = p.pciDomainID; break;
    case cudaDevAttrMemoryClockRate:              *value = p.memoryClockRate; break;
    case cudaDevAttrGlobalMemoryBusWidth:         *value = p.memoryBusWidth; break;
    case cudaDevAttrStreamPrioritiesSupported:    *value = p.streamPrioritiesSupported; break;
    case cudaDevAttrGlobalL1CacheSupported:       *value = p.globalL1CacheSupported; break;
    case cudaDevAttrLocalL1CacheSupported:        *value = p.localL1CacheSupported; break;
    case cudaDevAttrIsMultiGpuBoard:              *value = p.isMultiGpuBoard; break;
    case cudaDevAttrHostNativeAtomicSupported:    *value = p.hostNativeAtomicSupported; break;
    case cudaDevAttrPageableMemoryAccess:         *value = p.pageableMemoryAccess; break;
    case cudaDevAttrCanUseHostPointerForRegisteredMem: *value = p.canUseHostPointerForRegisteredMem; break;
    case cudaDevAttrCooperativeLaunch:            *value = p.cooperativeLaunch; break;
    case cudaDevAttrHostRegisterSupported:        *value = p.hostRegisterSupported; break;
    default:
        return detail::record_error(cudaErrorInvalidValue);
    }
    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache /* cacheConfig */)
{
    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache *cacheConfig)
{
    *cacheConfig = cudaFuncCachePreferNone;
    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig /* config */)
{
    return cudaSuccess;
}

static inline
cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig *config)
{
    *config = cudaSharedMemBankSizeFourByte;
    return cudaSuccess;
}

template <typename T>
static inline
cudaError_t cudaFuncSetCacheConfig(T * /* func */, cudaFuncCache /* cacheConfig */)
{
    return cudaSuccess;
}

template <typename T>
static inline
cudaError_t cudaFuncSetSharedMemConfig(T * /* func */, cudaSharedMemConfig /* config */)
{
    return cudaSuccess;
}

//! Dynamic shared memory has no 48 KiB limit to raise, so these are accepted
//! and ignored
template <typename T>
static inline
cudaError_t cudaFuncSetAttribute(T * /* func */, cudaFuncAttribute /* attr */, int /* value */)
{
    return cudaSuccess;
}

template <typename T>
static inline
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr, T * /* func */)
{
    *attr = cudaFuncAttributes{};
    attr->maxThreadsPerBlock = 1024;
    attr->maxDynamicSharedSizeBytes = 48 * 1024;
    attr->ptxVersion = attr->binaryVersion = 60;
    return cudaSuccess;
}

static inline
cudaError_t cudaStreamSetAttribute(cudaStream_t /* stream */, cudaStreamAttrID /* attr */,
                                   const cudaStreamAttrValue * /* value */)
{
    return cudaSuccess;
}

static inline
cudaError_t cudaStreamGetAttribute(cudaStream_t /* stream */, cudaStreamAttrID /* attr */,
                                   cudaStreamAttrValue *value)
{
    *value = cudaStreamAttrValue{};
    return cudaSuccess;
}

static inline
cudaError_t cudaCtxResetPersistingL2Cache()
{
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
