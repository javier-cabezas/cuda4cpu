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

//
// HIP's runtime API, on top of cuda4cpu's CUDA runtime API, as HIP's NVIDIA
// platform does it. Most HIP names are CUDA names with another prefix
// (hipMalloc, hipStream_t, hipMemcpyHostToDevice): they are macros for the
// CUDA names, so they keep their overloads, default arguments and templates.
// What HIP does differently is in the namespace cuda4cpu::hip_api, exported to
// global scope like cuda_api: hipDeviceProp_t, device attributes, HIP's error
// names, host allocation flags, and the copies of HIP's driver API.
//

#pragma once

#include <climits>
#include <cstdint>
#include <cstdio>

#include "hip_common.h"
#include "hip_version.h"

#include "../cuda_runtime.h"
#include "../cuda_profiler_api.h"

namespace cuda4cpu {

inline namespace hip_api {

//! Converts between HIP's and CUDA's error codes, which are the same
inline cudaError_t hipCUDAErrorTohipError(cudaError_t error) { return error; }
inline cudaError_t hipErrorToCudaError(cudaError_t error) { return error; }

//! HIP's names for CUDA's error codes
inline const char *hipGetErrorName(cudaError_t error)
{
    switch (error) {
    case cudaSuccess:                        return "hipSuccess";
    case cudaErrorInvalidValue:              return "hipErrorInvalidValue";
    case cudaErrorMemoryAllocation:          return "hipErrorOutOfMemory";
    case cudaErrorInitializationError:       return "hipErrorNotInitialized";
    case cudaErrorInvalidConfiguration:      return "hipErrorInvalidConfiguration";
    case cudaErrorInvalidSymbol:             return "hipErrorInvalidSymbol";
    case cudaErrorInvalidDevicePointer:      return "hipErrorInvalidDevicePointer";
    case cudaErrorInvalidMemcpyDirection:    return "hipErrorInvalidMemcpyDirection";
    case cudaErrorInvalidDeviceFunction:     return "hipErrorInvalidDeviceFunction";
    case cudaErrorNoDevice:                  return "hipErrorNoDevice";
    case cudaErrorInvalidDevice:             return "hipErrorInvalidDevice";
    case cudaErrorInvalidResourceHandle:     return "hipErrorInvalidHandle";
    case cudaErrorIllegalState:              return "hipErrorIllegalState";
    case cudaErrorNotReady:                  return "hipErrorNotReady";
    case cudaErrorAssert:                    return "hipErrorAssert";
    case cudaErrorLaunchFailure:             return "hipErrorLaunchFailure";
    case cudaErrorCooperativeLaunchTooLarge: return "hipErrorCooperativeLaunchTooLarge";
    case cudaErrorNotSupported:              return "hipErrorNotSupported";
    case cudaErrorStreamCaptureUnsupported:  return "hipErrorStreamCaptureUnsupported";
    case cudaErrorStreamCaptureInvalidated:  return "hipErrorStreamCaptureInvalidated";
    case cudaErrorStreamCaptureMerge:        return "hipErrorStreamCaptureMerge";
    case cudaErrorStreamCaptureUnmatched:    return "hipErrorStreamCaptureUnmatched";
    case cudaErrorStreamCaptureUnjoined:     return "hipErrorStreamCaptureUnjoined";
    case cudaErrorStreamCaptureIsolation:    return "hipErrorStreamCaptureIsolation";
    case cudaErrorStreamCaptureImplicit:     return "hipErrorStreamCaptureImplicit";
    case cudaErrorGraphExecUpdateFailure:    return "hipErrorGraphExecUpdateFailure";
    case cudaErrorUnknown:                   return "hipErrorUnknown";
    }
    return "hipErrorUnknown";
}

//
// Devices. hipDeviceProp_t is CUDA's cudaDeviceProp plus the fields HIP adds
// for AMD GPUs.
//

//! The features of the device's architecture, as the __HIP_ARCH_HAS_*
//! macros of hip_runtime.h describe them
struct hipDeviceArch_t {
    unsigned hasGlobalInt32Atomics     : 1;
    unsigned hasGlobalFloatAtomicExch  : 1;
    unsigned hasSharedInt32Atomics     : 1;
    unsigned hasSharedFloatAtomicExch  : 1;
    unsigned hasFloatAtomicAdd         : 1;
    unsigned hasGlobalInt64Atomics     : 1;
    unsigned hasSharedInt64Atomics     : 1;
    unsigned hasDoubles                : 1;
    unsigned hasWarpVote               : 1;
    unsigned hasWarpBallot             : 1;
    unsigned hasWarpShuffle            : 1;
    unsigned hasFunnelShift            : 1;
    unsigned hasThreadFenceSystem      : 1;
    unsigned hasSyncThreadsExt         : 1;
    unsigned hasSurfaceFuncs           : 1;
    unsigned has3dGrid                 : 1;
    unsigned hasDynamicParallelism     : 1;
};

struct hipDeviceProp_t : cudaDeviceProp {
    //! The CPU's architecture, where an AMD GPU has its gfx name
    char gcnArchName[256];
    size_t maxSharedMemoryPerMultiProcessor;
    int clockInstructionRate;
    hipDeviceArch_t arch;
    unsigned int *hdpMemFlushCntl;
    unsigned int *hdpRegFlushCntl;
    int cooperativeMultiDeviceUnmatchedFunc;
    int cooperativeMultiDeviceUnmatchedGridDim;
    int cooperativeMultiDeviceUnmatchedBlockDim;
    int cooperativeMultiDeviceUnmatchedSharedMem;
    int isLargeBar;
    int asicRevision;
};

inline cudaError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int device)
{
    if (prop == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    hipDeviceProp_t p{};
    if (cudaError_t err = cudaGetDeviceProperties(&p, device))
        return err;

#if defined(__x86_64__)
    std::snprintf(p.gcnArchName, sizeof(p.gcnArchName), "x86_64");
#elif defined(__aarch64__)
    std::snprintf(p.gcnArchName, sizeof(p.gcnArchName), "aarch64");
#else
    std::snprintf(p.gcnArchName, sizeof(p.gcnArchName), "cpu");
#endif
    p.maxSharedMemoryPerMultiProcessor = p.sharedMemPerMultiprocessor;
    p.clockInstructionRate = p.clockRate;
    p.arch = hipDeviceArch_t{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0};
    p.isLargeBar = 1;
    *prop = p;
    return cudaSuccess;
}

//! The name of the R0600 version of the structure, which HIP's headers
//! rename hipGetDeviceProperties to
inline cudaError_t hipGetDevicePropertiesR0600(hipDeviceProp_t *prop, int device)
{
    return hipGetDeviceProperties(prop, device);
}

//! There is one device, which matches every request
inline cudaError_t hipChooseDevice(int *device, const hipDeviceProp_t * /* prop */)
{
    if (device == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *device = 0;
    return cudaSuccess;
}

//! Device attributes, with HIP's names. The values are cuda4cpu's own:
//! programs use the names.
enum hipDeviceAttribute_t {
    hipDeviceAttributeCudaCompatibleBegin = 0,
    hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin,
    hipDeviceAttributeAccessPolicyMaxWindowSize,
    hipDeviceAttributeAsyncEngineCount,
    hipDeviceAttributeCanMapHostMemory,
    hipDeviceAttributeCanUseHostPointerForRegisteredMem,
    hipDeviceAttributeClockRate,
    hipDeviceAttributeComputeMode,
    hipDeviceAttributeComputePreemptionSupported,
    hipDeviceAttributeConcurrentKernels,
    hipDeviceAttributeConcurrentManagedAccess,
    hipDeviceAttributeCooperativeLaunch,
    hipDeviceAttributeCooperativeMultiDeviceLaunch,
    hipDeviceAttributeDeviceOverlap,
    hipDeviceAttributeDirectManagedMemAccessFromHost,
    hipDeviceAttributeGlobalL1CacheSupported,
    hipDeviceAttributeHostNativeAtomicSupported,
    hipDeviceAttributeIntegrated,
    hipDeviceAttributeIsMultiGpuBoard,
    hipDeviceAttributeKernelExecTimeout,
    hipDeviceAttributeL2CacheSize,
    hipDeviceAttributeLocalL1CacheSupported,
    hipDeviceAttributeLuid,
    hipDeviceAttributeLuidDeviceNodeMask,
    hipDeviceAttributeComputeCapabilityMajor,
    hipDeviceAttributeManagedMemory,
    hipDeviceAttributeMaxBlocksPerMultiProcessor,
    hipDeviceAttributeMaxBlockDimX,
    hipDeviceAttributeMaxBlockDimY,
    hipDeviceAttributeMaxBlockDimZ,
    hipDeviceAttributeMaxGridDimX,
    hipDeviceAttributeMaxGridDimY,
    hipDeviceAttributeMaxGridDimZ,
    hipDeviceAttributeMaxSurface1D,
    hipDeviceAttributeMaxSurface1DLayered,
    hipDeviceAttributeMaxSurface2D,
    hipDeviceAttributeMaxSurface2DLayered,
    hipDeviceAttributeMaxSurface3D,
    hipDeviceAttributeMaxSurfaceCubemap,
    hipDeviceAttributeMaxSurfaceCubemapLayered,
    hipDeviceAttributeMaxTexture1DWidth,
    hipDeviceAttributeMaxTexture1DLayered,
    hipDeviceAttributeMaxTexture1DLinear,
    hipDeviceAttributeMaxTexture1DMipmap,
    hipDeviceAttributeMaxTexture2DWidth,
    hipDeviceAttributeMaxTexture2DHeight,
    hipDeviceAttributeMaxTexture2DGather,
    hipDeviceAttributeMaxTexture2DLayered,
    hipDeviceAttributeMaxTexture2DLinear,
    hipDeviceAttributeMaxTexture2DMipmap,
    hipDeviceAttributeMaxTexture3DWidth,
    hipDeviceAttributeMaxTexture3DHeight,
    hipDeviceAttributeMaxTexture3DDepth,
    hipDeviceAttributeMaxTexture3DAlt,
    hipDeviceAttributeMaxTextureCubemap,
    hipDeviceAttributeMaxTextureCubemapLayered,
    hipDeviceAttributeMaxThreadsDim,
    hipDeviceAttributeMaxThreadsPerBlock,
    hipDeviceAttributeMaxThreadsPerMultiProcessor,
    hipDeviceAttributeMaxPitch,
    hipDeviceAttributeMemoryBusWidth,
    hipDeviceAttributeMemoryClockRate,
    hipDeviceAttributeComputeCapabilityMinor,
    hipDeviceAttributeMultiGpuBoardGroupID,
    hipDeviceAttributeMultiprocessorCount,
    hipDeviceAttributeUnused1,
    hipDeviceAttributePageableMemoryAccess,
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
    hipDeviceAttributePciBusId,
    hipDeviceAttributePciDeviceId,
    hipDeviceAttributePciDomainId,
    hipDeviceAttributePciDomainID = hipDeviceAttributePciDomainId,
    hipDeviceAttributePersistingL2CacheMaxSize,
    hipDeviceAttributeMaxRegistersPerBlock,
    hipDeviceAttributeMaxRegistersPerMultiprocessor,
    hipDeviceAttributeReservedSharedMemPerBlock,
    hipDeviceAttributeMaxSharedMemoryPerBlock,
    hipDeviceAttributeSharedMemPerBlockOptin,
    hipDeviceAttributeSharedMemPerMultiprocessor,
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio,
    hipDeviceAttributeStreamPrioritiesSupported,
    hipDeviceAttributeSurfaceAlignment,
    hipDeviceAttributeTccDriver,
    hipDeviceAttributeTextureAlignment,
    hipDeviceAttributeTexturePitchAlignment,
    hipDeviceAttributeTotalConstantMemory,
    hipDeviceAttributeTotalGlobalMem,
    hipDeviceAttributeUnifiedAddressing,
    hipDeviceAttributeUnused2,
    hipDeviceAttributeWarpSize,
    hipDeviceAttributeMemoryPoolsSupported,
    hipDeviceAttributeVirtualMemoryManagementSupported,
    hipDeviceAttributeHostRegisterSupported,
    hipDeviceAttributeMemoryPoolSupportedHandleTypes,
    hipDeviceAttributeCudaCompatibleEnd = 9999,

    hipDeviceAttributeAmdSpecificBegin = 10000,
    hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin,
    hipDeviceAttributeUnused3,
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
    hipDeviceAttributeUnused4,
    hipDeviceAttributeUnused5,
    hipDeviceAttributeHdpMemFlushCntl,
    hipDeviceAttributeHdpRegFlushCntl,
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
    hipDeviceAttributeIsLargeBar,
    hipDeviceAttributeAsicRevision,
    hipDeviceAttributeCanUseStreamWaitValue,
    hipDeviceAttributeImageSupport,
    hipDeviceAttributePhysicalMultiProcessorCount,
    hipDeviceAttributeFineGrainSupport,
    hipDeviceAttributeWallClockRate,
    hipDeviceAttributeAmdSpecificEnd = 19999,
    hipDeviceAttributeVendorSpecificBegin = 20000
};

//! Rate of wall_clock64(), in kHz: it counts nanoseconds
inline constexpr int wall_clock_rate_khz = 1000000;

inline cudaError_t hipDeviceGetAttribute(int *value, hipDeviceAttribute_t attr, int device)
{
    if (value == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    hipDeviceProp_t p;
    if (cudaError_t err = hipGetDeviceProperties(&p, device))
        return err;

    auto clamp = [](size_t v) { return v > size_t(INT_MAX) ? INT_MAX : int(v); };
    switch (attr) {
    case hipDeviceAttributeEccEnabled:                        *value = p.ECCEnabled; break;
    case hipDeviceAttributeAccessPolicyMaxWindowSize:         *value = p.accessPolicyMaxWindowSize; break;
    case hipDeviceAttributeAsyncEngineCount:                  *value = p.asyncEngineCount; break;
    case hipDeviceAttributeCanMapHostMemory:                  *value = p.canMapHostMemory; break;
    case hipDeviceAttributeCanUseHostPointerForRegisteredMem: *value = p.canUseHostPointerForRegisteredMem; break;
    case hipDeviceAttributeClockRate:                         *value = p.clockRate; break;
    case hipDeviceAttributeComputeMode:                       *value = p.computeMode; break;
    case hipDeviceAttributeComputePreemptionSupported:        *value = p.computePreemptionSupported; break;
    case hipDeviceAttributeConcurrentKernels:                 *value = p.concurrentKernels; break;
    case hipDeviceAttributeConcurrentManagedAccess:           *value = p.concurrentManagedAccess; break;
    case hipDeviceAttributeCooperativeLaunch:                 *value = p.cooperativeLaunch; break;
    case hipDeviceAttributeCooperativeMultiDeviceLaunch:      *value = p.cooperativeMultiDeviceLaunch; break;
    case hipDeviceAttributeDeviceOverlap:                     *value = p.deviceOverlap; break;
    case hipDeviceAttributeDirectManagedMemAccessFromHost:    *value = p.directManagedMemAccessFromHost; break;
    case hipDeviceAttributeGlobalL1CacheSupported:            *value = p.globalL1CacheSupported; break;
    case hipDeviceAttributeHostNativeAtomicSupported:         *value = p.hostNativeAtomicSupported; break;
    case hipDeviceAttributeIntegrated:                        *value = p.integrated; break;
    case hipDeviceAttributeIsMultiGpuBoard:                   *value = p.isMultiGpuBoard; break;
    case hipDeviceAttributeKernelExecTimeout:                 *value = p.kernelExecTimeoutEnabled; break;
    case hipDeviceAttributeL2CacheSize:                       *value = p.l2CacheSize; break;
    case hipDeviceAttributeLocalL1CacheSupported:             *value = p.localL1CacheSupported; break;
    case hipDeviceAttributeComputeCapabilityMajor:            *value = p.major; break;
    case hipDeviceAttributeComputeCapabilityMinor:            *value = p.minor; break;
    case hipDeviceAttributeManagedMemory:                     *value = p.managedMemory; break;
    case hipDeviceAttributeMaxBlocksPerMultiProcessor:        *value = p.maxBlocksPerMultiProcessor; break;
    case hipDeviceAttributeMaxBlockDimX:                      *value = p.maxThreadsDim[0]; break;
    case hipDeviceAttributeMaxBlockDimY:                      *value = p.maxThreadsDim[1]; break;
    case hipDeviceAttributeMaxBlockDimZ:                      *value = p.maxThreadsDim[2]; break;
    case hipDeviceAttributeMaxGridDimX:                       *value = p.maxGridSize[0]; break;
    case hipDeviceAttributeMaxGridDimY:                       *value = p.maxGridSize[1]; break;
    case hipDeviceAttributeMaxGridDimZ:                       *value = p.maxGridSize[2]; break;
    case hipDeviceAttributeMaxThreadsPerBlock:                *value = p.maxThreadsPerBlock; break;
    case hipDeviceAttributeMaxThreadsPerMultiProcessor:       *value = p.maxThreadsPerMultiProcessor; break;
    case hipDeviceAttributeMaxPitch:                          *value = clamp(p.memPitch); break;
    case hipDeviceAttributeMemoryBusWidth:                    *value = p.memoryBusWidth; break;
    case hipDeviceAttributeMemoryClockRate:                   *value = p.memoryClockRate; break;
    case hipDeviceAttributeMultiGpuBoardGroupID:              *value = p.multiGpuBoardGroupID; break;
    case hipDeviceAttributeMultiprocessorCount:               *value = p.multiProcessorCount; break;
    case hipDeviceAttributePhysicalMultiProcessorCount:       *value = p.multiProcessorCount; break;
    case hipDeviceAttributePageableMemoryAccess:              *value = p.pageableMemoryAccess; break;
    case hipDeviceAttributePageableMemoryAccessUsesHostPageTables:
        *value = p.pageableMemoryAccessUsesHostPageTables;
        break;
    case hipDeviceAttributePciBusId:                          *value = p.pciBusID; break;
    case hipDeviceAttributePciDeviceId:                       *value = p.pciDeviceID; break;
    case hipDeviceAttributePciDomainId:                       *value = p.pciDomainID; break;
    case hipDeviceAttributePersistingL2CacheMaxSize:          *value = p.persistingL2CacheMaxSize; break;
    case hipDeviceAttributeMaxRegistersPerBlock:              *value = p.regsPerBlock; break;
    case hipDeviceAttributeMaxRegistersPerMultiprocessor:     *value = p.regsPerMultiprocessor; break;
    case hipDeviceAttributeReservedSharedMemPerBlock:         *value = clamp(p.reservedSharedMemPerBlock); break;
    case hipDeviceAttributeMaxSharedMemoryPerBlock:           *value = clamp(p.sharedMemPerBlock); break;
    case hipDeviceAttributeSharedMemPerBlockOptin:            *value = clamp(p.sharedMemPerBlockOptin); break;
    case hipDeviceAttributeSharedMemPerMultiprocessor:        *value = clamp(p.sharedMemPerMultiprocessor); break;
    case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:  *value = clamp(p.maxSharedMemoryPerMultiProcessor); break;
    case hipDeviceAttributeSingleToDoublePrecisionPerfRatio:  *value = p.singleToDoublePrecisionPerfRatio; break;
    case hipDeviceAttributeStreamPrioritiesSupported:         *value = p.streamPrioritiesSupported; break;
    case hipDeviceAttributeSurfaceAlignment:                  *value = clamp(p.surfaceAlignment); break;
    case hipDeviceAttributeTccDriver:                         *value = p.tccDriver; break;
    case hipDeviceAttributeTextureAlignment:                  *value = clamp(p.textureAlignment); break;
    case hipDeviceAttributeTexturePitchAlignment:             *value = clamp(p.texturePitchAlignment); break;
    case hipDeviceAttributeTotalConstantMemory:               *value = clamp(p.totalConstMem); break;
    case hipDeviceAttributeTotalGlobalMem:                    *value = clamp(p.totalGlobalMem); break;
    case hipDeviceAttributeUnifiedAddressing:                 *value = p.unifiedAddressing; break;
    case hipDeviceAttributeWarpSize:                          *value = p.warpSize; break;
    case hipDeviceAttributeMemoryPoolsSupported:              *value = p.memoryPoolsSupported; break;
    case hipDeviceAttributeHostRegisterSupported:             *value = p.hostRegisterSupported; break;
    case hipDeviceAttributeMemoryPoolSupportedHandleTypes:    *value = int(p.memoryPoolSupportedHandleTypes); break;
    case hipDeviceAttributeClockInstructionRate:              *value = p.clockInstructionRate; break;
    case hipDeviceAttributeIsLargeBar:                        *value = p.isLargeBar; break;
    case hipDeviceAttributeAsicRevision:                      *value = p.asicRevision; break;
    case hipDeviceAttributeWallClockRate:                     *value = wall_clock_rate_khz; break;
    // Device memory is host memory, which the host accesses coherently
    case hipDeviceAttributeFineGrainSupport:                  *value = 1; break;
    // Textures, virtual memory management and stream wait values aren't
    // implemented
    case hipDeviceAttributeImageSupport:
    case hipDeviceAttributeVirtualMemoryManagementSupported:
    case hipDeviceAttributeCanUseStreamWaitValue:
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc:
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim:
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim:
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem:
        *value = 0;
        break;
    default:
        return detail::record_error(cudaErrorInvalidValue);
    }
    return cudaSuccess;
}

//
// HIP's driver API for devices: a device handle is its ordinal
//

using hipDevice_t = int;

//! Initializes HIP: there is nothing to initialize
inline cudaError_t hipInit(unsigned int flags)
{
    return flags == 0 ? cudaSuccess : detail::record_error(cudaErrorInvalidValue);
}

inline cudaError_t hipDeviceGet(hipDevice_t *device, int ordinal)
{
    if (device == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    if (ordinal != 0)
        return detail::record_error(cudaErrorInvalidDevice);
    *device = ordinal;
    return cudaSuccess;
}

inline cudaError_t hipDeviceGetName(char *name, int len, hipDevice_t device)
{
    if (name == nullptr || len <= 0)
        return detail::record_error(cudaErrorInvalidValue);
    hipDeviceProp_t p;
    if (cudaError_t err = hipGetDeviceProperties(&p, device))
        return err;
    std::snprintf(name, size_t(len), "%s", p.name);
    return cudaSuccess;
}

inline cudaError_t hipDeviceComputeCapability(int *major, int *minor, hipDevice_t device)
{
    if (major == nullptr || minor == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    cudaDeviceProp p;
    if (cudaError_t err = cudaGetDeviceProperties(&p, device))
        return err;
    *major = p.major;
    *minor = p.minor;
    return cudaSuccess;
}

inline cudaError_t hipDeviceTotalMem(size_t *bytes, hipDevice_t device)
{
    if (bytes == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    cudaDeviceProp p;
    if (cudaError_t err = cudaGetDeviceProperties(&p, device))
        return err;
    *bytes = p.totalGlobalMem;
    return cudaSuccess;
}

inline cudaError_t hipDeviceGetUuid(cudaUUID_t *uuid, hipDevice_t device)
{
    if (uuid == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    cudaDeviceProp p;
    if (cudaError_t err = cudaGetDeviceProperties(&p, device))
        return err;
    *uuid = p.uuid;
    return cudaSuccess;
}

//
// Host memory. hipHostMalloc takes the flags that cudaHostAlloc takes, with
// other names, and has typed overloads.
//

enum : unsigned int {
    hipHostMallocDefault       = 0x0,
    hipHostMallocPortable      = 0x1,
    hipHostMallocMapped        = 0x2,
    hipHostMallocWriteCombined = 0x4,
    hipHostMallocNumaUser      = 0x20000000,
    hipHostMallocCoherent      = 0x40000000,
    hipHostMallocNonCoherent   = 0x80000000,

    hipHostAllocDefault       = hipHostMallocDefault,
    hipHostAllocPortable      = hipHostMallocPortable,
    hipHostAllocMapped        = hipHostMallocMapped,
    hipHostAllocWriteCombined = hipHostMallocWriteCombined,

    //! Flags of hipExtMallocWithFlags: all memory is host memory
    hipDeviceMallocDefault     = 0x0,
    hipDeviceMallocFinegrained = 0x1,
    hipMallocSignalMemory      = 0x2,
    hipDeviceMallocUncached    = 0x3,

    //! Flags of hipEventCreateWithFlags that only AMD GPUs have
    hipEventDisableSystemFence = 0x20000000,
    hipEventReleaseToDevice    = 0x40000000,
    hipEventReleaseToSystem    = 0x80000000,

    hipDeviceScheduleMask = 0x7
};

inline cudaError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags = hipHostMallocDefault)
{
    return cudaHostAlloc(ptr, size, flags);
}

template <typename T>
inline cudaError_t hipHostMalloc(T **ptr, size_t size, unsigned int flags = hipHostMallocDefault)
{
    return hipHostMalloc(reinterpret_cast<void **>(ptr), size, flags);
}

//! The deprecated names of hipHostMalloc and hipHostFree
inline cudaError_t hipMallocHost(void **ptr, size_t size)
{
    return hipHostMalloc(ptr, size);
}

template <typename T>
inline cudaError_t hipMallocHost(T **ptr, size_t size)
{
    return hipHostMalloc(ptr, size);
}

inline cudaError_t hipHostAlloc(void **ptr, size_t size, unsigned int flags)
{
    return hipHostMalloc(ptr, size, flags);
}

template <typename T>
inline cudaError_t hipHostAlloc(T **ptr, size_t size, unsigned int flags)
{
    return hipHostMalloc(ptr, size, flags);
}

inline cudaError_t hipHostFree(void *ptr)
{
    return cudaFreeHost(ptr);
}

inline cudaError_t hipFreeHost(void *ptr)
{
    return cudaFreeHost(ptr);
}

//! AMD's allocation with flags: the flags choose the kind of GPU memory,
//! and all memory is host memory
inline cudaError_t hipExtMallocWithFlags(void **ptr, size_t size, unsigned int /* flags */)
{
    return cudaMalloc(ptr, size);
}

//
// HIP's driver API for memory: device pointers are void *
//

using hipDeviceptr_t = void *;

inline cudaError_t hipMemcpyHtoD(hipDeviceptr_t dst, const void *src, size_t count)
{
    return cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

inline cudaError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t count)
{
    return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}

inline cudaError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t count)
{
    return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}

inline cudaError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, const void *src, size_t count, cudaStream_t stream)
{
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
}

inline cudaError_t hipMemcpyDtoHAsync(void *dst, hipDeviceptr_t src, size_t count, cudaStream_t stream)
{
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
}

inline cudaError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t count, cudaStream_t stream)
{
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

//! A copy in stream that completes before it returns, as every copy does
inline cudaError_t hipMemcpyWithStream(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                                       cudaStream_t stream)
{
    if (cudaError_t err = cudaMemcpyAsync(dst, src, count, kind, stream))
        return err;
    return cudaStreamSynchronize(stream);
}

}

namespace detail {

//! A memset of count elements of size bytes, as the hipMemsetD* functions do
inline cudaError_t fill_elements(void *dst, unsigned int value, unsigned int size, size_t count,
                                 cudaStream_t stream)
{
    const cudaMemsetParams params{dst, count * size, value, size, count, 1};
    return capturing(stream) ? capture_fill(stream, params) : fill(params);
}

}

inline namespace hip_api {

inline cudaError_t hipMemsetD8(hipDeviceptr_t dst, unsigned char value, size_t count)
{
    return detail::fill_elements(dst, unsigned(value), 1, count, nullptr);
}

inline cudaError_t hipMemsetD16(hipDeviceptr_t dst, unsigned short value, size_t count)
{
    return detail::fill_elements(dst, unsigned(value), 2, count, nullptr);
}

inline cudaError_t hipMemsetD32(hipDeviceptr_t dst, int value, size_t count)
{
    return detail::fill_elements(dst, unsigned(value), 4, count, nullptr);
}

inline cudaError_t hipMemsetD8Async(hipDeviceptr_t dst, unsigned char value, size_t count,
                                    cudaStream_t stream = nullptr)
{
    return detail::fill_elements(dst, unsigned(value), 1, count, stream);
}

inline cudaError_t hipMemsetD16Async(hipDeviceptr_t dst, unsigned short value, size_t count,
                                     cudaStream_t stream = nullptr)
{
    return detail::fill_elements(dst, unsigned(value), 2, count, stream);
}

inline cudaError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, cudaStream_t stream = nullptr)
{
    return detail::fill_elements(dst, unsigned(value), 4, count, stream);
}

}

}

using namespace cuda4cpu::hip_api;

//! HIP_SYMBOL(x) names the symbol x in hipMemcpyToSymbol and the other symbol
//! functions, which take symbols by name, as in CUDA
#define HIP_SYMBOL(X) X

//! Calling convention of host callbacks, empty on Linux
#ifndef HIPRT_CB
#define HIPRT_CB
#endif

//
// The HIP names of CUDA's, by the header of cuda4cpu that defines them
//

// Errors (cuda/error.hpp)
#define hipGetLastError cudaGetLastError
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipPeekAtLastError cudaPeekAtLastError
#define hipErrorInvalidValue cudaErrorInvalidValue
#define hipErrorOutOfMemory cudaErrorMemoryAllocation
#define hipErrorMemoryAllocation cudaErrorMemoryAllocation
#define hipErrorNotInitialized cudaErrorInitializationError
#define hipErrorInitializationError cudaErrorInitializationError
#define hipErrorInvalidConfiguration cudaErrorInvalidConfiguration
#define hipErrorInvalidSymbol cudaErrorInvalidSymbol
#define hipErrorInvalidDevicePointer cudaErrorInvalidDevicePointer
#define hipErrorInvalidMemcpyDirection cudaErrorInvalidMemcpyDirection
#define hipErrorInvalidDeviceFunction cudaErrorInvalidDeviceFunction
#define hipErrorNoDevice cudaErrorNoDevice
#define hipErrorInvalidDevice cudaErrorInvalidDevice
#define hipErrorInvalidHandle cudaErrorInvalidResourceHandle
#define hipErrorInvalidResourceHandle cudaErrorInvalidResourceHandle
#define hipErrorIllegalState cudaErrorIllegalState
#define hipErrorNotReady cudaErrorNotReady
#define hipErrorAssert cudaErrorAssert
#define hipErrorLaunchFailure cudaErrorLaunchFailure
#define hipErrorCooperativeLaunchTooLarge cudaErrorCooperativeLaunchTooLarge
#define hipErrorNotSupported cudaErrorNotSupported
#define hipErrorStreamCaptureUnsupported cudaErrorStreamCaptureUnsupported
#define hipErrorStreamCaptureInvalidated cudaErrorStreamCaptureInvalidated
#define hipErrorStreamCaptureMerge cudaErrorStreamCaptureMerge
#define hipErrorStreamCaptureUnmatched cudaErrorStreamCaptureUnmatched
#define hipErrorStreamCaptureUnjoined cudaErrorStreamCaptureUnjoined
#define hipErrorStreamCaptureIsolation cudaErrorStreamCaptureIsolation
#define hipErrorStreamCaptureImplicit cudaErrorStreamCaptureImplicit
#define hipErrorGraphExecUpdateFailure cudaErrorGraphExecUpdateFailure
#define hipErrorUnknown cudaErrorUnknown
#define hipGetErrorString cudaGetErrorString

// Handles (cuda/types.hpp)
#define hipGraph_t cudaGraph_t
#define hipGraphNode_t cudaGraphNode_t
#define hipGraphExec_t cudaGraphExec_t
#define hipStream_t cudaStream_t
#define hipEvent_t cudaEvent_t
#define hipStreamCallback_t cudaStreamCallback_t

// Devices (cuda/device.hpp)
#define hipDeviceScheduleAuto cudaDeviceScheduleAuto
#define hipDeviceScheduleSpin cudaDeviceScheduleSpin
#define hipDeviceScheduleYield cudaDeviceScheduleYield
#define hipDeviceScheduleBlockingSync cudaDeviceScheduleBlockingSync
#define hipDeviceMapHost cudaDeviceMapHost
#define hipDeviceLmemResizeToMax cudaDeviceLmemResizeToMax
#define hipComputeMode cudaComputeMode
#define hipComputeModeDefault cudaComputeModeDefault
#define hipComputeModeExclusive cudaComputeModeExclusive
#define hipComputeModeProhibited cudaComputeModeProhibited
#define hipComputeModeExclusiveProcess cudaComputeModeExclusiveProcess
#define hipLimit_t cudaLimit
#define hipLimitStackSize cudaLimitStackSize
#define hipLimitPrintfFifoSize cudaLimitPrintfFifoSize
#define hipLimitMallocHeapSize cudaLimitMallocHeapSize
#define hipLimitDevRuntimeSyncDepth cudaLimitDevRuntimeSyncDepth
#define hipLimitDevRuntimePendingLaunchCount cudaLimitDevRuntimePendingLaunchCount
#define hipLimitMaxL2FetchGranularity cudaLimitMaxL2FetchGranularity
#define hipLimitPersistingL2CacheSize cudaLimitPersistingL2CacheSize
#define hipFuncCache_t cudaFuncCache
#define hipFuncCachePreferNone cudaFuncCachePreferNone
#define hipFuncCachePreferShared cudaFuncCachePreferShared
#define hipFuncCachePreferL1 cudaFuncCachePreferL1
#define hipFuncCachePreferEqual cudaFuncCachePreferEqual
#define hipSharedMemConfig cudaSharedMemConfig
#define hipSharedMemBankSizeDefault cudaSharedMemBankSizeDefault
#define hipSharedMemBankSizeFourByte cudaSharedMemBankSizeFourByte
#define hipSharedMemBankSizeEightByte cudaSharedMemBankSizeEightByte
#define hipFuncAttribute cudaFuncAttribute
#define hipFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#define hipFuncAttributePreferredSharedMemoryCarveout cudaFuncAttributePreferredSharedMemoryCarveout
#define hipFuncAttributes cudaFuncAttributes
#define hipAccessProperty cudaAccessProperty
#define hipAccessPropertyNormal cudaAccessPropertyNormal
#define hipAccessPropertyStreaming cudaAccessPropertyStreaming
#define hipAccessPropertyPersisting cudaAccessPropertyPersisting
#define hipAccessPolicyWindow cudaAccessPolicyWindow
#define hipLaunchAttributeID cudaLaunchAttributeID
#define hipLaunchAttributeAccessPolicyWindow cudaLaunchAttributeAccessPolicyWindow
#define hipLaunchAttributeSynchronizationPolicy cudaLaunchAttributeSynchronizationPolicy
#define hipLaunchAttributePriority cudaLaunchAttributePriority
#define hipLaunchAttributeValue cudaLaunchAttributeValue
#define hipStreamAttrID cudaStreamAttrID
#define hipStreamAttrValue cudaStreamAttrValue
#define hipStreamAttributeAccessPolicyWindow cudaStreamAttributeAccessPolicyWindow
#define hipStreamAttributePriority cudaStreamAttributePriority
#define hipUUID cudaUUID_t
#define hipSetDeviceFlags cudaSetDeviceFlags
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipDeviceReset cudaDeviceReset
#define hipGetDeviceCount cudaGetDeviceCount
#define hipSetDevice cudaSetDevice
#define hipGetDeviceFlags cudaGetDeviceFlags
#define hipDeviceGetStreamPriorityRange cudaDeviceGetStreamPriorityRange
#define hipDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define hipDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define hipDeviceDisablePeerAccess cudaDeviceDisablePeerAccess
#define hipGetDevice cudaGetDevice
#define hipDriverGetVersion cudaDriverGetVersion
#define hipRuntimeGetVersion cudaRuntimeGetVersion
#define hipDeviceSetCacheConfig cudaDeviceSetCacheConfig
#define hipDeviceGetCacheConfig cudaDeviceGetCacheConfig
#define hipDeviceSetSharedMemConfig cudaDeviceSetSharedMemConfig
#define hipDeviceGetSharedMemConfig cudaDeviceGetSharedMemConfig
#define hipFuncSetCacheConfig cudaFuncSetCacheConfig
#define hipFuncSetSharedMemConfig cudaFuncSetSharedMemConfig
#define hipFuncSetAttribute cudaFuncSetAttribute
#define hipFuncGetAttributes cudaFuncGetAttributes
#define hipStreamSetAttribute cudaStreamSetAttribute
#define hipStreamGetAttribute cudaStreamGetAttribute
#define hipDeviceSetLimit cudaDeviceSetLimit
#define hipDeviceGetLimit cudaDeviceGetLimit

// Memory (cuda/memory.hpp)
#define hipMemcpyKind cudaMemcpyKind
#define hipMemcpyHostToHost cudaMemcpyHostToHost
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipMemcpyDefault cudaMemcpyDefault
#define hipMemAttachGlobal cudaMemAttachGlobal
#define hipMemAttachHost cudaMemAttachHost
#define hipMemAttachSingle cudaMemAttachSingle
#define hipHostRegisterDefault cudaHostRegisterDefault
#define hipHostRegisterPortable cudaHostRegisterPortable
#define hipHostRegisterMapped cudaHostRegisterMapped
#define hipHostRegisterIoMemory cudaHostRegisterIoMemory
#define hipHostRegisterReadOnly cudaHostRegisterReadOnly
#define hipMemLocationType cudaMemLocationType
#define hipMemLocationTypeInvalid cudaMemLocationTypeInvalid
#define hipMemLocationTypeDevice cudaMemLocationTypeDevice
#define hipMemLocationTypeHost cudaMemLocationTypeHost
#define hipMemLocationTypeHostNuma cudaMemLocationTypeHostNuma
#define hipMemLocationTypeHostNumaCurrent cudaMemLocationTypeHostNumaCurrent
#define hipMemLocation cudaMemLocation
#define hipMemoryAdvise cudaMemoryAdvise
#define hipMemAdviseSetReadMostly cudaMemAdviseSetReadMostly
#define hipMemAdviseUnsetReadMostly cudaMemAdviseUnsetReadMostly
#define hipMemAdviseSetPreferredLocation cudaMemAdviseSetPreferredLocation
#define hipMemAdviseUnsetPreferredLocation cudaMemAdviseUnsetPreferredLocation
#define hipMemAdviseSetAccessedBy cudaMemAdviseSetAccessedBy
#define hipMemAdviseUnsetAccessedBy cudaMemAdviseUnsetAccessedBy
#define hipCpuDeviceId cudaCpuDeviceId
#define hipMemPool_t cudaMemPool_t
#define hipMemPoolAttr cudaMemPoolAttr
#define hipMemPoolReuseFollowEventDependencies cudaMemPoolReuseFollowEventDependencies
#define hipMemPoolReuseAllowOpportunistic cudaMemPoolReuseAllowOpportunistic
#define hipMemPoolReuseAllowInternalDependencies cudaMemPoolReuseAllowInternalDependencies
#define hipMemPoolAttrReleaseThreshold cudaMemPoolAttrReleaseThreshold
#define hipMemPoolAttrReservedMemCurrent cudaMemPoolAttrReservedMemCurrent
#define hipMemPoolAttrReservedMemHigh cudaMemPoolAttrReservedMemHigh
#define hipMemPoolAttrUsedMemCurrent cudaMemPoolAttrUsedMemCurrent
#define hipMemPoolAttrUsedMemHigh cudaMemPoolAttrUsedMemHigh
#define hipArray_t cudaArray_t
#define hipArray_const_t cudaArray_const_t
#define hipPitchedPtr cudaPitchedPtr
#define hipExtent cudaExtent
#define hipPos cudaPos
#define hipMemcpy3DParms cudaMemcpy3DParms
#define hipMemsetParams cudaMemsetParams
#define make_hipPitchedPtr make_cudaPitchedPtr
#define make_hipPos make_cudaPos
#define make_hipExtent make_cudaExtent
#define hipMalloc cudaMalloc
#define hipMallocManaged cudaMallocManaged
#define hipMallocAsync cudaMallocAsync
#define hipMallocFromPoolAsync cudaMallocFromPoolAsync
#define hipFreeAsync cudaFreeAsync
#define hipDeviceGetDefaultMemPool cudaDeviceGetDefaultMemPool
#define hipDeviceGetMemPool cudaDeviceGetMemPool
#define hipMemPoolCreate cudaMemPoolCreate
#define hipMemPoolDestroy cudaMemPoolDestroy
#define hipMemPoolSetAttribute cudaMemPoolSetAttribute
#define hipMemPoolGetAttribute cudaMemPoolGetAttribute
#define hipMemPoolTrimTo cudaMemPoolTrimTo
#define hipMemPrefetchAsync cudaMemPrefetchAsync
#define hipStreamAttachMemAsync cudaStreamAttachMemAsync
#define hipMemAdvise cudaMemAdvise
#define hipHostRegister cudaHostRegister
#define hipHostUnregister cudaHostUnregister
#define hipHostGetDevicePointer cudaHostGetDevicePointer
#define hipFree cudaFree
#define hipMemGetInfo cudaMemGetInfo
#define hipMemcpy cudaMemcpy
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemcpy2D cudaMemcpy2D
#define hipMemcpy2DAsync cudaMemcpy2DAsync
#define hipMemcpy3D cudaMemcpy3D
#define hipMemcpy3DAsync cudaMemcpy3DAsync
#define hipMallocPitch cudaMallocPitch
#define hipMalloc3D cudaMalloc3D
#define hipMemset cudaMemset
#define hipMemsetAsync cudaMemsetAsync
#define hipMemset2D cudaMemset2D
#define hipMemset2DAsync cudaMemset2DAsync
#define hipMemset3D cudaMemset3D
#define hipMemcpyToSymbol cudaMemcpyToSymbol
#define hipMemcpyFromSymbol cudaMemcpyFromSymbol
#define hipMemcpyToSymbolAsync cudaMemcpyToSymbolAsync
#define hipMemcpyFromSymbolAsync cudaMemcpyFromSymbolAsync
#define hipGetSymbolAddress cudaGetSymbolAddress
#define hipGetSymbolSize cudaGetSymbolSize

// Streams (cuda/streams.hpp)
#define hipStreamDefault cudaStreamDefault
#define hipStreamNonBlocking cudaStreamNonBlocking
#define hipStreamPerThread cudaStreamPerThread
#define hipStreamLegacy cudaStreamLegacy
#define hipHostFn_t cudaHostFn_t
#define hipLaunchHostFunc cudaLaunchHostFunc
#define hipStreamAddCallback cudaStreamAddCallback
#define hipStreamCreateWithPriority cudaStreamCreateWithPriority
#define hipStreamCreate cudaStreamCreate
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamGetFlags cudaStreamGetFlags
#define hipStreamGetPriority cudaStreamGetPriority
#define hipStreamQuery cudaStreamQuery
#define hipStreamSynchronize cudaStreamSynchronize

// Events (cuda/events.hpp)
#define hipEventDefault cudaEventDefault
#define hipEventBlockingSync cudaEventBlockingSync
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventInterprocess cudaEventInterprocess
#define hipEventCreate cudaEventCreate
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventDestroy cudaEventDestroy
#define hipEventElapsedTime cudaEventElapsedTime
#define hipEventQuery cudaEventQuery
#define hipEventRecord cudaEventRecord
#define hipEventRecordWithFlags cudaEventRecordWithFlags
#define hipStreamWaitEvent cudaStreamWaitEvent
#define hipEventSynchronize cudaEventSynchronize

// Launches and occupancy (launch.hpp)
#define hipLaunchCooperativeKernel cudaLaunchCooperativeKernel
#define hipKernelNodeParams cudaKernelNodeParams
#define hipLaunchKernel cudaLaunchKernel
#define hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define hipOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define hipOccupancyMaxPotentialBlockSize cudaOccupancyMaxPotentialBlockSize
#define hipOccupancyMaxPotentialBlockSizeVariableSMem cudaOccupancyMaxPotentialBlockSizeVariableSMem
#define hipOccupancyAvailableDynamicSMemPerBlock cudaOccupancyAvailableDynamicSMemPerBlock

// Graphs and stream capture (cuda/graphs.hpp)
#define hipGraphConditionalHandle cudaGraphConditionalHandle
#define hipGraphNodeType cudaGraphNodeType
#define hipGraphNodeTypeKernel cudaGraphNodeTypeKernel
#define hipGraphNodeTypeMemcpy cudaGraphNodeTypeMemcpy
#define hipGraphNodeTypeMemset cudaGraphNodeTypeMemset
#define hipGraphNodeTypeHost cudaGraphNodeTypeHost
#define hipGraphNodeTypeGraph cudaGraphNodeTypeGraph
#define hipGraphNodeTypeEmpty cudaGraphNodeTypeEmpty
#define hipGraphNodeTypeWaitEvent cudaGraphNodeTypeWaitEvent
#define hipGraphNodeTypeEventRecord cudaGraphNodeTypeEventRecord
#define hipGraphNodeTypeExtSemaphoreSignal cudaGraphNodeTypeExtSemaphoreSignal
#define hipGraphNodeTypeExtSemaphoreWait cudaGraphNodeTypeExtSemaphoreWait
#define hipGraphNodeTypeMemAlloc cudaGraphNodeTypeMemAlloc
#define hipGraphNodeTypeMemFree cudaGraphNodeTypeMemFree
#define hipGraphNodeTypeConditional cudaGraphNodeTypeConditional
#define hipGraphNodeTypeCount cudaGraphNodeTypeCount
#define hipStreamCaptureMode cudaStreamCaptureMode
#define hipStreamCaptureModeGlobal cudaStreamCaptureModeGlobal
#define hipStreamCaptureModeThreadLocal cudaStreamCaptureModeThreadLocal
#define hipStreamCaptureModeRelaxed cudaStreamCaptureModeRelaxed
#define hipStreamCaptureStatus cudaStreamCaptureStatus
#define hipStreamCaptureStatusNone cudaStreamCaptureStatusNone
#define hipStreamCaptureStatusActive cudaStreamCaptureStatusActive
#define hipStreamCaptureStatusInvalidated cudaStreamCaptureStatusInvalidated
#define hipStreamUpdateCaptureDependenciesFlags cudaStreamUpdateCaptureDependenciesFlags
#define hipStreamAddCaptureDependencies cudaStreamAddCaptureDependencies
#define hipStreamSetCaptureDependencies cudaStreamSetCaptureDependencies
#define hipGraphExecUpdateResult cudaGraphExecUpdateResult
#define hipGraphExecUpdateSuccess cudaGraphExecUpdateSuccess
#define hipGraphExecUpdateError cudaGraphExecUpdateError
#define hipGraphExecUpdateErrorTopologyChanged cudaGraphExecUpdateErrorTopologyChanged
#define hipGraphExecUpdateErrorNodeTypeChanged cudaGraphExecUpdateErrorNodeTypeChanged
#define hipGraphExecUpdateErrorFunctionChanged cudaGraphExecUpdateErrorFunctionChanged
#define hipGraphExecUpdateErrorParametersChanged cudaGraphExecUpdateErrorParametersChanged
#define hipGraphExecUpdateErrorNotSupported cudaGraphExecUpdateErrorNotSupported
#define hipGraphExecUpdateErrorUnsupportedFunctionChange cudaGraphExecUpdateErrorUnsupportedFunctionChange
#define hipGraphExecUpdateErrorAttributesChanged cudaGraphExecUpdateErrorAttributesChanged
#define hipGraphExecUpdateResultInfo cudaGraphExecUpdateResultInfo
#define hipGraphInstantiateFlags cudaGraphInstantiateFlags
#define hipGraphInstantiateFlagAutoFreeOnLaunch cudaGraphInstantiateFlagAutoFreeOnLaunch
#define hipGraphInstantiateFlagUpload cudaGraphInstantiateFlagUpload
#define hipGraphInstantiateFlagDeviceLaunch cudaGraphInstantiateFlagDeviceLaunch
#define hipGraphInstantiateFlagUseNodePriority cudaGraphInstantiateFlagUseNodePriority
#define hipGraphInstantiateResult cudaGraphInstantiateResult
#define hipGraphInstantiateSuccess cudaGraphInstantiateSuccess
#define hipGraphInstantiateError cudaGraphInstantiateError
#define hipGraphInstantiateInvalidStructure cudaGraphInstantiateInvalidStructure
#define hipGraphInstantiateNodeOperationNotSupported cudaGraphInstantiateNodeOperationNotSupported
#define hipGraphInstantiateMultipleDevicesNotSupported cudaGraphInstantiateMultipleDevicesNotSupported
#define hipGraphInstantiateParams cudaGraphInstantiateParams
#define hipGraphMemAttributeType cudaGraphMemAttributeType
#define hipGraphMemAttrUsedMemCurrent cudaGraphMemAttrUsedMemCurrent
#define hipGraphMemAttrUsedMemHigh cudaGraphMemAttrUsedMemHigh
#define hipGraphMemAttrReservedMemCurrent cudaGraphMemAttrReservedMemCurrent
#define hipGraphMemAttrReservedMemHigh cudaGraphMemAttrReservedMemHigh
#define hipGraphDebugDotFlags cudaGraphDebugDotFlags
#define hipGraphDebugDotFlagsVerbose cudaGraphDebugDotFlagsVerbose
#define hipGraphDependencyType cudaGraphDependencyType
#define hipGraphDependencyTypeDefault cudaGraphDependencyTypeDefault
#define hipGraphDependencyTypeProgrammatic cudaGraphDependencyTypeProgrammatic
#define hipGraphEdgeData cudaGraphEdgeData
#define hipKernelNodeParamsV2 cudaKernelNodeParamsV2
#define hipHostNodeParams cudaHostNodeParams
#define hipHostNodeParamsV2 cudaHostNodeParamsV2
#define hipMemsetParamsV2 cudaMemsetParamsV2
#define hipMemcpyNodeParams cudaMemcpyNodeParams
#define hipMemAllocationType cudaMemAllocationType
#define hipMemAllocationTypeInvalid cudaMemAllocationTypeInvalid
#define hipMemAllocationTypePinned cudaMemAllocationTypePinned
#define hipMemAllocationTypeMax cudaMemAllocationTypeMax
#define hipMemAllocationHandleType cudaMemAllocationHandleType
#define hipMemHandleTypeNone cudaMemHandleTypeNone
#define hipMemHandleTypePosixFileDescriptor cudaMemHandleTypePosixFileDescriptor
#define hipMemHandleTypeWin32 cudaMemHandleTypeWin32
#define hipMemHandleTypeWin32Kmt cudaMemHandleTypeWin32Kmt
#define hipMemHandleTypeFabric cudaMemHandleTypeFabric
#define hipMemAccessFlags cudaMemAccessFlags
#define hipMemAccessFlagsProtNone cudaMemAccessFlagsProtNone
#define hipMemAccessFlagsProtRead cudaMemAccessFlagsProtRead
#define hipMemAccessFlagsProtReadWrite cudaMemAccessFlagsProtReadWrite
#define hipMemAccessDesc cudaMemAccessDesc
#define hipMemPoolProps cudaMemPoolProps
#define hipMemAllocNodeParams cudaMemAllocNodeParams
#define hipMemAllocNodeParamsV2 cudaMemAllocNodeParamsV2
#define hipMemFreeNodeParams cudaMemFreeNodeParams
#define hipChildGraphNodeParams cudaChildGraphNodeParams
#define hipEventRecordNodeParams cudaEventRecordNodeParams
#define hipEventWaitNodeParams cudaEventWaitNodeParams
#define hipGraphSetConditional cudaGraphSetConditional
#define hipGraphConditionalHandleFlags cudaGraphConditionalHandleFlags
#define hipGraphCondAssignDefault cudaGraphCondAssignDefault
#define hipGraphConditionalNodeType cudaGraphConditionalNodeType
#define hipGraphCondTypeIf cudaGraphCondTypeIf
#define hipGraphCondTypeWhile cudaGraphCondTypeWhile
#define hipGraphCondTypeSwitch cudaGraphCondTypeSwitch
#define hipConditionalNodeParams cudaConditionalNodeParams
#define hipGraphNodeParams cudaGraphNodeParams
#define hipGraphCreate cudaGraphCreate
#define hipGraphDestroy cudaGraphDestroy
#define hipGraphClone cudaGraphClone
#define hipGraphAddKernelNode cudaGraphAddKernelNode
#define hipGraphKernelNodeGetParams cudaGraphKernelNodeGetParams
#define hipGraphKernelNodeSetParams cudaGraphKernelNodeSetParams
#define hipGraphAddMemcpyNode cudaGraphAddMemcpyNode
#define hipGraphAddMemcpyNode1D cudaGraphAddMemcpyNode1D
#define hipGraphMemcpyNodeGetParams cudaGraphMemcpyNodeGetParams
#define hipGraphMemcpyNodeSetParams cudaGraphMemcpyNodeSetParams
#define hipGraphMemcpyNodeSetParams1D cudaGraphMemcpyNodeSetParams1D
#define hipGraphAddMemsetNode cudaGraphAddMemsetNode
#define hipGraphMemsetNodeGetParams cudaGraphMemsetNodeGetParams
#define hipGraphMemsetNodeSetParams cudaGraphMemsetNodeSetParams
#define hipGraphAddHostNode cudaGraphAddHostNode
#define hipGraphHostNodeGetParams cudaGraphHostNodeGetParams
#define hipGraphHostNodeSetParams cudaGraphHostNodeSetParams
#define hipGraphAddEmptyNode cudaGraphAddEmptyNode
#define hipGraphAddChildGraphNode cudaGraphAddChildGraphNode
#define hipGraphChildGraphNodeGetGraph cudaGraphChildGraphNodeGetGraph
#define hipGraphAddEventRecordNode cudaGraphAddEventRecordNode
#define hipGraphAddEventWaitNode cudaGraphAddEventWaitNode
#define hipGraphEventRecordNodeGetEvent cudaGraphEventRecordNodeGetEvent
#define hipGraphEventWaitNodeGetEvent cudaGraphEventWaitNodeGetEvent
#define hipGraphAddMemAllocNode cudaGraphAddMemAllocNode
#define hipGraphMemAllocNodeGetParams cudaGraphMemAllocNodeGetParams
#define hipGraphAddMemFreeNode cudaGraphAddMemFreeNode
#define hipGraphMemFreeNodeGetParams cudaGraphMemFreeNodeGetParams
#define hipGraphConditionalHandleCreate cudaGraphConditionalHandleCreate
#define hipGraphAddNode cudaGraphAddNode
#define hipGraphAddDependencies cudaGraphAddDependencies
#define hipGraphRemoveDependencies cudaGraphRemoveDependencies
#define hipGraphDestroyNode cudaGraphDestroyNode
#define hipGraphGetNodes cudaGraphGetNodes
#define hipGraphGetRootNodes cudaGraphGetRootNodes
#define hipGraphGetEdges cudaGraphGetEdges
#define hipGraphNodeGetType cudaGraphNodeGetType
#define hipGraphNodeGetDependencies cudaGraphNodeGetDependencies
#define hipGraphNodeGetDependentNodes cudaGraphNodeGetDependentNodes
#define hipGraphDebugDotPrint cudaGraphDebugDotPrint
#define hipGraphInstantiate cudaGraphInstantiate
#define hipGraphInstantiateWithFlags cudaGraphInstantiateWithFlags
#define hipGraphInstantiateWithParams cudaGraphInstantiateWithParams
#define hipGraphExecDestroy cudaGraphExecDestroy
#define hipGraphLaunch cudaGraphLaunch
#define hipGraphUpload cudaGraphUpload
#define hipGraphExecUpdate cudaGraphExecUpdate
#define hipGraphExecKernelNodeSetParams cudaGraphExecKernelNodeSetParams
#define hipGraphExecMemcpyNodeSetParams cudaGraphExecMemcpyNodeSetParams
#define hipGraphExecMemcpyNodeSetParams1D cudaGraphExecMemcpyNodeSetParams1D
#define hipGraphExecMemsetNodeSetParams cudaGraphExecMemsetNodeSetParams
#define hipGraphExecHostNodeSetParams cudaGraphExecHostNodeSetParams
#define hipGraphExecChildGraphNodeSetParams cudaGraphExecChildGraphNodeSetParams
#define hipDeviceGetGraphMemAttribute cudaDeviceGetGraphMemAttribute
#define hipDeviceSetGraphMemAttribute cudaDeviceSetGraphMemAttribute
#define hipDeviceGraphMemTrim cudaDeviceGraphMemTrim
#define hipStreamBeginCapture cudaStreamBeginCapture
#define hipStreamBeginCaptureToGraph cudaStreamBeginCaptureToGraph
#define hipStreamEndCapture cudaStreamEndCapture
#define hipStreamIsCapturing cudaStreamIsCapturing
#define hipStreamGetCaptureInfo cudaStreamGetCaptureInfo
#define hipStreamUpdateCaptureDependencies cudaStreamUpdateCaptureDependencies
#define hipThreadExchangeStreamCaptureMode cudaThreadExchangeStreamCaptureMode

// Profiler (cuda_profiler_api.h)
#define hipProfilerStart cudaProfilerStart
#define hipProfilerStop cudaProfilerStop
