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
// Drop-in replacement for HIP's hip/hip_runtime.h: the runtime API of
// hip_runtime_api.h, plus what HIP adds to CUDA's kernel language. Kernels
// are written as in CUDA (__global__, threadIdx, __syncthreads(), atomicAdd,
// ...), and launched with kernel<<<...>>>(...), which cuda4cpu-rewrite
// rewrites, or with hipLaunchKernelGGL, a macro.
//

#pragma once

#include <chrono>
#include <cstdint>
#include <type_traits>

#include "hip_runtime_api.h"

namespace cuda4cpu {

inline namespace hip_api {

//
// Warp functions without a mask, as on AMD GPUs: every lane of the warp takes
// part, like the _sync functions with a full mask
//

template <typename T>
inline T __shfl(T var, int srcLane, int width = warpSize)
{
    return __shfl_sync(0xffffffffu, var, srcLane, width);
}

template <typename T>
inline T __shfl_up(T var, unsigned int delta, int width = warpSize)
{
    return __shfl_up_sync(0xffffffffu, var, delta, width);
}

template <typename T>
inline T __shfl_down(T var, unsigned int delta, int width = warpSize)
{
    return __shfl_down_sync(0xffffffffu, var, delta, width);
}

template <typename T>
inline T __shfl_xor(T var, int laneMask, int width = warpSize)
{
    return __shfl_xor_sync(0xffffffffu, var, laneMask, width);
}

//! Mask of the lanes whose predicate is non-zero. HIP's masks have 64 bits,
//! for AMD's wavefronts of 64 lanes; warps have 32.
inline unsigned long long __ballot(int predicate)
{
    return __ballot_sync(0xffffffffu, predicate);
}

inline int __any(int predicate)
{
    return __any_sync(0xffffffffu, predicate);
}

inline int __all(int predicate)
{
    return __all_sync(0xffffffffu, predicate);
}

//! Lane of the calling thread within its warp
inline unsigned int __lane_id()
{
    return thread_block::lane_id();
}

//
// Bit manipulation
//

//! The width bits of src0 that start at bit offset
inline unsigned int __bitextract_u32(unsigned int src0, unsigned int offset, unsigned int width)
{
    offset &= 31;
    width &= 31;
    return width == 0 ? 0 : (src0 >> offset) & ((1u << width) - 1);
}

inline uint64_t __bitextract_u64(uint64_t src0, unsigned int offset, unsigned int width)
{
    offset &= 63;
    width &= 63;
    return width == 0 ? 0 : (src0 >> offset) & ((uint64_t(1) << width) - 1);
}

//! src0 with the width bits that start at bit offset replaced by the low
//! bits of src1
inline unsigned int __bitinsert_u32(unsigned int src0, unsigned int src1, unsigned int offset,
                                    unsigned int width)
{
    offset &= 31;
    width &= 31;
    const unsigned int mask = ((1u << width) - 1) << offset;
    return (src0 & ~mask) | ((src1 << offset) & mask);
}

inline uint64_t __bitinsert_u64(uint64_t src0, uint64_t src1, unsigned int offset, unsigned int width)
{
    offset &= 63;
    width &= 63;
    const uint64_t mask = ((uint64_t(1) << width) - 1) << offset;
    return (src0 & ~mask) | ((src1 << offset) & mask);
}

//
// Clocks
//

//! HIP's name of clock64()
inline long long __clock64()
{
    return clock64();
}

//! A constant-rate clock, which counts at hipDeviceAttributeWallClockRate:
//! nanoseconds
inline long long wall_clock64()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

//
// AMD's atomics with other guarantees for floating-point additions: all of
// them are atomicAdd on the CPU
//

template <typename T>
inline T unsafeAtomicAdd(T *address, std::type_identity_t<T> val)
{
    return atomicAdd(address, val);
}

template <typename T>
inline T safeAtomicAdd(T *address, std::type_identity_t<T> val)
{
    return atomicAdd(address, val);
}

template <typename T>
inline void atomicAddNoRet(T *address, std::type_identity_t<T> val)
{
    atomicAdd(address, val);
}

}

}

//
// The device's features, which HIP defines when it compiles device code.
// cuda4cpu compiles host and device code at once: device code tests these.
// Funnel shifts, __syncthreads_count/_and/_or, surfaces and dynamic
// parallelism aren't implemented.
//

#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__     1
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__ 1
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__     1
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__ 1
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__         1
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__     1
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__     1
#define __HIP_ARCH_HAS_DOUBLES__                  1
#define __HIP_ARCH_HAS_WARP_VOTE__                1
#define __HIP_ARCH_HAS_WARP_BALLOT__              1
#define __HIP_ARCH_HAS_WARP_SHUFFLE__             1
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__        0
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__      1
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__          0
#define __HIP_ARCH_HAS_SURFACE_FUNCS__            0
#define __HIP_ARCH_HAS_3DGRID__                   1
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__         0

//
// The built-in variables under HIP's other names
//

#define hipThreadIdx_x (threadIdx.x)
#define hipThreadIdx_y (threadIdx.y)
#define hipThreadIdx_z (threadIdx.z)
#define hipBlockIdx_x  (blockIdx.x)
#define hipBlockIdx_y  (blockIdx.y)
#define hipBlockIdx_z  (blockIdx.z)
#define hipBlockDim_x  (blockDim.x)
#define hipBlockDim_y  (blockDim.y)
#define hipBlockDim_z  (blockDim.z)
#define hipGridDim_x   (gridDim.x)
#define hipGridDim_y   (gridDim.y)
#define hipGridDim_z   (gridDim.z)

//
// Launches. hipLaunchKernelGGL is a macro, as in HIP, so a kernel name with
// template arguments that contain commas goes in HIP_KERNEL_NAME(...). Like
// the launches cuda4cpu-rewrite writes, it calls the kernel from a lambda:
// template arguments are deduced from the arguments, default arguments
// apply, and the kernel can be inlined.
//
// As in HIP, a program that defines HIP_TEMPLATE_KERNEL_LAUNCH before
// including this header gets a function template instead, which takes the
// kernel as a function pointer: kernel<A, B> needs no HIP_KERNEL_NAME, but
// the template arguments can't be deduced.
//

#define HIP_KERNEL_NAME(...) __VA_ARGS__

#ifdef HIP_TEMPLATE_KERNEL_LAUNCH

namespace cuda4cpu {

inline namespace hip_api {

template <typename... Args, typename F = void (*)(Args...)>
inline void hipLaunchKernelGGL(F kernel, const dim3 &numBlocks, const dim3 &dimBlocks, uint32_t sharedMemBytes,
                               cudaStream_t stream, Args... args)
{
    launch(*kernel, numBlocks, dimBlocks, sharedMemBytes, stream).call(args...);
}

}

}

#else

#define hipLaunchKernelGGL(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...)               \
    ::cuda4cpu::launch([&](const auto &...cuda4cpu_args) { kernelName(cuda4cpu_args...); },            \
                       numBlocks, numThreads, memPerBlock, streamId).call(__VA_ARGS__)

#endif

//
// Dynamic shared memory. HIP_DYNAMIC_SHARED(type, name) declares
// extern __shared__ type name[]; cuda4cpu-rewrite rewrites it as it rewrites
// those declarations. Without the rewriter, this macro gives the pointer form,
// which works in functions only.
//

#define HIP_DYNAMIC_SHARED(type, var) type *var = ::cuda4cpu::dynamic_shared<type>()
#define HIP_DYNAMIC_SHARED_ATTRIBUTE
