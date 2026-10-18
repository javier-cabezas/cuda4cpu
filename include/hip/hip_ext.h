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
// AMD's extensions of HIP: launches between two events. The flags choose
// AMD GPU behavior and are ignored. hipExtLaunchKernelGGL is a macro, like
// hipLaunchKernelGGL, or a function template with HIP_TEMPLATE_KERNEL_LAUNCH.
//

#pragma once

#include <cstdint>

#include "hip_runtime.h"

namespace cuda4cpu {

namespace detail {

//! Records an event of a launch between events, if there is one
inline void record_launch_event(cudaEvent_t event, cudaStream_t stream)
{
    if (event != nullptr)
        cudaEventRecord(event, stream);
}

}

inline namespace hip_api {

inline cudaError_t hipExtLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args,
                                      size_t sharedMemBytes, cudaStream_t stream, cudaEvent_t startEvent,
                                      cudaEvent_t stopEvent, int /* flags */)
{
    detail::record_launch_event(startEvent, stream);
    const cudaError_t err = cudaLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    detail::record_launch_event(stopEvent, stream);
    return err;
}

#ifdef HIP_TEMPLATE_KERNEL_LAUNCH

template <typename... Args, typename F = void (*)(Args...)>
inline void hipExtLaunchKernelGGL(F kernel, const dim3 &numBlocks, const dim3 &dimBlocks, std::uint32_t sharedMemBytes,
                                  cudaStream_t stream, cudaEvent_t startEvent, cudaEvent_t stopEvent,
                                  std::uint32_t /* flags */, Args... args)
{
    detail::record_launch_event(startEvent, stream);
    hipLaunchKernelGGL(kernel, numBlocks, dimBlocks, sharedMemBytes, stream, args...);
    detail::record_launch_event(stopEvent, stream);
}

#endif

}

}

#ifndef HIP_TEMPLATE_KERNEL_LAUNCH

#define hipExtLaunchKernelGGL(kernelName, numBlocks, numThreads, memPerBlock, streamId, startEvent, stopEvent, \
                              flags, ...)                                                                   \
    do {                                                                                                    \
        ::cuda4cpu::detail::record_launch_event(startEvent, streamId);                                      \
        hipLaunchKernelGGL(kernelName, numBlocks, numThreads, memPerBlock, streamId, __VA_ARGS__);          \
        ::cuda4cpu::detail::record_launch_event(stopEvent, streamId);                                       \
        (void)(flags);                                                                                      \
    } while (0)

#endif
