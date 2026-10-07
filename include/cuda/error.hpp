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

#include "types.hpp"

namespace cuda4cpu {

namespace detail {

//! Last error returned by a runtime call on this thread, for cudaGetLastError
inline thread_local cudaError_t last_error = cudaSuccess;

//! Records err as the last error of the calling thread, unless it is
//! cudaSuccess, and returns it
inline cudaError_t record_error(cudaError_t err)
{
    if (err != cudaSuccess)
        last_error = err;
    return err;
}

}

inline namespace cuda_api {

//! Returns the last error produced by a runtime call on this thread, and
//! resets it to cudaSuccess
static inline
cudaError_t cudaGetLastError()
{
    cudaError_t err = detail::last_error;
    detail::last_error = cudaSuccess;
    return err;
}

//! Returns the last error produced by a runtime call on this thread
static inline
cudaError_t cudaPeekAtLastError()
{
    return detail::last_error;
}

static inline
const char *cudaGetErrorName(cudaError_t error)
{
    switch (error) {
    case cudaSuccess:                     return "cudaSuccess";
    case cudaErrorInvalidValue:           return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation:       return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:    return "cudaErrorInitializationError";
    case cudaErrorInvalidConfiguration:   return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidSymbol:          return "cudaErrorInvalidSymbol";
    case cudaErrorInvalidDevicePointer:   return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorNoDevice:               return "cudaErrorNoDevice";
    case cudaErrorInvalidDevice:          return "cudaErrorInvalidDevice";
    case cudaErrorInvalidResourceHandle:  return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady:               return "cudaErrorNotReady";
    case cudaErrorAssert:                 return "cudaErrorAssert";
    case cudaErrorLaunchFailure:          return "cudaErrorLaunchFailure";
    case cudaErrorCooperativeLaunchTooLarge: return "cudaErrorCooperativeLaunchTooLarge";
    case cudaErrorNotSupported:           return "cudaErrorNotSupported";
    case cudaErrorUnknown:                return "cudaErrorUnknown";
    }
    return "unrecognized error code";
}

static inline
const char *cudaGetErrorString(cudaError_t error)
{
    switch (error) {
    case cudaSuccess:                     return "no error";
    case cudaErrorInvalidValue:           return "invalid argument";
    case cudaErrorMemoryAllocation:       return "out of memory";
    case cudaErrorInitializationError:    return "initialization error";
    case cudaErrorInvalidConfiguration:   return "invalid configuration argument";
    case cudaErrorInvalidSymbol:          return "invalid device symbol";
    case cudaErrorInvalidDevicePointer:   return "invalid device pointer";
    case cudaErrorInvalidMemcpyDirection: return "invalid copy direction for memcpy";
    case cudaErrorNoDevice:               return "no CUDA-capable device is detected";
    case cudaErrorInvalidDevice:          return "invalid device ordinal";
    case cudaErrorInvalidResourceHandle:  return "invalid resource handle";
    case cudaErrorNotReady:               return "device not ready";
    case cudaErrorAssert:                 return "device-side assert triggered";
    case cudaErrorLaunchFailure:          return "unspecified launch failure";
    case cudaErrorCooperativeLaunchTooLarge:
        return "too many blocks in cooperative launch";
    case cudaErrorNotSupported:           return "operation not supported";
    case cudaErrorUnknown:                return "unknown error";
    }
    return "unrecognized error code";
}

}

}
