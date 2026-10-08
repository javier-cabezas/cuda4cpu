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

// Stream flags, which are macros as in CUDA: code passes cudaStreamDefault
// where a stream goes, which needs a null pointer constant
#define cudaStreamDefault     0x00
#define cudaStreamNonBlocking 0x01

//! The default stream of the calling thread. Work runs when it's issued, so
//! it differs from the legacy default stream (cudaStreamLegacy, the null
//! stream) only in stream capture, which works on it as on other streams.
#define cudaStreamPerThread (::cuda4cpu::detail::per_thread_stream())
#define cudaStreamLegacy    (static_cast<::cuda4cpu::cudaStream_t>(nullptr))

namespace cuda4cpu {

inline namespace cuda_api {

using cudaHostFn_t = void (*)(void *userData);

}

namespace detail {

// Defined in the library, with the graphs
cudaError_t capture_host(cudaStream_t stream, cudaHostFn_t fn, void *userData);
void abandon_capture(cudaStream_t stream);

//! The stream that cudaStreamPerThread names: one per OS thread
inline cudaStream_t per_thread_stream()
{
    static thread_local cudaStream__ stream{};
    return &stream;
}

}

inline namespace cuda_api {

//! Runs fn in order with the work in stream, which has completed
static inline
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
    if (detail::capturing(stream))
        return detail::capture_host(stream, fn, userData);
    fn(userData);
    return cudaSuccess;
}

//! Not allowed in stream capture, as in CUDA: use cudaLaunchHostFunc
static inline
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int /*flags*/)
{
    if (detail::capturing(stream))
        return detail::record_error(cudaErrorStreamCaptureUnsupported);
    callback(stream, cudaSuccess, userData);

    return cudaSuccess;
}

static inline
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority)
{
    *stream = new cudaStream__;
    (*stream)->flags    = flags;
    (*stream)->priority = priority;

    return cudaSuccess;
}

static inline
cudaError_t cudaStreamCreate(cudaStream_t *stream)
{
    return cudaStreamCreateWithPriority(stream, 0, 0);
}

static inline
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
{
    return cudaStreamCreateWithPriority(stream, flags, 0);
}

static inline
cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    if (stream == detail::per_thread_stream())
        return detail::record_error(cudaErrorInvalidResourceHandle);
    if (detail::capturing(stream))
        detail::abandon_capture(stream);
    delete stream;

    return cudaSuccess;
}

static inline
cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags)
{
    *flags = stream->flags;

    return cudaSuccess;
}

static inline
cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority)
{
    *priority = stream->priority;

    return cudaSuccess;
}

static inline
cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    return detail::capturing(stream) ? detail::record_error(cudaErrorStreamCaptureUnsupported) : cudaSuccess;
}

//! Work runs when it's issued, so there is nothing to wait for. A stream being
//! captured can't be synchronized, as in CUDA.
static inline
cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    return detail::capturing(stream) ? detail::record_error(cudaErrorStreamCaptureUnsupported) : cudaSuccess;
}

}

}
