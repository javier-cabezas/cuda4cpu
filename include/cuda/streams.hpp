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

inline namespace cuda_api {

enum : unsigned int {
    cudaStreamDefault     = 0x00,
    cudaStreamNonBlocking = 0x01
};

using cudaHostFn_t = void (*)(void *userData);

//! Runs fn in order with the work in stream, which has completed
static inline
cudaError_t cudaLaunchHostFunc(cudaStream_t /* stream */, cudaHostFn_t fn, void *userData)
{
    fn(userData);
    return cudaSuccess;
}

static inline
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int /*flags*/)
{
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
cudaError_t cudaStreamQuery(cudaStream_t /* stream */)
{
    return cudaSuccess;
}

static inline
cudaError_t cudaStreamSynchronize(cudaStream_t /* stream */)
{
    return cudaSuccess;
}

static inline
cudaError_t cudaStreamWaitEvent(cudaStream_t /* stream */, cudaEvent_t /* event */, unsigned int /* flags */ = 0)
{
    return cudaSuccess;
}

}

}
