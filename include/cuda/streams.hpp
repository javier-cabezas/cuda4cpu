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

namespace cuda4cpu {

static inline
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int /*flags*/)
{
    callback(stream, 0, userData);

    return 0;
}

static inline
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority)
{
    *stream = new cudaStream__;
    (*stream)->flags    = flags;
    (*stream)->priority = priority;

    return 0;
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

    return 0;
}

static inline
cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags)
{
    *flags = stream->flags;

    return 0;
}

static inline
cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority)
{
    *priority = stream->priority;

    return 0;
}

static inline
cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    return 0;
}

static inline
cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    return 0;
}

static inline
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int /* flags */)
{
    return 0;
}


}
