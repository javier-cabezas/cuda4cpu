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

static inline
cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    *event = new cudaEvent__;
    (*event)->stream = NULL;

    return 0;
}

static inline
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int /*flags*/)
{
    return cudaEventCreate(event);
}

static inline
cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    delete event;

    return 0;
}

static inline
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    *ms = std::chrono::duration_cast<std::chrono::microseconds>(end->tstamp - start->tstamp).count();

    return 0;
}

static inline
cudaError_t cudaEventQuery(cudaEvent_t event)
{
    return 0;
}

static inline
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    event->tstamp = std::chrono::system_clock::now();
    event->stream = stream;

    return 0;
}

static inline
cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    return 0;
}

}
