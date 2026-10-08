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
    cudaEventDefault       = 0x00,
    cudaEventBlockingSync  = 0x01,
    cudaEventDisableTiming = 0x02,
    cudaEventInterprocess  = 0x04
};

static inline
cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    *event = new cudaEvent__;
    (*event)->stream = nullptr;

    return cudaSuccess;
}

static inline
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int /*flags*/)
{
    return cudaEventCreate(event);
}

//! The C++ overload of cuda_runtime.h
static inline
cudaError_t cudaEventCreate(cudaEvent_t *event, unsigned int flags)
{
    return cudaEventCreateWithFlags(event, flags);
}

static inline
cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    delete event;

    return cudaSuccess;
}

static inline
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    *ms = std::chrono::duration<float, std::milli>(end->tstamp - start->tstamp).count();

    return cudaSuccess;
}

static inline
cudaError_t cudaEventQuery(cudaEvent_t /* event */)
{
    return cudaSuccess;
}

}

namespace detail {

// Defined in the library, with the graphs
cudaError_t capture_event_record(cudaEvent_t event, cudaStream_t stream);
cudaError_t capture_wait_event(cudaStream_t stream, cudaEvent_t event);

}

inline namespace cuda_api {

//! In a stream being captured, records which captured work the event follows,
//! for the streams that wait for it
static inline
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr)
{
    if (detail::capturing(stream))
        return detail::capture_event_record(event, stream);
    event->tstamp = std::chrono::system_clock::now();
    event->stream = stream;
    event->capture.reset();
    event->capture_deps.clear();

    return cudaSuccess;
}

static inline
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = nullptr, unsigned int /* flags */ = 0)
{
    return cudaEventRecord(event, stream);
}

//! Work runs when it's issued, so a stream never has to wait for an event,
//! except in stream capture: waiting for an event recorded in a stream being
//! captured makes stream join that capture, after the event's work.
static inline
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int /* flags */ = 0)
{
    if (event != nullptr && (event->capture || detail::capturing(stream)))
        return detail::capture_wait_event(stream, event);
    return cudaSuccess;
}

static inline
cudaError_t cudaEventSynchronize(cudaEvent_t /* event */)
{
    return cudaSuccess;
}

}

}
