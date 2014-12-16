/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
 */

#pragma once

#include "types.hpp"

namespace cuda4cpu {

static
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

static
cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    delete event;

    return 0;
}

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
