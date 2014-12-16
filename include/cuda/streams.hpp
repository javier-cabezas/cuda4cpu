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

namespace cuda4cpu {

static inline
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int /*flags*/)
{
    callback(stream, 0, userData);

    return 0;
}

static
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
