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


// A templated kernel with dynamic shared memory and a launch in a local
// header, as is common in CUDA code: cuda4cpu-rewrite rewrites headers
// included with #include "..." too.

#pragma once

template <typename T>
__global__ void block_sum(const T *in, T *out)
{
    extern __shared__ T partial[];
    partial[threadIdx.x] = in[threadIdx.x];
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *out = partial[0];
}

//! Sum of n elements, n a power of two of up to 1024
template <typename T>
T header_sum(const T *in, unsigned n)
{
    T out{};
    block_sum<<<1, n, n * sizeof(T)>>>(in, &out);
    return out;
}
