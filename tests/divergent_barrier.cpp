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
// Threads of a block that reach different __syncthreads() calls, which CUDA
// doesn't allow. cuda4cpu warns, aborts or ignores it depending on
// CUDA4CPU_DIVERGENT_BARRIERS; the tests check its output.
//

#include <cstdio>

#include "cuda4cpu.hpp"

using namespace cuda4cpu;

__global__ void uniform(int *out)
{
    out[threadIdx.x] = 1;
    __syncthreads();
    out[threadIdx.x] += 1;
}

__global__ void divergent(int *out)
{
    if (threadIdx.x % 2 == 0) {
        out[threadIdx.x] = 1;
        __syncthreads();
    } else {
        out[threadIdx.x] = 2;
        __syncthreads();
    }
}

int main()
{
    int out[64];
    launch(uniform, 4, 64).call(out);
    std::printf("uniform: done\n");
    launch(divergent, 4, 64).call(out);
    std::printf("divergent: done\n");
}
