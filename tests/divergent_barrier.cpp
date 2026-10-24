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
#include <cstring>

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

// Only thread 1 takes the other call, after a barrier that every thread
// reaches, which starts each one's fiber (`divergent_barrier later`). The
// first and last threads to arrive agree. tests/CMakeLists.txt expects the
// report to name the lines of these two calls.
__global__ void divergent_later(int *out)
{
    out[threadIdx.x] = 0;
    __syncthreads();
    if (threadIdx.x != 1) {
        out[threadIdx.x] = 1;
        __syncthreads();
    } else {
        out[threadIdx.x] = 2;
        __syncthreads();
    }
}

// Lanes of a warp that execute different warp operations with overlapping masks
__global__ void divergent_warp(int *out)
{
    if (threadIdx.x % 32 < 16)
        out[threadIdx.x] = __shfl_xor_sync(0xffffffff, int(threadIdx.x), 1);
    else
        out[threadIdx.x] = int(__ballot_sync(0xffffffff, 1));
}

// A lane that calls a warp operation with a mask that omits its own lane bit
__global__ void missing_self_mask(int *out)
{
    out[threadIdx.x] = __shfl_sync(0xfffffffe, int(threadIdx.x), 1);
}

int main(int argc, char **argv)
{
    int out[64];
    launch(uniform, 4, 64).call(out);
    std::printf("uniform: done\n");
    if (argc > 1 && std::strcmp(argv[1], "later") == 0)
        launch(divergent_later, 4, 64).call(out);
    else if (argc > 1 && std::strcmp(argv[1], "warp") == 0)
        launch(divergent_warp, 4, 64).call(out);
    else if (argc > 1 && std::strcmp(argv[1], "warp_mask") == 0)
        launch(missing_self_mask, 4, 64).call(out);
    else
        launch(divergent, 4, 64).call(out);
    std::printf("divergent: done\n");
}
