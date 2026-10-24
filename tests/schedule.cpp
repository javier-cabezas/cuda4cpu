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
// Missing __syncthreads() and __syncwarp(): each thread reads the shared
// memory that the previous thread wrote, without a barrier. In the default
// order the previous thread has always run first, which hides both bugs;
// warp_reverse and warp_random expose the missing __syncthreads() between
// warps while keeping lanes within each warp in forward order; reverse,
// random and mixed expose both. The same kernels with the barriers are
// correct in every order. Usage: schedule hidden|warp_exposed|exposed
//

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda4cpu.hpp"

using namespace cuda4cpu;

constexpr unsigned N = 64;

template <bool Barrier>
__global__ void shift(int *out, const int *in)
{
    __shared__ int s[N];
    unsigned t = threadIdx.x;
    s[t] = in[blockIdx.x * N + t];
    if (Barrier)
        __syncthreads();
    out[blockIdx.x * N + t] = t > 0 ? s[t - 1] : -1;
}

template <bool Sync>
__global__ void warp_shift(int *out, const int *in)
{
    __shared__ int s[N];
    unsigned t = threadIdx.x;
    s[t] = in[blockIdx.x * N + t];
    if (Sync)
        __syncwarp();
    out[blockIdx.x * N + t] = t % 32 > 0 ? s[t - 1] : -1;
}

template <bool Barrier>
static bool correct()
{
    const unsigned blocks = 4;
    int in[blocks * N], out[blocks * N];
    for (unsigned i = 0; i < blocks * N; ++i)
        in[i] = int(i) * 3;
    launch(shift<Barrier>, blocks, N).call(out, in);
    for (unsigned i = 0; i < blocks * N; ++i)
        if (out[i] != (i % N > 0 ? in[i - 1] : -1))
            return false;
    return true;
}

template <bool Sync>
static bool warp_correct()
{
    const unsigned blocks = 4;
    int in[blocks * N], out[blocks * N];
    for (unsigned i = 0; i < blocks * N; ++i)
        in[i] = int(i) * 3;
    launch(warp_shift<Sync>, blocks, N).call(out, in);
    for (unsigned i = 0; i < blocks * N; ++i)
        if (out[i] != (i % 32 > 0 ? in[i - 1] : -1))
            return false;
    return true;
}

int main(int argc, char **argv)
{
    if (argc != 2 || (std::strcmp(argv[1], "hidden") != 0 &&
                      std::strcmp(argv[1], "warp_exposed") != 0 &&
                      std::strcmp(argv[1], "exposed") != 0)) {
        std::fprintf(stderr, "usage: schedule hidden|warp_exposed|exposed\n");
        return EXIT_FAILURE;
    }
    const bool expect_block_exposed = std::strcmp(argv[1], "hidden") != 0;
    const bool expect_warp_exposed  = std::strcmp(argv[1], "exposed") == 0;

    int errors = 0;
    if (!correct<true>() || !warp_correct<true>()) {
        std::printf("the kernel with the barrier gave wrong results\n");
        ++errors;
    }
    if (correct<false>() == expect_block_exposed) {
        std::printf("the missing __syncthreads() was %s\n", expect_block_exposed ? "hidden" : "exposed");
        ++errors;
    }
    if (warp_correct<false>() == expect_warp_exposed) {
        std::printf("the missing __syncwarp() was %s\n", expect_warp_exposed ? "hidden" : "exposed");
        ++errors;
    }
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
