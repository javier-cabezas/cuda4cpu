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


#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda4cpu.hpp"

using namespace cuda4cpu;

static constexpr unsigned N = 64;

static unsigned errors = 0;

// A broken scheduler can end the process from inside a kernel with exit(0),
// which would look like a pass
static bool finished = false;

static void fail_if_unfinished()
{
    if (!finished) {
        std::cout << "process exited before the tests completed" << std::endl;
        std::_Exit(EXIT_FAILURE);
    }
}

static void check(const char *test, const std::vector<int> &out, const std::vector<int> &expected)
{
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] != expected[i]) {
            std::cout << test << ": out[" << i << "] = " << out[i]
                      << ", expected " << expected[i] << "\n";
            ++errors;
            return;
        }
    }
}

// Thread `skip` returns before the barrier. Like on GPUs, the remaining
// threads must still get past it.
__global__ void early_exit(int *out, unsigned skip)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == skip)
        return;
    out[t] = 1;
    __syncthreads();
    out[t] = 2;
}

// Odd threads return between barriers; the even ones keep exchanging data
// through shared memory, which is only correct if every barrier holds.
__global__ void exit_between_barriers(int *out)
{
    __shared__ int s[N];
    unsigned t = threadIdx.x;

    s[t] = int(t) + 1;
    __syncthreads();
    if (t % 2 == 1)
        return;

    int v = s[(t + N - 2) % N];
    __syncthreads();
    s[t] = v * 10;
    __syncthreads();
    out[blockIdx.x * N + t] = s[(t + 2) % N];
}

// Only odd blocks use barriers, and their first threads return before
// reaching one, so the switch to fiber mode happens in the middle of a block.
__global__ void barrier_in_some_blocks(int *out)
{
    __shared__ int s[N];
    unsigned t = threadIdx.x;
    unsigned g = blockIdx.x * N + t;

    if (blockIdx.x % 2 == 0) {
        out[g] = int(t);
        return;
    }
    if (t < 5)
        return;
    s[t] = int(t);
    __syncthreads();
    out[g] = s[5 + (N - 1 - t)];
}

// Only the last thread reaches the barrier
__global__ void lone_barrier(int *out)
{
    if (threadIdx.x != N - 1)
        return;
    __syncthreads();
    out[blockIdx.x] = 7;
}

// In a 3D block, the threads before `first` (in index order) return early, so
// the block reaches the barrier in the middle of direct mode, at a thread
// whose y and z are not 0. Every other thread must get past it with its own
// index, and the threads that returned must not run again: they count their
// runs.
__global__ void barrier_in_3d_block(int *out, unsigned first)
{
    const unsigned t = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    if (t < first) {
        out[t] += 1;
        return;
    }
    out[t] = 0;
    __syncthreads();
    out[t] = int(threadIdx.x + 10 * threadIdx.y + 100 * threadIdx.z);
}

// The same with a shuffle as the first warp operation: the lanes of a 3D
// block must come out right in direct mode too. Lanes before `first` aren't
// in the mask.
__global__ void shuffle_in_3d_block(int *out, unsigned first)
{
    const unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    const unsigned t = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    if (t < first) {
        out[t] += 1;
        return;
    }
    const unsigned warp = t / 32 * 32;
    const unsigned begin = first > warp ? first - warp : 0, end = threads - warp < 32 ? threads - warp : 32;
    const unsigned mask = (end == 32 ? ~0u : (1u << end) - 1) & ~((1u << begin) - 1);
    out[t] = __shfl_xor_sync(mask, int(t), 1);
}

int main()
{
    std::atexit(fail_if_unfinished);

    const unsigned blocks = 8;

    for (unsigned skip : {0u, 5u, N - 1}) {
        std::vector<int> out(blocks * N, 0), expected(blocks * N, 2);
        for (unsigned b = 0; b < blocks; ++b) expected[b * N + skip] = 0;
        launch(early_exit, blocks, N).call(out.data(), skip);
        check("early_exit", out, expected);
    }

    {
        std::vector<int> out(blocks * N, 0), expected(blocks * N, 0);
        for (unsigned b = 0; b < blocks; ++b)
            for (unsigned t = 0; t < N; t += 2) expected[b * N + t] = int(t + 1) * 10;
        launch(exit_between_barriers, blocks, N).call(out.data());
        check("exit_between_barriers", out, expected);
    }

    {
        std::vector<int> out(blocks * N, -1), expected(blocks * N, -1);
        for (unsigned b = 0; b < blocks; ++b) {
            for (unsigned t = 0; t < N; ++t) {
                if (b % 2 == 0)  expected[b * N + t] = int(t);
                else if (t >= 5) expected[b * N + t] = int(5 + (N - 1 - t));
            }
        }
        launch(barrier_in_some_blocks, blocks, N).call(out.data());
        check("barrier_in_some_blocks", out, expected);
    }

    {
        std::vector<int> out(blocks, 0), expected(blocks, 7);
        launch(lone_barrier, blocks, N).call(out.data());
        check("lone_barrier", out, expected);
    }

    // 60 threads: a full warp and one of 28 lanes. Thread 46 is (2, 2, 3).
    for (unsigned first : {0u, 46u}) {
        const dim3 block(4, 3, 5);
        std::vector<int> out(60, 0), expected(60, 1);
        for (unsigned t = first; t < 60; ++t)
            expected[t] = int(t % 4 + 10 * (t / 4 % 3) + 100 * (t / 12));
        launch(barrier_in_3d_block, 1, block).call(out.data(), first);
        check("barrier_in_3d_block", out, expected);

        std::fill(out.begin(), out.end(), 0);
        for (unsigned t = first; t < 60; ++t)
            expected[t] = int(t ^ 1);
        launch(shuffle_in_3d_block, 1, block).call(out.data(), first);
        check("shuffle_in_3d_block", out, expected);
    }

    finished = true;
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
