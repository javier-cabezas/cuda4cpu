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


// Loops with interleaved threads: cuda4cpu-rewrite inserts loop ticks in the
// outermost loops of device functions, and CUDA4CPU_INTERLEAVE fixes how often
// a thread lets the next one run (1: every iteration). Results must be the same
// with any interleaving: in grid-stride loops, around barriers and shuffles,
// with threads that leave a loop early, in device and host functions, and in
// constant evaluation.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

__global__ void grid_stride_copy(unsigned char *out, const unsigned char *in, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = in[i] + 1;
    }
}

// A running sum over rounds through shared memory, with barriers in the loop
__global__ void rounds_with_barriers(int *out, int rounds)
{
    __shared__ int s[128];
    int acc = 0;
    for (int r = 0; r < rounds; ++r) {
        s[threadIdx.x] = int(threadIdx.x) + r;
        __syncthreads();
        acc += s[(threadIdx.x + 1) % blockDim.x];
        __syncthreads();
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// Shuffles in the loop, and threads that leave it early
__global__ void shuffles_and_early_exits(int *out, int rounds)
{
    const int t = int(threadIdx.x);
    int v = t;
    int r = 0;
    while (r < rounds) {
        v += __shfl_xor_sync(0xffffffff, v, 1);
        ++r;
    }
    for (int k = 0; k < 64; ++k) {
        if (k == t % 7)
            break;
        v += 1;
    }
    out[t] = v;
}

__device__ int sum_below(int n)
{
    int s = 0;
    for (int i = 0; i < n; ++i) {
        s += i;
    }
    return s;
}

__global__ void device_function_loops(int *out)
{
    out[threadIdx.x] = sum_below(int(threadIdx.x));
}

// Thread 0 waits for thread 1, which runs only if thread 0 lets it: this
// finishes only when the loop ticks interleave the threads
__global__ void handoff(int *flag)
{
    if (threadIdx.x == 0) {
        while (atomicAdd(flag, 0) == 0) {
        }
    } else {
        atomicExch(flag, 1);
    }
}

// Evaluated at compile time: the tick does nothing there
__host__ __device__ constexpr int triangle(int n)
{
    int s = 0;
    do {
        s += n;
    } while (--n > 0);
    return s;
}
static_assert(triangle(4) == 10);

int main()
{
    {
        const int n = 1 << 20;
        std::vector<unsigned char> in(n), out(n, 0);
        for (int i = 0; i < n; ++i)
            in[i] = (unsigned char)(i * 7);
        grid_stride_copy<<<8, 64>>>(out.data(), in.data(), n);
        bool same = true;
        for (int i = 0; i < n; ++i)
            same = same && out[i] == (unsigned char)(in[i] + 1);
        EXPECT(same);
    }
    {
        const int blocks = 4, threads = 128, rounds = 10;
        std::vector<int> out(blocks * threads, -1);
        rounds_with_barriers<<<blocks, threads>>>(out.data(), rounds);
        bool same = true;
        for (int b = 0; b < blocks; ++b) {
            for (int t = 0; t < threads; ++t) {
                int expected = 0;
                for (int r = 0; r < rounds; ++r)
                    expected += (t + 1) % threads + r;
                same = same && out[b * threads + t] == expected;
            }
        }
        EXPECT(same);
    }
    {
        const int threads = 64, rounds = 5;
        std::vector<int> out(threads, -1);
        shuffles_and_early_exits<<<1, threads>>>(out.data(), rounds);
        std::vector<int> v(threads);
        for (int t = 0; t < threads; ++t)
            v[t] = t;
        for (int r = 0; r < rounds; ++r) {
            std::vector<int> next(threads);
            for (int t = 0; t < threads; ++t)
                next[t] = v[t] + v[t ^ 1];
            v = next;
        }
        bool same = true;
        for (int t = 0; t < threads; ++t)
            same = same && out[t] == v[t] + t % 7;
        EXPECT(same);
    }
    {
        std::vector<int> out(256, -1);
        device_function_loops<<<1, 256>>>(out.data());
        bool same = true;
        for (int t = 0; t < 256; ++t)
            same = same && out[t] == t * (t - 1) / 2;
        EXPECT(same);
    }
    if (const char *mode = std::getenv("CUDA4CPU_INTERLEAVE"); mode != nullptr && std::atoi(mode) > 0) {
        int flag = 0;
        handoff<<<1, 2>>>(&flag);
        EXPECT(flag == 1);
    }

    // On the host, the ticks of a __host__ __device__ function do nothing
    int total = 0;
    for (int i = 1; i <= 100000; ++i)
        total += triangle(i % 5 + 1) > 0;
    EXPECT(total == 100000);

    if (errors == 0)
        std::printf("interleave: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
