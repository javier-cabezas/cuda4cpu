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

// Kernels that crash, one per argument: crash_test.cmake checks that each
// report names the CUDA thread, its block and the line of the launch, which
// the program prints before launching, and that the program still dies of
// the signal (or exits in its own handler).

#undef NDEBUG
#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#define LAUNCH_LINE() std::fprintf(stderr, "launch at line %d\n", __LINE__)

__global__ void invalid_access(int *p)
{
    if (blockIdx.x == 3 && threadIdx.x == 17)
        p[1 << 30] = 1;
}

//! Crashes after a barrier, in fiber mode, with 2D blocks
__global__ void after_barrier(int *p)
{
    __syncthreads();
    if (blockIdx.x == 1 && blockIdx.y == 2 && threadIdx.x == 5 && threadIdx.y == 1)
        p[1 << 30] = 1;
    __syncthreads();
}

__device__ int deep(int n)
{
    volatile char frame[4096];
    frame[0] = char(n);
    return n == 0 ? frame[0] : deep(n - 1) + frame[0];
}

__global__ void overflow(int *p, int depth)
{
    if (threadIdx.x == 5)
        p[0] = deep(depth);
}

__global__ void divide(int *p, int divisor)
{
    if (threadIdx.x == 2)
        p[0] = 7 / divisor;
}

__global__ void asserts(int *p)
{
    assert(threadIdx.x != 9);
    p[threadIdx.x] = 1;
}

void own_handler(int)
{
    const char text[] = "the program's handler\n";
    ssize_t written = write(STDERR_FILENO, text, sizeof(text) - 1);
    (void)written;
    _exit(3);
}

int main(int argc, char **argv)
{
    if (argc != 2)
        return 2;
    const char *what = argv[1];
    int *p;
    cudaMalloc(&p, 1024);
    int zero = argc - 2, depth = 1 << 20;
    if (std::strcmp(what, "invalid_access") == 0) {
        LAUNCH_LINE(); invalid_access<<<4, 32>>>(p);
    } else if (std::strcmp(what, "after_barrier") == 0) {
        LAUNCH_LINE(); after_barrier<<<dim3(2, 3), dim3(8, 4)>>>(p);
    } else if (std::strcmp(what, "overflow") == 0) {
        LAUNCH_LINE(); overflow<<<1, 32>>>(p, depth);
    } else if (std::strcmp(what, "divide") == 0) {
        LAUNCH_LINE(); divide<<<1, 32>>>(p, zero);
    } else if (std::strcmp(what, "assert") == 0) {
        void *args[] = {&p};
        LAUNCH_LINE(); cudaLaunchKernel(asserts, dim3(1), dim3(32), args);
    } else if (std::strcmp(what, "own_handler") == 0) {
        std::signal(SIGSEGV, own_handler);
        LAUNCH_LINE(); invalid_access<<<4, 32>>>(p);
    } else if (std::strcmp(what, "host") == 0) {
        // No report for a crash outside kernels
        invalid_access<<<1, 1>>>(p);
        std::raise(SIGSEGV);
    }
    cudaDeviceSynchronize();
    std::printf("no crash\n");
    return 0;
}
