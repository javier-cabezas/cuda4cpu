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


#include <csignal>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "cuda4cpu.hpp"

using namespace cuda4cpu;

// Uses `Bytes` of stack across a barrier, so that every fiber's stack is live at once
template <size_t Bytes>
__global__ void big_frame(int *out)
{
    volatile unsigned char buf[Bytes];
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = 0; i < Bytes; i += 512) buf[i] = (unsigned char)(t + i);
    __syncthreads();

    int ok = 1;
    for (size_t i = 0; i < Bytes; i += 512) ok &= buf[i] == (unsigned char)(t + i);
    out[t] = ok;
}

__device__ unsigned recurse(unsigned depth)
{
    volatile char frame[1024];
    frame[0] = char(depth);
    if (depth == 0)
        return frame[0];
    return recurse(depth - 1) + frame[0];
}

__global__ void overflow(unsigned *out, unsigned depth)
{
    out[threadIdx.x] = recurse(depth);
}

static unsigned errors = 0;

template <size_t Bytes>
static void run_big_frame(const char *name)
{
    const unsigned blocks = 4, threads = 64;
    std::vector<int> out(blocks * threads, 0);
    launch(big_frame<Bytes>, blocks, threads).call(out.data());
    for (int ok : out) {
        if (!ok) {
            std::cout << name << ": stack contents were corrupted\n";
            ++errors;
            return;
        }
    }
}

static void expect_limit(size_t expected)
{
    size_t value = 0;
    cudaDeviceGetLimit(&value, cudaLimitStackSize);
    if (value != expected) {
        std::cout << "cudaLimitStackSize is " << value << ", expected " << expected << "\n";
        ++errors;
    }
}

int main()
{
    // A stack overflow must hit the guard page and crash, not corrupt memory.
    // Fork before any kernel launch, while the process is still single threaded.
    pid_t pid = fork();
    if (pid == 0) {
        unsigned out[32];
        launch(overflow, 1, 32).call(out, 1u << 20);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGSEGV) {
        std::cout << "stack overflow did not raise SIGSEGV (status " << status << ")\n";
        ++errors;
    }

    expect_limit(detail::min_stack_size);
    run_big_frame<48 * 1024>("48 KiB frames with the default stack size");

    // Requests below the minimum are rounded up
    cudaDeviceSetLimit(cudaLimitStackSize, 1024);
    expect_limit(detail::min_stack_size);

    cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 1024);
    expect_limit(1024 * 1024);
    run_big_frame<768 * 1024>("768 KiB frames with a 1 MiB stack size");

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
