/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
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
// Micro-benchmarks of the execution engine. Each result is the median of
// several runs. Run on an otherwise idle machine.
//

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "cuda4cpu.hpp"

using namespace cuda4cpu;

namespace {

template <typename F>
double median_ms(F &&f, int runs = 7)
{
    std::vector<double> times;
    for (int r = 0; r < runs; ++r) {
        auto start = std::chrono::steady_clock::now();
        f();
        auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void row(const char *name, double value, const char *unit)
{
    std::printf("%-58s %10.2f %s\n", name, value, unit);
}

__global__ void empty_kernel(int *) {}

__global__ void barriers(int n)
{
    for (int i = 0; i < n; ++i)
        __syncthreads();
}

__global__ void shuffles(int *out, int n)
{
    int v = threadIdx.x;
    for (int i = 0; i < n; ++i)
        v += __shfl_xor_sync(0xffffffff, v, i & 31);
    if (v == -1)
        *out = v;   // keep the shuffles alive
}

__global__ void vecadd(float *C, const float *A, const float *B, size_t n)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

// Block b does work proportional to b, like a triangular loop nest
__global__ void triangular(float *out, int scale)
{
    float acc = 0.f;
    for (int i = 0; i < int(blockIdx.x) * scale; ++i)
        acc += float(i) * 1e-7f;
    out[size_t(blockIdx.x) * blockDim.x + threadIdx.x] = acc;
}

}

int main()
{
    std::printf("cuda4cpu micro-benchmarks, %d OS threads\n\n", system::get_system().get_num_procs());

    // Launch overhead with cached fibers
    {
        launch(empty_kernel, 1, 256).call(nullptr);
        const int reps = 1000;
        double ms = median_ms([&] { for (int r = 0; r < reps; ++r) launch(empty_kernel, 1, 256).call(nullptr); });
        row("launch, 1 block x 256 threads, empty kernel", ms * 1e3 / reps, "us");
    }

    // A new block shape each time, so that every OS thread creates its fibers
    {
        unsigned shape = 1000;
        double ms = median_ms([&] { launch(empty_kernel, 1024, shape++).call(nullptr); });
        row("first launch, 1024 blocks x ~1000 threads (creates fibers)", ms, "ms");
    }

    // One fiber switch per thread per barrier
    {
        const int n = 1000, threads = 256, blocks = 64;
        double base = median_ms([&] { launch(barriers, blocks, threads).call(0); });
        double ms   = median_ms([&] { launch(barriers, blocks, threads).call(n); });
        double per_core = double(blocks) * threads * n / system::get_system().get_num_procs();
        row("__syncthreads, per thread", (ms - base) * 1e6 / per_core, "ns");
    }

    {
        const int n = 1000, threads = 256, blocks = 64;
        int out = 0;
        double base = median_ms([&] { launch(shuffles, blocks, threads).call(&out, 0); });
        double ms   = median_ms([&] { launch(shuffles, blocks, threads).call(&out, n); });
        double per_core = double(blocks) * threads * n / system::get_system().get_num_procs();
        row("__shfl_xor_sync, per lane", (ms - base) * 1e6 / per_core, "ns");
    }

    // Barrier-free kernel against the equivalent OpenMP loop
    {
        const size_t n = size_t(1) << 24;
        std::vector<float> A(n, 1.f), B(n, 2.f), C(n);
        double ms = median_ms([&] { launch(vecadd, unsigned(n / 256), 256).call(C.data(), A.data(), B.data(), n); });
        double omp = median_ms([&] {
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) C[i] = A[i] + B[i];
        });
        row("vecadd 16M floats", ms, "ms");
        row("  same loop with OpenMP", omp, "ms");
    }

    // Uneven blocks, against an OpenMP loop with dynamic scheduling
    {
        const unsigned blocks = 512, threads = 64;
        const int scale = 40;
        std::vector<float> out(blocks * threads);
        double ms = median_ms([&] { launch(triangular, blocks, threads).call(out.data(), scale); });
        double omp = median_ms([&] {
            #pragma omp parallel for schedule(dynamic)
            for (unsigned b = 0; b < blocks; ++b) {
                for (unsigned t = 0; t < threads; ++t) {
                    float acc = 0.f;
                    for (int i = 0; i < int(b) * scale; ++i) acc += float(i) * 1e-7f;
                    out[b * threads + t] = acc;
                }
            }
        });
        row("triangular, 512 blocks of increasing cost", ms, "ms");
        row("  same loop with OpenMP schedule(dynamic)", omp, "ms");
    }
}
