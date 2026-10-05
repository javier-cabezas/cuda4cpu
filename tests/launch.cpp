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


#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda4cpu.hpp"

using namespace cuda4cpu;

// Every CUDA thread writes its global linear id, so that a missing, repeated
// or misnumbered thread shows up in the output
__global__ void write_ids(unsigned *out, unsigned *count)
{
    size_t block  = (size_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    size_t thread = (size_t(threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    size_t nthreads = size_t(blockDim.x) * blockDim.y * blockDim.z;
    size_t id = block * nthreads + thread;

    __syncthreads();
    out[id] = unsigned(id);
    count[id] += 1;
}

static unsigned errors = 0;

static void run(dim3 grid, dim3 block)
{
    size_t n = size_t(grid.x) * grid.y * grid.z * block.x * block.y * block.z;
    std::vector<unsigned> out(n, ~0u), count(n, 0);

    launch(write_ids, grid, block).call(out.data(), count.data());

    for (size_t i = 0; i < n; ++i) {
        if (out[i] != i || count[i] != 1) {
            std::cout << "grid (" << grid.x << "," << grid.y << "," << grid.z << ") block ("
                      << block.x << "," << block.y << "," << block.z << "): thread " << i
                      << " wrote " << out[i] << " " << count[i] << " times\n";
            ++errors;
            return;
        }
    }
}

int main()
{
    // More block shapes than fibers cached per OS thread, twice, so that
    // cached fibers are both reused and evicted
    const dim3 blocks[] = { dim3(32), dim3(8, 8), dim3(4, 4, 4), dim3(1), dim3(1024), dim3(3, 5, 7) };
    for (int round = 0; round < 2; ++round) {
        for (const dim3 &block : blocks) {
            run(dim3(1), block);            // fewer blocks than cores
            run(dim3(37), block);
            run(dim3(5, 3, 2), block);
        }
    }

    // Empty launches do nothing
    unsigned dummy = 0;
    launch(write_ids, dim3(0), dim3(32)).call(&dummy, &dummy);
    launch(write_ids, dim3(4), dim3(0)).call(&dummy, &dummy);
    if (dummy != 0) {
        std::cout << "empty launch ran threads\n";
        ++errors;
    }

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
