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
// 256-bin histogram of bytes, as in the CUDA samples' "histogram": each block
// accumulates a private histogram in shared memory with shared-memory
// atomics, then merges it into the global one with global atomics.
//

#include <random>
#include <vector>

#include <cuda_runtime.h>
#include "sample.hpp"

constexpr int BIN_COUNT = 256;

__global__ void histogram256(unsigned int *hist, const unsigned char *data, size_t n)
{
    __shared__ unsigned int s_hist[BIN_COUNT];

    for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x)
        s_hist[i] = 0;
    __syncthreads();

    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x)
        atomicAdd(&s_hist[data[i]], 1u);
    __syncthreads();

    for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x)
        atomicAdd(&hist[i], s_hist[i]);
}

int main()
{
    const size_t n = size_t(1) << 24;

    std::vector<unsigned char> data(n);
    std::mt19937 gen(42);
    std::binomial_distribution<int> dist(BIN_COUNT - 1, 0.3);   // uneven bins
    for (auto &v : data) v = (unsigned char)dist(gen);

    std::vector<unsigned int> hist(BIN_COUNT, 0), hist_gold(BIN_COUNT, 0);

    double ms = sample::time_ms([&] {
        histogram256<<<240, 192>>>(hist.data(), data.data(), n);
    });

    double host_ms = sample::time_ms([&] {
        for (size_t i = 0; i < n; ++i) ++hist_gold[data[i]];
    });

    sample::report("histogram256", ms, host_ms);
    return sample::finish(sample::count_mismatches(hist.data(), hist_gold.data(), BIN_COUNT));
}
