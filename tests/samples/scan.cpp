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
// Inclusive prefix sum of ints, in the style of the CUDA samples' "shfl_scan":
// each warp scans with __shfl_up_sync, the first warp scans the per-warp
// sums, and a second pass adds the scanned per-block sums.
//

#include <random>
#include <vector>

#include "cuda4cpu.hpp"
#include "sample.hpp"

using namespace cuda4cpu;

__device__ int warp_inclusive_scan(int value)
{
    int lane = threadIdx.x & (warpSize - 1);
    for (int offset = 1; offset < warpSize; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, value, offset);
        if (lane >= offset)
            value += n;
    }
    return value;
}

__global__ void block_scan(int *out, int *block_sums, const int *in, size_t n)
{
    __shared__ int warp_sums[32];

    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    int value = i < n ? in[i] : 0;
    value = warp_inclusive_scan(value);
    if (lane == warpSize - 1)
        warp_sums[warp] = value;
    __syncthreads();

    if (warp == 0) {
        int sum = lane < int(blockDim.x / warpSize) ? warp_sums[lane] : 0;
        warp_sums[lane] = warp_inclusive_scan(sum);
    }
    __syncthreads();

    if (warp > 0)
        value += warp_sums[warp - 1];
    if (i < n)
        out[i] = value;
    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = value;
}

__global__ void add_block_offsets(int *out, const int *scanned_block_sums, size_t n)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && i < n)
        out[i] += scanned_block_sums[blockIdx.x - 1];
}

int main()
{
    const unsigned threads = 1024;
    const size_t n = size_t(threads) * threads - 12345;   // not a multiple of the block size
    const unsigned blocks = unsigned((n + threads - 1) / threads);

    std::vector<int> in(n), out(n), out_gold(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 9);
    for (auto &v : in) v = dist(gen);

    std::vector<int> block_sums(blocks), scanned_sums(blocks), unused(1);

    double ms = sample::time_ms([&] {
        launch(block_scan, blocks, threads).call(out.data(), block_sums.data(), in.data(), n);
        launch(block_scan, 1, threads).call(scanned_sums.data(), unused.data(), block_sums.data(), size_t(blocks));
        launch(add_block_offsets, blocks, threads).call(out.data(), scanned_sums.data(), n);
    });

    double host_ms = sample::time_ms([&] {
        int sum = 0;
        for (size_t i = 0; i < n; ++i) out_gold[i] = sum += in[i];
    });

    sample::report("shfl_scan", ms, host_ms);
    return sample::finish(sample::count_mismatches(out.data(), out_gold.data(), n));
}
