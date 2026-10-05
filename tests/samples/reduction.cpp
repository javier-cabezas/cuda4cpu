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
// Sum of an array of floats, in the style of the CUDA samples' "reduction":
//  - reduce_shared: tree reduction in dynamic shared memory, finished with
//    warp shuffles by the first warp, one partial sum per block
//  - reduce_warp: grid-stride loop, warp shuffles, one atomicAdd per warp
//

#include <random>
#include <vector>

#include "cuda4cpu.hpp"
#include "sample.hpp"

using namespace cuda4cpu;

__global__ void reduce_shared(float *partial, const float *in, size_t n)
{
    float *sdata = dynamic_shared<float>();   // CUDA: extern __shared__ float sdata[];

    unsigned tid = threadIdx.x;
    size_t i = size_t(blockIdx.x) * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.f;
    if (i < n) sum += in[i];
    if (i + blockDim.x < n) sum += in[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] = sum = sum + sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        if (blockDim.x >= 64)
            sum += sdata[tid + 32];
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0)
        partial[blockIdx.x] = sum;
}

__global__ void reduce_warp(float *out, const float *in, size_t n)
{
    float sum = 0.f;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x)
        sum += in[i];

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if ((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(out, sum);
}

int main()
{
    const size_t n = size_t(1) << 24;
    const unsigned threads = 256;

    std::vector<float> in(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto &v : in) v = dist(gen);

    double expected = 0.0;
    double host_ms = sample::time_ms([&] {
        #pragma omp parallel for reduction(+:expected)
        for (size_t i = 0; i < n; ++i) expected += in[i];
    });

    size_t errors = 0;

    // Two elements per thread, then the host adds the per-block partial sums
    {
        const unsigned blocks = unsigned((n + threads * 2 - 1) / (threads * 2));
        std::vector<float> partial(blocks);
        double ms = sample::time_ms([&] {
            launch(reduce_shared, blocks, threads, threads * sizeof(float))
                .call(partial.data(), in.data(), n);
        });
        double sum = 0.0;
        for (float p : partial) sum += p;
        sample::report("reduce_shared", ms, host_ms);
        errors += sample::count_mismatches(&sum, &expected, 1, 1e-5);
    }

    {
        float sum = 0.f;
        double ms = sample::time_ms([&] {
            launch(reduce_warp, 1024, threads).call(&sum, in.data(), n);
        });
        sample::report("reduce_warp", ms, host_ms);
        errors += sample::count_mismatches(&sum, &expected, 1, 1e-5);
    }

    return sample::finish(errors);
}
