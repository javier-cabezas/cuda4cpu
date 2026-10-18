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

// A missing __syncthreads() that goes unnoticed with the default order:
// thread t reads what thread t - 1 wrote, and thread t - 1 runs first. Run it
// with CUDA4CPU_SCHEDULE=reverse or random to expose it.

#include <cstdio>

// Each element becomes the sum of itself and its left neighbor in the block
__global__ void pair_sum(const int *in, int *out)
{
    __shared__ int tile[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    tile[t] = in[i];
    // Bug: a __syncthreads() is missing here
    out[i] = tile[t] + (t > 0 ? tile[t - 1] : 0);
}

int main()
{
    const int n = 1 << 16;
    int *in, *out;
    cudaMallocManaged(&in, n * sizeof(int));
    cudaMallocManaged(&out, n * sizeof(int));
    for (int i = 0; i < n; ++i)
        in[i] = i;

    pair_sum<<<n / 256, 256>>>(in, out);
    cudaDeviceSynchronize();

    int wrong = 0;
    for (int i = 0; i < n; ++i)
        wrong += out[i] != i + (i % 256 ? i - 1 : 0);
    std::printf("%s: %d wrong results\n", wrong ? "FAILED" : "PASSED", wrong);

    cudaFree(in);
    cudaFree(out);
    return wrong != 0;
}
