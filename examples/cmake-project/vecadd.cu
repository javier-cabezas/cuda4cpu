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

#include <cstdio>

__global__ void vecadd(float *c, const float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    const int n = 1 << 20;
    float *a, *b, *c;
    cudaMallocManaged(&a, n * sizeof(float));
    cudaMallocManaged(&b, n * sizeof(float));
    cudaMallocManaged(&c, n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    vecadd<<<(n + 255) / 256, 256>>>(c, a, b, n);
    cudaDeviceSynchronize();

    int wrong = 0;
    for (int i = 0; i < n; ++i)
        wrong += c[i] != 3.0f * i;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("c[42] = %.0f, computed on %s\n", c[42], prop.name);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return wrong != 0;
}
