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

// A HIP program

#include <hip/hip_runtime.h>

#include <cstdio>

__global__ void add(const int *a, const int *b, int *c, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    const int n = 300;
    int *a, *b, *c;
    hipMallocManaged(&a, n * sizeof(int));
    hipMallocManaged(&b, n * sizeof(int));
    hipMallocManaged(&c, n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }
    hipLaunchKernelGGL(add, dim3((n + 63) / 64), dim3(64), 0, 0, a, b, c, n);
    if (hipDeviceSynchronize() != hipSuccess)
        return 1;
    for (int i = 0; i < n; ++i)
        if (c[i] != 3 * i)
            return 1;
    hipFree(a);
    hipFree(b);
    hipFree(c);
    std::printf("hip: PASS\n");
    return 0;
}
