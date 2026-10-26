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

// A program of one .cu file, with nvcc's version macros

#include <cstdio>

#if !defined(__NVCC__) || __CUDACC_VER_MAJOR__ < 12
#error "compiled by cuda4cpu-nvcc, which CMake identifies as nvcc 12"
#endif

__global__ void square(int *v)
{
    v[threadIdx.x] *= v[threadIdx.x];
}

int main()
{
    int *v;
    cudaMallocManaged(&v, 32 * sizeof(int));
    for (int i = 0; i < 32; ++i)
        v[i] = i;
    square<<<1, 32>>>(v);
    cudaDeviceSynchronize();
    for (int i = 0; i < 32; ++i)
        if (v[i] != i * i)
            return 1;
    cudaFree(v);
    std::printf("single: PASS\n");
    return 0;
}
