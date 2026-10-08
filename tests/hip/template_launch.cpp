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
// hipLaunchKernelGGL as a function template, which HIP declares when
// HIP_TEMPLATE_KERNEL_LAUNCH is defined before hip_runtime.h: kernel names
// with template arguments need no HIP_KERNEL_NAME. A .cpp file, as HIP
// projects often name their sources, compiled with cuda4cpu_add_hip_sources.
//

#define HIP_TEMPLATE_KERNEL_LAUNCH

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

template <int BlockSize, int Factor, typename T>
__global__ void scale(T *data, int n)
{
    const int i = blockIdx.x * BlockSize + threadIdx.x;
    if (i < n)
        data[i] *= Factor;
}

int main()
{
    int data[100];
    for (int i = 0; i < 100; ++i)
        data[i] = i;

    constexpr int block = 32;
    hipLaunchKernelGGL(scale<block, 3, int>, dim3((100 + block - 1) / block), dim3(block), 0, hipStreamDefault,
                       data, 100);
    if (hipGetLastError() != hipSuccess)
        return EXIT_FAILURE;

    int wrong = 0;
    for (int i = 0; i < 100; ++i)
        wrong += data[i] != 3 * i;
    std::printf("hip_template_launch: %s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
