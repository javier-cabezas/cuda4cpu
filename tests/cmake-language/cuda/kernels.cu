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

// Kernels in a static library, compiled for separate compilation, called
// from a C++ file

#include <cuda_runtime.h>

#include "saxpy.cuh"

//! y = a * x + y on the device, for host vectors of n elements
cudaError_t device_saxpy(float a, const float *x, float *y, int n)
{
    float *dx, *dy;
    cudaMalloc(&dx, n * sizeof(float));
    cudaMalloc(&dy, n * sizeof(float));
    cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyHostToDevice);
    saxpy<<<(n + 127) / 128, 128>>>(a, dx, dy, n);
    cudaMemcpy(y, dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    return cudaGetLastError();
}
