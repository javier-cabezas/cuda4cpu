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

// A kernel without a bounds check. Compiled with -fsanitize=address, the
// first access past the end of the allocation is reported with its line.

__global__ void scale(float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    x[i] *= 2.0f; // Bug: no bounds check against n
}

int main()
{
    const int n = 1100;
    float *x;
    cudaMalloc(&x, n * sizeof(float));
    cudaMemset(x, 0, n * sizeof(float));

    scale<<<(n + 255) / 256, 256>>>(x, n);
    cudaDeviceSynchronize();

    cudaFree(x);
}
