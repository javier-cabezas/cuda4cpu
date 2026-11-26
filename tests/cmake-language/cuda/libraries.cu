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

// cuBLAS and cuFFT through CMake's CUDA::cublas and CUDA::cufft targets

#include <cstdio>

#include <cublas_v2.h>
#include <cufft.h>

int main()
{
    // y = 2 x + y
    cublasHandle_t handle;
    float x[3] = {1, 2, 3}, y[3] = {1, 1, 1};
    const float two = 2;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS || cublasSaxpy(handle, 3, &two, x, 1, y, 1) != CUBLAS_STATUS_SUCCESS)
        return 1;
    cublasDestroy(handle);
    if (y[2] != 7)
        return 2;
    // The DFT of a constant is its sum at frequency 0
    cufftHandle plan;
    cufftComplex data[4] = {{1, 0}, {1, 0}, {1, 0}, {1, 0}};
    if (cufftPlan1d(&plan, 4, CUFFT_C2C, 1) != CUFFT_SUCCESS ||
        cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS)
        return 3;
    cufftDestroy(plan);
    if (data[0].x != 4 || data[1].x != 0)
        return 4;
    std::printf("libraries: PASS\n");
    return 0;
}
