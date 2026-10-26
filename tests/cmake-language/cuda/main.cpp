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

// Host code in a .cpp file, compiled as C++17 by the project's C++ compiler,
// which uses the CUDA runtime API

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

cudaError_t device_saxpy(float a, const float *x, float *y, int n);

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices != 1)
        return 1;
    const int n = 1000;
    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = float(i);
        y[i] = 1.f;
    }
    if (device_saxpy(2.f, x.data(), y.data(), n) != cudaSuccess)
        return 1;
    for (int i = 0; i < n; ++i) {
        if (y[i] != 2.f * float(i) + 1.f) {
            std::printf("y[%d] = %g\n", i, y[i]);
            return 1;
        }
    }
    std::printf("host: PASS\n");
    return 0;
}
