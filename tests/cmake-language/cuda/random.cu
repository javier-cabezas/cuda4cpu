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

// cuRAND through CMake's CUDA::curand target: the host API and the device API

#include <cstdio>

#include <curand.h>
#include <curand_kernel.h>

__global__ void draw(unsigned *out)
{
    curandState state;
    curand_init(1234, threadIdx.x, 0, &state);
    out[threadIdx.x] = curand(&state);
}

int main()
{
    unsigned *device, host[32];
    cudaMallocManaged(&device, 32 * sizeof(unsigned));
    draw<<<1, 32>>>(device);
    cudaDeviceSynchronize();
    // The host API's first 32 results are the first values of subsequences 0 to 31
    curandGenerator_t generator;
    if (curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW) != CURAND_STATUS_SUCCESS ||
        curandSetPseudoRandomGeneratorSeed(generator, 1234) != CURAND_STATUS_SUCCESS ||
        curandGenerate(generator, host, 32) != CURAND_STATUS_SUCCESS)
        return 1;
    curandDestroyGenerator(generator);
    for (int i = 0; i < 32; ++i)
        if (host[i] != device[i])
            return 2;
    cudaFree(device);
    std::printf("random: PASS\n");
    return 0;
}
