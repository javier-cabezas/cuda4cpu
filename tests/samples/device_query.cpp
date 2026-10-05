/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
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
// The essentials of the CUDA samples' "deviceQuery": enumerate the devices
// and print their properties. Exits with an error if a property that CUDA
// code commonly relies on has an implausible value.
//

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
        exit(EXIT_FAILURE);
    }
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);

    int failures = 0;
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10,
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
               (float)deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long)deviceProp.totalGlobalMem);
        printf("  (%03d) Multiprocessors\n", deviceProp.multiProcessorCount);
        printf("  GPU Max Clock rate:                            %.0f MHz\n", deviceProp.clockRate * 1e-3f);
        printf("  Total amount of constant memory:               %zu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n",
               (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Managed Memory:                %s\n", deviceProp.managedMemory ? "Yes" : "No");
        printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);

        int smCount = 0;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev);

        if (deviceProp.warpSize != 32 || deviceProp.maxThreadsPerBlock != 1024 ||
            deviceProp.multiProcessorCount <= 0 || smCount != deviceProp.multiProcessorCount ||
            deviceProp.totalGlobalMem == 0 || deviceProp.maxThreadsDim[2] != 64) {
            printf("  -> implausible device properties\n");
            ++failures;
        }
    }

    // Devices that don't exist are rejected
    if (cudaSetDevice(deviceCount) != cudaErrorInvalidDevice) {
        printf("cudaSetDevice(%d) did not return cudaErrorInvalidDevice\n", deviceCount);
        ++failures;
    }

    printf("Result = %s\n", failures ? "FAIL" : "PASS");
    exit(failures ? EXIT_FAILURE : EXIT_SUCCESS);
}
