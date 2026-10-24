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
// Copies whose pointers don't match their cudaMemcpyKind, with each
// CUDA4CPU_MEMCPY_DIRECTION (the test sets it): warn copies and reports each
// mistake once, error fails with cudaErrorInvalidMemcpyDirection, and ignore
// copies silently. Usage: memcpy_direction warn|error|ignore
//

#include <cstdio>
#include <cstring>

#include "cuda_runtime.h"

static int errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

int main(int argc, char **argv)
{
    if (argc != 2 || (std::strcmp(argv[1], "warn") != 0 && std::strcmp(argv[1], "error") != 0 &&
                      std::strcmp(argv[1], "ignore") != 0)) {
        std::fprintf(stderr, "usage: memcpy_direction warn|error|ignore\n");
        return 1;
    }
    const bool fails = std::strcmp(argv[1], "error") == 0;
    const cudaError_t mismatch = fails ? cudaErrorInvalidMemcpyDirection : cudaSuccess;

    int host[4] = {1, 2, 3, 4}, back[4] = {};
    int *dev = nullptr, *managed = nullptr, *pinned = nullptr;
    EXPECT(cudaMalloc(&dev, sizeof(host)) == cudaSuccess);
    EXPECT(cudaMallocManaged(&managed, sizeof(host)) == cudaSuccess);
    EXPECT(cudaMallocHost(&pinned, sizeof(host)) == cudaSuccess);

    // Copies that match their kind, in every mode. Managed and pinned memory
    // can be either side.
    EXPECT(cudaMemcpy(dev, host, sizeof(host), cudaMemcpyHostToDevice) == cudaSuccess);
    EXPECT(cudaMemcpy(back, dev, sizeof(host), cudaMemcpyDeviceToHost) == cudaSuccess);
    EXPECT(std::memcmp(back, host, sizeof(host)) == 0);
    EXPECT(cudaMemcpy(managed, host, sizeof(host), cudaMemcpyHostToDevice) == cudaSuccess);
    EXPECT(cudaMemcpy(managed, dev, sizeof(host), cudaMemcpyDeviceToHost) == cudaSuccess);
    EXPECT(cudaMemcpy(dev, pinned, sizeof(host), cudaMemcpyDeviceToDevice) == cudaSuccess);
    EXPECT(cudaMemcpy(back, host, sizeof(host), cudaMemcpyDefault) == cudaSuccess);

    // Swapped pointers, twice: warn reports it once, and the copy is made
    // unless the mode is error
    EXPECT(cudaMemcpy(dev, host, sizeof(host), cudaMemcpyHostToDevice) == cudaSuccess);
    for (int i = 0; i < 2; ++i) {
        std::memset(back, 0, sizeof(back));
        EXPECT(cudaMemcpy(back, dev, sizeof(host), cudaMemcpyHostToDevice) == mismatch);
        EXPECT((std::memcmp(back, host, sizeof(host)) == 0) == !fails);
    }
    EXPECT(cudaMemcpy(dev, host, sizeof(host), cudaMemcpyDeviceToHost) == mismatch);
    EXPECT(cudaMemcpy(host, back, sizeof(host), cudaMemcpyDeviceToDevice) == mismatch);
    EXPECT(cudaMemcpy(dev, back, sizeof(host), cudaMemcpyHostToHost) == mismatch);
    EXPECT(cudaMemcpyAsync(back, dev, sizeof(host), cudaMemcpyHostToDevice) == mismatch);
    EXPECT(cudaMemcpy2D(back, sizeof(host), dev, sizeof(host), sizeof(host), 1, cudaMemcpyHostToDevice) ==
           mismatch);
    cudaGetLastError();

    // Kinds that aren't kinds always fail
    EXPECT(cudaMemcpy(back, host, sizeof(host), cudaMemcpyKind(7)) == cudaErrorInvalidMemcpyDirection);
    cudaGetLastError();

    EXPECT(cudaFree(dev) == cudaSuccess);
    EXPECT(cudaFree(managed) == cudaSuccess);
    EXPECT(cudaFreeHost(pinned) == cudaSuccess);
    if (errors == 0)
        std::printf("memcpy_direction: passed\n");
    return errors == 0 ? 0 : 1;
}
