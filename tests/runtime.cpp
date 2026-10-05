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
// Runtime API semantics: errors, launch configuration limits, memory
// management, symbols and intrinsics. Uses only cuda_runtime.h, like CUDA code.
//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <cuda_runtime.h>

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

__global__ void set_flag(int *flag)
{
    *flag = 1;
}

__constant__ float table[4];
__device__ int counter;

__global__ void read_table(float *out)
{
    out[threadIdx.x] = table[threadIdx.x];
    if (threadIdx.x == 0)
        counter += 1;
}

__global__ void intrinsics(float *out)
{
    out[0] = __expf(0.f);
    out[1] = __fdividef(1.f, 4.f);
    out[2] = __saturatef(2.f);
    out[3] = float(__popc(0xf0f0u));
}

static bool launch_rejected(dim3 grid, dim3 block)
{
    int flag = 0;
    cuda4cpu::launch(set_flag, grid, block).call(&flag);
    return flag == 0 && cudaGetLastError() == cudaErrorInvalidConfiguration;
}

int main()
{
    // Last error: recorded by failing calls, peeked, then cleared
    EXPECT(cudaGetLastError() == cudaSuccess);
    EXPECT(cudaSetDevice(1) == cudaErrorInvalidDevice);
    EXPECT(cudaPeekAtLastError() == cudaErrorInvalidDevice);
    EXPECT(cudaGetLastError() == cudaErrorInvalidDevice);
    EXPECT(cudaGetLastError() == cudaSuccess);

    EXPECT(std::strcmp(cudaGetErrorName(cudaErrorInvalidValue), "cudaErrorInvalidValue") == 0);
    EXPECT(std::strcmp(cudaGetErrorString(cudaSuccess), "no error") == 0);
    EXPECT(std::strcmp(cudaGetErrorString(cudaErrorMemoryAllocation), "out of memory") == 0);
    EXPECT(std::strcmp(cudaGetErrorString(cudaError_t(12345)), "unrecognized error code") == 0);

    // Launch configurations outside CUDA's limits are rejected without running
    EXPECT(launch_rejected(1, 2048));
    EXPECT(launch_rejected(1, dim3(32, 64)));     // 2048 threads in total
    EXPECT(launch_rejected(1, dim3(1, 1, 65)));
    EXPECT(launch_rejected(dim3(1, 70000), 32));
    EXPECT(launch_rejected(0, 32));
    EXPECT(launch_rejected(1, 0));
    {
        int flag = 0;
        cuda4cpu::launch(set_flag, dim3(1, 65535), dim3(8, 8, 16)).call(&flag);
        EXPECT(flag == 1 && cudaGetLastError() == cudaSuccess);
    }

    // Allocation: 256-byte alignment, failures reported, size 0 allowed
    for (size_t size : {1, 100, 4096, 1000000}) {
        void *p = nullptr;
        EXPECT(cudaMalloc(&p, size) == cudaSuccess);
        EXPECT(reinterpret_cast<uintptr_t>(p) % 256 == 0);
        EXPECT(cudaFree(p) == cudaSuccess);
    }
    {
        void *p = reinterpret_cast<void *>(0x1);
        EXPECT(cudaMalloc(&p, size_t(1) << 60) == cudaErrorMemoryAllocation);
        EXPECT(p == reinterpret_cast<void *>(0x1));
        EXPECT(cudaGetLastError() == cudaErrorMemoryAllocation);
        // Rounding up to the alignment must not wrap around to a small size
        EXPECT(cudaMalloc(&p, std::numeric_limits<size_t>::max()) == cudaErrorMemoryAllocation);
        EXPECT(p == reinterpret_cast<void *>(0x1));
        cudaGetLastError();
        EXPECT(cudaMalloc(&p, 0) == cudaSuccess && p == nullptr);
    }

    // Typed overloads, managed and host memory
    {
        float *d = nullptr, *m = nullptr, *h = nullptr;
        EXPECT(cudaMalloc(&d, 16 * sizeof(float)) == cudaSuccess && d != nullptr);
        EXPECT(cudaMallocManaged(&m, 16 * sizeof(float)) == cudaSuccess && m != nullptr);
        EXPECT(cudaMallocHost(&h, 16 * sizeof(float)) == cudaSuccess && h != nullptr);
        EXPECT(cudaHostAlloc(&h, 16 * sizeof(float), cudaHostAllocMapped) == cudaSuccess);
        cudaFree(d);
        cudaFree(m);
        cudaFreeHost(h);
    }

    // Copies and memset
    {
        int src[4] = {1, 2, 3, 4}, dst[4] = {};
        EXPECT(cudaMemcpy(dst, src, sizeof(src), cudaMemcpyHostToHost) == cudaSuccess);
        EXPECT(std::memcmp(dst, src, sizeof(src)) == 0);
        EXPECT(cudaMemcpy(dst, src, sizeof(src), cudaMemcpyKind(7)) == cudaErrorInvalidMemcpyDirection);
        EXPECT(cudaGetLastError() == cudaErrorInvalidMemcpyDirection);
        EXPECT(cudaMemcpyAsync(dst, src, sizeof(src), cudaMemcpyDefault) == cudaSuccess);
        EXPECT(cudaMemset(dst, 0xff, sizeof(dst)) == cudaSuccess && dst[3] == -1);
        EXPECT(cudaMemsetAsync(dst, 0, sizeof(dst)) == cudaSuccess && dst[0] == 0);
    }

    // Symbols are passed by name
    {
        const float values[4] = {0.5f, 1.5f, 2.5f, 3.5f};
        EXPECT(cudaMemcpyToSymbol(table, values, sizeof(values)) == cudaSuccess);
        const float x = 9.f;
        EXPECT(cudaMemcpyToSymbol(table, &x, sizeof(float), 2 * sizeof(float)) == cudaSuccess);
        EXPECT(cudaMemcpyToSymbol(table, &x, sizeof(float), 4 * sizeof(float)) == cudaErrorInvalidValue);
        cudaGetLastError();

        const int zero = 0;
        EXPECT(cudaMemcpyToSymbol(counter, &zero, sizeof(int)) == cudaSuccess);

        float out[4] = {};
        cuda4cpu::launch(read_table, 1, 4).call(out);
        EXPECT(out[0] == 0.5f && out[1] == 1.5f && out[2] == 9.f && out[3] == 3.5f);

        int count = -1;
        EXPECT(cudaMemcpyFromSymbol(&count, counter, sizeof(int)) == cudaSuccess && count == 1);

        void *address = nullptr;
        size_t size = 0;
        EXPECT(cudaGetSymbolAddress(&address, table) == cudaSuccess && address == static_cast<void *>(table));
        EXPECT(cudaGetSymbolSize(&size, table) == cudaSuccess && size == sizeof(table));
    }

    // Events and streams with CUDA's default arguments
    {
        cudaEvent_t start, stop;
        cudaStream_t stream;
        EXPECT(cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess);
        EXPECT(cudaStreamCreate(&stream) == cudaSuccess);
        EXPECT(cudaEventRecord(start) == cudaSuccess);
        EXPECT(cudaEventRecord(stop, stream) == cudaSuccess);
        EXPECT(cudaStreamWaitEvent(stream, stop) == cudaSuccess);
        EXPECT(cudaEventQuery(stop) == cudaSuccess);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
    }

    // Device queries
    {
        int count = 0, device = -1, value = 0;
        EXPECT(cudaGetDeviceCount(&count) == cudaSuccess && count == 1);
        EXPECT(cudaGetDevice(&device) == cudaSuccess && device == 0);
        EXPECT(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, 0) == cudaSuccess && value == 32);
        EXPECT(cudaDeviceGetAttribute(&value, cudaDeviceAttr(9999), 0) == cudaErrorInvalidValue);
        cudaGetLastError();
        cudaDeviceProp prop;
        EXPECT(cudaGetDeviceProperties(&prop, 1) == cudaErrorInvalidDevice);
        cudaGetLastError();
        EXPECT(cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.major == 6);
    }

    // Intrinsics, on the host and in a kernel
    {
        EXPECT(__popc(0xf0f0u) == 8 && __popcll(~0ull) == 64);
        EXPECT(__clz(1) == 31 && __clz(0) == 32 && __clzll(1) == 63);
        EXPECT(__ffs(0) == 0 && __ffs(8) == 4 && __ffsll(1ll << 40) == 41);
        EXPECT(__brev(1u) == 0x80000000u && __brev(0x12345678u) == 0x1e6a2c48u);
        EXPECT(__brevll(1ull) == 1ull << 63);
        EXPECT(__saturatef(-1.f) == 0.f && __saturatef(0.25f) == 0.25f && __saturatef(NAN) == 0.f);
        float s = 0.f, c = 0.f;
        __sincosf(0.f, &s, &c);
        EXPECT(s == 0.f && c == 1.f);
        EXPECT(__logf(1.f) == 0.f && __powf(2.f, 3.f) == 8.f && __exp10f(2.f) == 100.f);

        int ints[3] = {7, 8, 9};
        EXPECT(__ldg(ints + 1) == 8);

        float out[4] = {};
        cuda4cpu::launch(intrinsics, 1, 1).call(out);
        EXPECT(out[0] == 1.f && out[1] == 0.25f && out[2] == 1.f && out[3] == 8.f);
    }

    if (errors == 0)
        std::printf("runtime: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
