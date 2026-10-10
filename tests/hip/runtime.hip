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
// HIP's runtime API, as HIP programs use it: through hip/hip_runtime.h and
// HIP's names only. Errors and their names, device queries, host memory,
// HIP's driver API copies and memsets, symbols, streams (with
// hipStreamDefault as a stream, and hipStreamPerThread), memory pools,
// launches through hipLaunchKernel, and stream capture into graphs.
//

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

__constant__ int coefficients[4];

__global__ void scale(int *out, int factor)
{
    out[hipThreadIdx_x] = coefficients[hipThreadIdx_x] * factor;
}

__global__ void set_flag(int *flag)
{
    *flag = 1;
}

int main()
{
    // Errors, with HIP's names
    {
        EXPECT(hipSuccess == 0 && hipErrorInvalidValue == 1 && hipErrorOutOfMemory == 2);
        EXPECT(hipErrorMemoryAllocation == hipErrorOutOfMemory && hipErrorInvalidResourceHandle == hipErrorInvalidHandle);
        EXPECT(std::strcmp(hipGetErrorName(hipErrorOutOfMemory), "hipErrorOutOfMemory") == 0);
        EXPECT(std::strcmp(hipGetErrorName(hipErrorInvalidHandle), "hipErrorInvalidHandle") == 0);
        EXPECT(std::strcmp(hipGetErrorName(hipSuccess), "hipSuccess") == 0);
        EXPECT(std::strcmp(hipGetErrorString(hipErrorInvalidValue), "invalid argument") == 0);
        EXPECT(hipCUDAErrorTohipError(cudaErrorNotReady) == hipErrorNotReady);

        EXPECT(hipSetDevice(1) == hipErrorInvalidDevice);
        EXPECT(hipPeekAtLastError() == hipErrorInvalidDevice);
        EXPECT(hipGetLastError() == hipErrorInvalidDevice);
        EXPECT(hipGetLastError() == hipSuccess);
    }

    // Devices
    {
        int count = 0, device = -1;
        EXPECT(hipInit(0) == hipSuccess);
        EXPECT(hipGetDeviceCount(&count) == hipSuccess && count == 1);
        EXPECT(hipGetDevice(&device) == hipSuccess && device == 0);

        hipDeviceProp_t props;
        EXPECT(hipGetDeviceProperties(&props, 0) == hipSuccess);
        EXPECT(props.warpSize == 32 && props.maxThreadsPerBlock == 1024 && props.multiProcessorCount > 0);
        EXPECT(std::strlen(props.gcnArchName) > 0 && std::strncmp(props.name, "cuda4cpu", 8) == 0);
        EXPECT(props.maxSharedMemoryPerMultiProcessor == props.sharedMemPerMultiprocessor);
        EXPECT(props.arch.hasDoubles == 1 && props.arch.hasWarpShuffle == 1 && props.arch.hasDynamicParallelism == 0);
        EXPECT(hipGetDeviceProperties(&props, 1) == hipErrorInvalidDevice);
        hipGetLastError();

        int value = 0;
        EXPECT(hipDeviceGetAttribute(&value, hipDeviceAttributeWarpSize, 0) == hipSuccess && value == 32);
        EXPECT(hipDeviceGetAttribute(&value, hipDeviceAttributeMultiprocessorCount, 0) == hipSuccess &&
               value == props.multiProcessorCount);
        EXPECT(hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, 0) == hipSuccess && value == 1024);
        EXPECT(hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMajor, 0) == hipSuccess &&
               value == props.major);
        EXPECT(hipDeviceGetAttribute(&value, hipDeviceAttributeCooperativeLaunch, 0) == hipSuccess && value == 1);
        EXPECT(hipDeviceGetAttribute(&value, hipDeviceAttributeImageSupport, 0) == hipSuccess && value == 0);
        EXPECT(hipDeviceGetAttribute(&value, hipDeviceAttributeWallClockRate, 0) == hipSuccess && value > 0);
        EXPECT(hipDeviceGetAttribute(&value, hipDeviceAttributeTotalGlobalMem, 0) == hipSuccess && value > 0);
        EXPECT(hipDeviceGetAttribute(&value, hipDeviceAttributeVendorSpecificBegin, 0) == hipErrorInvalidValue);
        hipGetLastError();

        hipDevice_t handle = -1;
        char name[256];
        int major = 0, minor = 0;
        size_t total = 0, free = 0;
        hipUUID uuid;
        EXPECT(hipDeviceGet(&handle, 0) == hipSuccess && handle == 0);
        EXPECT(hipDeviceGetName(name, sizeof(name), handle) == hipSuccess && std::strcmp(name, props.name) == 0);
        EXPECT(hipDeviceComputeCapability(&major, &minor, handle) == hipSuccess && major == props.major &&
               minor == props.minor);
        EXPECT(hipDeviceTotalMem(&total, handle) == hipSuccess && total == props.totalGlobalMem);
        EXPECT(hipDeviceGetUuid(&uuid, handle) == hipSuccess);
        EXPECT(hipMemGetInfo(&free, &total) == hipSuccess && free > 0 && free <= total);
        EXPECT(hipChooseDevice(&device, &props) == hipSuccess && device == 0);
        EXPECT(hipRuntimeGetVersion(&value) == hipSuccess && value > 0);
    }

    // Host memory, with HIP's flags and typed overloads
    {
        float *p = nullptr;
        EXPECT(hipHostMalloc(&p, 16 * sizeof(float)) == hipSuccess && p != nullptr);
        EXPECT(hipHostFree(p) == hipSuccess);
        EXPECT(hipHostMalloc(&p, 16 * sizeof(float), hipHostMallocMapped | hipHostMallocCoherent) == hipSuccess);
        float *device = nullptr;
        EXPECT(hipHostGetDevicePointer(reinterpret_cast<void **>(&device), p, 0) == hipSuccess && device == p);
        EXPECT(hipFreeHost(p) == hipSuccess);
        void *v = nullptr;
        EXPECT(hipMallocHost(&v, 64) == hipSuccess && v != nullptr);
        EXPECT(hipHostFree(v) == hipSuccess);
        EXPECT(hipExtMallocWithFlags(&v, 64, hipDeviceMallocFinegrained) == hipSuccess && v != nullptr);
        EXPECT(hipFree(v) == hipSuccess);
    }

    // Copies and memsets of HIP's driver API
    {
        int host[8] = {1, 2, 3, 4, 5, 6, 7, 8}, back[8] = {};
        hipDeviceptr_t a = nullptr, b = nullptr;
        EXPECT(hipMalloc(&a, sizeof(host)) == hipSuccess && hipMalloc(&b, sizeof(host)) == hipSuccess);
        EXPECT(hipMemcpyHtoD(a, host, sizeof(host)) == hipSuccess);
        EXPECT(hipMemcpyDtoD(b, a, sizeof(host)) == hipSuccess);
        EXPECT(hipMemcpyDtoH(back, b, sizeof(host)) == hipSuccess && std::memcmp(host, back, sizeof(host)) == 0);
        EXPECT(hipMemcpyDtoHAsync(back, a, sizeof(int), hipStreamDefault) == hipSuccess && back[0] == 1);

        EXPECT(hipMemsetD32(a, -7, 8) == hipSuccess);
        EXPECT(hipMemcpy(back, a, sizeof(back), hipMemcpyDeviceToHost) == hipSuccess);
        EXPECT(back[0] == -7 && back[7] == -7);
        EXPECT(hipMemsetD16Async(a, 0x1234, 3, hipStreamDefault) == hipSuccess);
        unsigned short halves[4];
        EXPECT(hipMemcpyWithStream(halves, a, sizeof(halves), hipMemcpyDeviceToHost, nullptr) == hipSuccess);
        EXPECT(halves[0] == 0x1234 && halves[2] == 0x1234 && halves[3] == 0xffff);
        EXPECT(hipMemsetD8(a, 0xab, 1) == hipSuccess);
        unsigned char byte = 0;
        EXPECT(hipMemcpy(&byte, a, 1, hipMemcpyDeviceToHost) == hipSuccess && byte == 0xab);
        hipFree(a);
        hipFree(b);
    }

    // Large memsets of 2- and 4-byte elements, which several OS threads share
    {
        const size_t count = size_t(4) << 20;
        hipDeviceptr_t a = nullptr;
        EXPECT(hipMalloc(&a, (count + 1) * sizeof(int)) == hipSuccess);
        auto *words = static_cast<int *>(a);
        words[count] = 3;
        EXPECT(hipMemsetD32(a, -7, count) == hipSuccess);
        bool all = true;
        for (size_t i = 0; i < count; ++i)
            all = all && words[i] == -7;
        EXPECT(all && words[count] == 3);
        EXPECT(hipMemsetD16(a, 0x1234, 2 * count - 1) == hipSuccess);
        auto *halves = static_cast<unsigned short *>(a);
        all = true;
        for (size_t i = 0; i < 2 * count - 1; ++i)
            all = all && halves[i] == 0x1234;
        EXPECT(all && halves[2 * count - 1] == 0xffff);
        hipFree(a);
    }

    // Symbols, named directly or with HIP_SYMBOL
    {
        const int host[4] = {1, 2, 3, 4};
        int *out = nullptr, back[4] = {};
        EXPECT(hipMemcpyToSymbol(HIP_SYMBOL(coefficients), host, sizeof(host)) == hipSuccess);
        EXPECT(hipMalloc(&out, sizeof(back)) == hipSuccess);
        hipLaunchKernelGGL(scale, 1, 4, 0, 0, out, 10);
        EXPECT(hipGetLastError() == hipSuccess);
        EXPECT(hipMemcpy(back, out, sizeof(back), hipMemcpyDeviceToHost) == hipSuccess);
        EXPECT(back[0] == 10 && back[3] == 40);
        EXPECT(hipMemcpyFromSymbol(back, coefficients, sizeof(back)) == hipSuccess && back[2] == 3);
        size_t size = 0;
        EXPECT(hipGetSymbolSize(&size, HIP_SYMBOL(coefficients)) == hipSuccess && size == sizeof(host));
        hipFree(out);
    }

    // Launches: invalid configurations, hipLaunchKernel with an argument array,
    // and kernels given as void *
    {
        int flag = 0;
        hipLaunchKernelGGL(set_flag, dim3(1), dim3(2048), 0, hipStreamDefault, &flag);
        EXPECT(hipGetLastError() == hipErrorInvalidConfiguration && flag == 0);

        int *p = &flag;
        void *args[] = {&p};
        EXPECT(hipLaunchKernel(reinterpret_cast<const void *>(set_flag), dim3(1), dim3(1), args, 0, nullptr) ==
               hipSuccess);
        EXPECT(flag == 1);
        flag = 0;
        EXPECT(hipLaunchCooperativeKernel((void *)set_flag, 1, 1, args, 0, 0) == hipSuccess && flag == 1);

        int blocks = 0, min_grid = 0, block = 0;
        EXPECT(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, set_flag, 256, 0) == hipSuccess && blocks == 1);
        EXPECT(hipOccupancyMaxPotentialBlockSize(&min_grid, &block, set_flag) == hipSuccess && block == 1024);
        hipFuncAttributes attributes;
        EXPECT(hipFuncGetAttributes(&attributes, reinterpret_cast<const void *>(set_flag)) == hipSuccess);
    }

    // Streams and events, with hipStreamDefault and 0 as streams
    {
        hipStream_t stream;
        hipEvent_t start, stop;
        EXPECT(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess);
        EXPECT(hipEventCreate(&start) == hipSuccess && hipEventCreateWithFlags(&stop, hipEventDisableTiming) == hipSuccess);
        EXPECT(hipEventRecord(start, 0) == hipSuccess);
        EXPECT(hipEventRecord(stop, hipStreamDefault) == hipSuccess);
        EXPECT(hipStreamWaitEvent(stream, stop, 0) == hipSuccess);
        EXPECT(hipEventSynchronize(stop) == hipSuccess && hipStreamSynchronize(stream) == hipSuccess);
        float ms = -1;
        EXPECT(hipEventElapsedTime(&ms, start, stop) == hipSuccess && ms >= 0);
        unsigned int flags = 0;
        EXPECT(hipStreamGetFlags(stream, &flags) == hipSuccess && flags == hipStreamNonBlocking);
        hipEventDestroy(start);
        hipEventDestroy(stop);
        EXPECT(hipStreamDestroy(stream) == hipSuccess);

        int *data = nullptr;
        EXPECT(hipMalloc(&data, sizeof(int)) == hipSuccess);
        EXPECT(hipMemsetAsync(data, 0, sizeof(int), hipStreamPerThread) == hipSuccess);
        EXPECT(hipStreamSynchronize(hipStreamPerThread) == hipSuccess);
        hipFree(data);
    }

    // Stream-ordered allocation from a pool of its own
    {
        hipMemPoolProps props{};
        props.allocType = hipMemAllocationTypePinned;
        props.location.type = hipMemLocationTypeDevice;
        props.location.id = 0;
        hipMemPool_t pool;
        hipStream_t stream;
        EXPECT(hipMemPoolCreate(&pool, &props) == hipSuccess);
        EXPECT(hipStreamCreate(&stream) == hipSuccess);
        void *p = nullptr;
        EXPECT(hipMallocFromPoolAsync(&p, 1024, pool, stream) == hipSuccess && p != nullptr);
        EXPECT(hipFreeAsync(p, stream) == hipSuccess);
        EXPECT(hipMemPoolDestroy(pool) == hipSuccess);
        hipStreamDestroy(stream);
    }

    // Stream capture into a graph, a memset of HIP's driver API included
    {
        hipStream_t stream;
        hipGraph_t graph;
        hipGraphExec_t exec;
        int *data = nullptr, back[4] = {};
        EXPECT(hipStreamCreate(&stream) == hipSuccess);
        EXPECT(hipMalloc(&data, sizeof(back)) == hipSuccess);
        EXPECT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) == hipSuccess);
        EXPECT(hipMemsetD32Async(data, 5, 4, stream) == hipSuccess);
        hipLaunchKernelGGL(set_flag, 1, 1, 0, stream, data + 3);
        EXPECT(hipStreamEndCapture(stream, &graph) == hipSuccess);
        size_t nodes = 0;
        EXPECT(hipGraphGetNodes(graph, nullptr, &nodes) == hipSuccess && nodes == 2);
        EXPECT(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == hipSuccess);
        EXPECT(hipGraphLaunch(exec, stream) == hipSuccess && hipStreamSynchronize(stream) == hipSuccess);
        EXPECT(hipMemcpy(back, data, sizeof(back), hipMemcpyDeviceToHost) == hipSuccess);
        EXPECT(back[0] == 5 && back[2] == 5 && back[3] == 1);
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
        hipStreamDestroy(stream);
        hipFree(data);
    }

    if (errors == 0)
        std::printf("hip_runtime: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
