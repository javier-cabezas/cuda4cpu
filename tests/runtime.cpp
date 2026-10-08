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
// Runtime API semantics: errors, launch configuration limits, memory
// management, symbols and intrinsics. Uses only cuda_runtime.h, like CUDA code.
//

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

#include <cuda.h>
#include <cuda_profiler_api.h>
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
        cudaFreeHost(h);
        EXPECT(cudaHostAlloc(&h, 16 * sizeof(float), cudaHostAllocMapped) == cudaSuccess && h != nullptr);
        cudaFreeHost(h);
        cudaFree(d);
        cudaFree(m);
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

    // Directed rounding and conversions, exact as in CUDA
    {
        const float one = 1.0f, tiny = 1e-8f;
        // Valgrind ignores the SSE rounding mode, so arithmetic always rounds to nearest under it.
        const bool honors_rounding_mode = __fadd_ru(one, tiny) > 1.0f;
        if (honors_rounding_mode) {
            EXPECT(__fadd_rd(one, tiny) == 1.0f && __fadd_ru(one, tiny) > 1.0f && __fadd_rn(one, tiny) == 1.0f);
            EXPECT(__fadd_rz(-one, -tiny) == -1.0f && __fadd_rd(-one, -tiny) < -1.0f);
            EXPECT(__fdiv_rd(1.0f, 3.0f) < __fdiv_ru(1.0f, 3.0f));
            EXPECT(__dmul_rd(0.1, 3.0) < __dmul_ru(0.1, 3.0));
            EXPECT(__fsqrt_rd(2.0f) < __fsqrt_ru(2.0f));
        } else {
            std::printf("runtime: the FPU ignores the rounding mode here (Valgrind?); skipping directed rounding checks\n");
        }
        EXPECT(__fadd_rn(one, tiny) == 1.0f && __dsqrt_rn(4.0) == 2.0);
        EXPECT(std::fegetround() == FE_TONEAREST);
        EXPECT(__float2int_rn(2.5f) == 2 && __float2int_rn(3.5f) == 4 && __float2int_rz(-2.7f) == -2);
        EXPECT(__float2int_ru(2.1f) == 3 && __float2int_rd(-2.1f) == -3);
        EXPECT(__float2int_rn(NAN) == 0 && __float2int_rn(1e20f) == std::numeric_limits<int>::max());
        EXPECT(__float2uint_rn(-5.0f) == 0u && __double2ll_rz(-1e30) == std::numeric_limits<long long>::min());
        EXPECT(__int2float_rd(16777217) == 16777216.0f && __int2float_ru(16777217) == 16777218.0f);
        EXPECT(__double2float_rd(0.1) < __double2float_ru(0.1));
        EXPECT(__int_as_float(0x3f800000) == 1.0f && __float_as_uint(-0.0f) == 0x80000000u);
        EXPECT(__longlong_as_double(__double_as_longlong(3.25)) == 3.25);
        EXPECT(__hiloint2double(__double2hiint(-7.5), __double2loint(-7.5)) == -7.5);
        EXPECT(__mul24(-3, 5) == -15 && __umul24(0x1000003u, 2u) == 6u);
        EXPECT(max(3, 7) == 7 && min(-1, 2u) == 2u && max(1.5f, 2.5) == 2.5 && min(4ll, -2ll) == -2ll);
    }

    // Scoped atomics behave like the plain ones
    {
        int counter = 0;
        atomicAdd_system(&counter, 2);
        atomicAdd_block(&counter, 3);
        EXPECT(counter == 5 && atomicMax_system(&counter, 9) == 5 && counter == 9);
    }

    // Mapped and registered host memory, stream-ordered allocation and pools
    {
        EXPECT(cudaSetDeviceFlags(cudaDeviceMapHost) == cudaSuccess);
        unsigned int flags = 0;
        EXPECT(cudaGetDeviceFlags(&flags) == cudaSuccess && flags == cudaDeviceMapHost);

        float *host = nullptr, *device = nullptr;
        EXPECT(cudaHostAlloc(&host, 64, cudaHostAllocMapped) == cudaSuccess);
        EXPECT(cudaHostGetDevicePointer(&device, host, 0) == cudaSuccess && device == host);
        cudaFreeHost(host);

        float local[4];
        EXPECT(cudaHostRegister(local, sizeof(local), cudaHostRegisterMapped) == cudaSuccess);
        EXPECT(cudaHostUnregister(local) == cudaSuccess);

        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        void *p = nullptr;
        EXPECT(cudaMallocAsync(&p, 1000, stream) == cudaSuccess && p != nullptr);
        EXPECT(cudaFreeAsync(p, stream) == cudaSuccess);
        cudaMemPool_t pool = nullptr;
        unsigned long long threshold = ~0ull;
        EXPECT(cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess && pool != nullptr);
        EXPECT(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold) == cudaSuccess);
        int value = 0;
        EXPECT(cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, 0) == cudaSuccess && value == 1);

        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location = {cudaMemLocationTypeDevice, 0};
        cudaMemPool_t created = nullptr;
        EXPECT(cudaMemPoolCreate(&created, &props) == cudaSuccess && created != nullptr && created != pool);
        EXPECT(cudaMallocFromPoolAsync(&p, 1000, created, stream) == cudaSuccess && p != nullptr);
        EXPECT(cudaFreeAsync(p, stream) == cudaSuccess);
        EXPECT(cudaMemPoolDestroy(created) == cudaSuccess);
        EXPECT(cudaMemPoolDestroy(pool) == cudaErrorInvalidValue);    // the default pool can't be destroyed
        cudaGetLastError();

        bool called = false;
        EXPECT(cudaLaunchHostFunc(stream, [](void *flag) { *static_cast<bool *>(flag) = true; }, &called) == cudaSuccess);
        EXPECT(called);
        cudaStreamDestroy(stream);
    }

    // The host's memory is the device's
    {
        size_t free = 0, total = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        EXPECT(cudaMemGetInfo(&free, &total) == cudaSuccess);
        EXPECT(total == prop.totalGlobalMem && free > 0 && free <= total);
        EXPECT(cudaMemGetInfo(nullptr, &total) == cudaErrorInvalidValue);
        cudaGetLastError();
    }

    // cudaStreamDefault is a null pointer constant, as in CUDA, so code passes
    // it where a stream goes. cudaStreamPerThread is each thread's own stream.
    {
        int flag = 0;
        cudaEvent_t event;
        cudaEventCreate(&event);
        EXPECT(cudaEventRecord(event, cudaStreamDefault) == cudaSuccess);
        EXPECT(cudaMemsetAsync(&flag, 0, sizeof(flag), cudaStreamDefault) == cudaSuccess);
        cuda4cpu::launch(set_flag, 1, 1, 0, cudaStreamDefault).call(&flag);
        EXPECT(flag == 1);
        EXPECT(cudaStreamSynchronize(cudaStreamLegacy) == cudaSuccess);
        cudaEventDestroy(event);

        cudaStream_t other = nullptr;
        std::thread([&] { other = cudaStreamPerThread; }).join();
        EXPECT(cudaStreamPerThread != nullptr && cudaStreamPerThread == cudaStreamPerThread);
        EXPECT(other != nullptr && other != cudaStreamPerThread);

        // Work issued to it is captured like any stream's
        cudaGraph_t graph = nullptr;
        EXPECT(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal) == cudaSuccess);
        flag = 0;
        cuda4cpu::launch(set_flag, 1, 1, 0, cudaStreamPerThread).call(&flag);
        EXPECT(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess && flag == 0);
        size_t nodes = 0;
        EXPECT(cudaGraphGetNodes(graph, nullptr, &nodes) == cudaSuccess && nodes == 1);
        cudaGraphExec_t exec = nullptr;
        EXPECT(cudaGraphInstantiate(&exec, graph) == cudaSuccess);
        EXPECT(cudaGraphLaunch(exec, cudaStreamPerThread) == cudaSuccess && flag == 1);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);

        EXPECT(cudaStreamDestroy(cudaStreamPerThread) == cudaErrorInvalidResourceHandle);
        cudaGetLastError();
    }

    // Device queries and configuration that have no effect on a CPU
    {
        int least = -1, greatest = -1, can_access = -1, overlap = -1;
        EXPECT(cudaDeviceGetStreamPriorityRange(&least, &greatest) == cudaSuccess && least == 0 && greatest == 0);
        EXPECT(cudaDeviceCanAccessPeer(&can_access, 0, 0) == cudaSuccess && can_access == 0);
        EXPECT(cudaDeviceGetAttribute(&overlap, cudaDevAttrGpuOverlap, 0) == cudaSuccess && overlap == 0);
        EXPECT(cudaFuncSetCacheConfig(set_flag, cudaFuncCachePreferL1) == cudaSuccess);
        EXPECT(cudaFuncSetAttribute(set_flag, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024) == cudaSuccess);
        cudaFuncAttributes attributes;
        EXPECT(cudaFuncGetAttributes(&attributes, set_flag) == cudaSuccess && attributes.maxThreadsPerBlock == 1024);
        EXPECT(cudaProfilerStart() == cudaSuccess && cudaProfilerStop() == cudaSuccess);
        EXPECT(CUDA_VERSION == CUDART_VERSION);
        EXPECT(std::strcmp(cudaGetErrorName(cudaErrorAssert), "cudaErrorAssert") == 0);
    }

    if (errors == 0)
        std::printf("runtime: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
