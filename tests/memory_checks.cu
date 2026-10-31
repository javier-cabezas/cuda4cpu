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

// CUDA4CPU_INIT_MEMORY and CUDA4CPU_LEAK_CHECK, run by
// memory_checks_test.cmake. "fill" checks that every kind of allocation
// starts filled with the byte that CUDA4CPU_INIT_MEMORY gives. "leaks" leaks
// allocations at known lines, which it prints, and frees the others, two of
// them after main returns: from a static object's destructor, and from a
// destructor function.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime_api.h>

#define LEAK_LINE(what) std::printf("%s leaked at line %d\n", what, __LINE__)

//! Frees its allocation after main returns: not a leak
struct holder {
    int *p = nullptr;
    ~holder() { cudaFree(p); }
} held;

static void *freed_last = nullptr;

[[gnu::destructor]] static void free_last()
{
    cudaFree(freed_last);
}

static int failures = 0;

static void expect_filled(const char *what, const void *p, size_t size, unsigned long byte)
{
    const auto *bytes = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < size; ++i) {
        if (bytes[i] != byte) {
            std::printf("%s: byte %zu is 0x%02x, expected 0x%02lx\n", what, i, bytes[i], byte);
            ++failures;
            return;
        }
    }
}

__global__ void copy(unsigned char *dst, const unsigned char *src)
{
    dst[threadIdx.x] = src[threadIdx.x];
}

static int fill()
{
    const char *setting = std::getenv("CUDA4CPU_INIT_MEMORY");
    char *end = nullptr;
    const unsigned long byte = setting != nullptr ? std::strtoul(setting, &end, 0) : 256;
    if (setting == nullptr || *end != '\0' || byte > 0xff) {
        // The library rejects it when the first allocation reads it
        void *p;
        cudaMalloc(&p, 1);
        cudaFree(p);
        std::printf("set CUDA4CPU_INIT_MEMORY to a byte\n");
        return 1;
    }

    const size_t n = 1000;
    void *device, *managed, *host, *alloc, *pitched, *async;
    cudaMalloc(&device, n);
    cudaMallocManaged(&managed, n);
    cudaMallocHost(&host, n);
    cudaHostAlloc(&alloc, n, cudaHostAllocDefault);
    size_t pitch;
    cudaMallocPitch(&pitched, &pitch, 100, 10);
    cudaMallocAsync(&async, n, nullptr);
    cudaPitchedPtr volume;
    cudaMalloc3D(&volume, make_cudaExtent(50, 4, 3));
    // The typed overloads
    int *typed_device, *typed_managed, *typed_host;
    cudaMalloc(&typed_device, n);
    cudaMallocManaged(&typed_managed, n);
    cudaMallocHost(&typed_host, n);
    // Large enough to be filled by several OS threads
    const size_t large = 64 << 20;
    void *big;
    cudaMalloc(&big, large);
    expect_filled("cudaMalloc", device, n, byte);
    expect_filled("cudaMallocManaged", managed, n, byte);
    expect_filled("cudaMallocHost", host, n, byte);
    expect_filled("cudaHostAlloc", alloc, n, byte);
    expect_filled("cudaMallocPitch", pitched, pitch * 10, byte);
    expect_filled("cudaMallocAsync", async, n, byte);
    expect_filled("cudaMalloc3D", volume.ptr, volume.pitch * 4 * 3, byte);
    expect_filled("a large cudaMalloc", big, large, byte);
    expect_filled("cudaMalloc(int **)", typed_device, n, byte);
    expect_filled("cudaMallocManaged(int **)", typed_managed, n, byte);
    expect_filled("cudaMallocHost(int **)", typed_host, n, byte);

    // A kernel sees the fill too
    unsigned char *copied;
    cudaMallocManaged(&copied, 64);
    copy<<<1, 64>>>(copied, static_cast<unsigned char *>(device));
    cudaDeviceSynchronize();
    expect_filled("a kernel's copy", copied, 64, byte);

    // A graph allocation, each time the graph allocates it
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    void *in_graph;
    cudaMallocAsync(&in_graph, n, stream);
    copy<<<1, 64, 0, stream>>>(copied, static_cast<unsigned char *>(in_graph));
    cudaMemsetAsync(in_graph, byte ^ 0xff, n, stream);
    cudaFreeAsync(in_graph, stream);
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, 0);
    for (int launch = 0; launch < 2; ++launch) {
        std::memset(copied, 0, 64);
        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
        expect_filled("a graph allocation", copied, 64, byte);
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    for (void *p : {device, pitched, async, volume.ptr, big, static_cast<void *>(copied),
                    static_cast<void *>(typed_device)})
        cudaFree(p);
    cudaFree(managed);
    cudaFree(typed_managed);
    cudaFreeHost(host);
    cudaFreeHost(typed_host);
    cudaFreeHost(alloc);
    if (failures == 0)
        std::printf("filled\n");
    return failures != 0;
}

static int leaks()
{
    for (int i = 0; i < 2; ++i) {
        int *p;
        cudaMalloc(&p, 100); LEAK_LINE("device");
    }
    float *managed;
    cudaMallocManaged(&managed, 4); LEAK_LINE("managed");
    char *pinned;
    cudaMallocHost(&pinned, 50); LEAK_LINE("pinned");
    double *hip_pinned;
    hipHostMalloc(&hip_pinned, 30); LEAK_LINE("hip_pinned");

    // A graph allocation that the graph doesn't free
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    void *in_graph;
    cudaMallocAsync(&in_graph, 1000, stream); LEAK_LINE("graph");
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    // Freed: not leaks
    void *device, *host;
    cudaMalloc(&device, 1000);
    cudaMallocHost(&host, 1000);
    cudaFree(device);
    cudaFreeHost(host);
    cudaMalloc(&held.p, 1000);
    cudaMalloc(&freed_last, 1000);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && std::strcmp(argv[1], "fill") == 0)
        return fill();
    if (argc == 2 && std::strcmp(argv[1], "leaks") == 0)
        return leaks();
    std::printf("usage: memory_checks fill|leaks\n");
    return 2;
}
