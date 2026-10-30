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
// allocations at known lines, which it prints, and frees the others, one of
// them from a static object's destructor, after main returns.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define LEAK_LINE(what) std::printf("%s leaked at line %d\n", what, __LINE__)

//! Frees its allocation after main returns: not a leak
struct holder {
    int *p = nullptr;
    ~holder() { cudaFree(p); }
} held;

static int failures = 0;

static void expect_filled(const char *what, const void *p, size_t size, int byte)
{
    const auto *bytes = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < size; ++i) {
        if (bytes[i] != byte) {
            std::printf("%s: byte %zu is 0x%02x, expected 0x%02x\n", what, i, bytes[i], byte);
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
    const int byte = setting != nullptr ? int(std::strtoul(setting, nullptr, 0)) : -1;
    if (byte < 0) {
        std::printf("set CUDA4CPU_INIT_MEMORY\n");
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

    for (void *p : {device, pitched, async, volume.ptr, big, static_cast<void *>(copied)})
        cudaFree(p);
    cudaFree(managed);
    cudaFreeHost(host);
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

    // Freed: not leaks
    void *device, *host;
    cudaMalloc(&device, 1000);
    cudaMallocHost(&host, 1000);
    cudaFree(device);
    cudaFreeHost(host);
    cudaMalloc(&held.p, 1000);
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
