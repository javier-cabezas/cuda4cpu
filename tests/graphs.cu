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



// CUDA graphs: stream capture (with fork and join), explicit graphs with every
// node type, updates, conditional nodes, graph memory, and the 2D/3D copies.
// It is a .cu file so that cuda4cpu-rewrite handles (void *)kernel and
// params.gridDim, as in real CUDA code.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

#define CHECK(call) EXPECT((call) == cudaSuccess)

__global__ void scale(float *x, float factor, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] *= factor;
}

__global__ void add(float *x, const float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += y[i];
}

// Decrements *counter and keeps the loop going while it is positive
__global__ void count_down(int *counter, int *iterations, cudaGraphConditionalHandle handle)
{
    --*counter;
    ++*iterations;
    cudaGraphSetConditional(handle, *counter > 0);
}

__global__ void set_condition(const int *value, cudaGraphConditionalHandle handle)
{
    cudaGraphSetConditional(handle, unsigned(*value));
}

__global__ void store(int *where, int value)
{
    *where = value;
}

// The device clock counts at clockRate kHz
__global__ void clocks(long long *ticks, long *legacy)
{
    ticks[0] = clock64();
    legacy[0] = clock();
    long long until = ticks[0] + 1000;
    while (clock64() < until) {
    }
    ticks[1] = clock64();
    legacy[1] = clock();
}

static std::vector<int> order;

static void record(void *tag)
{
    order.push_back(int(reinterpret_cast<intptr_t>(tag)));
}

static void test_capture()
{
    const int n = 1000;
    std::vector<float> host(n);
    float *x, *y;
    CHECK(cudaMalloc(&x, n * sizeof(float)));
    CHECK(cudaMalloc(&y, n * sizeof(float)));

    cudaStream_t s1, s2;
    cudaEvent_t fork, join;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));
    CHECK(cudaEventCreate(&fork));
    CHECK(cudaEventCreate(&join));
    for (int i = 0; i < n; ++i)
        host[i] = float(i);

    // Nothing runs while capturing: s1 copies and scales x; s2, forked from
    // s1, fills y; s1 joins s2 and adds y to x
    order.clear();
    cudaStreamCaptureStatus status;
    CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
    CHECK(cudaStreamIsCapturing(s1, &status));
    EXPECT(status == cudaStreamCaptureStatusActive);
    CHECK(cudaMemcpyAsync(x, host.data(), n * sizeof(float), cudaMemcpyHostToDevice, s1));
    CHECK(cudaEventRecord(fork, s1));
    CHECK(cudaStreamWaitEvent(s2, fork, 0));
    CHECK(cudaStreamIsCapturing(s2, &status));
    EXPECT(status == cudaStreamCaptureStatusActive);
    scale<<<(n + 255) / 256, 256, 0, s1>>>(x, 2.0f, n);
    CHECK(cudaMemsetAsync(y, 0, n * sizeof(float), s2));
    CHECK(cudaLaunchHostFunc(s2, record, reinterpret_cast<void *>(1)));
    CHECK(cudaEventRecord(join, s2));
    CHECK(cudaStreamWaitEvent(s1, join, 0));
    add<<<(n + 255) / 256, 256, 0, s1>>>(x, y, n);
    CHECK(cudaLaunchHostFunc(s1, record, reinterpret_cast<void *>(2)));
    EXPECT(cudaStreamSynchronize(s1) == cudaErrorStreamCaptureUnsupported);
    cudaGetLastError();
    EXPECT(order.empty());

    unsigned long long id = 0;
    cudaGraph_t captured = nullptr;
    const cudaGraphNode_t *deps = nullptr;
    size_t ndeps = 0;
    CHECK(cudaStreamGetCaptureInfo(s1, &status, &id, &captured, &deps, &ndeps));
    EXPECT(id != 0 && captured != nullptr && ndeps == 1);

    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(s1, &graph));
    EXPECT(graph == captured);
    CHECK(cudaStreamIsCapturing(s2, &status));
    EXPECT(status == cudaStreamCaptureStatusNone);

    size_t nodes = 0, roots = 0, edges = 0;
    CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CHECK(cudaGraphGetRootNodes(graph, nullptr, &roots));
    CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, &edges));
    // memcpy -> {scale, memset -> host 1} -> add -> host 2
    EXPECT(nodes == 6 && roots == 1 && edges == 6);

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    for (int launch = 0; launch < 3; ++launch) {
        CHECK(cudaGraphLaunch(exec, s1));
        CHECK(cudaMemcpy(host.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool right = true;
        for (int i = 0; i < n; ++i)
            right = right && host[i] == 2.0f * float(i);
        EXPECT(right);
        for (int i = 0; i < n; ++i)
            host[i] = float(i);
    }
    EXPECT((order == std::vector<int>{1, 2, 1, 2, 1, 2}));

    // Update the executable graph from a capture with other arguments
    CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
    CHECK(cudaMemcpyAsync(x, host.data(), n * sizeof(float), cudaMemcpyHostToDevice, s1));
    CHECK(cudaEventRecord(fork, s1));
    CHECK(cudaStreamWaitEvent(s2, fork, 0));
    scale<<<(n + 255) / 256, 256, 0, s1>>>(x, 3.0f, n);
    CHECK(cudaMemsetAsync(y, 0, n * sizeof(float), s2));
    CHECK(cudaLaunchHostFunc(s2, record, reinterpret_cast<void *>(1)));
    CHECK(cudaEventRecord(join, s2));
    CHECK(cudaStreamWaitEvent(s1, join, 0));
    add<<<(n + 255) / 256, 256, 0, s1>>>(x, y, n);
    CHECK(cudaLaunchHostFunc(s1, record, reinterpret_cast<void *>(2)));
    cudaGraph_t updated;
    CHECK(cudaStreamEndCapture(s1, &updated));
    cudaGraphExecUpdateResultInfo info;
    CHECK(cudaGraphExecUpdate(exec, updated, &info));
    EXPECT(info.result == cudaGraphExecUpdateSuccess);
    CHECK(cudaGraphLaunch(exec, s1));
    CHECK(cudaMemcpy(host.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT(host[10] == 30.0f && host[n - 1] == 3.0f * float(n - 1));

    // A different topology can't be updated
    CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
    scale<<<1, 1, 0, s1>>>(x, 1.0f, 1);
    cudaGraph_t other;
    CHECK(cudaStreamEndCapture(s1, &other));
    cudaGraphNode_t error_node;
    cudaGraphExecUpdateResult result;
    EXPECT(cudaGraphExecUpdate(exec, other, &error_node, &result) == cudaErrorGraphExecUpdateFailure);
    EXPECT(result == cudaGraphExecUpdateErrorTopologyChanged);
    cudaGetLastError();

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaGraphDestroy(updated));
    CHECK(cudaGraphDestroy(other));
    CHECK(cudaEventDestroy(fork));
    CHECK(cudaEventDestroy(join));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));
    CHECK(cudaFree(x));
    CHECK(cudaFree(y));
}

static void test_explicit()
{
    const int n = 300;
    float *x, *y;
    CHECK(cudaMalloc(&x, n * sizeof(float)));
    CHECK(cudaMalloc(&y, n * sizeof(float)));
    std::vector<float> ones(n, 1.0f), host(n);

    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));

    // Added out of order: the dependencies decide when each node runs. A
    // diamond: memset x and copy y, then scale x, then add y to x.
    cudaGraphNode_t memset_node, copy_node, scale_node, add_node, host_node, empty_node;

    cudaMemsetParams memset_params = {0};
    memset_params.dst         = x;
    memset_params.value       = 0x40400000;   // 3.0f
    memset_params.elementSize = sizeof(float);
    memset_params.width       = n;
    memset_params.height      = 1;
    CHECK(cudaGraphAddMemsetNode(&memset_node, graph, nullptr, 0, &memset_params));

    cudaMemcpy3DParms copy_params = {0};
    copy_params.srcPtr = make_cudaPitchedPtr(ones.data(), n * sizeof(float), n, 1);
    copy_params.dstPtr = make_cudaPitchedPtr(y, n * sizeof(float), n, 1);
    copy_params.extent = make_cudaExtent(n * sizeof(float), 1, 1);
    copy_params.kind   = cudaMemcpyHostToDevice;
    CHECK(cudaGraphAddMemcpyNode(&copy_node, graph, nullptr, 0, &copy_params));

    int count = n;
    float factor = 2.0f;
    void *scale_args[] = {&x, &factor, &count};
    cudaKernelNodeParams scale_params = {0};
    scale_params.func           = (void *)scale;
    scale_params.gridDim        = dim3((n + 127) / 128);
    scale_params.blockDim       = dim3(128);
    scale_params.kernelParams   = scale_args;
    CHECK(cudaGraphAddKernelNode(&scale_node, graph, &memset_node, 1, &scale_params));
    factor = 100.0f;   // the arguments were copied

    void *add_args[] = {&x, &y, &count};
    cudaKernelNodeParams add_params = scale_params;
    add_params.func         = (void *)add;
    add_params.kernelParams = add_args;
    cudaGraphNode_t add_deps[] = {scale_node, copy_node};
    CHECK(cudaGraphAddKernelNode(&add_node, graph, add_deps, 2, &add_params));

    cudaHostNodeParams host_params = {record, reinterpret_cast<void *>(7)};
    CHECK(cudaGraphAddHostNode(&host_node, graph, &add_node, 1, &host_params));
    CHECK(cudaGraphAddEmptyNode(&empty_node, graph, &host_node, 1));

    cudaGraphNodeType type;
    CHECK(cudaGraphNodeGetType(add_node, &type));
    EXPECT(type == cudaGraphNodeTypeKernel);
    cudaKernelNodeParams got;
    CHECK(cudaGraphKernelNodeGetParams(scale_node, &got));
    EXPECT(got.func == (void *)scale && got.gridDim.x == 3u && *static_cast<float *>(got.kernelParams[1]) == 2.0f);
    size_t ndeps = 0;
    CHECK(cudaGraphNodeGetDependencies(add_node, nullptr, &ndeps));
    EXPECT(ndeps == 2);

    // A graph with this one as a child node, after another kernel
    cudaGraph_t parent;
    CHECK(cudaGraphCreate(&parent, 0));
    cudaGraphNode_t first, child;
    float half = 0.5f;
    void *half_args[] = {&x, &half, &count};
    cudaKernelNodeParams half_params = scale_params;
    half_params.kernelParams = half_args;
    CHECK(cudaGraphAddKernelNode(&first, parent, nullptr, 0, &half_params));
    CHECK(cudaGraphAddChildGraphNode(&child, parent, &first, 1, graph));
    CHECK(cudaGraphAddKernelNode(&first, parent, &child, 1, &half_params));

    order.clear();
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, parent, nullptr, nullptr, 0));
    CHECK(cudaGraphDestroy(graph));    // the executable graph has its own copy
    CHECK(cudaGraphLaunch(exec, nullptr));
    CHECK(cudaMemcpy(host.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT(host[0] == 3.5f && host[n - 1] == 3.5f);   // (3 * 2 + 1) / 2
    EXPECT(order == std::vector<int>{7});

    // Change a kernel node of the executable graph
    factor = 4.0f;
    CHECK(cudaGraphExecKernelNodeSetParams(exec, first, &half_params));
    CHECK(cudaGraphLaunch(exec, nullptr));

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(parent));

    // cudaLaunchKernel with the kernel as a void *
    factor = 10.0f;
    CHECK(cudaLaunchKernel((const void *)scale, dim3(3), dim3(128), scale_args, 0, nullptr));
    CHECK(cudaMemcpy(host.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT(host[5] == 35.0f);
    // A void * that isn't a known kernel
    int not_a_kernel = 0;
    EXPECT(cudaLaunchKernel(static_cast<const void *>(&not_a_kernel), dim3(1), dim3(1), scale_args) ==
           cudaErrorInvalidDeviceFunction);
    cudaGetLastError();

    CHECK(cudaFree(x));
    CHECK(cudaFree(y));
}

static void test_conditionals()
{
    int *counter, *iterations, *value, *out;
    CHECK(cudaMallocManaged(&counter, sizeof(int)));
    CHECK(cudaMallocManaged(&iterations, sizeof(int)));
    CHECK(cudaMallocManaged(&value, sizeof(int)));
    CHECK(cudaMallocManaged(&out, sizeof(int)));

    // while (counter > 0) count_down; the condition starts at 1 on every launch
    {
        cudaGraph_t graph;
        CHECK(cudaGraphCreate(&graph, 0));
        cudaGraphConditionalHandle handle;
        CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));
        cudaGraphNodeParams params = {cudaGraphNodeTypeConditional};
        params.conditional.handle = handle;
        params.conditional.type   = cudaGraphCondTypeWhile;
        params.conditional.size   = 1;
        cudaGraphNode_t node;
        CHECK(cudaGraphAddNode(&node, graph, nullptr, 0, &params));
        cudaGraph_t body = params.conditional.phGraph_out[0];

        // The body is captured into the existing graph
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        CHECK(cudaStreamBeginCaptureToGraph(stream, body, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
        count_down<<<1, 1, 0, stream>>>(counter, iterations, handle);
        CHECK(cudaStreamEndCapture(stream, nullptr));
        CHECK(cudaStreamDestroy(stream));

        cudaGraphExec_t exec;
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        for (int start : {5, 1, 3}) {
            *counter    = start;
            *iterations = 0;
            CHECK(cudaGraphLaunch(exec, nullptr));
            EXPECT(*counter == 0 && *iterations == start);
        }
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(graph));
    }

    // if/else and switch, on a condition set by a kernel
    for (cudaGraphConditionalNodeType type : {cudaGraphCondTypeIf, cudaGraphCondTypeSwitch}) {
        cudaGraph_t graph;
        CHECK(cudaGraphCreate(&graph, 0));
        cudaGraphConditionalHandle handle;
        CHECK(cudaGraphConditionalHandleCreate(&handle, graph));

        cudaGraphNodeParams params = {cudaGraphNodeTypeKernel};
        void *set_args[] = {&value, &handle};
        params.kernel.func         = (void *)set_condition;
        params.kernel.gridDim      = dim3(1);
        params.kernel.blockDim     = dim3(1);
        params.kernel.kernelParams = set_args;
        cudaGraphNode_t setter, conditional;
        CHECK(cudaGraphAddNode(&setter, graph, nullptr, 0, &params));

        const unsigned bodies = type == cudaGraphCondTypeIf ? 2 : 3;
        cudaGraphNodeParams cparams = {cudaGraphNodeTypeConditional};
        cparams.conditional.handle = handle;
        cparams.conditional.type   = type;
        cparams.conditional.size   = bodies;
        CHECK(cudaGraphAddNode(&conditional, graph, &setter, 1, &cparams));
        for (unsigned b = 0; b < bodies; ++b) {
            int tag = 10 + int(b);
            void *store_args[] = {&out, &tag};
            params.kernel.func         = (void *)store;
            params.kernel.kernelParams = store_args;
            cudaGraphNode_t node;
            CHECK(cudaGraphAddNode(&node, cparams.conditional.phGraph_out[b], nullptr, 0, &params));
        }

        cudaGraphExec_t exec;
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        for (int v : {0, 1, 2, 7}) {
            *value = v;
            *out   = -1;
            CHECK(cudaGraphLaunch(exec, nullptr));
            const int expected = type == cudaGraphCondTypeIf ? (v != 0 ? 10 : 11) : (v < 3 ? 10 + v : -1);
            EXPECT(*out == expected);
        }
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(graph));
    }

    CHECK(cudaFree(counter));
    CHECK(cudaFree(iterations));
    CHECK(cudaFree(value));
    CHECK(cudaFree(out));
}

static uint64_t used_memory()
{
    uint64_t used = 0;
    CHECK(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &used));
    return used;
}

static void test_memory()
{
    const size_t bytes = 1 << 20;
    const uint64_t before = used_memory();

    // Allocated by the graph, freed by the program, allocated again by the
    // next launch at the same address
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaMemAllocNodeParams alloc = {};
    alloc.poolProps.allocType     = cudaMemAllocationTypePinned;
    alloc.poolProps.location.type = cudaMemLocationTypeDevice;
    alloc.bytesize                = bytes;
    cudaGraphNode_t alloc_node, fill_node;
    CHECK(cudaGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc));
    void *ptr = alloc.dptr;
    EXPECT(ptr != nullptr);
    cudaMemsetParams fill = {ptr, 0, 0x5a, 1, bytes, 1};
    CHECK(cudaGraphAddMemsetNode(&fill_node, graph, &alloc_node, 1, &fill));

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CHECK(cudaGraphDestroy(graph));
    EXPECT(used_memory() == before);
    for (int launch = 0; launch < 2; ++launch) {
        CHECK(cudaGraphLaunch(exec, nullptr));
        EXPECT(used_memory() == before + bytes);
        EXPECT(static_cast<unsigned char *>(ptr)[bytes - 1] == 0x5a);
        CHECK(cudaFree(ptr));
        EXPECT(used_memory() == before);
    }
    EXPECT(cudaFree(ptr) == cudaErrorInvalidValue);   // already free
    cudaGetLastError();
    CHECK(cudaGraphExecDestroy(exec));

    // cudaMallocAsync and cudaFreeAsync in a capture become memory nodes
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    int *result;
    CHECK(cudaMallocManaged(&result, sizeof(int)));
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    int *scratch = nullptr;
    CHECK(cudaMallocAsync(&scratch, sizeof(int), stream));
    store<<<1, 1, 0, stream>>>(scratch, 42);
    CHECK(cudaMemcpyAsync(result, scratch, sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CHECK(cudaFreeAsync(scratch, stream));
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    *result = 0;
    CHECK(cudaGraphLaunch(exec, stream));
    EXPECT(*result == 42 && used_memory() == before);
    uint64_t high = 0;
    CHECK(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh, &high));
    EXPECT(high >= before + bytes);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(result));
}

static void test_pitched_memory()
{
    const size_t width = 10, height = 4;
    float *dev;
    size_t pitch = 0;
    CHECK(cudaMallocPitch(&dev, &pitch, width * sizeof(float), height));
    EXPECT(pitch >= width * sizeof(float) && pitch % 256 == 0);
    CHECK(cudaMemset2D(dev, pitch, 0, width * sizeof(float), height));

    float host[height][width];
    for (size_t r = 0; r < height; ++r)
        for (size_t c = 0; c < width; ++c)
            host[r][c] = float(r * 100 + c);
    CHECK(cudaMemcpy2D(dev, pitch, host, width * sizeof(float), width * sizeof(float), height,
                       cudaMemcpyHostToDevice));
    EXPECT(reinterpret_cast<float *>(reinterpret_cast<char *>(dev) + 3 * pitch)[9] == 309.0f);

    // A 2 x 2 x 2 box of a 3D allocation
    cudaPitchedPtr volume;
    CHECK(cudaMalloc3D(&volume, make_cudaExtent(4 * sizeof(int), 3, 2)));
    int box[2][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    cudaMemcpy3DParms p = {0};
    p.srcPtr = make_cudaPitchedPtr(box, 2 * sizeof(int), 2, 2);
    p.dstPtr = volume;
    p.dstPos = make_cudaPos(1 * sizeof(int), 1, 0);
    p.extent = make_cudaExtent(2 * sizeof(int), 2, 2);
    p.kind   = cudaMemcpyHostToDevice;
    CHECK(cudaMemcpy3D(&p));
    auto at = [&](size_t x, size_t y, size_t z) {
        return reinterpret_cast<int *>(static_cast<char *>(volume.ptr) + (z * volume.ysize + y) * volume.pitch)[x];
    };
    EXPECT(at(1, 1, 0) == 1 && at(2, 2, 0) == 4 && at(1, 1, 1) == 5 && at(2, 2, 1) == 8);

    CHECK(cudaFree(dev));
    CHECK(cudaFree(volume.ptr));
}

static void test_clock()
{
    long long *ticks;
    long *legacy;
    CHECK(cudaMallocManaged(&ticks, 2 * sizeof(long long)));
    CHECK(cudaMallocManaged(&legacy, 2 * sizeof(long)));
    clocks<<<1, 1>>>(ticks, legacy);
    EXPECT(ticks[1] - ticks[0] >= 1000);
    // clock() in a kernel reads the same clock, just after clock64()
    EXPECT(legacy[0] >= ticks[0] && legacy[1] >= ticks[1] && legacy[1] > legacy[0]);
    CHECK(cudaFree(ticks));
    CHECK(cudaFree(legacy));
}

int main()
{
    test_capture();
    test_explicit();
    test_conditionals();
    test_memory();
    test_pitched_memory();
    test_clock();
    std::printf("graphs: %s\n", errors == 0 ? "PASS" : "FAIL");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
