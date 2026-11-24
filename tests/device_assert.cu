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

// A failed assert() in a kernel, as CUDA's device-side assert: each failing
// thread prints CUDA's message with its block and thread, and ends, while the
// others go on; the error (cudaErrorAssert) is sticky: the synchronizing
// calls, copies, memsets and cudaGetLastError return it, and launches don't
// run, until cudaDeviceReset(). A failed assert in host code still aborts.

// The asserts are the test, in Release builds too
#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <set>
#include <signal.h>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#include <omp.h>
#include <cuda_runtime.h>

static int failures = 0;

#define EXPECT(cond)                                                          \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

constexpr int threads = 64;

constexpr int fails_line = __LINE__ + 4;        // its assert's
__global__ void fails(int *out, int first, int second)
{
    const int t = threadIdx.x;
    assert(t != first && t != second);
    out[t] = t + 1;
}

// Threads wait at barriers before and after the one that fails
__global__ void fails_between_barriers(int *out, int failing)
{
    __shared__ int tile[threads];
    const int t = threadIdx.x;
    tile[t] = t;
    __syncthreads();
    assert(t != failing);
    __syncthreads();
    out[t] = tile[(t + 1) % threads] + 1;
}

__global__ void writes(int *out, int value)
{
    out[threadIdx.x] = value;
}

__global__ void fails_in_one_block(int *out, int block)
{
    assert(int(blockIdx.x) != block || threadIdx.x != 3);
    out[blockIdx.x * blockDim.x + threadIdx.x] = 1;
}

__host__ __device__ int checked_half(int n)
{
    assert(n % 2 == 0);
    return n / 2;
}

//! What the kernels print to stderr while f runs
template <typename F>
static std::string stderr_of(F f)
{
    char name[] = "/tmp/device_assert_XXXXXX";
    const int file = mkstemp(name);
    unlink(name);
    std::fflush(stderr);
    const int saved = dup(STDERR_FILENO);
    dup2(file, STDERR_FILENO);
    f();
    std::fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    std::string text;
    char buffer[4096];
    lseek(file, 0, SEEK_SET);
    for (ssize_t n; (n = read(file, buffer, sizeof(buffer))) > 0;)
        text.append(buffer, size_t(n));
    close(file);
    return text;
}

//! The lines printed, without cuda4cpu's notes (a random order's seed)
static std::set<std::string> lines_of(const std::string &text)
{
    std::set<std::string> lines;
    size_t start = 0;
    for (size_t end; (end = text.find('\n', start)) != std::string::npos; start = end + 1) {
        if (text.compare(start, 9, "cuda4cpu:") != 0)
            lines.insert(text.substr(start, end - start));
    }
    return lines;
}

//! Whether line is CUDA's message for thread t of block b at an assert of
//! function at this file's line (compilers spell the function's type
//! differently: int* or int *)
static bool is_message(const std::string &text, const char *function, const char *assertion, int line, int b,
                       int t)
{
    char start[256], end[256];
    std::snprintf(start, sizeof(start), "%s:%d: void %s(", __FILE__, line, function);
    std::snprintf(end, sizeof(end), "): block: [%d,0,0], thread: [%d,0,0] Assertion `%s` failed.", b, t, assertion);
    return text.starts_with(start) && text.ends_with(end);
}

static void threads_fail_and_the_rest_go_on(int *out)
{
    cudaMemset(out, 0, threads * sizeof(int));
    const std::string printed = stderr_of([&] { fails<<<1, threads>>>(out, 7, 40); });
    const std::set<std::string> lines = lines_of(printed);
    EXPECT(lines.size() == 2);
    for (int t : {7, 40}) {
        EXPECT(std::any_of(lines.begin(), lines.end(), [t](const std::string &line) {
            return is_message(line, "fails", "t != first && t != second", fails_line, 0, t);
        }));
    }
    for (int t = 0; t < threads; ++t)
        EXPECT(out[t] == (t == 7 || t == 40 ? 0 : t + 1));
}

static void the_error_is_sticky(int *out)
{
    EXPECT(cudaGetLastError() == cudaErrorAssert);
    EXPECT(cudaGetLastError() == cudaErrorAssert);           // it stays
    EXPECT(cudaPeekAtLastError() == cudaErrorAssert);
    EXPECT(cudaDeviceSynchronize() == cudaErrorAssert);
    EXPECT(cudaStreamSynchronize(nullptr) == cudaErrorAssert);
    EXPECT(cudaStreamQuery(nullptr) == cudaErrorAssert);
    cudaEvent_t event;
    EXPECT(cudaEventCreate(&event) == cudaSuccess);
    EXPECT(cudaEventSynchronize(event) == cudaErrorAssert);
    EXPECT(cudaEventQuery(event) == cudaErrorAssert);
    cudaEventDestroy(event);
    EXPECT(std::strcmp(cudaGetErrorString(cudaErrorAssert), "device-side assert triggered") == 0);

    // Copies, memsets and launches do nothing
    int host[threads];
    std::memset(host, 0x5a, sizeof(host));
    EXPECT(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost) == cudaErrorAssert);
    EXPECT(host[1] == 0x5a5a5a5a);
    EXPECT(cudaMemset(out, 0xff, threads * sizeof(int)) == cudaErrorAssert);
    EXPECT(out[1] == 2);
    writes<<<1, threads>>>(out, 99);
    EXPECT(out[1] == 2);
    void *args[] = {&out, &failures};
    EXPECT(cudaLaunchKernel(writes, dim3(1), dim3(threads), args) == cudaErrorAssert);
    EXPECT(out[1] == 2);
}

static void reset_clears_it(int *out)
{
    EXPECT(cudaDeviceReset() == cudaSuccess);
    EXPECT(cudaGetLastError() == cudaSuccess);
    EXPECT(cudaDeviceSynchronize() == cudaSuccess);
    writes<<<1, threads>>>(out, 99);
    EXPECT(cudaDeviceSynchronize() == cudaSuccess);
    EXPECT(out[1] == 99);
}

static void between_barriers(int *out)
{
    const std::string printed = stderr_of([&] { fails_between_barriers<<<1, threads>>>(out, 5); });
    EXPECT(lines_of(printed).size() == 1);
    EXPECT(printed.find("block: [0,0,0], thread: [5,0,0] Assertion `t != failing` failed.") != std::string::npos);
    EXPECT(cudaDeviceSynchronize() == cudaErrorAssert);
    for (int t = 0; t < threads; ++t)
        EXPECT(out[t] == (t == 5 ? 99 : (t + 1) % threads + 1));
    cudaDeviceReset();
}

static void many_blocks()
{
    int *out;
    const int blocks = 256;
    cudaMallocManaged(&out, blocks * 32 * sizeof(int));
    cudaMemset(out, 0, blocks * 32 * sizeof(int));
    const std::string printed = stderr_of([&] { fails_in_one_block<<<blocks, 32>>>(out, 17); });
    EXPECT(lines_of(printed).size() == 1);
    EXPECT(printed.find("block: [17,0,0], thread: [3,0,0]") != std::string::npos);
    EXPECT(cudaDeviceSynchronize() == cudaErrorAssert);
    EXPECT(out[17 * 32 + 3] == 0);
    cudaDeviceReset();

    // On one OS thread, the blocks run in order: those after the failing one
    // don't run
    const int threads_before = omp_get_max_threads();
    omp_set_num_threads(1);
    cudaMemset(out, 0, blocks * 32 * sizeof(int));
    fails_in_one_block<<<blocks, 32>>>(out, 17);
    omp_set_num_threads(threads_before);
    EXPECT(cudaDeviceSynchronize() == cudaErrorAssert);
    EXPECT(out[16 * 32] == 1 && out[17 * 32 + 4] == 1 && out[18 * 32] == 0 && out[255 * 32] == 0);
    cudaDeviceReset();
    cudaFree(out);
}

//! A graph's launch stops at a kernel that fails, and doesn't start after
__global__ void fails_in_a_graph(int *out)
{
    assert(threadIdx.x != 2);
    out[threadIdx.x] = 5;
}

static void graphs(int *out)
{
    cudaMemset(out, 0, threads * sizeof(int));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    fails_in_a_graph<<<1, threads, 0, stream>>>(out);
    cudaMemsetAsync(out, 0x11, sizeof(int), stream);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t exec;
    EXPECT(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess);
    stderr_of([&] { EXPECT(cudaGraphLaunch(exec, stream) == cudaErrorAssert); });
    EXPECT(out[0] == 5);                // the memset after the kernel didn't run
    out[1] = 0;
    EXPECT(cudaGraphLaunch(exec, stream) == cudaErrorAssert);
    EXPECT(out[1] == 0);                // nor the graph, after
    cudaDeviceReset();
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

//! A message longer than a buffer of 1 KB would hold keeps its end
template <int N>
struct a_rather_long_name_for_a_type_that_makes_the_kernels_name_long {};

template <typename A, typename B, typename C>
__global__ void fails_with_a_long_name(A, B, C)
{
    assert(threadIdx.x != 1 &&
           "a failure in a kernel whose name and message are long a failure in a kernel whose name and"
           " message are long a failure in a kernel whose name and message are long a failure in a ker"
           "nel whose name and message are long a failure in a kernel whose name and message are long "
           "a failure in a kernel whose name and message are long a failure in a kernel whose name and"
           " message are long a failure in a kernel whose name and message are long a failure in a ker"
           "nel whose name and message are long a failure in a kernel whose name and message are long "
           "a failure in a kernel whose name and message are long a failure in a kernel whose name and"
           " message are long a failure in a kernel whose name and message are long a failure in a ker"
           "nel whose name and message are long a failure in a kernel whose name and message are long "
           "a failure in a kernel whose name and message are long a failure in a kernel whose name and"
           " message are long a failure in a kernel whose name and message are long a failure in a ker"
           "nel whose name and message are long a failure in a kernel whose name and message are long "
           "a failure in a kernel whose name and message are long a failure in a kernel whose name and"
           " message are long a failure in a kernel whose name and message are long a failure in a ker"
           "nel whose name and message are long" " and it is long");
}

static void long_message()
{
    using T = a_rather_long_name_for_a_type_that_makes_the_kernels_name_long<123456789>;
    const std::string printed = stderr_of([&] {
        fails_with_a_long_name<<<1, 4>>>(std::vector<T>{}, std::vector<std::vector<T>>{}, std::set<T *>{});
    });
    EXPECT(printed.size() > 1024);
    EXPECT(printed.ends_with("is long\"` failed.\n"));
    cudaDeviceReset();
}

//! A failed assert in host code, in a __host__ __device__ function, still
//! aborts: in a child process
static void host_assert_aborts()
{
    std::fflush(nullptr);
    const pid_t child = fork();
    if (child == 0) {
        const int null = open("/dev/null", O_WRONLY);
        dup2(null, STDERR_FILENO);
        std::exit(checked_half(3));
    }
    int status = 0;
    waitpid(child, &status, 0);
    EXPECT(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

int main()
{
    int *out;
    cudaMallocManaged(&out, threads * sizeof(int));
    threads_fail_and_the_rest_go_on(out);
    the_error_is_sticky(out);
    reset_clears_it(out);
    between_barriers(out);
    many_blocks();
    graphs(out);
    long_message();
    host_assert_aborts();
    EXPECT(cudaGetLastError() == cudaSuccess);
    cudaFree(out);
    if (failures == 0)
        std::printf("device_assert: all passed\n");
    return failures == 0 ? 0 : 1;
}
