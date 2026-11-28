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
// Micro-benchmarks of the execution engine. Each result is the median of
// several runs. Run on an otherwise idle machine.
//

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "cuda4cpu.hpp"
#include "cublas_v2.h"
#include "cufft.h"
#include "curand.h"

using namespace cuda4cpu;

namespace {

template <typename F>
double median_ms(F &&f, int runs = 7)
{
    std::vector<double> times;
    for (int r = 0; r < runs; ++r) {
        auto start = std::chrono::steady_clock::now();
        f();
        auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

//! Median time of op(ptr) on a fresh allocation of bytes, as programs copy
//! their inputs into memory they just allocated
template <typename F>
double fresh_ms(size_t bytes, F &&op, int runs = 7)
{
    std::vector<double> times;
    for (int r = 0; r < runs; ++r) {
        void *ptr = nullptr;
        cudaMalloc(&ptr, bytes);
        auto start = std::chrono::steady_clock::now();
        op(ptr);
        auto end = std::chrono::steady_clock::now();
        cudaFree(ptr);
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void row(const char *name, double value, const char *unit)
{
    std::printf("%-58s %10.2f %s\n", name, value, unit);
}

__global__ void empty_kernel(int *) {}

__global__ void barriers(int n)
{
    for (int i = 0; i < n; ++i)
        __syncthreads();
}

__global__ void shuffles(int *out, int n)
{
    int v = threadIdx.x;
    for (int i = 0; i < n; ++i)
        v += __shfl_xor_sync(0xffffffff, v, i & 31);
    if (v == -1)
        *out = v;   // keep the shuffles alive
}

__global__ void vecadd(float *C, const float *A, const float *B, size_t n)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

// Most threads of a launch with more threads than elements only check bounds
__global__ void bounds_check(float *out, size_t n)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = 1.f;
}

// A grid-stride copy, with and without the loop tick that cuda4cpu-rewrite
// inserts in the outermost loops of device functions
__global__ void grid_stride(unsigned char *out, const unsigned char *in, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = in[i];
}

__global__ void grid_stride_ticks(unsigned char *out, const unsigned char *in, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        cuda4cpu::loop_tick();
        out[i] = in[i];
    }
}

// Compute-bound: a polynomial per element
__global__ void poly(float *out, const float *in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i], acc = 0.f;
        for (int k = 0; k < 16; ++k)
            acc = acc * v + 0.5f;
        out[i] = acc;
    }
}

// A 2D, 5-point stencil without shared memory
__global__ void stencil(float *out, const float *in, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && y > 0 && x < w - 1 && y < h - 1)
        out[y * w + x] = 0.2f * (in[y * w + x] + in[y * w + x - 1] + in[y * w + x + 1] +
                                 in[(y - 1) * w + x] + in[(y + 1) * w + x]);
}

// Block b does work proportional to b, like a triangular loop nest, or to
// the number of blocks after it
__global__ void triangular(float *out, int scale, bool decreasing)
{
    const int work = int(decreasing ? gridDim.x - 1 - blockIdx.x : blockIdx.x) * scale;
    float acc = 0.f;
    for (int i = 0; i < work; ++i)
        acc += float(i) * 1e-7f;
    out[size_t(blockIdx.x) * blockDim.x + threadIdx.x] = acc;
}

}

int main()
{
    std::printf("cuda4cpu micro-benchmarks, %d OS threads\n\n", omp_get_max_threads());

    // Launch overhead with cached fibers
    {
        launch(empty_kernel, 1, 256).call(nullptr);
        const int reps = 1000;
        double ms = median_ms([&] { for (int r = 0; r < reps; ++r) launch(empty_kernel, 1, 256).call(nullptr); });
        row("launch, 1 block x 256 threads, empty kernel", ms * 1e3 / reps, "us");
    }

    // Fewer blocks than OS threads
    {
        launch(empty_kernel, 4, 256).call(nullptr);
        const int reps = 1000;
        double ms = median_ms([&] { for (int r = 0; r < reps; ++r) launch(empty_kernel, 4, 256).call(nullptr); });
        row("launch, 4 blocks x 256 threads, empty kernel", ms * 1e3 / reps, "us");
    }

    // A new block shape each time, so that every OS thread creates its fibers
    {
        unsigned shape = 1000;
        double ms = median_ms([&] { launch(empty_kernel, 1024, shape++).call(nullptr); });
        row("first launch, 1024 blocks x ~1000 threads (creates fibers)", ms, "ms");
    }

    // One fiber switch per thread per barrier
    {
        const int n = 1000, threads = 256, blocks = 64;
        double base = median_ms([&] { launch(barriers, blocks, threads).call(0); });
        double ms   = median_ms([&] { launch(barriers, blocks, threads).call(n); });
        double per_core = double(blocks) * threads * n / omp_get_max_threads();
        row("__syncthreads, per thread", (ms - base) * 1e6 / per_core, "ns");
    }

    {
        const int n = 1000, threads = 256, blocks = 64;
        int out = 0;
        double base = median_ms([&] { launch(shuffles, blocks, threads).call(&out, 0); });
        double ms   = median_ms([&] { launch(shuffles, blocks, threads).call(&out, n); });
        double per_core = double(blocks) * threads * n / omp_get_max_threads();
        row("__shfl_xor_sync, per lane", (ms - base) * 1e6 / per_core, "ns");
    }

    // Copies and memsets, into fresh allocations (bound by page faults) and
    // into memory already touched (bound by memory bandwidth)
    {
        const size_t bytes = size_t(256) << 20;
        std::vector<char> host(bytes, 1);
        std::vector<char> touched(bytes, 0);
        double fresh_copy = fresh_ms(bytes, [&](void *dev) {
            cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice);
        });
        double fresh_set = fresh_ms(bytes, [&](void *dev) { cudaMemset(dev, 0, bytes); });
        // Both are host vectors, which cudaMemcpyDefault allows
        double copy = median_ms([&] { cudaMemcpy(touched.data(), host.data(), bytes, cudaMemcpyDefault); });
        row("cudaMemcpy 256 MB into a new allocation", fresh_copy, "ms");
        row("cudaMemset 256 MB of a new allocation", fresh_set, "ms");
        row("cudaMemcpy 256 MB into touched memory", copy, "ms");
    }

    // Barrier-free kernel against the equivalent OpenMP loop
    {
        const size_t n = size_t(1) << 24;
        std::vector<float> A(n, 1.f), B(n, 2.f), C(n);
        double ms = median_ms([&] { launch(vecadd, unsigned(n / 256), 256).call(C.data(), A.data(), B.data(), n); });
        double omp = median_ms([&] {
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) C[i] = A[i] + B[i];
        });
        // The form that cuda4cpu-rewrite generates for vecadd<<<...>>>(...): the
        // kernel is known at compile time, so it can be inlined
        double ms_rewritten = median_ms([&] {
            launch([&](const auto &...a) { vecadd(a...); }, unsigned(n / 256), 256)
                .call(C.data(), A.data(), B.data(), n);
        });
        row("vecadd 16M floats", ms, "ms");
        row("  launched as rewritten from vecadd<<<...>>>", ms_rewritten, "ms");
        row("  same loop with OpenMP", omp, "ms");
    }

    // Per-thread cost of barrier-free kernels: threads that only check bounds,
    // and vecadd on data that fits in the caches
    {
        const unsigned blocks = 16384, threads = 1024;
        // A real buffer, although no thread writes it: in
        // microbench_vectorized, the threads still issue masked stores. At
        // nullptr, each took a microcode assist (2.3 instead of 0.1 ns per
        // thread). On a Ryzen 8845HS, stores masked off still cost the
        // bandwidth of their cache lines: the vectorized loop runs at
        // 0.14 ns per thread on one core, and doesn't get faster with more.
        std::vector<float> out(size_t(blocks) * threads, 1.f);
        auto run = [&] {
            launch([&](const auto &...a) { bounds_check(a...); }, blocks, threads).call(out.data(), size_t(0));
        };
        run();
        double ms = median_ms(run);
        row("CUDA thread that only checks bounds, per thread and core", ms * 1e6 * omp_get_max_threads() /
                                                                         (double(blocks) * threads), "ns");

        const size_t n = size_t(1) << 20;
        const int reps = 50;
        std::vector<float> A(n, 1.f), B(n, 2.f), C(n);
        double cached = median_ms([&] {
            for (int r = 0; r < reps; ++r)
                launch([&](const auto &...a) { vecadd(a...); }, unsigned(n / 256), 256)
                    .call(C.data(), A.data(), B.data(), n);
        });
        double omp = median_ms([&] {
            for (int r = 0; r < reps; ++r) {
                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) C[i] = A[i] + B[i];
            }
        });
        row("vecadd 1M floats (in cache), as rewritten", cached / reps, "ms");
        row("  same loop with OpenMP", omp / reps, "ms");
    }

    // Barrier-free kernels on data that fits in the caches, on one OS thread.
    // In microbench_vectorized, Clang vectorizes the loop over their CUDA
    // threads.
    {
        const int saved = omp_get_max_threads();
        omp_set_num_threads(1);
        const int w = 512, h = 512, n = w * h, reps = 20;
        std::vector<float> in(n, 1.f), out(n);
        double poly_ms = median_ms([&] {
            for (int r = 0; r < reps; ++r)
                launch([&](const auto &...a) { poly(a...); }, unsigned(n / 256), 256).call(out.data(), in.data(), n);
        });
        double stencil_ms = median_ms([&] {
            for (int r = 0; r < reps; ++r)
                launch([&](const auto &...a) { stencil(a...); }, dim3(w / 64, h / 4), dim3(64, 4))
                    .call(out.data(), in.data(), w, h);
        });
        omp_set_num_threads(saved);
        row("polynomial per element, in cache, one OS thread", poly_ms * 1e6 / (double(reps) * n), "ns");
        row("2D stencil per element, in cache, one OS thread", stencil_ms * 1e6 / (double(reps) * n), "ns");
    }

    // A grid-stride loop run one thread after another touches a new page per
    // iteration; with loop ticks, the threads take turns every few iterations
    {
        const int n = 50000000 & ~255;
        std::vector<unsigned char> in(n, 1), out(n);
        double plain = median_ms([&] {
            launch([&](const auto &...a) { grid_stride(a...); }, 64, 256).call(out.data(), in.data(), n);
        });
        double ticks = median_ms([&] {
            launch([&](const auto &...a) { grid_stride_ticks(a...); }, 64, 256).call(out.data(), in.data(), n);
        });
        row("grid-stride copy 50 MB, <<<64, 256>>>, one thread after another", plain, "ms");
        row("  threads taking turns (loop ticks)", ticks, "ms");
    }

    // Uneven blocks, against an OpenMP loop with dynamic scheduling
    {
        const unsigned blocks = 512, threads = 64;
        const int scale = 40;
        std::vector<float> out(blocks * threads);
        double ms = median_ms([&] { launch(triangular, blocks, threads).call(out.data(), scale, false); });
        double down = median_ms([&] { launch(triangular, blocks, threads).call(out.data(), scale, true); });
        double omp = median_ms([&] {
            #pragma omp parallel for schedule(dynamic)
            for (unsigned b = 0; b < blocks; ++b) {
                for (unsigned t = 0; t < threads; ++t) {
                    float acc = 0.f;
                    for (int i = 0; i < int(b) * scale; ++i) acc += float(i) * 1e-7f;
                    out[b * threads + t] = acc;
                }
            }
        });
        row("triangular, 512 blocks of increasing cost", ms, "ms");
        row("  same loop with OpenMP schedule(dynamic)", omp, "ms");
        row("triangular, 512 blocks of decreasing cost", down, "ms");
    }

    // The libraries' host calls, alone and right after a kernel, while the
    // OpenMP threads that ran it may still be spinning
    {
        const int n = 1 << 16;
        float *scratch = nullptr;
        cudaMalloc(&scratch, n * sizeof(float));
        auto kernels = [&] {
            for (int r = 0; r < 20; ++r)
                launch(bounds_check, n / 256, 256).call(scratch, size_t(n));
        };
        const double kernels_ms = median_ms(kernels);
        auto after_kernel = [&](auto &&op) {
            return (median_ms([&] {
                for (int r = 0; r < 20; ++r) {
                    launch(bounds_check, n / 256, 256).call(scratch, size_t(n));
                    op();
                }
            }) - kernels_ms) / 20;
        };
        // Alone: after the threads of the previous calls have gone to sleep
        // (libomp's spin for 200 ms)
        auto alone = [&](auto &&op) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            return median_ms(op);
        };

        cufftComplex *data = nullptr;
        cudaMalloc(&data, sizeof(cufftComplex) * 1024 * 1024);
        cudaMemset(data, 0, sizeof(cufftComplex) * 1024 * 1024);
        cufftHandle plan;
        cufftPlan2d(&plan, 1024, 1024, CUFFT_C2C);
        auto fft = [&] { cufftExecC2C(plan, data, data, CUFFT_FORWARD); };
        fft();
        row("cuFFT C2C 1024 x 1024", alone(fft), "ms");
        row("  alternating with kernels", after_kernel(fft), "ms");
        cufftDestroy(plan);
        cufftPlan1d(&plan, 256, CUFFT_C2C, 256);
        fft();
        row("cuFFT C2C 256, batch 256", alone(fft) * 1e3, "us");
        cufftDestroy(plan);
        cudaFree(data);

        // On the CBLAS library that CUDA4CPU_BLAS names or cuda4cpu finds,
        // or its own (CUDA4CPU_BLAS=builtin)
        const int m = 512;
        float *a = nullptr, *b = nullptr, *c = nullptr;
        cudaMalloc(&a, sizeof(float) * m * m);
        cudaMalloc(&b, sizeof(float) * m * m);
        cudaMalloc(&c, sizeof(float) * m * m);
        cudaMemset(a, 0, sizeof(float) * m * m);
        cudaMemset(b, 0, sizeof(float) * m * m);
        cublasHandle_t handle;
        cublasCreate(&handle);
        const float one = 1.f, zero = 0.f;
        auto gemm = [&] { cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &one, a, m, b, m, &zero, c, m); };
        gemm();
        row("cublasSgemm 512 x 512 x 512", alone(gemm), "ms");
        row("  alternating with kernels", after_kernel(gemm), "ms");
        cublasDestroy(handle);
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);

        // The host API: 4096 lanes of XORWOW states
        const size_t count = size_t(1) << 24;
        float *out = nullptr;
        cudaMalloc(&out, count * sizeof(float));
        curandGenerator_t generator;
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
        auto generate = [&] { curandGenerateUniform(generator, out, count); };
        generate();
        row("curandGenerateUniform, XORWOW, 16 M floats", alone(generate), "ms");
        curandDestroyGenerator(generator);
        cudaFree(out);
        cudaFree(scratch);
    }
}
