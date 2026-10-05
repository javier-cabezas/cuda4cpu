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
// Row-wise softmax with temperature, as found in machine-learning code: one
// warp per row, warp-shuffle reductions, fast-math intrinsics, read-only
// loads and a __constant__ parameter set with cudaMemcpyToSymbol. The only
// porting change is the kernel launch.
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s:%d: %s failed: %s (%s)\n", __FILE__, __LINE__, \
                         #call, cudaGetErrorString(err__), cudaGetErrorName(err__)); \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

__constant__ float inv_temperature;

__device__ __forceinline__ float warp_max(float v)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}

__device__ __forceinline__ float warp_sum(float v)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// One warp per row
__global__ void __launch_bounds__(256)
softmax_rows(float *__restrict__ out, const float *__restrict__ in, int rows, int cols)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    if (warp >= rows)
        return;

    const float *row = in + (size_t)warp * cols;

    float m = -INFINITY;
    for (int c = lane; c < cols; c += warpSize)
        m = fmaxf(m, __ldg(row + c) * inv_temperature);
    m = warp_max(m);

    float s = 0.f;
    for (int c = lane; c < cols; c += warpSize)
        s += __expf(__ldg(row + c) * inv_temperature - m);
    s = warp_sum(s);

    for (int c = lane; c < cols; c += warpSize)
        out[(size_t)warp * cols + c] = __fdividef(__expf(__ldg(row + c) * inv_temperature - m), s);
}

int main()
{
    const int rows = 4099, cols = 1000;   // the last block is partially used
    const float temperature = 0.5f;
    const size_t bytes = size_t(rows) * cols * sizeof(float);

    std::vector<float> h_in(size_t(rows) * cols), h_out(h_in.size());
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.f, 3.f);
    for (auto &v : h_in) v = dist(gen);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    const float inv_t = 1.f / temperature;
    CUDA_CHECK(cudaMemcpyToSymbol(inv_temperature, &inv_t, sizeof(float)));

    const int threads = 256, warps_per_block = threads / 32;
    const int blocks = (rows + warps_per_block - 1) / warps_per_block;
    cuda4cpu::launch(softmax_rows, blocks, threads).call(d_out, d_in, rows, cols);   // CUDA: softmax_rows<<<blocks, threads>>>(...)
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // Host reference in double precision
    size_t errors = 0;
    for (int r = 0; r < rows; ++r) {
        const float *x = &h_in[size_t(r) * cols];
        double m = -INFINITY, s = 0.0;
        for (int c = 0; c < cols; ++c) m = std::max(m, double(x[c]) * inv_t);
        for (int c = 0; c < cols; ++c) s += std::exp(double(x[c]) * inv_t - m);
        for (int c = 0; c < cols; ++c) {
            double expected = std::exp(double(x[c]) * inv_t - m) / s;
            double got = h_out[size_t(r) * cols + c];
            if (std::abs(got - expected) > 1e-5 * std::max(expected, 1e-3)) {
                if (errors++ < 10)
                    std::printf("[%d, %d] = %g, expected %g\n", r, c, got, expected);
            }
        }
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    std::printf("softmax %dx%d: %s\n", rows, cols, errors ? "FAIL" : "PASS");
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
