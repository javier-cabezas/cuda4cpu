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
// HIP's additions to the kernel language, checked against host references:
// hipLaunchKernelGGL (with HIP_KERNEL_NAME, deduced template arguments and no
// arguments), launches with <<<...>>>, HIP's names of the built-in variables,
// HIP_DYNAMIC_SHARED in a kernel and at namespace scope, the warp functions
// without masks, bit manipulation, clocks, atomics, cooperative groups, and
// the platform and feature macros.
//

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#if !defined(__HIPCC__) || !defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__)
#error "HIP sources are compiled as HIP's NVIDIA platform"
#endif
#if __HIP_ARCH_HAS_DOUBLES__ != 1 || __HIP_ARCH_HAS_WARP_SHUFFLE__ != 1 || HIP_VERSION_MAJOR < 6
#error "the device's features and HIP's version"
#endif

namespace cg = cooperative_groups;

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

template <typename T, int Factor>
__global__ void scale(T *out, const T *in, int n)
{
    const int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < n)
        out[i] = in[i] * Factor;
}

//! Template arguments deduced from the launch's arguments, and a default argument
template <typename T>
__global__ void fill(T *out, T value, int stride = 1)
{
    out[(blockIdx.x * blockDim.x + threadIdx.x) * stride] = value;
}

__device__ int launches;

__global__ void count_launch()
{
    atomicAdd(&launches, 1);
}

//! Reverses each block's elements through dynamic shared memory
__global__ void reverse_block(int *data)
{
    HIP_DYNAMIC_SHARED(int, tile)
    const unsigned t = threadIdx.x, n = blockDim.x;
    tile[t] = data[blockIdx.x * n + t];
    __syncthreads();
    data[blockIdx.x * n + t] = tile[n - 1 - t];
}

// At namespace scope, as extern __shared__ arrays can be
HIP_DYNAMIC_SHARED(float, global_tile);

__global__ void rotate_block(float *data)
{
    const unsigned t = threadIdx.x, n = blockDim.x;
    global_tile[t] = data[t];
    __syncthreads();
    data[t] = global_tile[(t + 1) % n];
}

//! The warp functions without masks, per lane
__global__ void warp_functions(int *shfl, int *up, int *down, int *xor_, unsigned long long *ballot, int *any,
                               int *all, unsigned *lane)
{
    const int t = int(threadIdx.x);
    shfl[t] = __shfl(t * 10, 3);
    up[t] = __shfl_up(t, 2, 16);
    down[t] = __shfl_down(t, 4, 8);
    xor_[t] = __shfl_xor(double(t), 1);
    ballot[t] = __ballot(t % 3 == 0);
    any[t] = __any(t == 17);
    all[t] = __all(t < 64);
    lane[t] = __lane_id();
}

//! A block reduction with __shfl_down and warpSize
__global__ void reduce(const float *in, float *out)
{
    __shared__ float partial[32];
    float v = in[blockIdx.x * blockDim.x + threadIdx.x];
    for (int delta = warpSize / 2; delta > 0; delta /= 2)
        v += __shfl_down(v, delta);
    if (__lane_id() == 0)
        partial[threadIdx.x / warpSize] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0;
        for (unsigned w = 0; w < blockDim.x / warpSize; ++w)
            sum += partial[w];
        out[blockIdx.x] = sum;
    }
}

//! A reduction over tiles of cooperative groups, from hip_cooperative_groups.h
__global__ void tile_sums(const int *in, int *out)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<16> tile = cg::tiled_partition<16>(block);
    int sum = in[block.thread_rank()];
    for (unsigned delta = tile.size() / 2; delta > 0; delta /= 2)
        sum += tile.shfl_down(sum, delta);
    if (tile.thread_rank() == 0)
        out[block.thread_rank() / 16] = sum;
}

__global__ void atomics(float *sum, int *count)
{
    unsafeAtomicAdd(sum, 0.5f);
    safeAtomicAdd(sum, 0.25f);
    atomicAddNoRet(count, 1);
}

__global__ void clocks(long long *ticks)
{
    const long long wall = wall_clock64(), clock = __clock64();
    ticks[0] = wall_clock64() - wall;
    ticks[1] = __clock64() - clock;
}

int main()
{
    // hipLaunchKernelGGL with template arguments in HIP_KERNEL_NAME
    {
        const int n = 1000;
        std::vector<float> in(n), out(n);
        std::iota(in.begin(), in.end(), 0.0f);
        float *d_in, *d_out;
        hipMalloc(&d_in, n * sizeof(float));
        hipMalloc(&d_out, n * sizeof(float));
        hipMemcpy(d_in, in.data(), n * sizeof(float), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(scale<float, 3>), dim3((n + 255) / 256), dim3(256), 0, hipStreamDefault,
                           d_out, d_in, n);
        EXPECT(hipGetLastError() == hipSuccess);
        hipMemcpy(out.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost);
        int wrong = 0;
        for (int i = 0; i < n; ++i)
            wrong += out[i] != 3.0f * float(i);
        EXPECT(wrong == 0);
        hipFree(d_in);
        hipFree(d_out);
    }

    // Deduced template arguments, a default argument, no arguments, <<<...>>>
    {
        double values[8] = {};
        hipLaunchKernelGGL(fill, 2, 2, 0, 0, values, 2.5, 2);
        EXPECT(values[0] == 2.5 && values[1] == 0 && values[6] == 2.5);
        hipLaunchKernelGGL(fill, 1, 8, 0, 0, values, -1.0);
        EXPECT(values[1] == -1.0 && values[7] == -1.0);

        int zero = 0;
        hipMemcpyToSymbol(launches, &zero, sizeof(zero));
        hipLaunchKernelGGL(count_launch, dim3(4), dim3(8), 0, 0);
        count_launch<<<2, 8>>>();
        int count = 0;
        hipMemcpyFromSymbol(&count, launches, sizeof(count));
        EXPECT(count == 48);
    }

    // HIP_DYNAMIC_SHARED in a kernel and at namespace scope
    {
        std::vector<int> data(256);
        std::iota(data.begin(), data.end(), 0);
        hipLaunchKernelGGL(reverse_block, 4, 64, 64 * sizeof(int), 0, data.data());
        EXPECT(data[0] == 63 && data[63] == 0 && data[64] == 127 && data[255] == 192);

        float rotated[32];
        std::iota(rotated, rotated + 32, 0.0f);
        rotate_block<<<1, 32, 32 * sizeof(float)>>>(rotated);
        EXPECT(rotated[0] == 1.0f && rotated[30] == 31.0f && rotated[31] == 0.0f);
    }

    // Warp functions without masks, against what each lane must get
    {
        const int n = 64;
        std::vector<int> shfl(n), up(n), down(n), xor_(n), any(n), all(n);
        std::vector<unsigned long long> ballot(n);
        std::vector<unsigned> lane(n);
        hipLaunchKernelGGL(warp_functions, 1, n, 0, 0, shfl.data(), up.data(), down.data(), xor_.data(),
                           ballot.data(), any.data(), all.data(), lane.data());
        unsigned long long ballots[2] = {};
        for (int t = 0; t < n; ++t)
            if (t % 3 == 0)
                ballots[t / 32] |= 1ull << (t % 32);
        int wrong = 0;
        for (int t = 0; t < n; ++t) {
            const int warp = t / 32 * 32, l = t % 32;
            wrong += shfl[t] != (warp + 3) * 10;
            wrong += up[t] != (l % 16 < 2 ? t : t - 2);
            wrong += down[t] != (l % 8 + 4 >= 8 ? t : t + 4);
            wrong += xor_[t] != (t ^ 1);
            wrong += ballot[t] != ballots[t / 32];
            wrong += any[t] != (t / 32 == 0);       // lane 17 of the first warp
            wrong += all[t] != 1;
            wrong += lane[t] != unsigned(l);
        }
        EXPECT(wrong == 0);
    }

    // A reduction with __shfl_down
    {
        const int blocks = 8, threads = 256;
        std::vector<float> in(blocks * threads), out(blocks);
        for (size_t i = 0; i < in.size(); ++i)
            in[i] = float(i % 7);
        hipLaunchKernelGGL(reduce, blocks, threads, 0, 0, in.data(), out.data());
        int wrong = 0;
        for (int b = 0; b < blocks; ++b) {
            float expected = 0;
            for (int t = 0; t < threads; ++t)
                expected += in[b * threads + t];
            wrong += out[b] != expected;
        }
        EXPECT(wrong == 0);
    }

    // Cooperative groups
    {
        int in[64], out[4] = {};
        std::iota(in, in + 64, 1);
        hipLaunchKernelGGL(tile_sums, 1, 64, 0, 0, in, out);
        for (int i = 0; i < 4; ++i)
            EXPECT(out[i] == std::accumulate(in + 16 * i, in + 16 * (i + 1), 0));
    }

    // Atomics, clocks and bit manipulation
    {
        float sum = 0;
        int count = 0;
        hipLaunchKernelGGL(atomics, 16, 64, 0, 0, &sum, &count);
        EXPECT(sum == 0.75f * 1024 && count == 1024);

        long long ticks[2] = {-1, -1};
        hipLaunchKernelGGL(clocks, 1, 1, 0, 0, ticks);
        EXPECT(ticks[0] >= 0 && ticks[1] >= 0);

        EXPECT(__bitextract_u32(0xabcd1234u, 8, 4) == 0x2);
        EXPECT(__bitextract_u32(0xabcd1234u, 16, 16) == 0xabcd);
        EXPECT(__bitextract_u64(0x0123456789abcdefull, 32, 12) == 0x567);
        EXPECT(__bitinsert_u32(0xffffffffu, 0, 4, 8) == 0xfffff00fu);
        EXPECT(__bitinsert_u64(0, 0x3f, 60, 4) == 0xf000000000000000ull);
    }

    if (errors == 0)
        std::printf("hip_kernels: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
