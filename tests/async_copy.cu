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
// Asynchronous copies (libcu++'s <cuda/pipeline> and <cuda/barrier>, and
// cooperative_groups/memcpy_async.h): copies into shared memory that other
// threads read after waiting, with a thread's pipeline, a group, and a
// barrier of the block; barriers of some of the block's threads, with a
// completion function, over several phases and with threads that drop out;
// and <cuda/cmath> and cuda::std.
//

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda/cmath>
#include <cuda/pipeline>
#include <cuda/std/type_traits>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

static_assert(cuda::ceil_div(10, 3) == 4 && cuda::ceil_div(9u, 3) == 3 && cuda::ceil_div(0, 7) == 0);
static_assert(cuda::ceil_div(size_t(1) << 40, 1024) == size_t(1) << 30);
static_assert(cuda::round_up(10, 4) == 12 && cuda::round_up(12, 4) == 12 && cuda::round_down(11, 4) == 8);
static_assert(cuda::std::is_same_v<cuda::std::remove_const_t<const int>, int>);
static_assert(cuda::aligned_size_t<16>(64) == 64 && cuda::aligned_size_t<16>::align == 16);

constexpr int threads = 128;

//! Each thread copies four floats into shared memory through its pipeline,
//! and after the wait and a barrier, reads the ones another thread copied
__global__ void through_pipeline(const float *in, float *out)
{
    float *tile = cuda4cpu::dynamic_shared<float>();
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    pipe.producer_acquire();
    cuda::memcpy_async(&tile[threadIdx.x * 4], &in[threadIdx.x * 4], shape, pipe);
    pipe.producer_commit();
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();
    const unsigned other = (threadIdx.x + 37) % threads;
    for (int i = 0; i < 4; ++i)
        out[threadIdx.x * 4 + i] = tile[other * 4 + i];
}

//! The block copies its part, each tile of 32 threads its own, and after
//! cg::wait every thread reads what others copied
__global__ void through_groups(const float *in, float *out)
{
    __shared__ float block_tile[threads * 4];
    __shared__ float warp_tiles[threads];
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    cg::memcpy_async(block, block_tile, in, cuda::aligned_size_t<16>(sizeof(float) * threads * 4));
    const unsigned first = threadIdx.x / 32 * 32;
    cg::memcpy_async(warp, warp_tiles + first, size_t(32), in + first, size_t(32));
    cg::wait(warp);
    cg::wait(block);
    out[threadIdx.x] = block_tile[(threadIdx.x * 3 + 5) % (threads * 4)] + warp_tiles[first + (threadIdx.x + 7) % 32];
}

//! A barrier of the block, initialized by one thread: the copies it waits
//! for are done
__global__ void through_barrier(const float *in, float *out)
{
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    __shared__ float tile[threads];
    if (threadIdx.x == 0)
        init(&bar, blockDim.x);
    __syncthreads();
    cuda::memcpy_async(&tile[threadIdx.x], &in[threadIdx.x], sizeof(float), bar);
    bar.arrive_and_wait();
    out[threadIdx.x] = tile[(threadIdx.x + 1) % threads];
}

struct count_phases {
    int *phases = nullptr;
    void operator()() noexcept { ++*phases; }
};

//! The even threads meet at a barrier of their own, with a completion
//! function, three times; thread 2 drops out after the first
__global__ void even_threads(int *phases, int *seen)
{
    __shared__ cuda::barrier<cuda::thread_scope_block, count_phases> bar;
    __shared__ int counter;
    __shared__ int values[threads];
    if (threadIdx.x == 0) {
        counter = 0;
        init(&bar, blockDim.x / 2, count_phases{&counter});
    }
    __syncthreads();
    if (threadIdx.x % 2 == 0) {
        for (int round = 0; round < 3; ++round) {
            values[threadIdx.x] = round * 1000 + int(threadIdx.x);
            if (threadIdx.x == 2 && round == 1) {
                bar.arrive_and_drop();
                break;
            }
            bar.arrive_and_wait();
            // Every even thread has written this round's value
            const int other = (int(threadIdx.x) + 10) % threads;
            seen[threadIdx.x * 3 + round] = values[other] == round * 1000 + other || other == 2;
            bar.arrive_and_wait();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        *phases = counter;
}

int main()
{
    std::vector<float> in(threads * 4), out(threads * 4);
    for (int i = 0; i < threads * 4; ++i)
        in[i] = float(i);

    through_pipeline<<<2, threads, threads * 4 * sizeof(float)>>>(in.data(), out.data());
    unsigned wrong = 0;
    for (int t = 0; t < threads; ++t)
        for (int i = 0; i < 4; ++i)
            wrong += out[t * 4 + i] != in[(t + 37) % threads * 4 + i];
    EXPECT(wrong == 0);

    std::fill(out.begin(), out.end(), -1.0f);
    through_groups<<<3, threads>>>(in.data(), out.data());
    wrong = 0;
    for (int t = 0; t < threads; ++t)
        wrong += out[t] != in[(t * 3 + 5) % (threads * 4)] + in[t / 32 * 32 + (t + 7) % 32];
    EXPECT(wrong == 0);

    std::fill(out.begin(), out.end(), -1.0f);
    through_barrier<<<2, threads>>>(in.data(), out.data());
    wrong = 0;
    for (int t = 0; t < threads; ++t)
        wrong += out[t] != in[(t + 1) % threads];
    EXPECT(wrong == 0);

    // Phases: 2 per round, the first round with 64 threads and the next two
    // with 63, plus the arrival of thread 2 in round 1 (which completes no
    // phase on its own)
    int phases = 0;
    std::vector<int> seen(threads * 3, -1);
    even_threads<<<1, threads>>>(&phases, seen.data());
    EXPECT(phases == 6);
    wrong = 0;
    for (int t = 0; t < threads; t += 2)
        for (int round = 0; round < 3; ++round)
            wrong += t == 2 && round > 0 ? seen[t * 3 + round] != -1 : seen[t * 3 + round] != 1;
    EXPECT(wrong == 0);

    if (errors == 0)
        std::printf("async_copy: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
