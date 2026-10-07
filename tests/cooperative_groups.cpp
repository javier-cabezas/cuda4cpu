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



// Cooperative groups: every group type and operation, in blocks with partial
// warps and partial tiles, checked against values computed on the host.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

namespace cg = cooperative_groups;

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

static void check(const char *test, const std::vector<int> &out, const std::vector<int> &expected,
                  int per_thread)
{
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] != expected[i]) {
            std::printf("%s: thread %zu, value %zu = %d, expected %d\n", test, i / per_thread,
                        i % per_thread, out[i], expected[i]);
            ++errors;
            return;
        }
    }
}

//
// thread_block and tiles of up to a warp, in blocks of 8 x 5 threads: the
// second warp has 8 threads
//

constexpr int small_results = 23;

__global__ void small_tiles(int *out)
{
    cg::thread_block cta = cg::this_thread_block();
    const int t = int(cta.thread_rank());
    const int n = int(cta.num_threads());
    int *o = out + (blockIdx.x * n + t) * small_results;

    cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(cta);
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    auto tile16 = cg::tiled_partition<16>(tile32);

    o[0] = t;
    o[1] = n;
    o[2] = int(tile4.thread_rank());
    o[3] = int(tile4.meta_group_rank());
    o[4] = int(tile4.meta_group_size());
    o[5] = tile4.shfl(t * 10, 2);
    o[6] = tile4.shfl_down(t, 1);
    o[7] = tile4.shfl_up(t, 1);
    o[8] = tile4.shfl_xor(t, 1);
    o[9] = int(tile4.ballot(t % 3 == 0));
    o[10] = tile4.any(t == 5);
    o[11] = tile4.all(t < 36);
    o[12] = cg::reduce(tile4, t, cg::plus<int>());
    o[13] = cg::reduce(tile32, t, cg::greater<int>());
    o[14] = cg::reduce(tile32, t, cg::plus<int>());
    o[15] = cg::inclusive_scan(tile4, t);
    o[16] = cg::exclusive_scan(tile4, t);
    o[17] = int(tile16.meta_group_rank());
    o[18] = int(tile16.thread_rank());
    o[19] = int(tile32.match_any(t % 4));

    __shared__ int buf[40];
    buf[t] = t;
    cta.sync();
    o[20] = buf[n - 1 - t];
    cg::sync(cta);
    o[21] = int(cta.group_index().x);
    o[22] = int(cta.thread_index().y);
}

static void test_small_tiles()
{
    const dim3 block(8, 5);
    const int n = 40, blocks = 2;
    std::vector<int> out(blocks * n * small_results, -1), expected;
    cuda4cpu::launch(small_tiles, blocks, block).call(out.data());

    for (int b = 0; b < blocks; ++b) {
        for (int t = 0; t < n; ++t) {
            const int base4 = t / 4 * 4, warp = t / 32 * 32, warp_end = std::min(warp + 32, n);
            int ballot = 0, sum4 = 0, sum32 = 0, inclusive = 0, match = 0;
            for (int r = 0; r < 4; ++r) {
                ballot |= ((base4 + r) % 3 == 0) << r;
                sum4 += base4 + r;
                if (base4 + r <= t)
                    inclusive += base4 + r;
            }
            for (int u = warp; u < warp_end; ++u) {
                sum32 += u;
                if (u % 4 == t % 4)
                    match |= 1 << (u - warp);
            }
            const int v[small_results] = {
                t, n, t % 4, t / 4, n / 4, (base4 + 2) * 10,
                t % 4 < 3 ? t + 1 : t, t % 4 > 0 ? t - 1 : t, t ^ 1, ballot,
                base4 == 4, base4 + 3 < 36, sum4, warp_end - 1, sum32, inclusive, inclusive - t,
                (t % 32) / 16, t % 16, match, n - 1 - t, b, t / 8};
            expected.insert(expected.end(), v, v + small_results);
        }
    }
    check("small tiles", out, expected, small_results);
}

//
// Tiles of several warps, in blocks of 256 threads and of 200 threads (the
// last tile of 64 has 8 threads, the last tile of 128 has 72)
//

constexpr int large_results = 7;

__global__ void large_tiles(int *out)
{
    cg::thread_block cta = cg::this_thread_block();
    const int t = int(cta.thread_rank());
    const int n = int(cta.num_threads());
    int *o = out + t * large_results;

    auto tile64 = cg::tiled_partition<64>(cta);
    auto tile128 = cg::tiled_partition<128>(cta);
    o[0] = cg::reduce(tile64, t, cg::plus<int>());
    o[1] = int(tile64.meta_group_rank());
    o[2] = int(tile64.thread_rank());
    o[3] = cg::reduce(tile128, t, cg::greater<int>());

    // Only even tiles of 64 exchange values, three times: their barriers
    // don't wait for the other tiles
    __shared__ int buf[256];
    int v = t;
    if (tile64.meta_group_rank() % 2 == 0) {
        const int base = t / 64 * 64, extent = std::min(64, n - base);
        for (int round = 0; round < 3; ++round) {
            buf[t] = v;
            tile64.sync();
            v = buf[base + (t - base + 1) % extent];
            tile64.sync();
        }
    }
    o[4] = v;
    cta.sync();

    // The same through a thread_group from the dynamic tiled_partition
    cg::thread_group g = cg::tiled_partition(cta, 64);
    o[5] = int(g.thread_rank());
    buf[t] = t;
    g.sync();
    o[6] = (t ^ 32) < n ? buf[t ^ 32] : -1;
}

static void test_large_tiles()
{
    for (int n : {256, 200}) {
        std::vector<int> out(n * large_results, -1), expected;
        cuda4cpu::launch(large_tiles, 1, n).call(out.data());
        for (int t = 0; t < n; ++t) {
            const int base64 = t / 64 * 64, end64 = std::min(base64 + 64, n);
            const int end128 = std::min(t / 128 * 128 + 128, n);
            int sum = 0;
            for (int u = base64; u < end64; ++u)
                sum += u;
            const int extent = end64 - base64;
            const int rotated = (t / 64) % 2 == 0 ? base64 + (t - base64 + 3) % extent : t;
            const int v[large_results] = {sum, t / 64, t % 64, end128 - 1, rotated, t % 64,
                                          (t ^ 32) < n ? t ^ 32 : -1};
            expected.insert(expected.end(), v, v + large_results);
        }
        check(n == 256 ? "large tiles (256 threads)" : "large tiles (200 threads)", out, expected,
              large_results);
    }
}

//
// Coalesced groups: the lanes that take a branch together
//

constexpr int coalesced_results = 10;

__global__ void coalesced(int *out, int *counter, int *slots)
{
    const int t = int(threadIdx.x);
    int *o = out + t * coalesced_results;
    std::fill(o, o + coalesced_results, -1);

    if (t % 3 == 0) {
        cg::coalesced_group g = cg::coalesced_threads();
        o[0] = int(g.num_threads());
        o[1] = int(g.thread_rank());
        o[2] = g.shfl(t, 0);
        o[3] = cg::reduce(g, t, cg::plus<int>());
        o[4] = int(g.ballot(t % 2 == 0));
        o[5] = g.shfl_down(t, 1);
        o[6] = cg::inclusive_scan(g, 1);

        // Warp-aggregated atomic increment
        int first = 0;
        if (g.thread_rank() == 0)
            first = atomicAdd(counter, int(g.num_threads()));
        slots[g.shfl(first, 0) + int(g.thread_rank())] = t;

        cg::coalesced_group quad = cg::tiled_partition(g, 4);
        o[7] = int(quad.thread_rank());
    } else {
        __syncwarp(__activemask());
    }

    auto tile32 = cg::tiled_partition<32>(cg::this_thread_block());
    cg::coalesced_group parity = cg::binary_partition(tile32, t & 1);
    o[8] = cg::reduce(parity, t, cg::plus<int>());
    cg::coalesced_group label = cg::labeled_partition(tile32, t % 5);
    o[9] = int(label.num_threads());
}

static void test_coalesced()
{
    const int n = 64;
    std::vector<int> out(n * coalesced_results, -2), expected, slots(n, -1);
    int counter = 0;
    cuda4cpu::launch(coalesced, 1, n).call(out.data(), &counter, slots.data());

    for (int t = 0; t < n; ++t) {
        const int warp = t / 32 * 32;
        std::vector<int> members;
        for (int u = warp; u < warp + 32; ++u) {
            if (u % 3 == 0)
                members.push_back(u);
        }
        int parity = 0, label = 0;
        for (int u = warp; u < warp + 32; ++u) {
            parity += (u & 1) == (t & 1) ? u : 0;
            label += u % 5 == t % 5;
        }
        std::vector<int> v(coalesced_results, -1);
        if (t % 3 == 0) {
            const int rank = int(std::find(members.begin(), members.end(), t) - members.begin());
            int sum = 0, ballot = 0;
            for (size_t r = 0; r < members.size(); ++r) {
                sum += members[r];
                ballot |= (members[r] % 2 == 0) << r;
            }
            v = {int(members.size()), rank, members[0], sum, ballot,
                 rank + 1 < int(members.size()) ? members[rank + 1] : t, rank + 1, rank % 4, -1, -1};
        }
        v[8] = parity;
        v[9] = label;
        expected.insert(expected.end(), v.begin(), v.end());
    }
    check("coalesced groups", out, expected, coalesced_results);

    // Every thread of the branch got its own slot
    std::vector<int> got(slots.begin(), slots.begin() + counter), want;
    for (int t = 0; t < n; t += 3)
        want.push_back(t);
    std::sort(got.begin(), got.end());
    EXPECT(got == want);
}

//
// __match_any_sync and __match_all_sync
//

__global__ void match(unsigned *out)
{
    const unsigned t = threadIdx.x;
    int pred_same = -1, pred_different = -1;
    out[t * 4 + 0] = __match_any_sync(0xffffffff, t / 8);
    out[t * 4 + 1] = __match_all_sync(0xffffffff, 42.0, &pred_same);
    out[t * 4 + 2] = __match_all_sync(0xffffffff, t, &pred_different);
    out[t * 4 + 3] = unsigned(pred_same * 2 + pred_different);
}

static void test_match()
{
    const unsigned n = 40;
    std::vector<unsigned> out(n * 4);
    cuda4cpu::launch(match, 1, n).call(out.data());
    for (unsigned t = 0; t < n; ++t) {
        const unsigned live = t < 32 ? 0xffffffffu : 0xffu;
        EXPECT(out[t * 4 + 0] == (0xffu << (t % 32 / 8 * 8) & live));
        EXPECT(out[t * 4 + 1] == live);
        EXPECT(out[t * 4 + 2] == (t >= 32 && n - 32 == 1 ? live : 0u));
        EXPECT(out[t * 4 + 3] == 2);
    }
}

//
// grid_group: ranks in any launch, and grid.sync() in a cooperative launch
//

__global__ void grid_ranks(unsigned long long *out)
{
    cg::grid_group grid = cg::this_grid();
    const unsigned long long t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t * 4 + 0] = grid.thread_rank();
    out[t * 4 + 1] = grid.num_threads();
    out[t * 4 + 2] = grid.block_rank() * 1000 + grid.num_blocks();
    out[t * 4 + 3] = grid.is_valid();
}

constexpr int rounds = 5;

// Each round, every thread reads the value that the next thread of the grid
// wrote before the grid barrier
__global__ void grid_rounds(int *data, int *out)
{
    cg::grid_group grid = cg::this_grid();
    const int rank = int(grid.thread_rank()), n = int(grid.num_threads());
    int v = rank;
    for (int round = 0; round < rounds; ++round) {
        data[rank] = v;
        grid.sync();
        v = data[(rank + 1) % n] + 1;
        __syncthreads();
        grid.sync();
    }
    out[rank] = grid.is_valid() ? v : -1;
}

static void test_grid()
{
    {
        const int blocks = 3, threads = 50;
        std::vector<unsigned long long> out(blocks * threads * 4);
        cuda4cpu::launch(grid_ranks, blocks, threads).call(out.data());
        for (int t = 0; t < blocks * threads; ++t) {
            EXPECT(out[t * 4 + 0] == unsigned(t));
            EXPECT(out[t * 4 + 1] == unsigned(blocks * threads));
            EXPECT(out[t * 4 + 2] == unsigned(t / threads * 1000 + blocks));
            EXPECT(out[t * 4 + 3] == 0);
        }
    }

    int attribute = 0, sms = 0;
    EXPECT(cudaDeviceGetAttribute(&attribute, cudaDevAttrCooperativeLaunch, 0) == cudaSuccess && attribute == 1);
    EXPECT(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess && sms > 0);

    const int blocks = std::min(sms, 6), threads = 64, n = blocks * threads;
    std::vector<int> data(n), out(n, -2);
    int *data_ptr = data.data(), *out_ptr = out.data();
    void *args[] = {&data_ptr, &out_ptr};
    EXPECT(cudaLaunchCooperativeKernel(grid_rounds, blocks, threads, args) == cudaSuccess);
    for (int rank = 0; rank < n; ++rank)
        EXPECT(out[rank] == (rank + rounds) % n + rounds);

    // A grid larger than the device can run at once
    EXPECT(cudaLaunchCooperativeKernel(grid_rounds, sms + 1, threads, args) == cudaErrorCooperativeLaunchTooLarge);
    EXPECT(cudaGetLastError() == cudaErrorCooperativeLaunchTooLarge);

    // Occupancy: one block of any size per multiprocessor
    int per_sm = 0, min_grid = 0, block_size = 0;
    EXPECT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, grid_rounds, 256, 0) == cudaSuccess && per_sm == 1);
    EXPECT(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, grid_rounds) == cudaSuccess);
    EXPECT(min_grid == sms && block_size == 1024);
    EXPECT(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, grid_rounds, 0, 128) == cudaSuccess);
    EXPECT(block_size == 128);
}

int main()
{
    test_small_tiles();
    test_large_tiles();
    test_coalesced();
    test_match();
    test_grid();
    std::printf("cooperative_groups: %s\n", errors == 0 ? "PASS" : "FAIL");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
