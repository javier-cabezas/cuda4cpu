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


#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda4cpu.hpp"

using namespace cuda4cpu;

static unsigned errors = 0;

template <typename T>
static void check(const char *test, const std::vector<T> &out, const std::vector<T> &expected)
{
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] != expected[i]) {
            std::cout << test << ": [" << i << "] = " << out[i] << ", expected " << expected[i] << "\n";
            ++errors;
            return;
        }
    }
}

constexpr unsigned full = 0xffffffff;
constexpr int nresults = 9;

// Every shuffle and vote variant; out holds nresults values per thread
__global__ void variants(int *out, double *out_double)
{
    int t = threadIdx.x;
    int lane = t % warpSize;
    int v = t * 10;
    int *o = out + t * nresults;

    o[0] = __shfl_sync(full, v, (lane + 3) % 32);
    o[1] = __shfl_sync(full, v, 5, 8);
    o[2] = __shfl_up_sync(full, v, 2, 16);
    o[3] = __shfl_down_sync(full, v, 3);
    o[4] = __shfl_xor_sync(full, v, 1);
    o[5] = __shfl_xor_sync(full, v, 4, 4);
    o[6] = int(__ballot_sync(full, lane % 3 == 0));
    o[7] = __any_sync(full, lane == 7);
    o[8] = __all_sync(full, lane < 31);
    out_double[t] = __shfl_down_sync(full, t + 0.5, 1);
}

// Host model of the expected results. A source lane that does not exist (in
// a partial warp) does not take part, so the caller keeps its own value.
static void expected_variants(unsigned nthreads, std::vector<int> &out, std::vector<double> &out_double)
{
    out.assign(nthreads * nresults, 0);
    out_double.assign(nthreads, 0.0);
    for (unsigned t = 0; t < nthreads; ++t) {
        unsigned lane = t % 32, base = t - lane;
        unsigned lanes = std::min(32u, nthreads - base);
        auto value = [&](unsigned src) { return src < lanes ? int(base + src) * 10 : int(t) * 10; };
        int *o = &out[t * nresults];

        o[0] = value((lane + 3) % 32);
        o[1] = value((lane & ~7u) + 5);
        o[2] = (lane % 16) < 2 ? int(t) * 10 : value(lane - 2);
        o[3] = lane + 3 >= 32 ? int(t) * 10 : value(lane + 3);
        o[4] = value(lane ^ 1);
        o[5] = ((lane ^ 4) / 4 > lane / 4) ? int(t) * 10 : value(lane ^ 4);
        unsigned ballot = 0;
        for (unsigned l = 0; l < lanes; ++l) if (l % 3 == 0) ballot |= 1u << l;
        o[6] = int(ballot);
        o[7] = lanes > 7;
        o[8] = lanes <= 31;
        out_double[t] = (lane + 1 < lanes) ? (t + 1) + 0.5 : t + 0.5;
    }
}

// Lanes with lane % 4 == 3 return first; shuffles from them return the
// caller's own value, and __activemask leaves them out
__global__ void exited_lanes(int *out, unsigned *mask)
{
    int t = threadIdx.x;
    int lane = t % warpSize;
    if (lane % 4 == 3)
        return;

    __syncwarp();
    mask[t] = __activemask();
    out[t] = __shfl_down_sync(__activemask(), t, 1);
}

// Only some lanes run the shuffle; the others go straight to __syncthreads(),
// which takes them out of the warp operation (the __activemask() idiom)
__global__ void branch_then_barrier(int *out)
{
    int t = threadIdx.x;
    int lane = t % warpSize;
    int v = t;
    if (lane < 16) {
        unsigned mask = __activemask();
        v = __shfl_xor_sync(mask, v, 1);
    }
    __syncthreads();
    out[t] = v;
}

int main()
{
    // Two full warps, and a full warp plus a partial one
    for (unsigned nthreads : {64u, 40u}) {
        std::vector<int> out(nthreads * nresults), expected;
        std::vector<double> out_double(nthreads), expected_double;
        launch(variants, 3, nthreads).call(out.data(), out_double.data());
        expected_variants(nthreads, expected, expected_double);
        check("variants", out, expected);
        check("variants (double)", out_double, expected_double);
    }

    {
        const unsigned nthreads = 64;
        std::vector<int> out(nthreads, -1), expected(nthreads, -1);
        std::vector<unsigned> mask(nthreads, 0), expected_mask(nthreads, 0);
        for (unsigned t = 0; t < nthreads; ++t) {
            if (t % 4 == 3) continue;
            expected[t] = ((t + 1) % 4 == 3 || (t % 32) == 31) ? int(t) : int(t + 1);
            expected_mask[t] = 0x77777777;
        }
        launch(exited_lanes, 2, nthreads).call(out.data(), mask.data());
        check("exited_lanes", out, expected);
        check("exited_lanes (__activemask)", mask, expected_mask);
    }

    {
        const unsigned nthreads = 96;
        std::vector<int> out(nthreads, -1), expected(nthreads);
        for (unsigned t = 0; t < nthreads; ++t) expected[t] = (t % 32) < 16 ? int(t ^ 1) : int(t);
        launch(branch_then_barrier, 2, nthreads).call(out.data());
        check("branch_then_barrier", out, expected);
    }

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
