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
static void expect(const char *what, T got, T expected)
{
    if (got != expected) {
        std::cout << what << ": got " << got << ", expected " << expected << "\n";
        ++errors;
    }
}

struct counters {
    int add_int;
    unsigned add_uint;
    unsigned long long add_ull;
    float add_float;
    double add_double;
    int sub_int;
    int min_int;
    int max_int;
    unsigned long long max_ull;
    unsigned inc;
    unsigned dec;
    unsigned and_bits;
    unsigned or_bits;
    unsigned xor_bits;
    unsigned cas_winners;
    int cas_flag;
    unsigned exch;
};

// Every thread of every block updates the same counters, so blocks running in
// parallel on different cores race on them
__global__ void update(counters *c)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(&c->add_int, 1);
    atomicAdd(&c->add_uint, 2);         // int literal, unsigned overload
    atomicAdd(&c->add_ull, 3ull);
    atomicAdd(&c->add_float, 0.5f);
    atomicAdd(&c->add_double, 0.25);
    atomicSub(&c->sub_int, 1);
    atomicMin(&c->min_int, -int(t));
    atomicMax(&c->max_int, int(t));
    atomicMax(&c->max_ull, (unsigned long long)t << 32);
    atomicInc(&c->inc, 9);              // counts modulo 10
    atomicDec(&c->dec, 6);              // counts down modulo 7
    atomicAnd(&c->and_bits, ~(1u << (t % 32)));
    atomicOr(&c->or_bits, 1u << (t % 32));
    atomicXor(&c->xor_bits, 1u << (t % 2));
    if (atomicCAS(&c->cas_flag, 0, 1) == 0)
        atomicAdd(&c->cas_winners, 1);
    atomicExch(&c->exch, t);
    __threadfence();
}

// Shared-memory atomics, then one global atomic per block
__global__ void block_histogram(unsigned *hist)
{
    __shared__ unsigned bins[4];
    if (threadIdx.x < 4)
        bins[threadIdx.x] = 0;
    __syncthreads();
    atomicAdd(&bins[threadIdx.x % 4], 1);
    __syncthreads();
    if (threadIdx.x < 4)
        atomicAdd(&hist[threadIdx.x], bins[threadIdx.x]);
}

int main()
{
    const unsigned blocks = 256, threads = 128, n = blocks * threads;

    counters c{};
    c.min_int = 0;
    c.max_int = -1;
    c.and_bits = ~0u;
    launch(update, blocks, threads).call(&c);

    expect("atomicAdd(int)", c.add_int, int(n));
    expect("atomicAdd(unsigned)", c.add_uint, 2 * n);
    expect("atomicAdd(unsigned long long)", c.add_ull, 3ull * n);
    expect("atomicAdd(float)", c.add_float, 0.5f * n);
    expect("atomicAdd(double)", c.add_double, 0.25 * n);
    expect("atomicSub", c.sub_int, -int(n));
    expect("atomicMin", c.min_int, -int(n - 1));
    expect("atomicMax(int)", c.max_int, int(n - 1));
    expect("atomicMax(unsigned long long)", c.max_ull, (unsigned long long)(n - 1) << 32);
    expect("atomicInc", c.inc, n % 10);
    expect("atomicDec", c.dec, (7 - n % 7) % 7);
    expect("atomicAnd", c.and_bits, 0u);
    expect("atomicOr", c.or_bits, ~0u);
    expect("atomicXor", c.xor_bits, 0u);    // each bit flipped n / 2 times, an even number
    expect("atomicCAS winners", c.cas_winners, 1u);
    expect("atomicExch", c.exch < n, true);

    std::vector<unsigned> hist(4, 0);
    launch(block_histogram, blocks, threads).call(hist.data());
    for (unsigned b = 0; b < 4; ++b) expect("shared-memory atomics", hist[b], n / 4);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
