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

// cuRAND (curand.h, curand_kernel.h) against references and against itself:
// - XORWOW's raw values and uniforms are CUDA's: the expected values are
//   mojolearn's reference streams (Apache-2.0, isolation_forest/checks in
//   github.com/mojolearn/mojolearn), whose jump matrices equal cuRAND's tables
// - Philox4x32-10's rounds give Random123's known answers (kat_vectors)
// - Sobol' points are Joe and Kuo's: the first 10 in 3 dimensions, as their
//   sobol.cc prints them, and the vectors of their last dimension
// - skipping ahead equals stepping, the host API's results are the device
//   API's in cuRAND's documented orderings, the distributions have their
//   moments, Sobol' points are equidistributed, errors are cuRAND's, and a
//   kernel estimates pi with per-thread states

#include <curand.h>
#include <curand_kernel.h>
#include <hiprand/hiprand.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

#define EXPECT(cond)                                                                 \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond);          \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

static unsigned bits_of(float f)
{
    unsigned b;
    std::memcpy(&b, &f, 4);
    return b;
}

static void xorwow_references()
{
    struct reference {
        unsigned long long seed, subsequence, offset;
        unsigned first[3], last;            // values 0, 1, 2 and 1023
        unsigned uniform_bits;              // of value 0
    };
    const reference references[] = {
        {0, 0, 0, {3179217846u, 1883133293u, 2220552389u}, 3436001202u, 0x3f3d7f04},
        {42, 0, 0, {300737663u, 2363150160u, 799977142u}, 628482242u, 0x3d8f6724},
        {42, 1, 0, {687819970u, 2592620328u, 2975550928u}, 4118673307u, 0x3e23fd33},
        {42, 7, 0, {467091836u, 1666773505u, 3985452410u}, 3054373917u, 0x3ddeba0c},
        {16045690984503111693ull, 3, 0, {4161790554u, 4271150030u, 583588182u}, 4148310150u, 0x3f780fe2},
        {123456789, 255, 0, {1949855887u, 979880767u, 3811427084u}, 1504334012u, 0x3ee870e1},
    };
    for (const reference &r : references) {
        curandState state;
        curand_init(r.seed, r.subsequence, r.offset, &state);
        unsigned values[1024];
        for (unsigned &v : values)
            v = curand(&state);
        EXPECT(values[0] == r.first[0] && values[1] == r.first[1] && values[2] == r.first[2]);
        EXPECT(values[1023] == r.last);
        curand_init(r.seed, r.subsequence, r.offset, &state);
        EXPECT(bits_of(curand_uniform(&state)) == r.uniform_bits);
    }
    // With offsets: the first value
    const struct { unsigned long long seed, subsequence, offset; unsigned value; } offsets[] = {
        {0, 0, 1, 1883133293u}, {42, 0, 3, 2317951200u}, {42, 1, 4, 1172961726u},
        {42, 7, 64, 1617146888u}, {16045690984503111693ull, 3, 4095, 722283759u}, {123456789, 255, 1, 979880767u},
    };
    for (const auto &o : offsets) {
        curandState state;
        curand_init(o.seed, o.subsequence, o.offset, &state);
        EXPECT(curand(&state) == o.value);
    }
}

static void skipping_equals_stepping()
{
    // Values ahead, by every means
    curandState a, b;
    curand_init(7, 3, 0, &a);
    curand_init(7, 3, 0, &b);
    for (int i = 0; i < 1000; ++i)
        curand(&a);
    skipahead(1000ull, &b);
    EXPECT(curand(&a) == curand(&b));
    // Subsequences: one from 2^40 + 5 is five from 2^40
    curand_init(7, (1ull << 40) + 5, 9, &a);
    curand_init(7, 1ull << 40, 0, &b);
    skipahead_sequence(5ull, &b);
    skipahead(9ull, &b);
    EXPECT(curand(&a) == curand(&b) && curand(&a) == curand(&b));

    curandStatePhilox4_32_10_t p, q;
    curand_init(99, 12, 0, &p);
    curand_init(99, 12, 0, &q);
    for (int i = 0; i < 4097; ++i)
        curand(&p);
    skipahead(4097ull, &q);
    EXPECT(curand(&p) == curand(&q));
    curand_init(99, 12, 3, &p);
    curand_init(99, 12, 0, &q);
    for (int i = 0; i < 3; ++i)
        curand(&q);
    const uint4 four = curand4(&p);                  // straddles two counters
    EXPECT(four.x == curand(&q) && four.y == curand(&q) && four.z == curand(&q) && four.w == curand(&q));
    EXPECT(curand(&p) == curand(&q));
    // A subsequence is the counter's upper half: 2^64 values
    curand_init(99, 1, 0, &p);
    curand_init(99, 0, 0, &q);
    skipahead_sequence(1ull, &q);
    EXPECT(curand(&p) == curand(&q));

    curandStateMRG32k3a_t m, n;
    curand_init(5, 2, 0, &m);
    curand_init(5, 2, 0, &n);
    for (int i = 0; i < 777; ++i)
        curand(&m);
    skipahead(777ull, &n);
    EXPECT(curand(&m) == curand(&n));
    curand_init(5, 3, 0, &m);
    curand_init(5, 2, 0, &n);
    skipahead_sequence(1ull, &n);
    EXPECT(curand(&m) == curand(&n));
}

static void philox_known_answers()
{
    using cuda4cpu::detail::rand::philox4x32_10;
    uint4 r = philox4x32_10(make_uint4(0, 0, 0, 0), make_uint2(0, 0));
    EXPECT(r.x == 0x6627e8d5 && r.y == 0xe169c58d && r.z == 0xbc57ac4c && r.w == 0x9b00dbd8);
    r = philox4x32_10(make_uint4(~0u, ~0u, ~0u, ~0u), make_uint2(~0u, ~0u));
    EXPECT(r.x == 0x408f276d && r.y == 0x41c83b0e && r.z == 0xa20bc7c6 && r.w == 0x6d5451fd);
    r = philox4x32_10(make_uint4(0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), make_uint2(0xa4093822, 0x299f31d0));
    EXPECT(r.x == 0xd16cfe09 && r.y == 0x94fdcceb && r.z == 0x5001e420 && r.w == 0x24126ea1);
    // curand_init(0, 0, 0) starts at the zero counter and key
    curandStatePhilox4_32_10_t state;
    curand_init(0, 0, 0, &state);
    EXPECT(curand(&state) == 0x6627e8d5 && curand(&state) == 0xe169c58d);
}

static void sobol_points()
{
    curandDirectionVectors32_t *vectors;
    EXPECT(curandGetDirectionVectors32(&vectors, CURAND_DIRECTION_VECTORS_32_JOEKUO6) == CURAND_STATUS_SUCCESS);
    const double expected[3][10] = {
        {0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125, 0.1875, 0.6875},
        {0, 0.5, 0.25, 0.75, 0.375, 0.875, 0.125, 0.625, 0.3125, 0.8125},
        {0, 0.5, 0.25, 0.75, 0.625, 0.125, 0.875, 0.375, 0.9375, 0.4375},
    };
    for (int d = 0; d < 3; ++d) {
        curandStateSobol32_t state;
        curand_init(vectors[d], 0, &state);
        for (int i = 0; i < 10; ++i)
            EXPECT(curand(&state) * 0x1p-32 == expected[d][i]);
    }
    // The last dimension (Joe and Kuo's line 20000: s 18, m 1 3 1 15 ... 253425)
    EXPECT(vectors[19999][0] == 1u << 31 && vectors[19999][1] == 3u << 30 && vectors[19999][3] == 15u << 28);
    EXPECT(vectors[19999][17] == 253425u << 14);
    // 64-bit vectors are the same numbers, wider
    curandDirectionVectors64_t *wide;
    EXPECT(curandGetDirectionVectors64(&wide, CURAND_DIRECTION_VECTORS_64_JOEKUO6) == CURAND_STATUS_SUCCESS);
    for (int d : {0, 1, 2, 777, 19999})
        for (int k = 0; k < 32; ++k)
            EXPECT((unsigned)(wide[d][k] >> 32) == vectors[d][k]);
    // Every dyadic interval of 1/1024 holds one of the first 1024 points of a
    // dimension, scrambled or not; skipping ahead is stepping
    curandDirectionVectors32_t *scrambled;
    unsigned int *constants;
    curandGetDirectionVectors32(&scrambled, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    curandGetScrambleConstants32(&constants);
    for (int d : {0, 5, 1000, 19999}) {
        curandStateSobol32_t plain;
        curandStateScrambledSobol32_t mixed;
        curand_init(vectors[d], 0, &plain);
        curand_init(scrambled[d], constants[d], 0, &mixed);
        std::vector<int> a(1024), b(1024);
        for (int i = 0; i < 1024; ++i) {
            ++a[curand(&plain) >> 22];
            ++b[curand(&mixed) >> 22];
        }
        for (int i = 0; i < 1024; ++i)
            EXPECT(a[i] == 1 && b[i] == 1);
        curandStateSobol32_t later;
        curand_init(vectors[d], 1000, &later);
        skipahead(24u, &later);
        EXPECT(curand(&later) == curand(&plain));
    }
}

//! Mean and variance of n values
template <typename F>
static void moments(size_t n, F next, double &mean, double &variance)
{
    double sum = 0, squares = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = next();
        sum += x;
        squares += x * x;
    }
    mean = sum / double(n);
    variance = squares / double(n) - mean * mean;
}

template <typename S>
static void distributions(const char *name)
{
    S state;
    curand_init(2024, 1, 0, &state);
    const size_t n = 400000;
    double mean, variance;
    float smallest = 1, largest = 0;
    moments(n, [&] { const float u = curand_uniform(&state); smallest = std::min(smallest, u); largest = std::max(largest, u); return u; },
            mean, variance);
    EXPECT(smallest > 0 && largest <= 1);
    EXPECT(std::fabs(mean - 0.5) < 0.003 && std::fabs(variance - 1.0 / 12) < 0.002);
    moments(n, [&] { return curand_uniform_double(&state); }, mean, variance);
    EXPECT(std::fabs(mean - 0.5) < 0.003 && std::fabs(variance - 1.0 / 12) < 0.002);
    moments(n, [&] { return curand_normal(&state); }, mean, variance);
    EXPECT(std::fabs(mean) < 0.01 && std::fabs(variance - 1) < 0.01);
    moments(n, [&] { return curand_normal_double(&state); }, mean, variance);
    EXPECT(std::fabs(mean) < 0.01 && std::fabs(variance - 1) < 0.01);
    // A log-normal of (0, 0.5): mean e^0.125
    moments(n, [&] { return curand_log_normal(&state, 0.0f, 0.5f); }, mean, variance);
    EXPECT(std::fabs(mean - std::exp(0.125)) < 0.01);
    for (double lambda : {0.5, 7.0, 100.0, 5000.0}) {
        moments(n / 4, [&] { return double(curand_poisson(&state, lambda)); }, mean, variance);
        if (!(std::fabs(mean - lambda) < 0.02 * lambda + 0.02 && std::fabs(variance - lambda) < 0.05 * lambda + 0.05))
            std::printf("%s: poisson(%g) has mean %g, variance %g\n", name, lambda, mean, variance);
        EXPECT(std::fabs(mean - lambda) < 0.02 * lambda + 0.02 && std::fabs(variance - lambda) < 0.05 * lambda + 0.05);
    }
    curandDiscreteDistribution_t poisson;
    EXPECT(curandCreatePoissonDistribution(30, &poisson) == CURAND_STATUS_SUCCESS);
    moments(n / 4, [&] { return double(curand_discrete(&state, poisson)); }, mean, variance);
    EXPECT(std::fabs(mean - 30) < 0.3 && std::fabs(variance - 30) < 1.5);
    curandDestroyDistribution(poisson);
}

//! The ends of ranges: the last Sobol' point, uniforms below 1, large means,
//! seeds that zero a component, offsets near 2^64
static void edges()
{
    curandDirectionVectors32_t *vectors;
    curandGetDirectionVectors32(&vectors, CURAND_DIRECTION_VECTORS_32_JOEKUO6);
    curandStateSobol32_t last;
    curand_init(vectors[3], 0xffffffffu, &last);
    curand(&last);
    EXPECT(curand(&last) == 0);                      // the sequence starts again
    curandDirectionVectors64_t *wide;
    curandGetDirectionVectors64(&wide, CURAND_DIRECTION_VECTORS_64_JOEKUO6);
    curandStateSobol64_t last64;
    curand_init(wide[3], ~0ull, &last64);
    curand(&last64);
    EXPECT(curand(&last64) == 0);

    using cuda4cpu::detail::rand::uniform_double;
    EXPECT(uniform_double(~0ull) < 1 && uniform_double(~0u, ~0u) < 1 && uniform_double(0ull) > 0);
    EXPECT(std::isfinite(cuda4cpu::detail::rand::normal_quantile(uniform_double(~0ull))));
    EXPECT(std::isfinite(cuda4cpu::detail::rand::normal_quantile(uniform_double(0ull))));

    // Quasirandom Poisson values of any mean: by inversion from the mode
    for (double lambda : {0.3, 12.0, 800.0, 5000.0, 2e6}) {
        curandStateSobol64_t q;
        curand_init(wide[1], 0ull, &q);
        double sum = 0;
        const int n = 4096;
        for (int i = 0; i < n; ++i)
            sum += curand_poisson(&q, lambda);
        EXPECT(std::fabs(sum / n - lambda) < 0.01 * lambda + 0.01);
        curand_init(wide[2], 0ull, &q);
        for (int i = 0; i < 64; ++i)
            EXPECT(std::isfinite(curand_normal_double(&q)) && std::isfinite(curand_log_normal(&q, 0.0f, 1.0f)));
    }

    // Seeds whose terms are all zero in a component still give a sequence
    for (unsigned long long seed : {0xAAAAAAAA55555555ull, 18446585346252035941ull}) {
        curandStateMRG32k3a_t m;
        curand_init(seed, 0, 0, &m);
        const unsigned a = curand(&m), b = curand(&m), c = curand(&m);
        EXPECT(!(a == b && b == c));
    }

    curandGenerator_t g;
    curandCreateGenerator(&g, CURAND_RNG_PSEUDO_XORWOW);
    std::vector<unsigned> k(8);
    EXPECT(curandGeneratePoisson(g, k.data(), 8, 5e9) == CURAND_STATUS_OUT_OF_RANGE);
    EXPECT(curandGeneratePoisson(g, k.data(), 8, 1e8) == CURAND_STATUS_SUCCESS);
    // 17 results before 2^64, and some after: result m is still lane m % 4096's
    EXPECT(curandSetGeneratorOffset(g, ~0ull - 16) == CURAND_STATUS_SUCCESS);
    std::vector<unsigned> late(40);
    EXPECT(curandGenerate(g, late.data(), late.size()) == CURAND_STATUS_SUCCESS);
    for (int i : {0, 16, 17, 39}) {
        const unsigned __int128 m = (unsigned __int128)(~0ull - 16) + i;
        curandState s;
        curand_init(0, (unsigned long long)(m % 4096), (unsigned long long)(m / 4096), &s);
        EXPECT(late[i] == curand(&s));
    }
    curandDestroyGenerator(g);
    // cuRAND can't skip ahead in Mersenne twisters
    curandCreateGenerator(&g, CURAND_RNG_PSEUDO_MT19937);
    EXPECT(curandSetGeneratorOffset(g, 10) == CURAND_STATUS_TYPE_ERROR);
    curandDestroyGenerator(g);

    // hipRAND's C++ API, with a 64-bit engine
    hiprand_cpp::sobol64 engine;
    hiprand_cpp::uniform_int_distribution<unsigned long long> integers;
    std::vector<unsigned long long> points(4);
    integers(engine, points.data(), points.size());
    EXPECT(points[0] == 0 && points[1] == 1ull << 63 && points[2] == 3ull << 62 && points[3] == 1ull << 62);
    hiprand_cpp::default_random_engine pseudo(1234);
    hiprand_cpp::normal_distribution<double> normal(1.0, 2.0);
    std::vector<double> values(10000);
    normal(pseudo, values.data(), values.size());
    double sum = 0;
    for (double v : values)
        sum += v;
    EXPECT(std::fabs(sum / values.size() - 1) < 0.1);
}

//! The host API's results are the device API's, in cuRAND's orderings
static void host_orderings()
{
    curandGenerator_t g;
    // XORWOW: result m is value m / 4096 of subsequence m % 4096
    EXPECT(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS);
    EXPECT(curandSetPseudoRandomGeneratorSeed(g, 1234) == CURAND_STATUS_SUCCESS);
    std::vector<unsigned> first(10000), second(3000);
    EXPECT(curandGenerate(g, first.data(), first.size()) == CURAND_STATUS_SUCCESS);
    EXPECT(curandGenerate(g, second.data(), second.size()) == CURAND_STATUS_SUCCESS);   // continues
    auto xorwow_at = [](unsigned long long seed, unsigned long long m) {
        curandState s;
        curand_init(seed, m % 4096, m / 4096, &s);
        return curand(&s);
    };
    for (unsigned long long m : {0ull, 1ull, 4095ull, 4096ull, 9999ull})
        EXPECT(first[m] == xorwow_at(1234, m));
    for (unsigned long long m : {0ull, 1ull, 2999ull})
        EXPECT(second[m] == xorwow_at(1234, 10000 + m));
    // An offset starts later in the same results; uniforms are curand_uniform's
    EXPECT(curandSetGeneratorOffset(g, 4100) == CURAND_STATUS_SUCCESS);
    std::vector<float> u(5000);
    EXPECT(curandGenerateUniform(g, u.data(), u.size()) == CURAND_STATUS_SUCCESS);
    for (unsigned long long m : {0ull, 7ull, 4999ull}) {
        curandState s;
        curand_init(1234, (4100 + m) % 4096, (4100 + m) / 4096, &s);
        EXPECT(bits_of(u[m]) == bits_of(curand_uniform(&s)));
    }
    curandDestroyGenerator(g);

    // Philox: four values of a sequence, then four of the next, of 65536
    EXPECT(curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_PHILOX4_32_10) == CURAND_STATUS_SUCCESS);
    curandSetPseudoRandomGeneratorSeed(g, 77);
    std::vector<unsigned> p(600000);
    EXPECT(curandGenerate(g, p.data(), p.size()) == CURAND_STATUS_SUCCESS);
    for (unsigned long long m : {0ull, 3ull, 4ull, 262143ull, 262144ull, 262149ull, 599999ull}) {
        curandStatePhilox4_32_10_t s;
        curand_init(77, (m / 4) % 65536, m / 262144 * 4 + m % 4, &s);
        EXPECT(p[m] == curand(&s));
    }
    curandDestroyGenerator(g);

    // Every generator's distributions, by the host API
    for (curandRngType_t type : {CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MTGP32,
                                 CURAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_PHILOX4_32_10}) {
        EXPECT(curandCreateGenerator(&g, type) == CURAND_STATUS_SUCCESS);
        const size_t n = 200000;
        std::vector<float> f(n);
        std::vector<double> d(n);
        std::vector<unsigned> k(n);
        auto mean_of = [n](const auto &v) { double s = 0; for (auto x : v) s += double(x); return s / double(n); };
        EXPECT(curandGenerateUniform(g, f.data(), n) == CURAND_STATUS_SUCCESS);
        EXPECT(std::fabs(mean_of(f) - 0.5) < 0.005);
        EXPECT(curandGenerateUniformDouble(g, d.data(), n) == CURAND_STATUS_SUCCESS);
        EXPECT(std::fabs(mean_of(d) - 0.5) < 0.005);
        EXPECT(curandGenerateNormal(g, f.data(), n, 3.0f, 2.0f) == CURAND_STATUS_SUCCESS);
        EXPECT(std::fabs(mean_of(f) - 3) < 0.03);
        EXPECT(curandGenerateLogNormalDouble(g, d.data(), n, 0.0, 0.5) == CURAND_STATUS_SUCCESS);
        EXPECT(std::fabs(mean_of(d) - std::exp(0.125)) < 0.01);
        EXPECT(curandGeneratePoisson(g, k.data(), n, 12.5) == CURAND_STATUS_SUCCESS);
        EXPECT(std::fabs(mean_of(k) - 12.5) < 0.1);
        EXPECT(curandGenerate(g, k.data(), n) == CURAND_STATUS_SUCCESS);
        EXPECT(std::fabs(mean_of(k) / 4294967296.0 - 0.5) < 0.005);
        curandDestroyGenerator(g);
    }

    // Sobol': n / d points of each dimension, dimension after dimension
    EXPECT(curandCreateGenerator(&g, CURAND_RNG_QUASI_SOBOL32) == CURAND_STATUS_SUCCESS);
    EXPECT(curandSetQuasiRandomGeneratorDimensions(g, 3) == CURAND_STATUS_SUCCESS);
    std::vector<unsigned> q(30);
    EXPECT(curandGenerate(g, q.data(), q.size()) == CURAND_STATUS_SUCCESS);
    EXPECT(q[10] * 0x1p-32 == 0 && q[11] * 0x1p-32 == 0.5 && q[12] * 0x1p-32 == 0.25 && q[29] * 0x1p-32 == 0.4375);
    EXPECT(curandGenerate(g, q.data(), 29) == CURAND_STATUS_LENGTH_NOT_MULTIPLE);
    EXPECT(curandSetPseudoRandomGeneratorSeed(g, 1) == CURAND_STATUS_TYPE_ERROR);
    std::vector<double> qd(3000);
    EXPECT(curandGenerateNormalDouble(g, qd.data(), qd.size(), 0, 1) == CURAND_STATUS_SUCCESS);
    curandDestroyGenerator(g);

    EXPECT(curandCreateGenerator(&g, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64) == CURAND_STATUS_SUCCESS);
    std::vector<unsigned long long> w(1024);
    EXPECT(curandGenerateLongLong(g, w.data(), w.size()) == CURAND_STATUS_SUCCESS);
    std::vector<int> bins(1024);
    for (auto x : w)
        ++bins[x >> 54];
    for (int b : bins)
        EXPECT(b == 1);
    EXPECT(curandGenerate(g, q.data(), 3) == CURAND_STATUS_TYPE_ERROR);
    curandDestroyGenerator(g);

    // Errors
    EXPECT(curandCreateGenerator(&g, curandRngType_t(7)) == CURAND_STATUS_TYPE_ERROR);
    EXPECT(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_XORWOW) == CURAND_STATUS_SUCCESS);
    EXPECT(curandGenerateNormal(g, u.data(), 3, 0, 1) == CURAND_STATUS_LENGTH_NOT_MULTIPLE);
    EXPECT(curandSetQuasiRandomGeneratorDimensions(g, 2) == CURAND_STATUS_TYPE_ERROR);
    EXPECT(curandSetGeneratorOrdering(g, CURAND_ORDERING_QUASI_DEFAULT) == CURAND_STATUS_OUT_OF_RANGE);
    EXPECT(curandSetGeneratorOrdering(g, CURAND_ORDERING_PSEUDO_SEEDED) == CURAND_STATUS_SUCCESS);
    EXPECT(curandGenerateLongLong(g, w.data(), 4) == CURAND_STATUS_TYPE_ERROR);
    curandDestroyGenerator(g);
    EXPECT(curandCreateGenerator(&g, CURAND_RNG_QUASI_SOBOL32) == CURAND_STATUS_SUCCESS);
    EXPECT(curandSetQuasiRandomGeneratorDimensions(g, 0) == CURAND_STATUS_OUT_OF_RANGE);
    EXPECT(curandSetQuasiRandomGeneratorDimensions(g, 20001) == CURAND_STATUS_OUT_OF_RANGE);
    curandDestroyGenerator(g);
    int version = 0;
    EXPECT(curandGetVersion(&version) == CURAND_STATUS_SUCCESS && version == CURAND_VERSION);
}

//! Per-thread states in a kernel, as programs use them
__global__ void setup(curandState *states, unsigned long long seed)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}

__global__ void pi(curandState *states, int samples, unsigned *inside)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state = states[id];
    unsigned count = 0;
    for (int i = 0; i < samples; ++i) {
        const float x = curand_uniform(&state), y = curand_uniform(&state);
        count += x * x + y * y <= 1;
    }
    states[id] = state;
    atomicAdd(inside, count);
}

static void kernels()
{
    const int blocks = 64, threads = 128, samples = 500;
    curandState *states;
    unsigned *inside;
    cudaMalloc(&states, blocks * threads * sizeof(curandState));
    cudaMallocManaged(&inside, sizeof(unsigned));
    *inside = 0;
    setup<<<blocks, threads>>>(states, 31337);
    pi<<<blocks, threads>>>(states, samples, inside);
    cudaDeviceSynchronize();
    const double estimate = 4.0 * *inside / (double(blocks) * threads * samples);
    EXPECT(std::fabs(estimate - 3.14159265) < 0.01);
    // Thread 77's state is subsequence 77's, after 2 * samples values
    curandState host;
    cudaMemcpy(&host, &states[77], sizeof(host), cudaMemcpyDeviceToHost);
    curandState expected;
    curand_init(31337, 77, 2 * samples, &expected);
    EXPECT(curand(&host) == curand(&expected));
    cudaFree(states);
    cudaFree(inside);
}

int main()
{
    xorwow_references();
    skipping_equals_stepping();
    philox_known_answers();
    sobol_points();
    distributions<curandStateXORWOW_t>("XORWOW");
    distributions<curandStatePhilox4_32_10_t>("Philox4_32_10");
    distributions<curandStateMRG32k3a_t>("MRG32k3a");
    host_orderings();
    edges();
    kernels();
    if (failures == 0)
        std::printf("curand: all passed\n");
    return failures == 0 ? 0 : 1;
}
