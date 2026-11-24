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

// cuRAND's device API: generator states that each CUDA thread initializes
// and draws from, at global scope as in CUDA.
//
// - XORWOW is cuRAND's: its seeding and its subsequences, 2^67 values apart,
//   so curand() and curand_uniform() give CUDA's numbers bit for bit.
// - Philox4_32_10 is Random123's, with cuRAND's use of the counter (the
//   subsequence in its upper half, the offset in its lower half) and key
//   (the seed): curand() and curand_uniform() are CUDA's too.
// - MRG32k3a is L'Ecuyer's, with rocRAND's seeding and subsequences 2^76
//   values apart: cuRAND's seeding isn't public, so its numbers differ.
// - Sobol' sequences (32 and 64 bits, scrambled or not) take the direction
//   vectors and scramble constants of curand.h.
// - Doubles take 53 bits from two values, normals use the Box-Muller
//   transform (pseudorandom generators, which cache the second result) or
//   the inverse normal distribution (Sobol'), and Poisson values Knuth's
//   method below a mean of 64 and Hormann's PTRS above. They are distributed
//   as CUDA's, with rocRAND's arithmetic where cuRAND's isn't public, but are
//   not the same numbers: a GPU's logf and sincosf round differently anyway.
//
// MTGP32's device API (curand_mtgp32_kernel.h) isn't provided.
#pragma once

#ifndef CURAND_KERNEL_H_
#define CURAND_KERNEL_H_
#endif

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <type_traits>

#include "curand.h"

namespace cuda4cpu::detail::rand {

//! The library's jumps: n values ahead, or n subsequences of 2^67 values
//! (XORWOW's state words, without its Weyl sequence), and n values or
//! subsequences of 2^76 values (MRG32k3a's two components)
void xorwow_jump(unsigned int v[5], unsigned long long n, bool subsequences);
void mrg32k3a_jump(unsigned int g1[3], unsigned int g2[3], unsigned long long n, bool subsequences);

//! cuRAND's uniform float of 32 bits: (0, 1], rounded once
inline float uniform(unsigned int x)
{
    return float(x) * 0x1p-32f + 0x1p-33f;
}

//! A uniform double of 53 bits: (0, 1). The largest, 1 - 2^-54, would round
//! to 1, so it is the double below 1 instead.
inline double uniform_double(unsigned long long x)
{
    return std::min(double(x >> 11) * 0x1p-53 + 0x1p-54, 0x1.fffffffffffffp-1);
}

//! From two values
inline double uniform_double(unsigned int x, unsigned int y)
{
    const unsigned long long z = (unsigned long long)x ^ ((unsigned long long)y << 21);
    return uniform_double((z & ((1ull << 53) - 1)) << 11);
}

//! The Box-Muller transform of two values: two independent standard normals
inline float2 box_muller(unsigned int x, unsigned int y)
{
    const float u = uniform(x);
    const float v = float(y) * (0x1p-32f * 2 * std::numbers::pi_v<float>) + 0x1p-32f * std::numbers::pi_v<float>;
    const float s = std::sqrt(-2.0f * std::log(u));
    return make_float2(std::sin(v) * s, std::cos(v) * s);
}

inline double2 box_muller_double(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
    const double u = uniform_double(x, y);
    const double v = uniform_double(z, w) * 2 * std::numbers::pi;
    const double s = std::sqrt(-2.0 * std::log(u));
    return make_double2(std::sin(v) * s, std::cos(v) * s);
}

//! The inverse of the standard normal distribution, for 0 < p < 1: Wichura's
//! algorithm AS 241 (PPND16), accurate to about 1e-16
inline double normal_quantile(double p)
{
    const double q = p - 0.5;
    if (std::fabs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        return q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r +
                         45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r +
                      133.14166789178437745) * r + 3.387132872796366608) /
               (((((((r * 5226.495278852545925 + 28729.085735721942674) * r + 39307.89580009271061) * r +
                    21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r +
                 42.313330701600911252) * r + 1.0);
    }
    double r = q < 0 ? p : 1.0 - p;
    r = std::sqrt(-std::log(r));
    double value;
    if (r <= 5.0) {
        r -= 1.6;
        value = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) * r +
                     1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r +
                  4.6303378461565452959) * r + 1.42343711074968357734) /
                (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r +
                     0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r +
                  2.05319162663775882187) * r + 1.0);
    } else {
        r -= 5.0;
        value = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r +
                     0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r +
                  5.4637849111641143699) * r + 6.6579046435011037772) /
                (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r +
                     7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r +
                  0.59983220655588793769) * r + 1.0);
    }
    return q < 0 ? -value : value;
}

//! A Poisson value of mean lambda from uniform doubles that next() draws
template <typename Next>
inline unsigned int poisson(double lambda, Next next)
{
    if (!(lambda > 0))
        return 0;
    if (lambda < 64) {
        // Knuth: the number of uniforms whose product stays above e^-lambda
        const double limit = std::exp(-lambda);
        unsigned int k = 0;
        for (double product = next(); product > limit; product *= next())
            ++k;
        return k;
    }
    // Hormann's transformed rejection with squeeze (PTRS)
    const double slam = std::sqrt(lambda), loglam = std::log(lambda);
    const double b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
    const double invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
    for (;;) {
        const double u = next() - 0.5, v = next(), us = 0.5 - std::fabs(u);
        const double k = std::floor((2 * a / us + b) * u + lambda + 0.43);
        if (us >= 0.07 && v <= vr)
            return (unsigned int)k;
        if (k < 0 || (us < 0.013 && v > us))
            continue;
        if (std::log(v) + std::log(invalpha) - std::log(a / (us * us) + b) <= -lambda + k * loglam - std::lgamma(k + 1))
            return (unsigned int)k;
    }
}

//! A Poisson value of mean lambda from one uniform u, by inversion: for the
//! quasirandom generators, whose dimensions must take one value each. The
//! search starts at the mode, whose probability doesn't underflow, and goes
//! down or up the distribution from there.
inline unsigned int poisson_quantile(double lambda, double u)
{
    if (!(lambda > 0))
        return 0;
    const double mode = std::floor(lambda);
    const double p_mode = std::exp(mode * std::log(lambda) - lambda - std::lgamma(mode + 1));
    // P(X <= mode), summed down from the mode until the terms vanish
    double cdf = p_mode, p = p_mode;
    for (double k = mode; k > 0 && p > 1e-17 * cdf; --k) {
        p *= k / lambda;
        cdf += p;
    }
    double k = mode;
    p = p_mode;
    if (u <= cdf) {
        while (k > 0 && cdf - p >= u) {             // P(X <= k - 1) still reaches u
            cdf -= p;
            p *= k / lambda;
            --k;
        }
    } else {
        const double last = mode + 40 * std::sqrt(lambda) + 100;
        while (cdf < u && k < last) {
            ++k;
            p *= lambda / k;
            cdf += p;
        }
    }
    return k > 4294967295.0 ? 4294967295u : (unsigned int)k;
}

//! Philox4x32-10 (Random123): ten rounds of the counter under the key
inline uint4 philox4x32_10(uint4 c, uint2 k)
{
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k.x += 0x9E3779B9u;
            k.y += 0xBB67AE85u;
        }
        const unsigned long long p0 = 0xD2511F53ull * c.x, p1 = 0xCD9E8D57ull * c.z;
        c = make_uint4(unsigned(p1 >> 32) ^ c.y ^ k.x, unsigned(p1), unsigned(p0 >> 32) ^ c.w ^ k.y, unsigned(p0));
    }
    return c;
}

//! Adds n to the 128-bit counter, from word `from` (0: the offset's half, 2:
//! the subsequence's half)
inline void philox_add(uint4 &c, unsigned long long n, int from)
{
    unsigned int *w[4] = {&c.x, &c.y, &c.z, &c.w};
    unsigned long long carry = n;
    for (int i = from; i < 4 && carry != 0; ++i) {
        const unsigned long long sum = (unsigned long long)*w[i] + (carry & 0xffffffffu);
        *w[i] = unsigned(sum);
        carry = (carry >> 32) + (sum >> 32);
    }
}

// MRG32k3a's moduli and multipliers
constexpr unsigned long long mrg_m1 = 4294967087ull, mrg_m2 = 4294944443ull;
constexpr unsigned long long mrg_a12 = 1403580, mrg_a13n = 810728, mrg_a21 = 527612, mrg_a23n = 1370589;

}

//
// States
//

struct curandStateXORWOW {
    unsigned int d, v[5];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};
typedef struct curandStateXORWOW curandStateXORWOW_t;
typedef struct curandStateXORWOW curandState_t;
typedef struct curandStateXORWOW curandState;

struct curandStatePhilox4_32_10 {
    uint4 ctr;
    uint4 output;
    uint2 key;
    unsigned int STATE;
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};
typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

struct curandStateMRG32k3a {
    unsigned int s1[3], s2[3];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};
typedef struct curandStateMRG32k3a curandStateMRG32k3a_t;

struct curandStateSobol32 {
    unsigned int i, x, c;
    unsigned int direction_vectors[32];
};
typedef struct curandStateSobol32 curandStateSobol32_t;

struct curandStateScrambledSobol32 {
    unsigned int i, x, c;
    unsigned int direction_vectors[32];
};
typedef struct curandStateScrambledSobol32 curandStateScrambledSobol32_t;

struct curandStateSobol64 {
    unsigned long long i, x, c;
    unsigned long long direction_vectors[64];
};
typedef struct curandStateSobol64 curandStateSobol64_t;

struct curandStateScrambledSobol64 {
    unsigned long long i, x, c;
    unsigned long long direction_vectors[64];
};
typedef struct curandStateScrambledSobol64 curandStateScrambledSobol64_t;

namespace cuda4cpu::detail::rand {

template <typename S>
concept pseudo = std::is_same_v<S, curandStateXORWOW> || std::is_same_v<S, curandStatePhilox4_32_10> ||
                 std::is_same_v<S, curandStateMRG32k3a>;
template <typename S>
concept sobol32 = std::is_same_v<S, curandStateSobol32> || std::is_same_v<S, curandStateScrambledSobol32>;
template <typename S>
concept sobol64 = std::is_same_v<S, curandStateSobol64> || std::is_same_v<S, curandStateScrambledSobol64>;
template <typename S>
concept quasi = sobol32<S> || sobol64<S>;

//! MRG32k3a's next value, in [1, m1]
inline unsigned int mrg32k3a_next(curandStateMRG32k3a *s)
{
    const unsigned long long p1 = (mrg_a12 * s->s1[1] + mrg_a13n * (mrg_m1 - s->s1[0])) % mrg_m1;
    s->s1[0] = s->s1[1];
    s->s1[1] = s->s1[2];
    s->s1[2] = unsigned(p1);
    const unsigned long long p2 = (mrg_a21 * s->s2[2] + mrg_a23n * (mrg_m2 - s->s2[0])) % mrg_m2;
    s->s2[0] = s->s2[1];
    s->s2[1] = s->s2[2];
    s->s2[2] = unsigned(p2);
    return unsigned(p1 > p2 ? p1 - p2 : p1 - p2 + mrg_m1);
}

//! A Sobol' point's value: the constant, and the direction vectors of the
//! Gray code of its index
template <typename T>
inline T sobol_value(const T *vectors, T c, T index)
{
    T x = c;
    for (T g = index ^ (index >> 1), k = 0; g != 0; g >>= 1, ++k) {
        if (g & 1)
            x ^= vectors[k];
    }
    return x;
}

}

//
// Initialization and skipping ahead
//

//! The state of the subsequence-th subsequence, offset values in
inline void curand_init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset,
                        curandStateXORWOW_t *state)
{
    const unsigned int s0 = unsigned(seed) ^ 0xaad26b49u, s1 = unsigned(seed >> 32) ^ 0xf7dcefddu;
    const unsigned int t0 = 1099087573u * s0, t1 = 2591861531u * s1;
    state->d = 6615241u + t1 + t0;
    state->v[0] = 123456789u + t0;
    state->v[1] = 362436069u ^ t0;
    state->v[2] = 521288629u + t1;
    state->v[3] = 88675123u ^ t1;
    state->v[4] = 5783321u + t0;
    cuda4cpu::detail::rand::xorwow_jump(state->v, subsequence, true);
    cuda4cpu::detail::rand::xorwow_jump(state->v, offset, false);
    state->d += 362437u * unsigned(offset);
    state->boxmuller_flag = state->boxmuller_flag_double = 0;
}

inline void curand_init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset,
                        curandStatePhilox4_32_10_t *state)
{
    state->ctr = make_uint4(0, 0, 0, 0);
    state->key = make_uint2(unsigned(seed), unsigned(seed >> 32));
    state->STATE = 0;
    state->boxmuller_flag = state->boxmuller_flag_double = 0;
    cuda4cpu::detail::rand::philox_add(state->ctr, subsequence, 2);
    state->STATE = unsigned(offset & 3);
    cuda4cpu::detail::rand::philox_add(state->ctr, offset / 4, 0);
    state->output = cuda4cpu::detail::rand::philox4x32_10(state->ctr, state->key);
}

inline void curand_init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset,
                        curandStateMRG32k3a_t *state)
{
    using namespace cuda4cpu::detail::rand;
    if (seed == 0)
        seed = 12345;
    const unsigned long long x = unsigned(seed) ^ 0x55555555u, y = unsigned(seed >> 32) ^ 0xAAAAAAAAu;
    const unsigned __int128 s = seed;
    state->s1[0] = unsigned(x * s % mrg_m1);
    state->s1[1] = unsigned(y * s % mrg_m1);
    state->s1[2] = unsigned(x * s % mrg_m1);
    state->s2[0] = unsigned(y * s % mrg_m2);
    state->s2[1] = unsigned(x * s % mrg_m2);
    state->s2[2] = unsigned(y * s % mrg_m2);
    // A component that is all zeros would stay so: start it as seed 12345's
    if ((state->s1[0] | state->s1[1] | state->s1[2]) == 0)
        state->s1[0] = state->s1[1] = state->s1[2] = 12345;
    if ((state->s2[0] | state->s2[1] | state->s2[2]) == 0)
        state->s2[0] = state->s2[1] = state->s2[2] = 12345;
    mrg32k3a_jump(state->s1, state->s2, subsequence, true);
    mrg32k3a_jump(state->s1, state->s2, offset, false);
    state->boxmuller_flag = state->boxmuller_flag_double = 0;
}

//! The Sobol' sequence of a dimension's direction vectors, from point offset
inline void curand_init(curandDirectionVectors32_t direction_vectors, unsigned int offset, curandStateSobol32_t *state)
{
    std::copy_n(direction_vectors, 32, state->direction_vectors);
    state->i = offset;
    state->c = 0;
    state->x = cuda4cpu::detail::rand::sobol_value(state->direction_vectors, 0u, offset);
}

inline void curand_init(curandDirectionVectors32_t direction_vectors, unsigned int scramble_c, unsigned int offset,
                        curandStateScrambledSobol32_t *state)
{
    std::copy_n(direction_vectors, 32, state->direction_vectors);
    state->i = offset;
    state->c = scramble_c;
    state->x = cuda4cpu::detail::rand::sobol_value(state->direction_vectors, scramble_c, offset);
}

inline void curand_init(curandDirectionVectors64_t direction_vectors, unsigned long long offset,
                        curandStateSobol64_t *state)
{
    std::copy_n(direction_vectors, 64, state->direction_vectors);
    state->i = offset;
    state->c = 0;
    state->x = cuda4cpu::detail::rand::sobol_value(state->direction_vectors, 0ull, offset);
}

inline void curand_init(curandDirectionVectors64_t direction_vectors, unsigned long long scramble_c,
                        unsigned long long offset, curandStateScrambledSobol64_t *state)
{
    std::copy_n(direction_vectors, 64, state->direction_vectors);
    state->i = offset;
    state->c = scramble_c;
    state->x = cuda4cpu::detail::rand::sobol_value(state->direction_vectors, scramble_c, offset);
}

inline void skipahead(unsigned long long n, curandStateXORWOW_t *state)
{
    cuda4cpu::detail::rand::xorwow_jump(state->v, n, false);
    state->d += 362437u * unsigned(n);
}

inline void skipahead_sequence(unsigned long long n, curandStateXORWOW_t *state)
{
    cuda4cpu::detail::rand::xorwow_jump(state->v, n, true);
}

inline void skipahead(unsigned long long n, curandStatePhilox4_32_10_t *state)
{
    const unsigned long long total = state->STATE + (n & 3);
    state->STATE = unsigned(total & 3);
    cuda4cpu::detail::rand::philox_add(state->ctr, n / 4 + total / 4, 0);
    state->output = cuda4cpu::detail::rand::philox4x32_10(state->ctr, state->key);
}

inline void skipahead_sequence(unsigned long long n, curandStatePhilox4_32_10_t *state)
{
    cuda4cpu::detail::rand::philox_add(state->ctr, n, 2);
    state->output = cuda4cpu::detail::rand::philox4x32_10(state->ctr, state->key);
}

inline void skipahead(unsigned long long n, curandStateMRG32k3a_t *state)
{
    cuda4cpu::detail::rand::mrg32k3a_jump(state->s1, state->s2, n, false);
}

inline void skipahead_sequence(unsigned long long n, curandStateMRG32k3a_t *state)
{
    cuda4cpu::detail::rand::mrg32k3a_jump(state->s1, state->s2, n, true);
}

template <typename S> requires cuda4cpu::detail::rand::quasi<S>
inline void skipahead(decltype(S::i) n, S *state)
{
    state->i += n;
    state->x = cuda4cpu::detail::rand::sobol_value(state->direction_vectors, state->c, state->i);
}

//
// Raw values
//

inline unsigned int curand(curandStateXORWOW_t *state)
{
    const unsigned int t = state->v[0] ^ (state->v[0] >> 2);
    state->v[0] = state->v[1];
    state->v[1] = state->v[2];
    state->v[2] = state->v[3];
    state->v[3] = state->v[4];
    state->v[4] = (state->v[4] ^ (state->v[4] << 4)) ^ (t ^ (t << 1));
    state->d += 362437u;
    return state->v[4] + state->d;
}

inline unsigned int curand(curandStatePhilox4_32_10_t *state)
{
    const unsigned int words[4] = {state->output.x, state->output.y, state->output.z, state->output.w};
    const unsigned int value = words[state->STATE];
    if (++state->STATE == 4) {
        state->STATE = 0;
        cuda4cpu::detail::rand::philox_add(state->ctr, 1, 0);
        state->output = cuda4cpu::detail::rand::philox4x32_10(state->ctr, state->key);
    }
    return value;
}

//! Four values at once, from where the state is
inline uint4 curand4(curandStatePhilox4_32_10_t *state)
{
    const uint4 current = state->output;
    cuda4cpu::detail::rand::philox_add(state->ctr, 1, 0);
    state->output = cuda4cpu::detail::rand::philox4x32_10(state->ctr, state->key);
    const unsigned int words[8] = {current.x, current.y, current.z, current.w,
                                   state->output.x, state->output.y, state->output.z, state->output.w};
    const unsigned int s = state->STATE;
    return make_uint4(words[s], words[s + 1], words[s + 2], words[s + 3]);
}

//! A 32-bit value: MRG32k3a's, scaled from [1, m1]
inline unsigned int curand(curandStateMRG32k3a_t *state)
{
    const unsigned int r = cuda4cpu::detail::rand::mrg32k3a_next(state);
    return unsigned(double(r - 1) * (4294967295.0 / double(cuda4cpu::detail::rand::mrg_m1 - 1)));
}

template <typename S> requires cuda4cpu::detail::rand::quasi<S>
inline decltype(S::x) curand(S *state)
{
    const auto value = state->x;
    // Gray code: the vector of the lowest zero bit of the index. After the
    // last point, the sequence starts again.
    const int k = std::countr_one(state->i);
    if (k < int(sizeof(state->x) * 8))
        state->x ^= state->direction_vectors[k];
    else
        state->x = state->c;
    ++state->i;
    return value;
}

//
// Uniform distributions: (0, 1]
//

template <typename S> requires std::is_same_v<S, curandStateXORWOW> || std::is_same_v<S, curandStatePhilox4_32_10> ||
                               cuda4cpu::detail::rand::sobol32<S>
inline float curand_uniform(S *state)
{
    return cuda4cpu::detail::rand::uniform(curand(state));
}

inline float curand_uniform(curandStateMRG32k3a_t *state)
{
    return float(double(cuda4cpu::detail::rand::mrg32k3a_next(state)) * (1.0 / double(cuda4cpu::detail::rand::mrg_m1 + 1)));
}

template <typename S> requires cuda4cpu::detail::rand::sobol64<S>
inline float curand_uniform(S *state)
{
    return cuda4cpu::detail::rand::uniform(unsigned(curand(state) >> 32));
}

template <typename S> requires std::is_same_v<S, curandStateXORWOW> || std::is_same_v<S, curandStatePhilox4_32_10>
inline double curand_uniform_double(S *state)
{
    const unsigned int x = curand(state), y = curand(state);
    return cuda4cpu::detail::rand::uniform_double(x, y);
}

inline double curand_uniform_double(curandStateMRG32k3a_t *state)
{
    return double(cuda4cpu::detail::rand::mrg32k3a_next(state)) * (1.0 / double(cuda4cpu::detail::rand::mrg_m1 + 1));
}

//! 32 bits of precision, as CUDA's
template <typename S> requires cuda4cpu::detail::rand::sobol32<S>
inline double curand_uniform_double(S *state)
{
    return double(curand(state)) * 0x1p-32 + 0x1p-33;
}

//! 53 bits of precision: the high ones
template <typename S> requires cuda4cpu::detail::rand::sobol64<S>
inline double curand_uniform_double(S *state)
{
    return cuda4cpu::detail::rand::uniform_double(curand(state));
}

inline float4 curand_uniform4(curandStatePhilox4_32_10_t *state)
{
    using cuda4cpu::detail::rand::uniform;
    const uint4 v = curand4(state);
    return make_float4(uniform(v.x), uniform(v.y), uniform(v.z), uniform(v.w));
}

inline double2 curand_uniform2_double(curandStatePhilox4_32_10_t *state)
{
    using cuda4cpu::detail::rand::uniform_double;
    const uint4 v = curand4(state);
    return make_double2(uniform_double(v.x, v.y), uniform_double(v.z, v.w));
}

inline double4 curand_uniform4_double(curandStatePhilox4_32_10_t *state)
{
    const double2 a = curand_uniform2_double(state), b = curand_uniform2_double(state);
    return make_double4(a.x, a.y, b.x, b.y);
}

//
// Normal distributions: mean 0, standard deviation 1
//

template <typename S> requires cuda4cpu::detail::rand::pseudo<S>
inline float2 curand_normal2(S *state)
{
    if constexpr (std::is_same_v<S, curandStateMRG32k3a>) {
        const double u = curand_uniform_double(state), v = curand_uniform_double(state);
        const double s = std::sqrt(-2.0 * std::log(u));
        return make_float2(float(std::sin(2 * std::numbers::pi * v) * s), float(std::cos(2 * std::numbers::pi * v) * s));
    } else {
        const unsigned int x = curand(state), y = curand(state);
        return cuda4cpu::detail::rand::box_muller(x, y);
    }
}

template <typename S> requires cuda4cpu::detail::rand::pseudo<S>
inline double2 curand_normal2_double(S *state)
{
    if constexpr (std::is_same_v<S, curandStateMRG32k3a>) {
        const double u = curand_uniform_double(state), v = curand_uniform_double(state);
        const double s = std::sqrt(-2.0 * std::log(u));
        return make_double2(std::sin(2 * std::numbers::pi * v) * s, std::cos(2 * std::numbers::pi * v) * s);
    } else {
        const unsigned int x = curand(state), y = curand(state), z = curand(state), w = curand(state);
        return cuda4cpu::detail::rand::box_muller_double(x, y, z, w);
    }
}

//! One normal: the Box-Muller transform makes two, and the state keeps the
//! second for the next call
template <typename S> requires cuda4cpu::detail::rand::pseudo<S>
inline float curand_normal(S *state)
{
    if (state->boxmuller_flag) {
        state->boxmuller_flag = 0;
        return state->boxmuller_extra;
    }
    const float2 v = curand_normal2(state);
    state->boxmuller_extra = v.y;
    state->boxmuller_flag = 1;
    return v.x;
}

template <typename S> requires cuda4cpu::detail::rand::pseudo<S>
inline double curand_normal_double(S *state)
{
    if (state->boxmuller_flag_double) {
        state->boxmuller_flag_double = 0;
        return state->boxmuller_extra_double;
    }
    const double2 v = curand_normal2_double(state);
    state->boxmuller_extra_double = v.y;
    state->boxmuller_flag_double = 1;
    return v.x;
}

//! Quasirandom normals invert the distribution, so that each point of a
//! dimension takes one value
template <typename S> requires cuda4cpu::detail::rand::quasi<S>
inline float curand_normal(S *state)
{
    return float(cuda4cpu::detail::rand::normal_quantile(curand_uniform_double(state)));
}

template <typename S> requires cuda4cpu::detail::rand::quasi<S>
inline double curand_normal_double(S *state)
{
    return cuda4cpu::detail::rand::normal_quantile(curand_uniform_double(state));
}

inline float4 curand_normal4(curandStatePhilox4_32_10_t *state)
{
    const uint4 v = curand4(state);
    const float2 a = cuda4cpu::detail::rand::box_muller(v.x, v.y), b = cuda4cpu::detail::rand::box_muller(v.z, v.w);
    return make_float4(a.x, a.y, b.x, b.y);
}

inline double4 curand_normal4_double(curandStatePhilox4_32_10_t *state)
{
    const double2 a = curand_normal2_double(state), b = curand_normal2_double(state);
    return make_double4(a.x, a.y, b.x, b.y);
}

//
// Log-normal distributions: e to a normal of mean and standard deviation
//

template <typename S>
inline float curand_log_normal(S *state, float mean, float stddev)
{
    return std::exp(mean + stddev * curand_normal(state));
}

template <typename S>
inline double curand_log_normal_double(S *state, double mean, double stddev)
{
    return std::exp(mean + stddev * curand_normal_double(state));
}

template <typename S> requires cuda4cpu::detail::rand::pseudo<S>
inline float2 curand_log_normal2(S *state, float mean, float stddev)
{
    const float2 v = curand_normal2(state);
    return make_float2(std::exp(mean + stddev * v.x), std::exp(mean + stddev * v.y));
}

template <typename S> requires cuda4cpu::detail::rand::pseudo<S>
inline double2 curand_log_normal2_double(S *state, double mean, double stddev)
{
    const double2 v = curand_normal2_double(state);
    return make_double2(std::exp(mean + stddev * v.x), std::exp(mean + stddev * v.y));
}

inline float4 curand_log_normal4(curandStatePhilox4_32_10_t *state, float mean, float stddev)
{
    const float4 v = curand_normal4(state);
    return make_float4(std::exp(mean + stddev * v.x), std::exp(mean + stddev * v.y), std::exp(mean + stddev * v.z),
                       std::exp(mean + stddev * v.w));
}

inline double4 curand_log_normal4_double(curandStatePhilox4_32_10_t *state, double mean, double stddev)
{
    const double4 v = curand_normal4_double(state);
    return make_double4(std::exp(mean + stddev * v.x), std::exp(mean + stddev * v.y), std::exp(mean + stddev * v.z),
                        std::exp(mean + stddev * v.w));
}

//
// Poisson and other discrete distributions
//

template <typename S>
inline unsigned int curand_poisson(S *state, double lambda)
{
    if constexpr (cuda4cpu::detail::rand::quasi<S>)
        return cuda4cpu::detail::rand::poisson_quantile(lambda, curand_uniform_double(state));
    else
        return cuda4cpu::detail::rand::poisson(lambda, [state] { return curand_uniform_double(state); });
}

inline uint4 curand_poisson4(curandStatePhilox4_32_10_t *state, double lambda)
{
    const unsigned int a = curand_poisson(state, lambda), b = curand_poisson(state, lambda);
    const unsigned int c = curand_poisson(state, lambda), d = curand_poisson(state, lambda);
    return make_uint4(a, b, c, d);
}

//! A value of a distribution from curandCreatePoissonDistribution: the
//! inverse of its cumulative distribution at a uniform
template <typename S>
inline unsigned int curand_discrete(S *state, curandDiscreteDistribution_t distribution)
{
    const double u = curand_uniform_double(state);
    const double *cdf = distribution->probability;
    const double *found = std::lower_bound(cdf, cdf + distribution->length, u);
    const unsigned int index = unsigned(std::min<ptrdiff_t>(found - cdf, distribution->length - 1));
    return distribution->shift + index;
}

inline uint4 curand_discrete4(curandStatePhilox4_32_10_t *state, curandDiscreteDistribution_t distribution)
{
    const unsigned int a = curand_discrete(state, distribution), b = curand_discrete(state, distribution);
    const unsigned int c = curand_discrete(state, distribution), d = curand_discrete(state, distribution);
    return make_uint4(a, b, c, d);
}
