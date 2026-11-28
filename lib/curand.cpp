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

// cuRAND's host API (curand.h) and the tables of its device API
// (curand_kernel.h): XORWOW's and MRG32k3a's jumps, Sobol' direction vectors
// and scramble constants, and the generators.

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string_view>
#include <vector>

#include <omp.h>

#include "curand_kernel.h"

namespace cuda4cpu::detail::rand {

namespace {

//
// XORWOW: its state words (without the Weyl sequence) are a linear map over
// GF(2) of 160 bits. A matrix holds the image of each basis bit (word i,
// bit j, row i * 32 + j), as 5 words.
//

using xorwow_matrix = std::array<unsigned int, 160 * 5>;

void apply(const xorwow_matrix &m, unsigned int v[5])
{
    unsigned int r[5] = {};
    for (int i = 0; i < 5; ++i) {
        for (unsigned int bits = v[i]; bits != 0; bits &= bits - 1) {
            const unsigned int *row = &m[(i * 32 + unsigned(std::countr_zero(bits))) * 5];
            for (int k = 0; k < 5; ++k)
                r[k] ^= row[k];
        }
    }
    std::copy_n(r, 5, v);
}

//! a then b (powers of one matrix commute)
xorwow_matrix product(const xorwow_matrix &a, const xorwow_matrix &b)
{
    xorwow_matrix c;
    for (int row = 0; row < 160; ++row) {
        unsigned int v[5];
        std::copy_n(&a[row * 5], 5, v);
        apply(b, v);
        std::copy_n(v, 5, &c[row * 5]);
    }
    return c;
}

xorwow_matrix xorwow_step()
{
    xorwow_matrix m;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            unsigned int v[5] = {};
            v[i] = 1u << j;
            const unsigned int t = v[0] ^ (v[0] >> 2);
            const unsigned int next[5] = {v[1], v[2], v[3], v[4], (v[4] ^ (v[4] << 4)) ^ (t ^ (t << 1))};
            std::copy_n(next, 5, &m[(i * 32 + j) * 5]);
        }
    }
    return m;
}

//! Jumps by base-16 digits: table[k][d - 1] is the step to the power d * 16^k
struct jump_table {
    std::vector<std::array<xorwow_matrix, 15>> digits;

    explicit jump_table(xorwow_matrix base)
    {
        digits.resize(16);
        for (auto &level : digits) {
            level[0] = base;
            for (int d = 1; d < 15; ++d)
                level[d] = product(level[d - 1], base);
            base = product(level[14], base);
        }
    }
};

const jump_table &xorwow_table(bool subsequences)
{
    static std::once_flag once;
    static std::unique_ptr<jump_table> values, sequences;
    std::call_once(once, [] {
        const xorwow_matrix step = xorwow_step();
        values = std::make_unique<jump_table>(step);
        xorwow_matrix spacing = step;          // the step to the power 2^67
        for (int i = 0; i < 67; ++i)
            spacing = product(spacing, spacing);
        sequences = std::make_unique<jump_table>(spacing);
    });
    return subsequences ? *sequences : *values;
}

//
// MRG32k3a: each component's step is a 3 x 3 matrix, modulo its modulus
//

using mrg_matrix = std::array<std::array<unsigned long long, 3>, 3>;

mrg_matrix product(const mrg_matrix &a, const mrg_matrix &b, unsigned long long m)
{
    mrg_matrix c{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                c[i][j] = (c[i][j] + a[i][k] * b[k][j] % m) % m;
    return c;
}

void apply(const mrg_matrix &a, unsigned int g[3], unsigned long long m)
{
    unsigned long long r[3] = {};
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            r[i] = (r[i] + a[i][k] * g[k] % m) % m;
    for (int i = 0; i < 3; ++i)
        g[i] = unsigned(r[i]);
}

//! The steps to the powers 2^k (values) and 2^(76 + k) (subsequences)
struct mrg_tables {
    std::array<mrg_matrix, 64> values1, values2, sequences1, sequences2;

    mrg_tables()
    {
        // (g0, g1, g2) to (g1, g2, a12 g1 - a13 g0), and (g1, g2, a21 g2 - a23 g0)
        mrg_matrix a1{{{0, 1, 0}, {0, 0, 1}, {mrg_m1 - mrg_a13n, mrg_a12, 0}}};
        mrg_matrix a2{{{0, 1, 0}, {0, 0, 1}, {mrg_m2 - mrg_a23n, 0, mrg_a21}}};
        for (int k = 0; k < 140; ++k) {
            if (k < 64) {
                values1[k] = a1;
                values2[k] = a2;
            }
            if (k >= 76) {
                sequences1[k - 76] = a1;
                sequences2[k - 76] = a2;
            }
            a1 = product(a1, a1, mrg_m1);
            a2 = product(a2, a2, mrg_m2);
        }
    }
};

const mrg_tables &mrg_table()
{
    static const mrg_tables tables;
    return tables;
}

}

void xorwow_jump(unsigned int v[5], unsigned long long n, bool subsequences)
{
    if (n == 0)
        return;
    const jump_table &table = xorwow_table(subsequences);
    for (int k = 0; n != 0; ++k, n >>= 4) {
        if (const unsigned d = unsigned(n & 15))
            apply(table.digits[k][d - 1], v);
    }
}

void mrg32k3a_jump(unsigned int g1[3], unsigned int g2[3], unsigned long long n, bool subsequences)
{
    if (n == 0)
        return;
    const mrg_tables &t = mrg_table();
    for (int k = 0; n != 0; ++k, n >>= 1) {
        if (n & 1) {
            apply(subsequences ? t.sequences1[k] : t.values1[k], g1, mrg_m1);
            apply(subsequences ? t.sequences2[k] : t.values2[k], g2, mrg_m2);
        }
    }
}

namespace {

//
// Sobol': Joe and Kuo's direction numbers (sobol_directions.inc), and the
// scrambled sets, which cuRAND makes with constants of its own: these are
// cuda4cpu's, from a fixed XORWOW sequence
//

#include "sobol_directions.inc"

//! Each dimension's s, a and m_1 ... m_s
struct direction_numbers {
    std::vector<unsigned> s, a;
    std::vector<std::array<unsigned, 18>> m;

    direction_numbers()
    {
        static constexpr std::string_view alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::vector<unsigned char> bytes;
        unsigned buffer = 0, held = 0;
        for (const char ch : std::string_view(sobol_direction_numbers)) {
            const size_t value = alphabet.find(ch);
            if (value == std::string_view::npos)
                continue;                       // padding
            buffer = buffer << 6 | unsigned(value);
            held += 6;
            if (held >= 8) {
                held -= 8;
                bytes.push_back((unsigned char)(buffer >> held));
            }
        }
        size_t bit = 0;
        auto take = [&](unsigned width) {
            unsigned value = 0;
            for (unsigned i = 0; i < width; ++i, ++bit)
                value |= unsigned(bytes[bit / 8] >> (bit % 8) & 1) << i;
            return value;
        };
        for (unsigned d = 1; d < sobol_dimensions; ++d) {
            const unsigned degree = take(5);
            s.push_back(degree);
            a.push_back(take(degree - 1));
            std::array<unsigned, 18> values{};
            for (unsigned k = 1; k <= degree; ++k)
                values[k - 1] = take(k);
            m.push_back(values);
        }
    }
};

//! The direction vectors of every dimension, with Bits bits
template <typename T, int Bits>
std::vector<std::array<T, Bits>> direction_vectors()
{
    static const direction_numbers numbers;
    std::vector<std::array<T, Bits>> vectors(sobol_dimensions);
    for (int k = 0; k < Bits; ++k)
        vectors[0][k] = T(1) << (Bits - 1 - k);
    for (unsigned d = 1; d < sobol_dimensions; ++d) {
        auto &v = vectors[d];
        const unsigned s = numbers.s[d - 1], a = numbers.a[d - 1];
        for (unsigned k = 0; k < s && k < unsigned(Bits); ++k)
            v[k] = T(numbers.m[d - 1][k]) << (Bits - 1 - k);
        for (unsigned k = s; k < unsigned(Bits); ++k) {
            T value = v[k - s] ^ (v[k - s] >> s);
            for (unsigned l = 1; l < s; ++l) {
                if (a >> (s - 1 - l) & 1)
                    value ^= v[k - l];
            }
            v[k] = value;
        }
    }
    return vectors;
}

//! A fixed sequence for the scrambled sets
struct scrambler {
    curandStateXORWOW state;
    scrambler() { curand_init(0x6375646134637075ull, 0, 0, &state); }
    unsigned int next32() { return curand(&state); }
    unsigned long long next64() { return (unsigned long long)curand(&state) << 32 | curand(&state); }
};

//! The vectors scrambled by a random lower-triangular matrix of bits (with
//! ones on its diagonal) per dimension: a linear scrambling, as Matousek's,
//! which keeps the sequence's equidistribution
template <typename T, int Bits>
std::vector<std::array<T, Bits>> scrambled_vectors()
{
    std::vector<std::array<T, Bits>> vectors = direction_vectors<T, Bits>();
    scrambler random;
    for (auto &v : vectors) {
        // row[p] has the bits that output bit p (from the most significant)
        // takes from the input's bits at positions q <= p
        std::array<T, Bits> row;
        for (int p = 0; p < Bits; ++p) {
            const T bits = Bits == 32 ? T(random.next32()) : T(random.next64());
            const T above = p == 0 ? T(0) : T(bits & ~(~T(0) >> p));   // positions q < p
            row[p] = above | (T(1) << (Bits - 1 - p));
        }
        for (T &x : v) {
            T y = 0;
            for (int p = 0; p < Bits; ++p) {
                if (std::popcount(T(x & row[p])) & 1)
                    y |= T(1) << (Bits - 1 - p);
            }
            x = y;
        }
    }
    return vectors;
}

}

}

using namespace cuda4cpu::detail::rand;

//
// The generators. A pseudorandom generator's results come from lanes, as
// cuRAND documents its orderings: the result at offset m is the next value
// of lane (m / group) % lanes. Each lane is a subsequence of the generator's
// device-API state, so XORWOW's (4096 lanes) and Philox's (65536 lanes, in
// groups of 4) are CUDA's numbers.
//

struct curandGenerator_st {
    curandRngType_t type;
    bool host;
    unsigned long long seed = 0;
    unsigned long long offset = 0;
    curandOrdering_t ordering;
    unsigned int dimensions = 1;
    cudaStream_t stream = nullptr;

    // Results made so far, from the offset, and the lanes' states (empty
    // until the first results, or after the seed, offset or ordering change)
    unsigned long long position = 0;
    std::vector<curandStateXORWOW> xorwow;
    std::vector<curandStatePhilox4_32_10> philox;
    std::vector<curandStateMRG32k3a> mrg;
    std::unique_ptr<std::mt19937> mt;

    bool quasi() const { return type >= CURAND_RNG_QUASI_SOBOL32; }
    bool sobol64() const { return type == CURAND_RNG_QUASI_SOBOL64 || type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64; }
    bool scrambled() const
    {
        return type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 || type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
    }

    void reset()
    {
        position = 0;
        xorwow.clear();
        philox.clear();
        mrg.clear();
        mt.reset();
    }

    size_t lanes() const
    {
        switch (type) {
        case CURAND_RNG_PSEUDO_XORWOW:
            return 4096;
        case CURAND_RNG_PSEUDO_PHILOX4_32_10:
            return 65536;
        case CURAND_RNG_PSEUDO_MRG32K3A:
            return ordering == CURAND_ORDERING_PSEUDO_LEGACY ? 4096 : 81920;
        default:
            return 1;
        }
    }

    size_t group() const { return type == CURAND_RNG_PSEUDO_PHILOX4_32_10 ? 4 : 1; }

    //! How many of the first `count` results lane k makes
    unsigned long long made_by(size_t k, unsigned long long count) const
    {
        const unsigned long long g = group(), round = g * lanes();
        const unsigned long long rest = count % round, start = k * g;
        return count / round * g + (rest > start ? std::min(rest - start, g) : 0);
    }

    void start()
    {
        const size_t n = lanes();
        switch (type) {
        case CURAND_RNG_PSEUDO_XORWOW:
            xorwow.resize(n);
            #pragma omp parallel for
            for (size_t k = 0; k < n; ++k) {
                if (ordering == CURAND_ORDERING_PSEUDO_SEEDED)
                    curand_init(seed ^ (k * 0x9E3779B97F4A7C15ull), 0, made_by(k, offset), &xorwow[k]);
                else
                    curand_init(seed, k, made_by(k, offset), &xorwow[k]);
            }
            break;
        case CURAND_RNG_PSEUDO_PHILOX4_32_10:
            philox.resize(n);
            #pragma omp parallel for
            for (size_t k = 0; k < n; ++k)
                curand_init(seed, k, made_by(k, offset), &philox[k]);
            break;
        case CURAND_RNG_PSEUDO_MRG32K3A:
            mrg.resize(n);
            #pragma omp parallel for
            for (size_t k = 0; k < n; ++k)
                curand_init(seed, k, made_by(k, offset), &mrg[k]);
            break;
        default:   // MT19937 and MTGP32: one std::mt19937 (not CUDA's numbers)
            mt = std::make_unique<std::mt19937>(unsigned(seed ^ (seed >> 32)));
            mt->discard(offset);
            break;
        }
    }

    //! The next n raw values in the generator's order (32 bits each,
    //! MRG32k3a's in [1, m1]): store(i, value) for each, in parallel. The
    //! lanes run round by round, each thread a contiguous range of them, so
    //! that a round's values are written next to each other: lane by lane,
    //! each value went to a cache line written a whole pass before, and
    //! XORWOW's took about as long on 16 threads as on one.
    template <typename Store>
    void each_raw(size_t n, Store store)
    {
        if (xorwow.empty() && philox.empty() && mrg.empty() && !mt)
            start();
        if (mt) {
            for (size_t i = 0; i < n; ++i)
                store(i, unsigned((*mt)()));
        } else if (!xorwow.empty()) {
            in_rounds(n, xorwow, [](curandStateXORWOW *s) { return curand(s); }, store);
        } else if (!philox.empty()) {
            in_rounds(n, philox, [](curandStatePhilox4_32_10 *s) { return curand(s); }, store);
        } else {
            in_rounds(n, mrg, [](curandStateMRG32k3a *s) { return mrg32k3a_next(s); }, store);
        }
        position += n;
    }

    template <typename State, typename Next, typename Store>
    void in_rounds(size_t n, std::vector<State> &states, Next next, Store store)
    {
        // Results past 2^64 don't wrap around the index
        using index = unsigned __int128;
        const size_t g = group(), l = lanes();
        const index first = index(offset) + position, last = first + n, round = index(g) * l;
        const index begin = first / round, end = (last + round - 1) / round;
        #pragma omp parallel if (n >= 32768 && !omp_in_parallel())
        {
            const size_t t = size_t(omp_get_thread_num()), threads = size_t(omp_get_num_threads());
            const size_t k0 = l * t / threads, k1 = l * (t + 1) / threads;
            for (index r = begin; r < end; ++r) {
                const index base = r * round;
                if (base >= first && base + round <= last) {
                    const size_t at = size_t(base - first);
                    for (size_t k = k0; k < k1; ++k)
                        for (size_t j = 0; j < g; ++j)
                            store(at + k * g + j, next(&states[k]));
                } else {
                    // The first and last rounds: a lane skips the results
                    // made before, and stops at the last
                    for (size_t k = k0; k < k1; ++k)
                        for (size_t j = 0; j < g; ++j) {
                            const index m = base + k * g + j;
                            if (m >= first && m < last)
                                store(size_t(m - first), next(&states[k]));
                        }
                }
            }
        }
    }

    //! The next n raw values, into out
    void raw(unsigned int *out, size_t n)
    {
        each_raw(n, [out](size_t i, unsigned int r) { out[i] = r; });
    }

    //! The next n raw values, into the bytes of an output of another type,
    //! which conversions in place then read with raw_at: a pair of them
    //! takes the bytes of the double or of the two floats made from it
    void raw_bytes(void *out, size_t n)
    {
        each_raw(n, [out](size_t i, unsigned int r) { std::memcpy(static_cast<char *>(out) + 4 * i, &r, 4); });
    }

    static unsigned int raw_at(const void *out, size_t i)
    {
        unsigned int r;
        std::memcpy(&r, static_cast<const char *>(out) + 4 * i, 4);
        return r;
    }

    //! f(i) for every i below n, in parallel
    template <typename F>
    static void each(size_t n, F f)
    {
        #pragma omp parallel for schedule(static) if (n >= 32768 && !omp_in_parallel())
        for (size_t i = 0; i < n; ++i)
            f(i);
    }

    unsigned int to_uint(unsigned int r) const
    {
        return type == CURAND_RNG_PSEUDO_MRG32K3A ? unsigned(double(r - 1) * (4294967295.0 / double(mrg_m1 - 1))) : r;
    }

    float to_float(unsigned int r) const
    {
        return type == CURAND_RNG_PSEUDO_MRG32K3A ? float(double(r) * (1.0 / double(mrg_m1 + 1))) : uniform(r);
    }

    double to_double(unsigned int r1, unsigned int r2) const
    {
        if (type == CURAND_RNG_PSEUDO_MRG32K3A)
            return double(r1) * (1.0 / double(mrg_m1 + 1));
        return uniform_double(r1, r2);
    }

    //! Quasirandom: n / dimensions points of each dimension, dimension after
    //! dimension, from point offset + position; values(dimension, x) stores
    template <typename F>
    curandStatus_t points(size_t n, F store)
    {
        if (n % dimensions != 0)
            return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
        const size_t count = n / dimensions;
        const unsigned long long first = offset + position;
        // Each dimension in chunks, each from its own point (curand_init
        // skips ahead), so that a few dimensions run in parallel too
        const size_t chunks = count >= 65536 && !omp_in_parallel()
                                  ? std::max<size_t>(1, (size_t(omp_get_max_threads()) * 4 + dimensions - 1) / dimensions)
                                  : 1;
        const size_t per_chunk = (count + chunks - 1) / chunks;
        if (sobol64()) {
            unsigned long long *constants = nullptr;
            curandDirectionVectors64_t *vectors = nullptr;
            curandGetDirectionVectors64(&vectors, scrambled() ? CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6
                                                              : CURAND_DIRECTION_VECTORS_64_JOEKUO6);
            if (scrambled())
                curandGetScrambleConstants64(&constants);
            #pragma omp parallel for collapse(2) if (!omp_in_parallel())
            for (unsigned d = 0; d < dimensions; ++d) {
                for (size_t c = 0; c < chunks; ++c) {
                    const size_t lo = c * per_chunk, hi = std::min(count, lo + per_chunk);
                    if (lo >= hi)
                        continue;
                    curandStateScrambledSobol64 state;
                    curand_init(vectors[d], constants ? constants[d] : 0, first + lo, &state);
                    for (size_t i = lo; i < hi; ++i)
                        store(d * count + i, curand(&state));
                }
            }
        } else {
            unsigned int *constants = nullptr;
            curandDirectionVectors32_t *vectors = nullptr;
            curandGetDirectionVectors32(&vectors, scrambled() ? CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
                                                              : CURAND_DIRECTION_VECTORS_32_JOEKUO6);
            if (scrambled())
                curandGetScrambleConstants32(&constants);
            #pragma omp parallel for collapse(2) if (!omp_in_parallel())
            for (unsigned d = 0; d < dimensions; ++d) {
                for (size_t c = 0; c < chunks; ++c) {
                    const size_t lo = c * per_chunk, hi = std::min(count, lo + per_chunk);
                    if (lo >= hi)
                        continue;
                    curandStateScrambledSobol32 state;
                    curand_init(vectors[d], constants ? constants[d] : 0, unsigned(first + lo), &state);
                    for (size_t i = lo; i < hi; ++i)
                        store(d * count + i, (unsigned long long)curand(&state));
                }
            }
        }
        position += count;
        return CURAND_STATUS_SUCCESS;
    }
};

namespace {

bool known(curandRngType_t type)
{
    switch (type) {
    case CURAND_RNG_PSEUDO_DEFAULT:
    case CURAND_RNG_PSEUDO_XORWOW:
    case CURAND_RNG_PSEUDO_MRG32K3A:
    case CURAND_RNG_PSEUDO_MTGP32:
    case CURAND_RNG_PSEUDO_MT19937:
    case CURAND_RNG_PSEUDO_PHILOX4_32_10:
    case CURAND_RNG_QUASI_DEFAULT:
    case CURAND_RNG_QUASI_SOBOL32:
    case CURAND_RNG_QUASI_SCRAMBLED_SOBOL32:
    case CURAND_RNG_QUASI_SOBOL64:
    case CURAND_RNG_QUASI_SCRAMBLED_SOBOL64:
        return true;
    default:
        return false;
    }
}

curandStatus_t create(curandGenerator_t *generator, curandRngType_t type, bool host)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    if (!known(type))
        return CURAND_STATUS_TYPE_ERROR;
    if (type == CURAND_RNG_PSEUDO_DEFAULT)
        type = CURAND_RNG_PSEUDO_XORWOW;
    if (type == CURAND_RNG_QUASI_DEFAULT)
        type = CURAND_RNG_QUASI_SOBOL32;
    auto *g = new curandGenerator_st{};
    g->type = type;
    g->host = host;
    g->ordering = g->quasi() ? CURAND_ORDERING_QUASI_DEFAULT : CURAND_ORDERING_PSEUDO_DEFAULT;
    *generator = g;
    return CURAND_STATUS_SUCCESS;
}

//! The largest mean of a Poisson distribution whose values (to 20 standard
//! deviations) fit in 32 bits
constexpr double max_poisson_lambda = 4.29e9;

//! Fills the cumulative distribution of a Poisson distribution of mean lambda
//! (at most max_poisson_lambda) from the first value whose probability isn't
//! negligible
void poisson_table(double lambda, std::vector<double> &cdf, unsigned int &shift)
{
    const double spread = 20 * std::sqrt(lambda) + 20;
    shift = unsigned(std::max(0.0, std::floor(lambda - spread)));
    const double last = std::min(std::ceil(lambda + spread), 4294967295.0);
    cdf.clear();
    double total = 0;
    for (double k = shift; k <= last; ++k) {
        total += std::exp(k * std::log(lambda) - lambda - std::lgamma(k + 1.0));
        cdf.push_back(total);
    }
    for (double &c : cdf)
        c /= total;
    cdf.back() = 1;
}

}

extern "C" {

curandStatus_t curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type)
{
    return create(generator, rng_type, false);
}

curandStatus_t curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type)
{
    return create(generator, rng_type, true);
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    delete generator;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetVersion(int *version)
{
    if (version == nullptr)
        return CURAND_STATUS_OUT_OF_RANGE;
    *version = CURAND_VERSION;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetProperty(libraryPropertyType type, int *value)
{
    if (value == nullptr)
        return CURAND_STATUS_OUT_OF_RANGE;
    switch (type) {
    case MAJOR_VERSION: *value = CURAND_VER_MAJOR; break;
    case MINOR_VERSION: *value = CURAND_VER_MINOR; break;
    case PATCH_LEVEL:   *value = CURAND_VER_PATCH; break;
    default:            return CURAND_STATUS_OUT_OF_RANGE;
    }
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    generator->stream = stream;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    if (generator->quasi())
        return CURAND_STATUS_TYPE_ERROR;
    generator->seed = seed;
    generator->reset();
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    // As in cuRAND, which can't skip ahead in these
    if (generator->type == CURAND_RNG_PSEUDO_MTGP32 || generator->type == CURAND_RNG_PSEUDO_MT19937)
        return CURAND_STATUS_TYPE_ERROR;
    generator->offset = offset;
    generator->reset();
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    if (generator->quasi() != (order == CURAND_ORDERING_QUASI_DEFAULT))
        return CURAND_STATUS_OUT_OF_RANGE;
    if (order == CURAND_ORDERING_PSEUDO_DYNAMIC && generator->host)
        return CURAND_STATUS_OUT_OF_RANGE;
    if (order == CURAND_ORDERING_PSEUDO_SEEDED && generator->type != CURAND_RNG_PSEUDO_XORWOW)
        return CURAND_STATUS_OUT_OF_RANGE;
    if (order < CURAND_ORDERING_PSEUDO_BEST || order > CURAND_ORDERING_QUASI_DEFAULT ||
        (order > CURAND_ORDERING_PSEUDO_DYNAMIC && order != CURAND_ORDERING_QUASI_DEFAULT))
        return CURAND_STATUS_OUT_OF_RANGE;
    generator->ordering = order;
    generator->reset();
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    if (!generator->quasi())
        return CURAND_STATUS_TYPE_ERROR;
    if (num_dimensions < 1 || num_dimensions > sobol_dimensions)
        return CURAND_STATUS_OUT_OF_RANGE;
    generator->dimensions = num_dimensions;
    generator->reset();
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int *output_ptr, size_t num)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    if (generator->sobol64())
        return CURAND_STATUS_TYPE_ERROR;
    if (generator->quasi())
        return generator->points(num, [&](size_t i, unsigned long long x) { output_ptr[i] = unsigned(x); });
    generator->each_raw(num, [&](size_t i, unsigned int r) { output_ptr[i] = generator->to_uint(r); });
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLongLong(curandGenerator_t generator, unsigned long long *output_ptr, size_t num)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    if (!generator->sobol64())
        return CURAND_STATUS_TYPE_ERROR;
    return generator->points(num, [&](size_t i, unsigned long long x) { output_ptr[i] = x; });
}

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float *output_ptr, size_t num)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    if (generator->quasi()) {
        const bool wide = generator->sobol64();
        return generator->points(num, [&](size_t i, unsigned long long x) {
            output_ptr[i] = uniform(wide ? unsigned(x >> 32) : unsigned(x));
        });
    }
    generator->each_raw(num, [&](size_t i, unsigned int r) { output_ptr[i] = generator->to_float(r); });
    return CURAND_STATUS_SUCCESS;
}

//! Doubles of pseudorandom generators: two raw values each, of which
//! MRG32k3a uses one
static curandStatus_t uniform_doubles(curandGenerator_t generator, size_t num, double *out)
{
    if (generator->type == CURAND_RNG_PSEUDO_MRG32K3A) {
        generator->each_raw(num, [&](size_t i, unsigned int r) { out[i] = generator->to_double(r, 0); });
    } else {
        // Raw value 2i and 2i + 1 take the bytes of double i
        generator->raw_bytes(out, 2 * num);
        curandGenerator_st::each(num, [&](size_t i) {
            const unsigned int x = curandGenerator_st::raw_at(out, 2 * i), y = curandGenerator_st::raw_at(out, 2 * i + 1);
            out[i] = generator->to_double(x, y);
        });
    }
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double *output_ptr, size_t num)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    if (generator->quasi()) {
        const bool wide = generator->sobol64();
        return generator->points(num, [&](size_t i, unsigned long long x) {
            output_ptr[i] = wide ? uniform_double(x) : double(x) * 0x1p-32 + 0x1p-33;
        });
    }
    return uniform_doubles(generator, num, output_ptr);
}

}

//! Normals: by pairs of uniforms through the Box-Muller transform
//! (pseudorandom, n even), or by inversion (quasirandom)
template <typename T>
static curandStatus_t normals(curandGenerator_t generator, T *out, size_t n, T mean, T stddev, bool log_normal)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    auto finish = [&](double z) {
        const T value = T(mean + stddev * T(z));
        return log_normal ? T(std::exp(value)) : value;
    };
    if (generator->quasi()) {
        const bool wide = generator->sobol64();
        return generator->points(n, [&](size_t i, unsigned long long x) {
            out[i] = finish(normal_quantile(wide ? uniform_double(x) : double(x) * 0x1p-32 + 0x1p-33));
        });
    }
    if (n % 2 != 0)
        return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
    auto pair = [&](double u1, double u2, T *to) {
        const double s = std::sqrt(-2.0 * std::log(u1)), v = 2 * std::numbers::pi * u2;
        to[0] = finish(std::sin(v) * s);
        to[1] = finish(std::cos(v) * s);
    };
    if constexpr (std::is_same_v<T, float>) {
        // In place: raw values i and i + 1 become normals i and i + 1, by
        // float's Box-Muller transform, or MRG32k3a's from its doubles
        generator->raw_bytes(out, n);
        const bool mrg = generator->type == CURAND_RNG_PSEUDO_MRG32K3A;
        curandGenerator_st::each(n / 2, [&](size_t p) {
            const size_t i = 2 * p;
            const unsigned int x = curandGenerator_st::raw_at(out, i), y = curandGenerator_st::raw_at(out, i + 1);
            if (mrg) {
                pair(generator->to_double(x, 0), generator->to_double(y, 0), out + i);
            } else {
                const float2 z = box_muller(x, y);
                out[i] = finish(z.x);
                out[i + 1] = finish(z.y);
            }
        });
    } else {
        uniform_doubles(generator, n, out);
        curandGenerator_st::each(n / 2, [&](size_t p) { pair(out[2 * p], out[2 * p + 1], out + 2 * p); });
    }
    return CURAND_STATUS_SUCCESS;
}

extern "C" {

curandStatus_t curandGenerateNormal(curandGenerator_t generator, float *output_ptr, size_t n, float mean, float stddev)
{
    return normals(generator, output_ptr, n, mean, stddev, false);
}

curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double *output_ptr, size_t n, double mean,
                                          double stddev)
{
    return normals(generator, output_ptr, n, mean, stddev, false);
}

curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float *output_ptr, size_t n, float mean,
                                       float stddev)
{
    return normals(generator, output_ptr, n, mean, stddev, true);
}

curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double *output_ptr, size_t n,
                                             double mean, double stddev)
{
    return normals(generator, output_ptr, n, mean, stddev, true);
}

//! Poisson values: the inverse of the distribution at uniform doubles
curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int *output_ptr, size_t n, double lambda)
{
    if (generator == nullptr)
        return CURAND_STATUS_NOT_INITIALIZED;
    if (!(lambda > 0 && lambda <= max_poisson_lambda))
        return CURAND_STATUS_OUT_OF_RANGE;
    std::vector<double> cdf;
    unsigned int shift;
    poisson_table(lambda, cdf, shift);
    auto value = [&](double u) {
        const size_t i = size_t(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        return shift + unsigned(std::min(i, cdf.size() - 1));
    };
    if (generator->quasi()) {
        const bool wide = generator->sobol64();
        return generator->points(n, [&](size_t i, unsigned long long x) {
            output_ptr[i] = value(wide ? uniform_double(x) : double(x) * 0x1p-32 + 0x1p-33);
        });
    }
    std::vector<double> u(n);
    uniform_doubles(generator, n, u.data());
    curandGenerator_st::each(n, [&](size_t i) { output_ptr[i] = value(u[i]); });
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateSeeds(curandGenerator_t generator)
{
    return generator == nullptr ? CURAND_STATUS_NOT_INITIALIZED : CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t *vectors[], curandDirectionVectorSet_t set)
{
    static std::once_flag once[2];
    static std::vector<std::array<unsigned int, 32>> tables[2];
    if (vectors == nullptr)
        return CURAND_STATUS_OUT_OF_RANGE;
    if (set != CURAND_DIRECTION_VECTORS_32_JOEKUO6 && set != CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
        return CURAND_STATUS_OUT_OF_RANGE;
    const int i = set == CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
    std::call_once(once[i], [i] {
        tables[i] = i ? scrambled_vectors<unsigned int, 32>() : direction_vectors<unsigned int, 32>();
    });
    *vectors = reinterpret_cast<curandDirectionVectors32_t *>(tables[i].data());
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t *vectors[], curandDirectionVectorSet_t set)
{
    static std::once_flag once[2];
    static std::vector<std::array<unsigned long long, 64>> tables[2];
    if (vectors == nullptr)
        return CURAND_STATUS_OUT_OF_RANGE;
    if (set != CURAND_DIRECTION_VECTORS_64_JOEKUO6 && set != CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)
        return CURAND_STATUS_OUT_OF_RANGE;
    const int i = set == CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
    std::call_once(once[i], [i] {
        tables[i] = i ? scrambled_vectors<unsigned long long, 64>() : direction_vectors<unsigned long long, 64>();
    });
    *vectors = reinterpret_cast<curandDirectionVectors64_t *>(tables[i].data());
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetScrambleConstants32(unsigned int **constants)
{
    static const std::vector<unsigned int> table = [] {
        scrambler random;
        for (int i = 0; i < 1000; ++i)
            random.next32();                  // apart from the scrambled vectors' numbers
        std::vector<unsigned int> values(sobol_dimensions);
        for (unsigned int &v : values)
            v = random.next32();
        return values;
    }();
    if (constants == nullptr)
        return CURAND_STATUS_OUT_OF_RANGE;
    *constants = const_cast<unsigned int *>(table.data());
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetScrambleConstants64(unsigned long long **constants)
{
    static const std::vector<unsigned long long> table = [] {
        scrambler random;
        for (int i = 0; i < 1000; ++i)
            random.next32();
        std::vector<unsigned long long> values(sobol_dimensions);
        for (unsigned long long &v : values)
            v = random.next64();
        return values;
    }();
    if (constants == nullptr)
        return CURAND_STATUS_OUT_OF_RANGE;
    *constants = const_cast<unsigned long long *>(table.data());
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t *discrete_distribution)
{
    if (discrete_distribution == nullptr || !(lambda > 0 && lambda <= max_poisson_lambda))
        return CURAND_STATUS_OUT_OF_RANGE;
    std::vector<double> cdf;
    unsigned int shift;
    poisson_table(lambda, cdf, shift);
    auto *d = new curandDiscreteDistribution_st{};
    d->probability = new double[cdf.size()];
    std::copy(cdf.begin(), cdf.end(), d->probability);
    d->length = unsigned(cdf.size());
    d->shift = shift;
    d->lambda = lambda;
    *discrete_distribution = d;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution)
{
    if (discrete_distribution == nullptr)
        return CURAND_STATUS_OUT_OF_RANGE;
    delete[] discrete_distribution->probability;
    delete discrete_distribution;
    return CURAND_STATUS_SUCCESS;
}

}
