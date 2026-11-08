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
// AMD's wave matrix builtins (Clang only): every WMMA of RDNA3 and RDNA4 and
// every SWMMAC, against an exact host product. The fragments are packed
// with this file's own layouts, written from the register and bit formulas
// of AMD's Matrix Instruction Calculator (and rocjitsu's for the 4-bit
// SWMMAC with K = 64), not from amdgcn_wmma.h. Then the arithmetic that
// small integers don't show: 16-bit results rounded at each block of 2
// (RDNA3) or 4 (RDNA4) products, bfloat16's truncation, saturation at each
// stage with clamp, RDNA3's copies of A and B in each half of the wave,
// and C's other halves with opsel.
//

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef short v8s __attribute__((ext_vector_type(8)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

//
// Each lane's bytes of A, B, C and the index; and the kernel that calls one
// builtin with them
//

struct fragments {
    unsigned char a[32][32] = {}, b[32][32] = {}, c[32][32] = {}, d[32][32] = {};
    int index[32] = {};
};

template <typename T>
__device__ T load(const unsigned char *bytes)
{
    T value;
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

template <typename Op>
__global__ void call(fragments *f)
{
    const unsigned l = threadIdx.x;
    const auto d = Op::call(f->a[l], f->b[l], f->c[l], f->index[l]);
    std::memcpy(f->d[l], &d, sizeof d);
}

#define OP(name, expr)                                                                         \
    struct name {                                                                              \
        static __device__ auto call(const unsigned char *a, const unsigned char *b,            \
                                    const unsigned char *c, int index)                         \
        {                                                                                      \
            (void)index;                                                                       \
            return expr;                                                                       \
        }                                                                                      \
    };

// RDNA3
OP(f32_f16, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(load<v16h>(a), load<v16h>(b), load<v8f>(c)))
OP(f32_bf16, __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(load<v16s>(a), load<v16s>(b), load<v8f>(c)))
OP(f16_lo, __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(load<v16h>(a), load<v16h>(b), load<v16h>(c), false))
OP(f16_hi, __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(load<v16h>(a), load<v16h>(b), load<v16h>(c), true))
OP(f16_tied, __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32(load<v16h>(a), load<v16h>(b), load<v16h>(c), true))
OP(bf16_lo, __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(load<v16s>(a), load<v16s>(b), load<v16s>(c), false))
OP(bf16_tied, __builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w32(load<v16s>(a), load<v16s>(b), load<v16s>(c), true))
OP(iu8_ss, __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, load<v4i>(a), true, load<v4i>(b), load<v8i>(c), false))
OP(iu8_uu, __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, load<v4i>(a), false, load<v4i>(b), load<v8i>(c), false))
OP(iu8_clamp, __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, load<v4i>(a), true, load<v4i>(b), load<v8i>(c), true))
OP(iu4_su, __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, load<v2i>(a), false, load<v2i>(b), load<v8i>(c), false))
// RDNA4
OP(f32_f16_12, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(load<v8h>(a), load<v8h>(b), load<v8f>(c)))
OP(f32_bf16_12, __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(load<v8s>(a), load<v8s>(b), load<v8f>(c)))
OP(f16_12, __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(load<v8h>(a), load<v8h>(b), load<v8h>(c)))
OP(bf16_12, __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(load<v8s>(a), load<v8s>(b), load<v8s>(c)))
OP(fp8_fp8, __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(load<v2i>(a), load<v2i>(b), load<v8f>(c)))
OP(fp8_bf8, __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(load<v2i>(a), load<v2i>(b), load<v8f>(c)))
OP(bf8_fp8, __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(load<v2i>(a), load<v2i>(b), load<v8f>(c)))
OP(bf8_bf8, __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(load<v2i>(a), load<v2i>(b), load<v8f>(c)))
OP(iu8_12, __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, load<v2i>(a), false, load<v2i>(b), load<v8i>(c), false))
OP(iu8_12_clamp, __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, load<v2i>(a), true, load<v2i>(b), load<v8i>(c), true))
OP(iu4_12, __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(false, load<int>(a), true, load<int>(b), load<v8i>(c), false))
OP(iu4_k32, __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, load<v2i>(a), true, load<v2i>(b), load<v8i>(c), false))
// SWMMAC
OP(s_f32_f16, __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(load<v8h>(a), load<v16h>(b), load<v8f>(c), index))
OP(s_f32_bf16, __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(load<v8s>(a), load<v16s>(b), load<v8f>(c), index))
OP(s_f16, __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32(load<v8h>(a), load<v16h>(b), load<v8h>(c), index))
OP(s_bf16, __builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32(load<v8s>(a), load<v16s>(b), load<v8s>(c), index))
OP(s_fp8_fp8, __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(load<v2i>(a), load<v4i>(b), load<v8f>(c), index))
OP(s_fp8_bf8, __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(load<v2i>(a), load<v4i>(b), load<v8f>(c), index))
OP(s_bf8_fp8, __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(load<v2i>(a), load<v4i>(b), load<v8f>(c), index))
OP(s_bf8_bf8, __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(load<v2i>(a), load<v4i>(b), load<v8f>(c), index))
OP(s_iu8, __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, load<v2i>(a), true, load<v4i>(b), load<v8i>(c), index, false))
OP(s_iu4_k32, __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32(false, load<int>(a), true, load<v2i>(b), load<v8i>(c), index, false))
OP(s_iu4_k64, __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, load<v2i>(a), false, load<v4i>(b), load<v8i>(c), index, false))
// The names as functions
OP(f32_f16_function, (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)(load<v16h>(a), load<v16h>(b), load<v8f>(c)))

//
// Element types: their bits, and values to and from them
//

enum class type { f16, bf16, fp8, bf8, f32, i8, u8, i4, u4, i32 };

static unsigned bits_of(type t)
{
    switch (t) {
    case type::f16: case type::bf16: return 16;
    case type::fp8: case type::bf8: case type::i8: case type::u8: return 8;
    case type::i4: case type::u4: return 4;
    default: return 32;
    }
}

//! The bits of a small integer v, exact in every type
static uint32_t encode(type t, double v)
{
    switch (t) {
    case type::f16: { _Float16 h = _Float16(v); uint16_t b; std::memcpy(&b, &h, 2); return b; }
    case type::bf16: return std::bit_cast<uint32_t>(float(v)) >> 16;
    case type::f32: return std::bit_cast<uint32_t>(float(v));
    case type::fp8: case type::bf8: {
        // sign, exponent and mantissa of a small integer or half-integer
        const bool e5 = t == type::bf8;
        const int bias = e5 ? 15 : 7, m = e5 ? 2 : 3;
        if (v == 0) return 0;
        const uint32_t sign = v < 0 ? 0x80 : 0;
        int e;
        const double f = std::frexp(std::fabs(v), &e);          // [0.5, 1)
        const uint32_t mantissa = uint32_t(std::lround((f * 2 - 1) * (1 << m)));
        return sign | uint32_t(e - 1 + bias) << m | mantissa;
    }
    default: return uint32_t(int64_t(v)) & (bits_of(t) == 32 ? 0xffffffffu : (1u << bits_of(t)) - 1);
    }
}

static double decode(type t, uint32_t b)
{
    switch (t) {
    case type::f16: { uint16_t s = uint16_t(b); _Float16 h; std::memcpy(&h, &s, 2); return double(h); }
    case type::bf16: return double(std::bit_cast<float>(b << 16));
    case type::f32: return double(std::bit_cast<float>(b));
    case type::i32: return double(int32_t(b));
    default: return 0;
    }
}

//! Writes bits at element index (of bits_of(t)) of a lane's bytes
static void put(unsigned char *bytes, type t, unsigned index, uint32_t value)
{
    const unsigned bits = bits_of(t);
    if (bits == 4) {
        unsigned char &byte = bytes[index / 2];
        byte = (byte & ~(15 << (4 * (index % 2)))) | (value & 15) << (4 * (index % 2));
    } else
        std::memcpy(bytes + index * bits / 8, &value, bits / 8);
}

static uint32_t get(const unsigned char *bytes, type t, unsigned index)
{
    uint32_t value = 0;
    std::memcpy(&value, bytes + index * bits_of(t) / 8, bits_of(t) / 8);
    return value;
}

//
// Layouts, from the calculator's formulas: the register, the bits in it,
// and the lane, as an element index in units of the type
//

struct place {
    unsigned lane, index;
};

//! RDNA3: A's row i (B's column i), element k, in lane i and i + 16
static std::vector<place> gfx11_input(unsigned i, unsigned k)
{
    return {{i, k}, {i + 16, k}};
}

//! RDNA3's C and D: register i / 2, the lanes of row i's parity; 16-bit in
//! the half that opsel selects
static place gfx11_output(type t, unsigned i, unsigned j, bool opsel)
{
    const unsigned reg = i / 2, lane = 16 * (i % 2) + j;
    return bits_of(t) == 16 ? place{lane, 2 * reg + opsel} : place{lane, reg};
}

//! RDNA4's dense A or B (row or column i), element k, for a K of 16 or 32
static place gfx12_input(type t, unsigned K, unsigned i, unsigned k)
{
    switch (bits_of(t)) {
    case 16: return {16 * (k / 4 % 2) + i, 2 * (2 * (k / 8) + k / 2 % 2) + k % 2};
    case 8:  return {16 * (k / 8) + i, 4 * (k / 4 % 2) + k % 4};
    default:
        if (K == 16) return {16 * (k / 8) + i, k % 8};
        return {16 * (k / 16) + i, 8 * (k / 8 % 2) + k % 8};
    }
}

static place gfx12_output(unsigned i, unsigned j)
{
    return {16 * (i / 8) + j, i % 8};
}

//! SWMMAC's sparse A: value v (0 or 1) of row i's group of four at k (a
//! multiple of 4), and the bits of the index that give its place
struct sparse_place {
    unsigned lane, index, index_bit;
};

static sparse_place sparse_a(type t, unsigned K, unsigned i, unsigned k, unsigned v)
{
    switch (bits_of(t)) {
    case 16: {
        const unsigned reg = 2 * (k / 16) + k / 4 % 2, lane = 16 * (k / 8 % 2) + i;
        return {lane, 2 * reg + v, 8 * (k / 16) + 4 * (k / 4 % 2) + 2 * v};
    }
    case 8: {
        const unsigned reg = k / 8 % 2, lane = 16 * (k / 16) + i;
        return {lane, 4 * reg + 2 * (k / 4 % 2) + v, 4 * (k / 4 % 4) + 2 * v};
    }
    default:
        if (K == 32)
            return {16 * (k / 16) + i, 2 * (k / 4 % 4) + v, 4 * (k / 4 % 4) + 2 * v};
        {
            // rocjitsu's: the calculator puts k / 32 in the lane instead
            const unsigned g = k / 4, local = 8 * (g / 8) + 2 * (g % 4) + v;
            return {16 * (g / 4 % 2) + i, local, 2 * local};
        }
    }
}

//! SWMMAC's dense B, column j, element k
static place sparse_b(type t, unsigned K, unsigned j, unsigned k)
{
    switch (bits_of(t)) {
    case 16: return {16 * (k / 8 % 2) + j, 2 * (4 * (k / 16) + k / 2 % 4) + k % 2};
    case 8:  return {16 * (k / 16) + j, 4 * (k / 4 % 4) + k % 4};
    default:
        if (K == 32) return {16 * (k / 16) + j, 8 * (k / 8 % 2) + k % 8};
        return {16 * (k / 16 % 2) + j, 16 * (k / 32) + k % 16};   // rocjitsu's
    }
}

//
// The checks
//

enum class arch { gfx11, gfx12, sparse };

struct test_case {
    const char *name;
    void (*kernel)(fragments *);
    arch layout;
    type ta, tb, tc;
    unsigned K;
    bool opsel = false;
};

//! Values for the matrices: small integers that every type holds exactly
static double sample(type t, std::mt19937 &random)
{
    switch (t) {
    case type::i8:  return int(random() % 256) - 128;
    case type::u8:  return random() % 256;
    case type::i4:  return int(random() % 16) - 8;
    case type::u4:  return random() % 16;
    case type::i32: return int(random() % 2001) - 1000;
    case type::f32: return int(random() % 17) - 8;
    default:        return int(random() % 7) - 3;
    }
}

static void run(const test_case &tc, std::mt19937 &random)
{
    const unsigned K = tc.K;
    std::vector<double> A(16 * K), B(K * 16), C(256);
    // Sparse A: two values of each group of four, at random places
    std::vector<unsigned> chosen(16 * K / 2);
    for (unsigned i = 0; i < 16; ++i)
        for (unsigned k = 0; k < K; ++k)
            A[i * K + k] = tc.layout == arch::sparse ? 0 : sample(tc.ta, random);
    if (tc.layout == arch::sparse)
        for (unsigned i = 0; i < 16; ++i)
            for (unsigned g = 0; g < K / 4; ++g) {
                unsigned first = random() % 4, second = random() % 4;
                while (second == first)
                    second = random() % 4;
                if (first > second)
                    std::swap(first, second);
                chosen[i * K / 2 + 2 * g] = first;
                chosen[i * K / 2 + 2 * g + 1] = second;
                A[i * K + 4 * g + first] = sample(tc.ta, random);
                A[i * K + 4 * g + second] = sample(tc.ta, random);
            }
    for (double &b : B)
        b = sample(tc.tb, random);
    for (double &c : C)
        c = sample(tc.tc, random);

    static fragments f;
    f = fragments{};
    for (unsigned i = 0; i < 16; ++i)
        for (unsigned k = 0; k < K; ++k) {
            switch (tc.layout) {
            case arch::gfx11:
                for (place p : gfx11_input(i, k)) {
                    put(f.a[p.lane], tc.ta, p.index, encode(tc.ta, A[i * K + k]));
                    put(f.b[p.lane], tc.tb, p.index, encode(tc.tb, B[k * 16 + i]));
                }
                break;
            case arch::gfx12: {
                const place pa = gfx12_input(tc.ta, K, i, k), pb = gfx12_input(tc.tb, K, i, k);
                put(f.a[pa.lane], tc.ta, pa.index, encode(tc.ta, A[i * K + k]));
                put(f.b[pb.lane], tc.tb, pb.index, encode(tc.tb, B[k * 16 + i]));
                break;
            }
            case arch::sparse: {
                const place pb = sparse_b(tc.tb, K, i, k);
                put(f.b[pb.lane], tc.tb, pb.index, encode(tc.tb, B[k * 16 + i]));
                if (k % 4 == 0)
                    for (unsigned v = 0; v < 2; ++v) {
                        const unsigned at = chosen[i * K / 2 + k / 2 + v];
                        const sparse_place pa = sparse_a(tc.ta, K, i, k, v);
                        put(f.a[pa.lane], tc.ta, pa.index, encode(tc.ta, A[i * K + k + at]));
                        f.index[pa.lane] |= int(at << pa.index_bit);
                    }
                break;
            }
            }
        }
    // C, and on RDNA3 with a 16-bit C, other values in the other halves
    for (unsigned l = 0; l < 32 && tc.layout == arch::gfx11 && bits_of(tc.tc) == 16; ++l)
        for (unsigned e = 0; e < 16; ++e)
            put(f.c[l], tc.tc, e, encode(tc.tc, 100 + e));
    for (unsigned i = 0; i < 16; ++i)
        for (unsigned j = 0; j < 16; ++j) {
            const place p = tc.layout == arch::gfx11 ? gfx11_output(tc.tc, i, j, tc.opsel) : gfx12_output(i, j);
            put(f.c[p.lane], tc.tc, p.index, encode(tc.tc, C[i * 16 + j]));
        }

    tc.kernel(&f);

    unsigned wrong = 0;
    for (unsigned i = 0; i < 16; ++i)
        for (unsigned j = 0; j < 16; ++j) {
            double expected = C[i * 16 + j];
            for (unsigned k = 0; k < K; ++k)
                expected += A[i * K + k] * B[k * 16 + j];
            const place p = tc.layout == arch::gfx11 ? gfx11_output(tc.tc, i, j, tc.opsel) : gfx12_output(i, j);
            const double got = decode(tc.tc, get(f.d[p.lane], tc.tc, p.index));
            if (got != expected && wrong++ < 3)
                std::printf("%s: D[%u][%u] = %g, expected %g\n", tc.name, i, j, got, expected);
        }
    if (tc.layout == arch::gfx11 && bits_of(tc.tc) == 16)
        for (unsigned l = 0; l < 32; ++l)
            for (unsigned r = 0; r < 8; ++r)
                if (decode(tc.tc, get(f.d[l], tc.tc, 2 * r + !tc.opsel)) != 100 + 2 * r + !tc.opsel && wrong++ < 3)
                    std::printf("%s: lane %u's other half %u changed\n", tc.name, l, r);
    if (wrong != 0) {
        std::printf("%s: %u wrong\n", tc.name, wrong);
        ++errors;
    }
}

template <typename Op>
static void launch(fragments *f)
{
    call<Op><<<1, 32>>>(f);
}

//! D for A and B of all ones in some columns of K, and C's value: the
//! rounding of 16-bit results and saturation, which depend on the blocks
template <typename Op>
static double ones(arch layout, type ta, type tc, unsigned K, double a, bool (*nonzero)(unsigned), double c,
                   double b = 1)
{
    static fragments f;
    f = fragments{};
    for (unsigned i = 0; i < 16; ++i)
        for (unsigned k = 0; k < K; ++k) {
            const double av = nonzero(k) ? a : 0;
            if (layout == arch::gfx11)
                for (place p : gfx11_input(i, k)) {
                    put(f.a[p.lane], ta, p.index, encode(ta, av));
                    put(f.b[p.lane], ta, p.index, encode(ta, b));
                }
            else {
                const place p = gfx12_input(ta, K, i, k);
                put(f.a[p.lane], ta, p.index, encode(ta, av));
                put(f.b[p.lane], ta, p.index, encode(ta, b));
            }
        }
    for (unsigned i = 0; i < 16; ++i)
        for (unsigned j = 0; j < 16; ++j) {
            const place p = layout == arch::gfx11 ? gfx11_output(tc, i, j, false) : gfx12_output(i, j);
            put(f.c[p.lane], tc, p.index, encode(tc, c));
        }
    call<Op><<<1, 32>>>(&f);
    const place p = layout == arch::gfx11 ? gfx11_output(tc, 5, 7, false) : gfx12_output(5, 7);
    return decode(tc, get(f.d[p.lane], tc, p.index));
}

int main()
{
    std::mt19937 random(2026);
    const test_case cases[] = {
        {"wmma_f32_16x16x16_f16_w32", launch<f32_f16>, arch::gfx11, type::f16, type::f16, type::f32, 16},
        {"wmma_f32_16x16x16_bf16_w32", launch<f32_bf16>, arch::gfx11, type::bf16, type::bf16, type::f32, 16},
        {"wmma_f16_16x16x16_f16_w32 (low)", launch<f16_lo>, arch::gfx11, type::f16, type::f16, type::f16, 16, false},
        {"wmma_f16_16x16x16_f16_w32 (high)", launch<f16_hi>, arch::gfx11, type::f16, type::f16, type::f16, 16, true},
        {"wmma_f16_16x16x16_f16_tied_w32", launch<f16_tied>, arch::gfx11, type::f16, type::f16, type::f16, 16, true},
        {"wmma_bf16_16x16x16_bf16_w32", launch<bf16_lo>, arch::gfx11, type::bf16, type::bf16, type::bf16, 16, false},
        {"wmma_bf16_16x16x16_bf16_tied_w32", launch<bf16_tied>, arch::gfx11, type::bf16, type::bf16, type::bf16, 16, true},
        {"wmma_i32_16x16x16_iu8_w32 (signed)", launch<iu8_ss>, arch::gfx11, type::i8, type::i8, type::i32, 16},
        {"wmma_i32_16x16x16_iu8_w32 (unsigned)", launch<iu8_uu>, arch::gfx11, type::u8, type::u8, type::i32, 16},
        {"wmma_i32_16x16x16_iu4_w32", launch<iu4_su>, arch::gfx11, type::i4, type::u4, type::i32, 16},
        {"wmma_f32_16x16x16_f16_w32_gfx12", launch<f32_f16_12>, arch::gfx12, type::f16, type::f16, type::f32, 16},
        {"wmma_f32_16x16x16_bf16_w32_gfx12", launch<f32_bf16_12>, arch::gfx12, type::bf16, type::bf16, type::f32, 16},
        {"wmma_f16_16x16x16_f16_w32_gfx12", launch<f16_12>, arch::gfx12, type::f16, type::f16, type::f16, 16},
        {"wmma_bf16_16x16x16_bf16_w32_gfx12", launch<bf16_12>, arch::gfx12, type::bf16, type::bf16, type::bf16, 16},
        {"wmma_f32_16x16x16_fp8_fp8_w32_gfx12", launch<fp8_fp8>, arch::gfx12, type::fp8, type::fp8, type::f32, 16},
        {"wmma_f32_16x16x16_fp8_bf8_w32_gfx12", launch<fp8_bf8>, arch::gfx12, type::fp8, type::bf8, type::f32, 16},
        {"wmma_f32_16x16x16_bf8_fp8_w32_gfx12", launch<bf8_fp8>, arch::gfx12, type::bf8, type::fp8, type::f32, 16},
        {"wmma_f32_16x16x16_bf8_bf8_w32_gfx12", launch<bf8_bf8>, arch::gfx12, type::bf8, type::bf8, type::f32, 16},
        {"wmma_i32_16x16x16_iu8_w32_gfx12", launch<iu8_12>, arch::gfx12, type::i8, type::u8, type::i32, 16},
        {"wmma_i32_16x16x16_iu4_w32_gfx12", launch<iu4_12>, arch::gfx12, type::u4, type::i4, type::i32, 16},
        {"wmma_i32_16x16x32_iu4_w32_gfx12", launch<iu4_k32>, arch::gfx12, type::i4, type::i4, type::i32, 32},
        {"swmmac_f32_16x16x32_f16_w32", launch<s_f32_f16>, arch::sparse, type::f16, type::f16, type::f32, 32},
        {"swmmac_f32_16x16x32_bf16_w32", launch<s_f32_bf16>, arch::sparse, type::bf16, type::bf16, type::f32, 32},
        {"swmmac_f16_16x16x32_f16_w32", launch<s_f16>, arch::sparse, type::f16, type::f16, type::f16, 32},
        {"swmmac_bf16_16x16x32_bf16_w32", launch<s_bf16>, arch::sparse, type::bf16, type::bf16, type::bf16, 32},
        {"swmmac_f32_16x16x32_fp8_fp8_w32", launch<s_fp8_fp8>, arch::sparse, type::fp8, type::fp8, type::f32, 32},
        {"swmmac_f32_16x16x32_fp8_bf8_w32", launch<s_fp8_bf8>, arch::sparse, type::fp8, type::bf8, type::f32, 32},
        {"swmmac_f32_16x16x32_bf8_fp8_w32", launch<s_bf8_fp8>, arch::sparse, type::bf8, type::fp8, type::f32, 32},
        {"swmmac_f32_16x16x32_bf8_bf8_w32", launch<s_bf8_bf8>, arch::sparse, type::bf8, type::bf8, type::f32, 32},
        {"swmmac_i32_16x16x32_iu8_w32", launch<s_iu8>, arch::sparse, type::i8, type::i8, type::i32, 32},
        {"swmmac_i32_16x16x32_iu4_w32", launch<s_iu4_k32>, arch::sparse, type::u4, type::i4, type::i32, 32},
        {"swmmac_i32_16x16x64_iu4_w32", launch<s_iu4_k64>, arch::sparse, type::i4, type::u4, type::i32, 64},
        {"(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)", launch<f32_f16_function>, arch::gfx11, type::f16, type::f16, type::f32, 16},
    };
    for (const test_case &tc : cases)
        for (int round = 0; round < 3; ++round)
            run(tc, random);

    // 16-bit results are rounded after each block of products: 2048 + 1 is
    // a tie that rounds back to 2048, where the exact sum of 8 ones is 2056
    auto even = [](unsigned k) { return k % 2 == 0; };
    auto fourth = [](unsigned k) { return k % 4 == 0; };
    auto all = [](unsigned) { return true; };
    auto first_half = [](unsigned k) { return k < 8; };
    EXPECT(ones<f16_lo>(arch::gfx11, type::f16, type::f16, 16, 1, even, 2048) == 2048);   // blocks of 2
    EXPECT(ones<f16_12>(arch::gfx12, type::f16, type::f16, 16, 1, even, 2048) == 2056);   // blocks of 4: +2 each
    EXPECT(ones<f16_12>(arch::gfx12, type::f16, type::f16, 16, 1, fourth, 2048) == 2048);
    EXPECT(ones<f32_f16_12>(arch::gfx12, type::f16, type::f32, 16, 1, fourth, 2048) == 2052);
    // bfloat16 truncates: 256 + 1.5 is 256, where rounding would give 258
    EXPECT(ones<bf16_12>(arch::gfx12, type::bf16, type::bf16, 16, 1.5, fourth, 256) == 256);
    EXPECT(ones<bf16_lo>(arch::gfx11, type::bf16, type::bf16, 16, 1.5, [](unsigned k) { return k == 0; }, 256) == 256);
    // clamp saturates at each stage: RDNA3 after each pair of 8-bit products,
    // RDNA4 after K's halves. 127 * 127 twice saturates, then -127 * 127...
    const double max = 2147483647.0;
    EXPECT(ones<iu8_clamp>(arch::gfx11, type::i8, type::i32, 16, 127, [](unsigned k) { return k < 2; }, max - 100, 127) ==
           max);
    EXPECT(ones<iu8_ss>(arch::gfx11, type::i8, type::i32, 16, 127, [](unsigned k) { return k < 2; }, max - 100, 127) ==
           -2147483648.0 + 32258 - 101);                                                 // wraps without clamp
    EXPECT(ones<iu8_12_clamp>(arch::gfx12, type::i8, type::i32, 16, 127, first_half, max - 100, 127) == max);
    EXPECT(ones<iu8_12_clamp>(arch::gfx12, type::i8, type::i32, 16, 1, all, 100, 1) == 116);
    // ... and the next stage starts from the saturated value
    {
        static fragments f;
        f = fragments{};
        for (unsigned i = 0; i < 16; ++i)
            for (unsigned k = 0; k < 16; ++k) {
                const place p = gfx12_input(type::i8, 16, i, k);
                put(f.a[p.lane], type::i8, p.index, encode(type::i8, k < 8 ? 127 : -128));
                put(f.b[p.lane], type::i8, p.index, encode(type::i8, 127));
            }
        for (unsigned i = 0; i < 16; ++i)
            for (unsigned j = 0; j < 16; ++j) {
                const place p = gfx12_output(i, j);
                put(f.c[p.lane], type::i32, p.index, encode(type::i32, max - 100));
            }
        call<iu8_12_clamp><<<1, 32>>>(&f);
        const place p = gfx12_output(9, 3);
        EXPECT(decode(type::i32, get(f.d[p.lane], type::i32, p.index)) == max - 8 * 128 * 127);
    }
    // RDNA3 reads the copy of A and B in the half of D's lane: row 1 (and
    // column 0) from lanes 17 and 16
    {
        static fragments f;
        f = fragments{};
        for (unsigned k = 0; k < 16; ++k) {
            put(f.a[17], type::f16, k, encode(type::f16, 1));      // row 1, upper copy
            put(f.a[1], type::f16, k, encode(type::f16, 3));       // row 1, lower copy
            put(f.b[16], type::f16, k, encode(type::f16, 1));      // column 0, upper copy
            put(f.b[0], type::f16, k, encode(type::f16, 1));
        }
        call<f32_f16><<<1, 32>>>(&f);
        const place p = gfx11_output(type::f32, 1, 0, false);
        EXPECT(p.lane == 16 && decode(type::f32, get(f.d[p.lane], type::f32, p.index)) == 16);
    }

    if (errors == 0)
        std::printf("amdgcn_wmma: all checks passed\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
