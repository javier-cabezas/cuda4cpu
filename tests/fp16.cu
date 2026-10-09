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
// Half precision (cuda_fp16.h), checked against an exact reference: a table
// of every finite half, decoded here independently of the header, that
// rounds exact values in each rounding mode. Additions, subtractions,
// multiplications and fused multiply-adds of halves are exact as integers
// times 2^-48 (__int128); divisions and square roots are rounded first to
// long double, whose 64 bits make the second rounding harmless.
//
// Then the special cases (NaN, infinities, saturation, integer conversions),
// half2, operators, and kernels with half2 arithmetic, shuffles and atomics.
//

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

static_assert(sizeof(__half) == 2 && sizeof(__half2) == 4 && alignof(__half2) == 4);
static_assert(std::is_trivially_copyable_v<__half> && std::is_trivially_copyable_v<__half2>);
static_assert(std::numeric_limits<long double>::digits >= 64, "the reference rounds through long double");

enum mode { rn, rz, rd, ru };

//! Every positive finite half, as an integer multiple of 2^-48, in order (their
//! bits are their indices), and 65536, which stands for infinity when rounding
static std::vector<__int128> positive_halves()
{
    std::vector<__int128> table;
    for (unsigned bits = 0; bits <= 0x7c00; ++bits) {
        const unsigned exponent = bits >> 10, mantissa = bits & 0x3ff;
        if (bits == 0x7c00)
            table.push_back(__int128(65536) << 48);
        else if (exponent == 0)
            table.push_back(__int128(mantissa) << 24);                   // mantissa * 2^-24
        else
            table.push_back(__int128(1024 + mantissa) << (exponent - 25 + 48));
    }
    return table;
}

static const std::vector<__int128> table = positive_halves();

//! What an entry of the table counts in long doubles
static constexpr long double table_unit = 0x1p-48L;

static long double value_of(unsigned bits)
{
    const long double magnitude = std::ldexp((long double)(table[bits & 0x7fff]), -48);
    return bits & 0x8000 ? -magnitude : magnitude;
}

//! The bits of x rounded to half. scale converts table entries to x's units:
//! 1 for an __int128 in units of 2^-48, table_unit for a long double.
template <typename T>
static unsigned reference(T x, mode m, T scale)
{
    const bool negative = x < 0 || (x == 0 && std::is_floating_point_v<T> && std::signbit((long double)x));
    const T a = negative ? -x : x;
    auto entry = [&](unsigned i) { return T(table[i]) * scale; };

    unsigned lo = 0, hi = 0x7c00;                         // the largest entry <= a
    if (a >= entry(0x7c00)) {
        lo = 0x7bff;
    } else {
        while (lo < hi) {
            const unsigned mid = (lo + hi + 1) / 2;
            if (entry(mid) <= a)
                lo = mid;
            else
                hi = mid - 1;
        }
    }
    unsigned result = lo;
    if (entry(lo) != a) {
        const bool up_away = (m == ru && !negative) || (m == rd && negative);
        if (m == rn) {
            const T below = a - entry(lo), above = entry(lo + 1) - a;
            if (above < below || (above == below && (lo & 1)))
                result = lo + 1;
        } else if (up_away) {
            result = lo + 1;
        }
        if (a >= entry(0x7c00))                           // beyond 65536: infinity or the largest half
            result = m == rn || up_away ? 0x7c00 : 0x7bff;
    }
    return result | (negative ? 0x8000 : 0);
}

static unsigned bits_of(__half h)
{
    return __half_as_ushort(h);
}

//! The value of a finite half in units of 2^-24, of which every half is a
//! multiple: products of two are in units of 2^-48, and fit
static __int128 scaled(unsigned bits)
{
    const __int128 v = table[bits & 0x7fff] >> 24;
    return bits & 0x8000 ? -v : v;
}

//
// Kernels
//

__global__ void dot(const half2 *a, const half2 *b, float *out, int n)
{
    __shared__ half2 partial[4];
    half2 sum = __float2half2_rn(0.f);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum = __hfma2(a[i], b[i], sum);
    for (int delta = 16; delta > 0; delta /= 2)
        sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, delta));
    if (threadIdx.x % 32 == 0)
        partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        half2 total = partial[0] + partial[1] + partial[2] + partial[3];
        *out = __low2float(total) + __high2float(total);
    }
}

__global__ void accumulate(__half *h, __half2 *h2, const __half *in)
{
    atomicAdd(h, __ldg(in));
    atomicAdd(h2, __halves2half2(__float2half(1.f), __float2half(0.5f)));
}

__global__ void convert(const float *in, __half *out, int *back)
{
    const unsigned i = threadIdx.x;
    out[i] = __float2half_rn(in[i]);
    back[i] = __half2int_rn(out[i]);
}

int main()
{
    std::mt19937_64 random(20261008);

    // Every half decodes to its value and converts back to its bits
    {
        unsigned wrong = 0;
        for (unsigned bits = 0; bits <= 0xffff; ++bits) {
            const __half h = __ushort_as_half((unsigned short)bits);
            const unsigned exponent = (bits >> 10) & 0x1f, mantissa = bits & 0x3ff;
            const float f = __half2float(h);
            if (exponent == 0x1f && mantissa != 0) {
                wrong += !std::isnan(f) || !__hisnan(h) || !__hisnan(__float2half(f));
                continue;
            }
            if (exponent == 0x1f)
                wrong += !std::isinf(f) || std::signbit(f) != bool(bits & 0x8000) || __hisnan(h);
            else
                wrong += (long double)f != value_of(bits) || std::signbit(f) != bool(bits & 0x8000);
            wrong += bits_of(__float2half(f)) != bits;
            wrong += bits_of(__double2half(double(f))) != bits;
        }
        EXPECT(wrong == 0);
    }

    // Conversions from float in every mode, at and around each half, at the
    // midpoints between consecutive halves, and at random
    {
        std::vector<float> inputs;
        for (unsigned bits = 0; bits < 0x7c00; ++bits) {
            const float v = float(value_of(bits)), next = float(value_of(bits + 1));
            const float mid = float((value_of(bits) + value_of(bits + 1)) / 2);
            for (float x : {v, mid, std::nextafter(v, 0.f), std::nextafter(v, 1e30f), std::nextafter(mid, 0.f),
                            std::nextafter(mid, 1e30f), next})
                inputs.insert(inputs.end(), {x, -x});
        }
        std::uniform_int_distribution<uint32_t> any_bits;
        for (int i = 0; i < 1000000; ++i) {
            const float x = std::bit_cast<float>(any_bits(random));
            if (!std::isnan(x))
                inputs.push_back(x);
        }
        for (float x : {0.f, -0.f, 1e-30f, -1e-30f, 65504.f, 65519.f, 65520.f, 65536.f, 1e30f, -1e30f,
                        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::denorm_min()})
            inputs.push_back(x);

        unsigned wrong = 0;
        for (float x : inputs) {
            const long double lx = x;
            if (std::isinf(x)) {
                wrong += bits_of(__float2half_rn(x)) != (x > 0 ? 0x7c00u : 0xfc00u);
                wrong += bits_of(__float2half_rz(x)) != (x > 0 ? 0x7c00u : 0xfc00u);
                continue;
            }
            wrong += bits_of(__float2half_rn(x)) != reference(lx, rn, table_unit);
            wrong += bits_of(__float2half_rz(x)) != reference(lx, rz, table_unit);
            wrong += bits_of(__float2half_rd(x)) != reference(lx, rd, table_unit);
            wrong += bits_of(__float2half_ru(x)) != reference(lx, ru, table_unit);
            wrong += bits_of(__float2half(x)) != reference(lx, rn, table_unit);
            if (wrong > 0 && wrong < 4)
                std::printf("  conversion of %a\n", double(x));
        }
        EXPECT(wrong == 0);

        // A double just above a midpoint, which float can't hold
        EXPECT(bits_of(__double2half(1.0 + 0x1p-11 + 0x1p-40)) == 0x3c01);
        EXPECT(bits_of(__double2half(1.0 + 0x1p-11)) == 0x3c00);              // a tie, to even
    }

    // Arithmetic on random finite halves, rounded to nearest
    {
        std::uniform_int_distribution<unsigned> any_half(0, 0xffff);
        auto finite = [&] {
            unsigned bits;
            do
                bits = any_half(random);
            while ((bits & 0x7c00) == 0x7c00);
            return bits;
        };
        // Operands in a narrower range too, where additions round more often
        std::uniform_int_distribution<unsigned> near_one(0x3800, 0x4400);
        const __int128 one = 1;

        unsigned wrong = 0;
        auto check = [&](const char *op, unsigned got, unsigned expected, unsigned a, unsigned b, unsigned c) {
            if (got != expected && wrong++ < 5)
                std::printf("  %s of %04x %04x %04x: %04x, expected %04x\n", op, a, b, c, got, expected);
        };
        for (int i = 0; i < 1000000; ++i) {
            const bool narrow = i % 2 == 1;
            const unsigned a = narrow ? near_one(random) | (any_half(random) & 0x8000) : finite();
            const unsigned b = narrow ? near_one(random) : finite();
            const unsigned c = finite();
            const __half ha = __ushort_as_half(a), hb = __ushort_as_half(b), hc = __ushort_as_half(c);
            const __int128 sa = scaled(a), sb = scaled(b), sc = scaled(c);

            // Exact: sums in units of 2^-24, products in units of 2^-48. An
            // exact zero has the sign IEEE arithmetic gives it, as in double.
            const double da = double(value_of(a)), db = double(value_of(b)), dc = double(value_of(c));
            auto exact = [&](__int128 x, double ieee) {
                return x != 0 ? reference(x, rn, one) : std::signbit(ieee) ? 0x8000u : 0u;
            };
            check("add", bits_of(__hadd(ha, hb)), exact((sa + sb) << 24, da + db), a, b, 0);
            check("sub", bits_of(__hsub(ha, hb)), exact((sa - sb) << 24, da - db), a, b, 0);
            check("mul", bits_of(__hmul(ha, hb)), exact(sa * sb, da * db), a, b, 0);
            check("fma", bits_of(__hfma(ha, hb, hc)), exact(sa * sb + (sc << 24), std::fma(da, db, dc)), a, b, c);
            if ((b & 0x7fff) != 0)
                check("div", bits_of(__hdiv(ha, hb)), reference(value_of(a) / value_of(b), rn, table_unit), a, b, 0);
            if ((a & 0x8000) == 0)
                check("sqrt", bits_of(hsqrt(ha)), reference(std::sqrt(value_of(a)), rn, table_unit), a, 0, 0);

            const __half2 sum = __hadd2(__halves2half2(ha, hb), __halves2half2(hb, hc));
            check("add2.x", bits_of(sum.x), bits_of(__hadd(ha, hb)), a, b, 0);
            check("add2.y", bits_of(sum.y), bits_of(__hadd(hb, hc)), b, c, 0);
        }
        EXPECT(wrong == 0);

        // A fused multiply-add doesn't round the product: x * x - round(x * x)
        // is the product's rounding error, 2^-20 (a subnormal), not 0
        const __half x = __ushort_as_half(0x3c01);                            // 1 + 2^-10
        EXPECT(bits_of(__hfma(x, x, __hneg(__hmul(x, x)))) == 0x0010);
        EXPECT(bits_of(__hsub(__hmul(x, x), __hmul(x, x))) == 0);

        // A product that is exactly a tie, 1.5 * 683/1024 = 1 + 2^-11, which
        // the smallest addend decides either way
        const __half t = __float2half(1.5f), u = __float2half(683.f / 1024.f);
        EXPECT(bits_of(__hfma(t, u, CUDART_ZERO_FP16)) == 0x3c00);             // to even
        EXPECT(bits_of(__hfma(t, u, CUDART_MIN_DENORM_FP16)) == 0x3c01);
        EXPECT(bits_of(__hfma(t, u, __hneg(CUDART_MIN_DENORM_FP16))) == 0x3c00);
        EXPECT(bits_of(__hfma(__hneg(t), u, CUDART_MIN_DENORM_FP16)) == 0xbc00);
    }

    // Special values and saturation
    {
        const __half nan = CUDART_NAN_FP16, inf = CUDART_INF_FP16, one = CUDART_ONE_FP16;
        const __half two = __float2half(2.f), neg = __float2half(-1.5f);
        EXPECT(__hisnan(nan) && !__hisnan(inf) && __hisinf(inf) == 1 && __hisinf(__hneg(inf)) == -1);
        EXPECT(!__heq(nan, nan) && __hequ(nan, nan) && !__hne(nan, one) && __hneu(nan, one));
        EXPECT(!__hlt(nan, one) && __hltu(nan, one) && __hlt(one, two) && __hge(two, two));
        EXPECT(__heq(CUDART_ZERO_FP16, CUDART_NEG_ZERO_FP16));
        EXPECT(__heq(__hmax(nan, one), one) && __heq(__hmin(two, nan), two) && __hisnan(__hmax_nan(nan, one)));
        EXPECT(__heq(__hmax(neg, one), one) && __heq(__hmin(neg, one), neg));
        EXPECT(__heq(__habs(neg), __float2half(1.5f)) && __heq(__hneg(neg), __float2half(1.5f)));
        EXPECT(__heq(__hadd_sat(__float2half(0.75f), __float2half(0.5f)), one));
        EXPECT(bits_of(__hsub_sat(__float2half(0.25f), __float2half(0.5f))) == 0);
        EXPECT(bits_of(__hmul_sat(nan, one)) == 0);
        EXPECT(bits_of(__hfma_relu(neg, one, one)) == 0 && __heq(__hfma_relu(one, one, one), two));
        EXPECT(__hisinf(__hmul(__float2half(300.f), __float2half(300.f))) == 1);
        EXPECT(__hisnan(__hsub(inf, inf)) && __hisnan(__hdiv(CUDART_ZERO_FP16, CUDART_ZERO_FP16)));
        EXPECT(__hisinf(__hdiv(one, CUDART_NEG_ZERO_FP16)) == -1);
        EXPECT(bits_of(CUDART_MAX_NORMAL_FP16) == 0x7bff && __half2float(CUDART_MIN_DENORM_FP16) == 0x1p-24f);
    }

    // Conversions to and from integers, saturated, with NaN as 0
    {
        const __half h25 = __float2half(2.5f), m25 = __float2half(-2.5f);
        EXPECT(__half2int_rn(h25) == 2 && __half2int_rz(h25) == 2 && __half2int_rd(h25) == 2 && __half2int_ru(h25) == 3);
        EXPECT(__half2int_rn(m25) == -2 && __half2int_rz(m25) == -2 && __half2int_rd(m25) == -3 && __half2int_ru(m25) == -2);
        EXPECT(__half2int_rn(CUDART_NAN_FP16) == 0 && __half2int_rz(CUDART_INF_FP16) == std::numeric_limits<int>::max());
        EXPECT(__half2short_rz(CUDART_MAX_NORMAL_FP16) == 32767 && __half2ushort_rn(m25) == 0);
        EXPECT(__half2ll_rn(__float2half(-65504.f)) == -65504 && __half2ull_rz(h25) == 2u && __half2uint_ru(h25) == 3u);
        EXPECT(__half2char_rz(__float2half(300.f)) == 127 && __half2uchar_rz(__float2half(300.f)) == 255);
        EXPECT(__hisinf(__int2half_rn(65520)) == 1 && bits_of(__int2half_rz(65520)) == 0x7bff);
        EXPECT(__half2float(__uint2half_ru(2049u)) == 2050.f && __half2float(__uint2half_rd(2049u)) == 2048.f);
        EXPECT(__half2float(__ll2half_rd(-2049)) == -2050.f && __half2float(__short2half_rn(-3)) == -3.f);
        EXPECT(__half2float(__ushort2half_rn(4097)) == 4096.f && __half2float(__ull2half_ru(1ull << 40)) > 65504.f);
        EXPECT(__half_as_short(__short_as_half(-12345)) == -12345);
    }

    // half2, conversions between float2 and half2, and lane moves
    {
        const half2 a = __floats2half2_rn(1.f, -2.f), b = __halves2half2(__float2half(3.f), __float2half(-2.f));
        EXPECT(__low2float(a) == 1.f && __high2float(a) == -2.f);
        const float2 f = __half22float2(a * b + a);
        EXPECT(f.x == 4.f && f.y == 2.f);
        EXPECT(__hbeq2(__lowhigh2highlow(a), __floats2half2_rn(-2.f, 1.f)));
        EXPECT(__hbeq2(__lows2half2(a, b), __floats2half2_rn(1.f, 3.f)) && __hbeq2(__high2half2(b), __float2half2_rn(-2.f)));
        const half2 eq = __heq2(a, b);
        EXPECT(__low2float(eq) == 0.f && __high2float(eq) == 1.f);
        EXPECT(__heq2_mask(a, b) == 0xffff0000u && __hlt2_mask(a, b) == 0x0000ffffu);
        // half2's == and != hold only if they hold in both lanes, as in CUDA
        EXPECT(!__hbeq2(a, b) && !(a == b) && !(a != b) && a != __floats2half2_rn(0.f, 0.f));
        EXPECT(__hbeq2(__hmax2(a, b), __floats2half2_rn(3.f, -2.f)) && __hbeq2(__habs2(a), __floats2half2_rn(1.f, 2.f)));
        EXPECT(__hbeq2(h2sqrt(__floats2half2_rn(4.f, 9.f)), __floats2half2_rn(2.f, 3.f)));
        EXPECT(__hbeq2(__hfma2(a, a, a), __floats2half2_rn(2.f, 2.f)) && __hbeq2(__float22half2_rn(f), __floats2half2_rn(4.f, 2.f)));
    }

    // Operators and implicit conversions, as CUDA code uses them
    {
        __half h = 1.5f;
        h += __half(2);
        ++h;
        h = h * __half(2.0) - __half(1);
        EXPECT(__half2float(h) == 8.f && float(h) == 8.f && int(h) == 8 && bool(h) && !bool(__half(0)));
        __half u;
        u = 70000u;
        EXPECT(__hisinf(u) == 1 && -h < h && h >= __half(8) && h != __half(9));
        half2 v(0.f, 1.f);
        v += half2(1.f, 1.f);
        v++;
        EXPECT(v == half2(2.f, 3.f) && -v < v);
        EXPECT(__heq(hrcp(__half(4)), __half(0.25f)) && __heq(hexp2(__half(3)), __half(8)));
        EXPECT(__heq(hfloor(__half(-1.5f)), __half(-2)) && __heq(hrint(__half(2.5f)), __half(2)));
        EXPECT(std::fabs(__half2float(hexp(__half(1))) - 2.71875f) < 1e-3f);
    }

    // Kernels: a dot product with __hfma2 and shuffles, atomics, conversions
    {
        const int n = 1000;
        std::vector<half2> a(n), b(n);
        float expected = 0;
        for (int i = 0; i < n; ++i) {
            a[i] = __floats2half2_rn(float(i % 3), float(i % 2));
            b[i] = __floats2half2_rn(float(i % 4) * 0.5f, 1.f);
            expected += float(i % 3) * float(i % 4) * 0.5f + float(i % 2);
        }
        float result = 0;
        dot<<<1, 128>>>(a.data(), b.data(), &result, n);
        EXPECT(result == expected);

        __half h = __float2half(0.f), one = __float2half(1.f);
        __half2 h2 = __float2half2_rn(0.f);
        accumulate<<<8, 128>>>(&h, &h2, &one);
        EXPECT(__half2float(h) == 1024.f && __low2float(h2) == 1024.f && __high2float(h2) == 512.f);

        float in[4] = {1.4f, -2.6f, 70000.f, 0.0001f};
        __half out[4];
        int back[4];
        convert<<<1, 4>>>(in, out, back);
        EXPECT(back[0] == 1 && back[1] == -3 && back[2] == std::numeric_limits<int>::max() && back[3] == 0);
        EXPECT(bits_of(out[2]) == 0x7c00 && bits_of(out[3]) == reference((long double)0.0001f, rn, table_unit));
    }

    if (errors == 0)
        std::printf("fp16: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
