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
// bfloat16 (cuda_bf16.h), checked against an exact reference: a table of
// every finite bfloat16 as a long double, which holds each exactly, decoded
// here independently of the header, that rounds exact values in each
// rounding mode. Operations are rounded first to odd in long double (toward
// zero, then the last bit set if the result was inexact, from the FPU's
// inexact flag), whose 64 bits make rounding that to bfloat16 exact.
//
// Then the special cases (NaN, infinities, saturation, integer conversions),
// bfloat162, operators, and kernels with bfloat162 arithmetic, shuffles,
// atomics and loads with cache hints.
//

#include <cuda_bf16.h>

#include <cfenv>
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

static_assert(sizeof(__nv_bfloat16) == 2 && sizeof(__nv_bfloat162) == 4 && alignof(__nv_bfloat162) == 4);
static_assert(std::is_trivially_copyable_v<__nv_bfloat16> && std::is_trivially_copyable_v<__nv_bfloat162>);
static_assert(std::numeric_limits<long double>::digits >= 64, "the reference rounds through long double");

enum mode { rn, rz, rd, ru };

//! Every positive finite bfloat16, in order (their bits are their indices),
//! and 2^128, which stands for infinity when rounding
static std::vector<long double> positive_bfloat16s()
{
    std::vector<long double> table;
    for (unsigned bits = 0; bits <= 0x7f80; ++bits) {
        const int exponent = int(bits >> 7), mantissa = int(bits & 0x7f);
        if (exponent == 0)
            table.push_back(std::ldexp((long double)mantissa, -133));            // mantissa * 2^-133
        else
            table.push_back(std::ldexp((long double)(128 + mantissa), exponent - 127 - 7));
    }
    return table;
}

static const std::vector<long double> table = positive_bfloat16s();

static long double value_of(unsigned bits)
{
    const long double magnitude = table[bits & 0x7fff];
    return bits & 0x8000 ? -magnitude : magnitude;
}

//! The bits of the exact value x rounded to bfloat16
static unsigned reference(long double x, mode m)
{
    const bool negative = std::signbit(x);
    const long double a = negative ? -x : x;

    unsigned lo = 0, hi = 0x7f80;                         // the largest entry <= a
    if (a >= table[0x7f80]) {
        lo = 0x7f7f;
    } else {
        while (lo < hi) {
            const unsigned mid = (lo + hi + 1) / 2;
            if (table[mid] <= a)
                lo = mid;
            else
                hi = mid - 1;
        }
    }
    unsigned result = lo;
    if (table[lo] != a) {
        const bool up_away = (m == ru && !negative) || (m == rd && negative);
        if (m == rn) {
            const long double below = a - table[lo], above = table[lo + 1] - a;
            if (above < below || (above == below && (lo & 1)))
                result = lo + 1;
        } else if (up_away) {
            result = lo + 1;
        }
        if (a >= table[0x7f80])                           // beyond 2^128: infinity or the largest bfloat16
            result = m == rn || up_away ? 0x7f80 : 0x7f7f;
    }
    return result | (negative ? 0x8000 : 0);
}

//! op() in long double, rounded to odd: toward zero, with the last bit of
//! the 64-bit significand set if that was inexact. Rounding the result to
//! the 8 bits of bfloat16 gives the correctly rounded exact result.
template <typename Op>
static long double round_to_odd(Op op)
{
    const int saved = std::fegetround();
    std::fesetround(FE_TOWARDZERO);
    std::feclearexcept(FE_INEXACT);
    volatile long double result = op();
    const bool inexact = std::fetestexcept(FE_INEXACT) != 0;
    std::fesetround(saved);
    long double r = result;
    if (inexact) {
        uint64_t significand;
        std::memcpy(&significand, &r, sizeof(significand));       // x87 extended: the low 8 bytes
        significand |= 1;
        std::memcpy(&r, &significand, sizeof(significand));
    }
    return r;
}

static unsigned bits_of(__nv_bfloat16 h)
{
    return __bfloat16_as_ushort(h);
}

//
// Kernels
//

__global__ void dot(const nv_bfloat162 *a, const nv_bfloat162 *b, float *out, int n)
{
    __shared__ nv_bfloat162 partial[4];
    nv_bfloat162 sum = __float2bfloat162_rn(0.f);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum = __hfma2(__ldcg(&a[i]), __ldcs(&b[i]), sum);
    for (int delta = 16; delta > 0; delta /= 2)
        sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, delta));
    if (threadIdx.x % 32 == 0)
        partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        nv_bfloat162 total = partial[0] + partial[1] + partial[2] + partial[3];
        *out = __low2float(total) + __high2float(total);
    }
}

__global__ void accumulate(__nv_bfloat16 *h, __nv_bfloat162 *h2, const __nv_bfloat16 *in)
{
    atomicAdd(h, __ldg(in));
    atomicAdd(h2, __halves2bfloat162(__float2bfloat16(1.f), __float2bfloat16(0.5f)));
}

__global__ void convert(const float *in, __nv_bfloat16 *out, int *back)
{
    const unsigned i = threadIdx.x;
    __stwb(&out[i], __float2bfloat16_rn(in[i]));
    back[i] = __bfloat162int_rn(out[i]);
}

int main()
{
    std::mt19937_64 random(20261025);

    // Every bfloat16 decodes to its value and converts back to its bits
    {
        unsigned wrong = 0;
        for (unsigned bits = 0; bits <= 0xffff; ++bits) {
            const __nv_bfloat16 h = __ushort_as_bfloat16((unsigned short)bits);
            const unsigned exponent = (bits >> 7) & 0xff, mantissa = bits & 0x7f;
            const float f = __bfloat162float(h);
            // A bfloat16 is the top half of a float
            uint32_t float_bits;
            std::memcpy(&float_bits, &f, sizeof(f));
            if (exponent == 0xff && mantissa != 0) {
                wrong += !std::isnan(f) || !__hisnan(h) || !__hisnan(__float2bfloat16(f));
                continue;
            }
            wrong += float_bits != bits << 16;
            if (exponent == 0xff)
                wrong += !std::isinf(f) || std::signbit(f) != bool(bits & 0x8000) || __hisnan(h);
            else
                wrong += (long double)f != value_of(bits) || std::signbit(f) != bool(bits & 0x8000);
            wrong += bits_of(__float2bfloat16(f)) != bits;
            wrong += bits_of(__double2bfloat16(double(f))) != bits;
        }
        EXPECT(wrong == 0);
    }

    // Conversions from float in every mode, at and around each bfloat16, at
    // the midpoints between consecutive ones, and at random
    {
        std::vector<float> inputs;
        for (unsigned bits = 0; bits < 0x7f80; ++bits) {
            const float v = float(value_of(bits)), next = float(value_of(bits + 1));
            const float mid = float((value_of(bits) + value_of(bits + 1)) / 2);
            for (float x : {v, mid, std::nextafter(v, 0.f), std::nextafter(v, 1e38f), std::nextafter(mid, 0.f),
                            std::nextafter(mid, 1e38f), next})
                inputs.insert(inputs.end(), {x, -x});
        }
        std::uniform_int_distribution<uint32_t> any_bits;
        for (int i = 0; i < 1000000; ++i) {
            const float x = std::bit_cast<float>(any_bits(random));
            if (!std::isnan(x))
                inputs.push_back(x);
        }
        for (float x : {0.f, -0.f, 1e-45f, -1e-45f, 3.3895314e38f, std::numeric_limits<float>::max(),
                        -std::numeric_limits<float>::max(), std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::denorm_min()})
            inputs.push_back(x);

        unsigned wrong = 0;
        for (float x : inputs) {
            const long double lx = x;
            if (std::isinf(x)) {
                wrong += bits_of(__float2bfloat16_rn(x)) != (x > 0 ? 0x7f80u : 0xff80u);
                wrong += bits_of(__float2bfloat16_rz(x)) != (x > 0 ? 0x7f80u : 0xff80u);
                continue;
            }
            wrong += bits_of(__float2bfloat16_rn(x)) != reference(lx, rn);
            wrong += bits_of(__float2bfloat16_rz(x)) != reference(lx, rz);
            wrong += bits_of(__float2bfloat16_rd(x)) != reference(lx, rd);
            wrong += bits_of(__float2bfloat16_ru(x)) != reference(lx, ru);
            wrong += bits_of(__float2bfloat16(x)) != reference(lx, rn);
            if (wrong > 0 && wrong < 4)
                std::printf("  conversion of %a\n", double(x));
        }
        EXPECT(wrong == 0);

        // A double just above a midpoint, which float can't hold
        EXPECT(bits_of(__double2bfloat16(1.0 + 0x1p-8 + 0x1p-40)) == 0x3f81);
        EXPECT(bits_of(__double2bfloat16(1.0 + 0x1p-8)) == 0x3f80);            // a tie, to even
        // Rounded through float, it would round twice and land on the tie
        EXPECT(bits_of(__double2bfloat16(1.0 + 0x1p-8 + 0x1p-50)) == 0x3f81);
    }

    // Arithmetic on random finite bfloat16s, rounded to nearest
    {
        std::uniform_int_distribution<unsigned> any_bfloat16(0, 0xffff);
        auto finite = [&] {
            unsigned bits;
            do
                bits = any_bfloat16(random);
            while ((bits & 0x7f80) == 0x7f80);
            return bits;
        };
        // Operands in a narrower range too, where additions round more often
        std::uniform_int_distribution<unsigned> near_one(0x3f00, 0x4080);

        unsigned wrong = 0;
        auto check = [&](const char *op, unsigned got, unsigned expected, unsigned a, unsigned b, unsigned c) {
            if (got != expected && wrong++ < 5)
                std::printf("  %s of %04x %04x %04x: %04x, expected %04x\n", op, a, b, c, got, expected);
        };
        for (int i = 0; i < 1000000; ++i) {
            const bool narrow = i % 2 == 1;
            const unsigned a = narrow ? near_one(random) | (any_bfloat16(random) & 0x8000) : finite();
            const unsigned b = narrow ? near_one(random) : finite();
            const unsigned c = narrow ? near_one(random) | (any_bfloat16(random) & 0x8000) : finite();
            const __nv_bfloat16 ha = __ushort_as_bfloat16(a), hb = __ushort_as_bfloat16(b), hc = __ushort_as_bfloat16(c);
            const long double la = value_of(a), lb = value_of(b), lc = value_of(c);

            // An exact zero has the sign IEEE arithmetic gives it, rounding
            // to nearest
            auto exact = [&](long double odd, long double ieee) {
                return odd != 0 ? reference(odd, rn) : std::signbit(ieee) ? 0x8000u : 0u;
            };
            check("add", bits_of(__hadd(ha, hb)), exact(round_to_odd([&] { return la + lb; }), la + lb), a, b, 0);
            check("sub", bits_of(__hsub(ha, hb)), exact(round_to_odd([&] { return la - lb; }), la - lb), a, b, 0);
            check("mul", bits_of(__hmul(ha, hb)), exact(round_to_odd([&] { return la * lb; }), la * lb), a, b, 0);
            check("fma", bits_of(__hfma(ha, hb, hc)),
                  exact(round_to_odd([&] { return std::fma(la, lb, lc); }), std::fma(la, lb, lc)), a, b, c);
            if ((b & 0x7fff) != 0)
                check("div", bits_of(__hdiv(ha, hb)), reference(round_to_odd([&] { return la / lb; }), rn), a, b, 0);
            if ((a & 0x8000) == 0)
                check("sqrt", bits_of(hsqrt(ha)), reference(round_to_odd([&] { return std::sqrt(la); }), rn), a, 0, 0);

            const __nv_bfloat162 sum = __hadd2(__halves2bfloat162(ha, hb), __halves2bfloat162(hb, hc));
            check("add2.x", bits_of(sum.x), bits_of(__hadd(ha, hb)), a, b, 0);
            check("add2.y", bits_of(sum.y), bits_of(__hadd(hb, hc)), b, c, 0);
        }
        EXPECT(wrong == 0);

        // A fused multiply-add doesn't round the product: x * x - round(x * x)
        // is the product's rounding error, 2^-14, not 0
        const __nv_bfloat16 x = __ushort_as_bfloat16(0x3f81);                   // 1 + 2^-7
        EXPECT(bits_of(__hfma(x, x, __hneg(__hmul(x, x)))) == 0x3880);
        EXPECT(bits_of(__hsub(__hmul(x, x), __hmul(x, x))) == 0);

        // A product that is exactly a tie, 1.75 * 37/64 = 1 + 3 * 2^-8, which
        // the smallest addend decides either way. Rounded to nearest in double,
        // the sum with -2^-133 would land on the tie and round up.
        const __nv_bfloat16 t = __float2bfloat16(1.75f), u = __float2bfloat16(37.f / 64.f);
        EXPECT(bits_of(__hfma(t, u, CUDART_ZERO_BF16)) == 0x3f82);             // to even
        EXPECT(bits_of(__hfma(t, u, CUDART_MIN_DENORM_BF16)) == 0x3f82);
        EXPECT(bits_of(__hfma(t, u, __hneg(CUDART_MIN_DENORM_BF16))) == 0x3f81);
        EXPECT(bits_of(__hfma(__hneg(t), u, CUDART_MIN_DENORM_BF16)) == 0xbf81);
    }

    // Special values and saturation
    {
        const __nv_bfloat16 nan = CUDART_NAN_BF16, inf = CUDART_INF_BF16, one = CUDART_ONE_BF16;
        const __nv_bfloat16 two = __float2bfloat16(2.f), neg = __float2bfloat16(-1.5f);
        EXPECT(__hisnan(nan) && !__hisnan(inf) && __hisinf(inf) == 1 && __hisinf(__hneg(inf)) == -1);
        EXPECT(!__heq(nan, nan) && __hequ(nan, nan) && !__hne(nan, one) && __hneu(nan, one));
        EXPECT(!__hlt(nan, one) && __hltu(nan, one) && __hlt(one, two) && __hge(two, two));
        EXPECT(__heq(CUDART_ZERO_BF16, CUDART_NEG_ZERO_BF16));
        EXPECT(__heq(__hmax(nan, one), one) && __heq(__hmin(two, nan), two) && __hisnan(__hmax_nan(nan, one)));
        EXPECT(__heq(__hmax(neg, one), one) && __heq(__hmin(neg, one), neg));
        EXPECT(__heq(__habs(neg), __float2bfloat16(1.5f)) && __heq(__hneg(neg), __float2bfloat16(1.5f)));
        EXPECT(__heq(__hadd_sat(__float2bfloat16(0.75f), __float2bfloat16(0.5f)), one));
        EXPECT(bits_of(__hsub_sat(__float2bfloat16(0.25f), __float2bfloat16(0.5f))) == 0);
        EXPECT(bits_of(__hmul_sat(nan, one)) == 0);
        EXPECT(bits_of(__hfma_relu(neg, one, one)) == 0 && __heq(__hfma_relu(one, one, one), two));
        EXPECT(__hisinf(__hmul(__float2bfloat16(1e30f), __float2bfloat16(1e30f))) == 1);
        EXPECT(bits_of(__hmul(__float2bfloat16(1e-30f), __float2bfloat16(1e-30f))) == 0);
        EXPECT(__hisnan(__hsub(inf, inf)) && __hisnan(__hdiv(CUDART_ZERO_BF16, CUDART_ZERO_BF16)));
        EXPECT(__hisinf(__hdiv(one, CUDART_NEG_ZERO_BF16)) == -1);
        EXPECT(bits_of(CUDART_MAX_NORMAL_BF16) == 0x7f7f && __bfloat162float(CUDART_MIN_DENORM_BF16) == 0x1p-133f);
        // Beyond the largest bfloat16, toward zero gives it, to nearest infinity
        EXPECT(bits_of(__float2bfloat16_rz(std::numeric_limits<float>::max())) == 0x7f7f);
        EXPECT(bits_of(__float2bfloat16_rn(std::numeric_limits<float>::max())) == 0x7f80);
        EXPECT(bits_of(__float2bfloat16_rd(-std::numeric_limits<float>::max())) == 0xff80);
    }

    // Conversions to and from integers, saturated, with NaN as 0
    {
        const __nv_bfloat16 h25 = __float2bfloat16(2.5f), m25 = __float2bfloat16(-2.5f);
        EXPECT(__bfloat162int_rn(h25) == 2 && __bfloat162int_rz(h25) == 2 && __bfloat162int_rd(h25) == 2 && __bfloat162int_ru(h25) == 3);
        EXPECT(__bfloat162int_rn(m25) == -2 && __bfloat162int_rz(m25) == -2 && __bfloat162int_rd(m25) == -3 && __bfloat162int_ru(m25) == -2);
        EXPECT(__bfloat162int_rn(CUDART_NAN_BF16) == 0 && __bfloat162int_rz(CUDART_INF_BF16) == std::numeric_limits<int>::max());
        EXPECT(__bfloat162short_rz(CUDART_MAX_NORMAL_BF16) == 32767 && __bfloat162ushort_rn(m25) == 0);
        EXPECT(__bfloat162ll_rn(__float2bfloat16(-65536.f)) == -65536 && __bfloat162ull_rz(h25) == 2u && __bfloat162uint_ru(h25) == 3u);
        EXPECT(__bfloat162char_rz(__float2bfloat16(300.f)) == 127 && __bfloat162uchar_rz(__float2bfloat16(300.f)) == 255);
        EXPECT(__bfloat162float(__int2bfloat16_rn(257)) == 256.f && __bfloat162float(__int2bfloat16_ru(257)) == 258.f);
        EXPECT(__bfloat162float(__uint2bfloat16_ru(2049u)) == 2064.f && __bfloat162float(__uint2bfloat16_rd(2049u)) == 2048.f);
        EXPECT(__bfloat162float(__ll2bfloat16_rd(-2049)) == -2064.f && __bfloat162float(__short2bfloat16_rn(-3)) == -3.f);
        EXPECT(__bfloat162float(__ushort2bfloat16_rz(65535)) == 65280.f && __bfloat162float(__ull2bfloat16_rn(1ull << 60)) == 0x1p60f);
        EXPECT(__bfloat16_as_short(__short_as_bfloat16(-12345)) == -12345);
    }

    // bfloat162, conversions between float2 and bfloat162, and lane moves
    {
        const nv_bfloat162 a = __floats2bfloat162_rn(1.f, -2.f), b = __halves2bfloat162(__float2bfloat16(3.f), __float2bfloat16(-2.f));
        EXPECT(__low2float(a) == 1.f && __high2float(a) == -2.f);
        const float2 f = __bfloat1622float2(a * b + a);
        EXPECT(f.x == 4.f && f.y == 2.f);
        EXPECT(__hbeq2(__lowhigh2highlow(a), __floats2bfloat162_rn(-2.f, 1.f)));
        EXPECT(__hbeq2(__lows2bfloat162(a, b), __floats2bfloat162_rn(1.f, 3.f)) && __hbeq2(__high2bfloat162(b), __float2bfloat162_rn(-2.f)));
        EXPECT(__heq(__low2bfloat16(b), __float2bfloat16(3.f)) && __hbeq2(make_bfloat162(a.y, a.x), __lowhigh2highlow(a)));
        const nv_bfloat162 eq = __heq2(a, b);
        EXPECT(__low2float(eq) == 0.f && __high2float(eq) == 1.f);
        EXPECT(__heq2_mask(a, b) == 0xffff0000u && __hlt2_mask(a, b) == 0x0000ffffu);
        EXPECT(!__hbeq2(a, b) && !(a == b) && !(a != b) && a != __floats2bfloat162_rn(0.f, 0.f));
        EXPECT(__hbeq2(__hmax2(a, b), __floats2bfloat162_rn(3.f, -2.f)) && __hbeq2(__habs2(a), __floats2bfloat162_rn(1.f, 2.f)));
        EXPECT(__hbeq2(h2sqrt(__floats2bfloat162_rn(4.f, 9.f)), __floats2bfloat162_rn(2.f, 3.f)));
        EXPECT(__hbeq2(__hfma2(a, a, a), __floats2bfloat162_rn(2.f, 2.f)) && __hbeq2(__float22bfloat162_rn(f), __floats2bfloat162_rn(4.f, 2.f)));
        // (1 - 2i)(3 - 2i) + (1 - 2i) = -1 - 8i + 1 - 2i
        EXPECT(__hbeq2(__hcmadd(a, b, a), __floats2bfloat162_rn(0.f, -10.f)));
        const half2 ha = __floats2half2_rn(1.f, -2.f), hb = __floats2half2_rn(3.f, -2.f);
        EXPECT(__hbeq2(__hcmadd(ha, hb, ha), __floats2half2_rn(0.f, -10.f)));
    }

    // Operators and implicit conversions, as CUDA code uses them
    {
        __nv_bfloat16 h = 1.5f;
        h += __nv_bfloat16(2);
        ++h;
        h = h * __nv_bfloat16(2.0) - __nv_bfloat16(1);
        EXPECT(__bfloat162float(h) == 8.f && float(h) == 8.f && int(h) == 8 && bool(h) && !bool(__nv_bfloat16(0)));
        __nv_bfloat16 u;
        u = 1000u;
        EXPECT(float(u) == 1000.f && -h < h && h >= __nv_bfloat16(8) && h != __nv_bfloat16(9));
        EXPECT(float(__nv_bfloat16(__float2half(0.1f))) == float(__float2bfloat16(float(__float2half(0.1f)))));
        nv_bfloat162 v(0.f, 1.f);
        v += nv_bfloat162(1.f, 1.f);
        v++;
        EXPECT(v == nv_bfloat162(2.f, 3.f) && -v < v);
        EXPECT(__heq(hrcp(__nv_bfloat16(4)), __nv_bfloat16(0.25f)) && __heq(hexp2(__nv_bfloat16(3)), __nv_bfloat16(8)));
        EXPECT(__heq(hfloor(__nv_bfloat16(-1.5f)), __nv_bfloat16(-2)) && __heq(hrint(__nv_bfloat16(2.5f)), __nv_bfloat16(2)));
        EXPECT(std::fabs(__bfloat162float(hexp(__nv_bfloat16(1))) - 2.71875f) < 1e-2f);
    }

    // Kernels: a dot product with __hfma2 and shuffles, atomics, conversions
    {
        const int n = 64;                                // sums that bfloat16 holds exactly
        std::vector<nv_bfloat162> a(n), b(n);
        float expected = 0;
        for (int i = 0; i < n; ++i) {
            a[i] = __floats2bfloat162_rn(float(i % 3), float(i % 2));
            b[i] = __floats2bfloat162_rn(float(i % 4) * 0.5f, 1.f);
            expected += float(i % 3) * float(i % 4) * 0.5f + float(i % 2);
        }
        float result = 0;
        dot<<<1, 128>>>(a.data(), b.data(), &result, n);
        EXPECT(result == expected);

        __nv_bfloat16 h = __float2bfloat16(0.f), one = __float2bfloat16(1.f);
        __nv_bfloat162 h2 = __float2bfloat162_rn(0.f);
        accumulate<<<2, 64>>>(&h, &h2, &one);
        EXPECT(__bfloat162float(h) == 128.f && __low2float(h2) == 128.f && __high2float(h2) == 64.f);

        float in[4] = {1.4f, -2.6f, 3e38f, 0.0001f};
        __nv_bfloat16 out[4];
        int back[4];
        convert<<<1, 4>>>(in, out, back);
        EXPECT(back[0] == 1 && back[1] == -3 && back[2] == std::numeric_limits<int>::max() && back[3] == 0);
        EXPECT(bits_of(out[2]) == reference((long double)3e38f, rn) && bits_of(out[3]) == reference((long double)0.0001f, rn));
    }

    if (errors == 0)
        std::printf("bf16: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
