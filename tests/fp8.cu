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
// 8-bit floats, CUDA's (cuda_fp8.h) and AMD's (hip/hip_fp8.h), checked
// against a reference: each format's 256 codes decoded here from its
// exponent and mantissa widths and bias, rounding to nearest even over them,
// and each vendor's rules for NaN, infinity and values too large, as their
// own headers implement them. Inputs are every value of each format, the
// midpoints between them, their neighbors in double and float, a million
// random floats and the special values. E8M0, the classes, packed pairs and
// quadruples, and conversions in kernels too.
//

#include <cuda_fp8.h>
#include <hip/hip_fp8.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

static_assert(sizeof(__nv_fp8_e4m3) == 1 && sizeof(__nv_fp8x2_e5m2) == 2 && sizeof(__nv_fp8x4_e4m3) == 4);
static_assert(sizeof(__nv_fp8_e8m0) == 1 && alignof(__nv_fp8x4_e8m0) == 4);
static_assert(sizeof(__hip_fp8_e4m3_fnuz) == 1 && sizeof(__hip_fp8x4_e5m2) == 4);
static_assert(std::is_trivially_copyable_v<__nv_fp8_e4m3> && std::is_trivially_copyable_v<__hip_fp8x2_e5m2_fnuz>);

//! A format, as the OCP and AMD specifications describe it
struct format {
    const char *name;
    int exponent_bits, mantissa_bits, bias;
    enum { ieee, one_nan, fnuz } kind;
    std::vector<double> values;     //!< Of the codes 0 to 127 (NaN where a code is NaN or infinity)
    unsigned max_code;              //!< The largest finite value
    double max;

    format(const char *n, int e, int m, int b, decltype(kind) k) : name(n), exponent_bits(e), mantissa_bits(m), bias(b), kind(k)
    {
        for (unsigned code = 0; code < 128; ++code) {
            const int exponent = int(code >> m), mantissa = int(code & ((1u << m) - 1));
            const bool top = exponent == (1 << e) - 1;
            double v;
            if (exponent == 0)
                v = std::ldexp(double(mantissa), 1 - b - m);
            else
                v = std::ldexp(double((1 << m) + mantissa), exponent - b - m);
            if ((kind == ieee && top) || (kind == one_nan && code == 127))
                v = std::numeric_limits<double>::quiet_NaN();
            values.push_back(v);
            if (!std::isnan(v))
                max_code = code;
        }
        max = values[max_code];
    }

    bool is_nan(unsigned code) const
    {
        if (kind == fnuz)
            return code == 0x80;
        return std::isnan(values[code & 0x7f]) && !(kind == ieee && (code & 0x7f) == infinity());
    }

    unsigned infinity() const { return ((1u << exponent_bits) - 1) << mantissa_bits; }

    //! The code of |x| rounded to nearest even, or max_code + 1 if it rounds
    //! beyond max (to the next value the exponent would have)
    unsigned round(double a) const
    {
        unsigned lo = 0;
        while (lo < max_code && values[lo + 1] <= a)
            ++lo;
        if (values[lo] == a)
            return lo;
        const double next = lo < max_code ? values[lo + 1] : max + (max - values[max_code - 1]);
        const double below = a - values[lo], above = next - a;
        return above < below || (above == below && (lo & 1)) ? lo + 1 : lo;
    }

    //! The value of a code, signed (NaN for NaN, infinity for E5M2's)
    double decode(unsigned code) const
    {
        if (is_nan(code))
            return std::numeric_limits<double>::quiet_NaN();
        if (kind == ieee && (code & 0x7f) == infinity())
            return code & 0x80 ? -INFINITY : INFINITY;
        const double v = values[code & 0x7f];
        return code & 0x80 ? -v : v;
    }
};

static const format e4m3("E4M3", 4, 3, 7, format::one_nan), e5m2("E5M2", 5, 2, 15, format::ieee);
static const format e4m3_fnuz("E4M3_FNUZ", 4, 3, 8, format::fnuz), e5m2_fnuz("E5M2_FNUZ", 5, 2, 16, format::fnuz);

//! CUDA's rules: values that round beyond the largest finite one, and
//! infinities, saturate or become NaN (E4M3) or infinity (E5M2); NaN keeps
//! its sign, and in E5M2 the top two bits of its payload
static unsigned cuda_reference(double x, bool saturate, const format &f)
{
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(x));
    const unsigned sign = bits >> 63 ? 0x80 : 0;
    if (std::isnan(x))
        return f.kind == format::one_nan ? sign | 0x7f : sign | 0x7e | unsigned((bits >> 50) & 3);
    const double a = std::fabs(x);
    const unsigned code = std::isinf(a) ? f.max_code + 1 : f.round(a);
    if (code > f.max_code)
        return sign | (saturate ? f.max_code : f.kind == format::one_nan ? 0x7f : f.infinity());
    return sign | code;
}

//! AMD's rules: magnitudes above the largest finite value overflow before
//! rounding; infinity stays infinity in E5M2 and is NaN elsewhere, even with
//! saturation; NaN is 0x7f with the sign, or 0x80 in FNUZ, which has no
//! negative zero
static unsigned amd_reference(double x, bool saturate, const format &f)
{
    const unsigned sign = std::signbit(x) ? 0x80 : 0;
    const unsigned nan = f.kind == format::fnuz ? 0x80 : sign | 0x7f;
    if (std::isnan(x))
        return nan;
    const double a = std::fabs(x);
    if (std::isinf(a))
        return f.kind == format::ieee ? sign | f.infinity() : nan;
    if (a > f.max)
        return saturate ? sign | f.max_code : f.kind == format::ieee ? sign | f.infinity() : nan;
    const unsigned code = f.round(a);
    if (f.kind == format::fnuz && code == 0)
        return 0;
    return sign | code;
}

//! The E8M0 code: the power of two at or below |x| toward zero, at or above
//! it otherwise; 0 below 2^-127; beyond 2^127, NaN or (saturating, or
//! toward zero) 2^127
static unsigned e8m0_reference(double x, bool saturate, cudaRoundMode rounding)
{
    if (std::isnan(x))
        return 0xff;
    const double a = std::fabs(x);
    if (std::isinf(a))
        return saturate ? 0xfe : 0xff;
    if (a == 0)
        return 0;
    int exponent = std::ilogb(a);
    if (rounding != cudaRoundZero && std::ldexp(1.0, exponent) != a)
        ++exponent;
    if (exponent < -127)
        return 0;
    if (exponent > 127)
        return rounding == cudaRoundZero || saturate ? 0xfe : 0xff;
    return unsigned(exponent + 127);
}

static std::vector<double> inputs_for(const format &f, std::mt19937_64 &random)
{
    std::vector<double> inputs;
    for (unsigned code = 0; code <= f.max_code + 1; ++code) {
        const double v = code <= f.max_code ? f.values[code] : f.max * 2;
        const double next = code < f.max_code ? f.values[code + 1] : v * 2;
        const double mid = (v + next) / 2;
        for (double x : {v, mid, std::nextafter(v, 0.0), std::nextafter(v, 1e300), std::nextafter(mid, 0.0),
                         std::nextafter(mid, 1e300), double(std::nextafter(float(v), 0.f)),
                         double(std::nextafter(float(mid), 1e30f)), v / 2, v * 0.75})
            inputs.insert(inputs.end(), {x, -x});
    }
    std::uniform_int_distribution<uint32_t> any_bits;
    for (int i = 0; i < 1000000; ++i)
        inputs.push_back(double(std::bit_cast<float>(any_bits(random))));
    for (double x : {0.0, -0.0, 1e300, -1e300, 1e-300, -1e-300, double(INFINITY), -double(INFINITY), double(NAN), -double(NAN),
                     double(std::numeric_limits<float>::denorm_min())})
        inputs.push_back(x);
    return inputs;
}

//
// Kernels
//

__global__ void convert(const float *in, __nv_fp8_e4m3 *e4m3, __nv_fp8_e5m2 *e5m2, __hip_fp8_storage_t *fnuz,
                        float *back)
{
    const unsigned i = threadIdx.x;
    e4m3[i] = __nv_fp8_e4m3(in[i]);
    e5m2[i].__x = __nv_cvt_float_to_fp8(in[i], __NV_NOSAT, __NV_E5M2);
    fnuz[i] = __hip_cvt_float_to_fp8(in[i], __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    const float amd = __hip_fp8_e4m3_fnuz(in[i]);
    back[i] = float(e4m3[i]) + float(e5m2[i]) + amd;
}

int main()
{
    std::mt19937_64 random(20261025);

    // Every code decodes to its value, through half, float and double
    {
        unsigned wrong = 0;
        for (unsigned code = 0; code < 256; ++code) {
            const auto same = [](double got, double expected) {
                return std::isnan(expected) ? std::isnan(got) : got == expected && std::signbit(got) == std::signbit(expected);
            };
            __nv_fp8_e4m3 a;
            a.__x = __nv_fp8_storage_t(code);
            __nv_fp8_e5m2 b;
            b.__x = __nv_fp8_storage_t(code);
            wrong += !same(double(a), e4m3.decode(code)) || !same(float(a), e4m3.decode(code));
            wrong += !same(__half2float(__half(a)), e4m3.decode(code));
            wrong += !same(double(b), e5m2.decode(code)) || !same(double(__bfloat162float(__nv_bfloat16(b))), e5m2.decode(code));
            wrong += !same(__half2float(__half(__nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t(code), __NV_E4M3))), e4m3.decode(code));
            for (auto [interp, f] : {std::pair{__HIP_E4M3, &e4m3}, std::pair{__HIP_E5M2, &e5m2},
                                     std::pair{__HIP_E4M3_FNUZ, &e4m3_fnuz}, std::pair{__HIP_E5M2_FNUZ, &e5m2_fnuz}}) {
                const __half_raw h = __hip_cvt_fp8_to_halfraw(__hip_fp8_storage_t(code), interp);
                wrong += !same(__half2float(__half(h)), f->decode(code));
                // NaN converts to half's canonical NaN, without its sign
                wrong += f->is_nan(code) && h.x != 0x7fff;
            }
            __hip_fp8_e5m2_fnuz c;
            c.__x = __hip_fp8_storage_t(code);
            wrong += !same(float(c), e5m2_fnuz.decode(code));
        }
        EXPECT(wrong == 0);
        EXPECT(e4m3.max == 448 && e5m2.max == 57344 && e4m3_fnuz.max == 240 && e5m2_fnuz.max == 57344);
    }

    // Conversions to every format, with and without saturation, CUDA's and
    // AMD's, from double, float, half and bfloat16
    {
        unsigned wrong = 0;
        auto check = [&](const char *what, const format &f, double x, unsigned got, unsigned expected) {
            if (got != expected && wrong++ < 8)
                std::printf("  %s to %s of %a: %02x, expected %02x\n", what, f.name, x, got, expected);
        };
        for (auto [interp, f] : {std::pair{__NV_E4M3, &e4m3}, std::pair{__NV_E5M2, &e5m2}}) {
            for (double x : inputs_for(*f, random)) {
                for (auto sat : {__NV_NOSAT, __NV_SATFINITE}) {
                    check("double", *f, x, __nv_cvt_double_to_fp8(x, sat, interp), cuda_reference(x, sat == __NV_SATFINITE, *f));
                    const float fx = float(x);
                    check("float", *f, x, __nv_cvt_float_to_fp8(fx, sat, interp), cuda_reference(fx, sat == __NV_SATFINITE, *f));
                    const __half h = __double2half(x);
                    check("half", *f, x, __nv_cvt_halfraw_to_fp8(h, sat, interp),
                          cuda_reference(double(__half2float(h)), sat == __NV_SATFINITE, *f));
                    const __nv_bfloat16 b = __double2bfloat16(x);
                    check("bfloat16", *f, x, __nv_cvt_bfloat16raw_to_fp8(b, sat, interp),
                          cuda_reference(double(__bfloat162float(b)), sat == __NV_SATFINITE, *f));
                }
            }
        }
        for (auto [interp, f] : {std::pair{__HIP_E4M3, &e4m3}, std::pair{__HIP_E5M2, &e5m2},
                                 std::pair{__HIP_E4M3_FNUZ, &e4m3_fnuz}, std::pair{__HIP_E5M2_FNUZ, &e5m2_fnuz}}) {
            for (double x : inputs_for(*f, random)) {
                for (auto sat : {__HIP_NOSAT, __HIP_SATFINITE}) {
                    check("hip double", *f, x, __hip_cvt_double_to_fp8(x, sat, interp), amd_reference(x, sat == __HIP_SATFINITE, *f));
                    const float fx = float(x);
                    check("hip float", *f, x, __hip_cvt_float_to_fp8(fx, sat, interp), amd_reference(fx, sat == __HIP_SATFINITE, *f));
                    const __half h = __double2half(x);
                    check("hip half", *f, x, __hip_cvt_halfraw_to_fp8(h, sat, interp),
                          amd_reference(double(__half2float(h)), sat == __HIP_SATFINITE, *f));
                }
            }
        }
        EXPECT(wrong == 0);

        // The edges where the vendors differ
        EXPECT(__nv_cvt_float_to_fp8(449.f, __NV_NOSAT, __NV_E4M3) == 0x7e);          // rounds down to 448
        EXPECT(__hip_cvt_float_to_fp8(449.f, __HIP_NOSAT, __HIP_E4M3) == 0x7f);       // overflows to NaN
        EXPECT(__nv_cvt_float_to_fp8(464.f, __NV_NOSAT, __NV_E4M3) == 0x7e);          // a tie, to even
        EXPECT(__nv_cvt_float_to_fp8(-465.f, __NV_NOSAT, __NV_E4M3) == 0xff);
        EXPECT(__nv_cvt_float_to_fp8(61440.f, __NV_NOSAT, __NV_E5M2) == 0x7c);        // a tie, to even: infinity
        EXPECT(__nv_cvt_float_to_fp8(INFINITY, __NV_SATFINITE, __NV_E5M2) == 0x7b);
        EXPECT(__hip_cvt_float_to_fp8(INFINITY, __HIP_SATFINITE, __HIP_E5M2) == 0x7c);
        EXPECT(__hip_cvt_float_to_fp8(-INFINITY, __HIP_SATFINITE, __HIP_E4M3_FNUZ) == 0x80);
        EXPECT(__hip_cvt_float_to_fp8(-1e-9f, __HIP_NOSAT, __HIP_E4M3_FNUZ) == 0x00);     // no negative zero
        EXPECT(__nv_cvt_float_to_fp8(-1e-9f, __NV_NOSAT, __NV_E4M3) == 0x80);
    }

    // Pairs and quadruples, low lane in the low byte
    {
        EXPECT(__nv_cvt_float2_to_fp8x2(make_float2(1.f, -2.f), __NV_SATFINITE, __NV_E4M3) == 0xc038);
        EXPECT(__nv_cvt_double2_to_fp8x2(make_double2(1e9, 0.5), __NV_SATFINITE, __NV_E5M2) == 0x387b);
        EXPECT(__nv_cvt_halfraw2_to_fp8x2(__floats2half2_rn(1.f, 2.f), __NV_NOSAT, __NV_E5M2) == 0x403c);
        EXPECT(__nv_cvt_bfloat16raw2_to_fp8x2(__floats2bfloat162_rn(1.f, 2.f), __NV_NOSAT, __NV_E4M3) == 0x4038);
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(0xc038, __NV_E4M3);
        EXPECT(__half2float(__half(__half_raw{h.x})) == 1.f && __half2float(__half(__half_raw{h.y})) == -2.f);
        EXPECT(__hip_cvt_float2_to_fp8x2(make_float2(1.f, -2.f), __HIP_SATFINITE, __HIP_E4M3_FNUZ) == 0xc840);
        const __half2_raw g = __hip_cvt_fp8x2_to_halfraw2(0xc840, __HIP_E4M3_FNUZ);
        EXPECT(__half2float(__half(__half_raw{g.x})) == 1.f && __half2float(__half(__half_raw{g.y})) == -2.f);

        const __nv_fp8x4_e5m2 v(make_float4(1.f, 2.f, 3.f, 1e6f));
        const float4 f = float4(v);
        EXPECT(f.x == 1.f && f.y == 2.f && f.z == 3.f && f.w == 57344.f && v.__x == 0x7b42403cu);
        const __nv_fp8x2_e4m3 p(__floats2half2_rn(0.5f, 500.f));
        const float2 q = float2(p);
        EXPECT(q.x == 0.5f && q.y == 448.f);
        const __half2 hq = __half2(p);
        EXPECT(__low2float(hq) == 0.5f && __high2float(hq) == 448.f);
        const __nv_fp8x4_e4m3 r(__floats2bfloat162_rn(1.f, 2.f), __floats2bfloat162_rn(3.f, 4.f));
        EXPECT(float4(r).w == 4.f && __nv_fp8x4_e4m3(make_double4(1, 2, 3, 4)).__x == r.__x);

        const __hip_fp8x4_e4m3_fnuz s(make_float4(1.f, -2.f, 300.f, NAN));
        const float4 t = s;
        EXPECT(t.x == 1.f && t.y == -2.f && t.z == 240.f && std::isnan(t.w));
        const __hip_fp8x2_e5m2 u = make_float2(1.f, 3.f);
        const float2 w = u;
        EXPECT(w.x == 1.f && w.y == 3.f && __half2(u) == __floats2half2_rn(1.f, 3.f));
    }

    // The classes: CUDA's convert explicitly, AMD's implicitly, both
    // saturating; to integers they truncate and saturate, NaN as 0
    {
        const __nv_fp8_e4m3 a(3.3f), big(1e9), nan(NAN);
        EXPECT(float(a) == 3.25f && double(big) == 448 && int(a) == 3 && (unsigned char)(big) == 255);
        EXPECT(int(nan) == 0 && bool(a) && !bool(__nv_fp8_e4m3(0.f)) && !bool(__nv_fp8_e4m3(-0.f)));
        EXPECT((signed char)(__nv_fp8_e4m3(-300.f)) == -128 && (long long)(__nv_fp8_e5m2(-57344.f)) == -57344);
        EXPECT(float(__nv_fp8_e5m2(100000)) == 57344.f && float(__nv_fp8_e4m3(17u)) == 16.f && float(__nv_fp8_e4m3(short(-9))) == -9.f);
        EXPECT(float(__nv_fp8_e5m2(__float2half(0.3f))) == 0.3125f && float(__nv_fp8_e4m3(__float2bfloat16(0.3f))) == 0.3125f);
        EXPECT(__bfloat162float(__nv_bfloat16(__nv_fp8_e5m2(1.5f))) == 1.5f);

        __hip_fp8_e4m3 b = 3.3f;
        float fb = b;
        int ib = b;
        EXPECT(fb == 3.25f && ib == 3 && float(__hip_fp8_e4m3(449.f)) == 448.f);       // saturating
        // AMD's conversions are implicit, so copy-initialization picks one:
        // __half(c) would find both __half(float) and the copy constructor
        __hip_fp8_e4m3_fnuz c = -2.5;
        const __half hc = c;
        const __hip_bfloat16 bc = c;
        EXPECT(double(c) == -2.5 && __half2float(hc) == -2.5f && __bfloat162float(bc) == -2.5f);
        __hip_fp8_e4m3_fnuz n;
        n.__x = 0x80;
        EXPECT(!bool(n) && int(n) == 0 && std::isnan(float(n)) && bool(c));
        EXPECT(__hip_fp8_e5m2_fnuz::__default_interpret == __HIP_E5M2_FNUZ && __hip_fp8_e4m3::__we == 4 && __hip_fp8_e4m3::__wm == 3);
        EXPECT(__hip_fp8_e5m2(short(7)).__x == __hip_cvt_float_to_fp8(7.f, __HIP_SATFINITE, __HIP_E5M2));
    }

    // E8M0: powers of two, toward zero or up, from every source type
    {
        unsigned wrong = 0;
        std::vector<double> inputs;
        for (int e = -140; e <= 140; ++e) {
            const double p = std::ldexp(1.0, e);
            for (double x : {p, std::nextafter(p, 0.0), std::nextafter(p, INFINITY), p * 1.5})
                inputs.insert(inputs.end(), {x, -x});
        }
        for (double x : {0.0, -0.0, double(INFINITY), -double(INFINITY), double(NAN), 1e300, 1e-300})
            inputs.push_back(x);
        for (double x : inputs) {
            for (auto sat : {__NV_NOSAT, __NV_SATFINITE}) {
                for (auto mode : {cudaRoundZero, cudaRoundPosInf}) {
                    const bool s = sat == __NV_SATFINITE;
                    wrong += __nv_cvt_double_to_e8m0(x, sat, mode) != e8m0_reference(x, s, mode);
                    wrong += __nv_cvt_float_to_e8m0(float(x), sat, mode) != e8m0_reference(float(x), s, mode);
                    const __nv_bfloat16 b = __double2bfloat16(x);
                    wrong += __nv_cvt_bfloat16raw_to_e8m0(b, sat, mode) != e8m0_reference(__bfloat162float(b), s, mode);
                    wrong += __hip_cvt_double_to_e8m0(x, sat == __NV_SATFINITE ? __HIP_SATFINITE : __HIP_NOSAT,
                                                      mode == cudaRoundZero ? hipRoundZero : hipRoundPosInf) !=
                             e8m0_reference(x, s, mode);
                }
            }
        }
        for (unsigned code = 0; code < 256; ++code) {
            const float expected = code == 255 ? NAN : std::ldexp(1.f, int(code) - 127);
            const float got = __bfloat162float(__nv_bfloat16(__nv_cvt_e8m0_to_bf16raw(__nv_fp8_storage_t(code))));
            wrong += code == 255 ? !std::isnan(got) : got != expected;
            __nv_fp8_e8m0 e;
            e.__x = __nv_fp8_storage_t(code);
            wrong += code == 255 ? !std::isnan(float(e)) : float(e) != expected || double(e) != double(expected);
        }
        EXPECT(wrong == 0);
        EXPECT(__nv_cvt_e8m0_to_bf16raw(0).x == 0x0040 && __nv_cvt_e8m0_to_bf16raw(0xff).x == 0x7fff);
        EXPECT(__hip_cvt_e8m0_to_bf16raw(127).x == 0x3f80);

        EXPECT(float(__nv_fp8_e8m0(3.f)) == 4.f && float(__nv_fp8_e8m0(4.0)) == 4.f && float(__nv_fp8_e8m0(-0.3f)) == 0.5f);
        EXPECT(float(__nv_fp8_e8m0(5u)) == 8.f && float(__nv_fp8_e8m0(-1024)) == 1024.f && __nv_fp8_e8m0(0).__x == 0);
        EXPECT(float(__nv_fp8_e8m0((1ll << 60) + 1)) == 0x1p61f && float(__nv_fp8_e8m0(1ull << 63)) == 0x1p63f);
        EXPECT(__nv_fp8_e8m0(INFINITY).__x == 0xfe && int(__nv_fp8_e8m0(3.f)) == 4 && (unsigned char)(__nv_fp8_e8m0(1e10f)) == 255);
        EXPECT(__half2float(__half(__nv_fp8_e8m0(0.25f))) == 0.25f && bool(__nv_fp8_e8m0(0.f)));
        const __nv_fp8x2_e8m0 pair(make_float2(3.f, 0.25f));
        EXPECT(pair.__x == ((125u << 8) | 129u) && float2(pair).x == 4.f && __high2float(__half2(pair)) == 0.25f);
        EXPECT(__bfloat162float(__nv_bfloat162(pair).y) == 0.25f && __nv_fp8x2_e8m0(__floats2half2_rn(3.f, 0.25f)).__x == pair.__x);
        EXPECT(__nv_fp8x2_e8m0(__floats2bfloat162_rn(3.f, 0.25f)).__x == pair.__x);
        EXPECT(__nv_cvt_float2_to_e8m0x2(make_float2(3.f, 0.25f), __NV_SATFINITE, cudaRoundPosInf) == pair.__x);
        const __nv_fp8x4_e8m0 quad(make_double4(1, 2, 3, 1e300));
        const float4 q = float4(quad);
        EXPECT(q.x == 1.f && q.y == 2.f && q.z == 4.f && q.w == 0x1p127f);
        EXPECT(__nv_fp8x4_e8m0(__floats2half2_rn(1.f, 2.f), __floats2half2_rn(3.f, 0.5f)).__x == 0x7e81807fu);
    }

    // In a kernel, as the compatibility suite's rocm-examples program does
    {
        float in[4] = {1.1f, -300.f, 0.01f, 1e6f};
        __nv_fp8_e4m3 e4m3_out[4];
        __nv_fp8_e5m2 e5m2_out[4];
        __hip_fp8_storage_t fnuz[4];
        float back[4];
        convert<<<1, 4>>>(in, e4m3_out, e5m2_out, fnuz, back);
        unsigned wrong = 0;
        for (int i = 0; i < 4; ++i) {
            wrong += e4m3_out[i].__x != cuda_reference(in[i], true, e4m3);
            wrong += e5m2_out[i].__x != cuda_reference(in[i], false, e5m2);
            wrong += fnuz[i] != amd_reference(in[i], true, e4m3_fnuz);
        }
        EXPECT(wrong == 0);
        EXPECT(std::isinf(back[3]) && back[0] == 1.125f + 1.f + 1.125f);        // E5M2 has 1.0 and 1.25
    }

    if (errors == 0)
        std::printf("fp8: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
