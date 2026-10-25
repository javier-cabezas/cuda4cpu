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
// Floating-point formats smaller than float: half (binary16), bfloat16 and
// the 8-bit formats. A value is held as its bits; to_bits rounds a double to
// them, exactly, in any of the four rounding modes, and from_bits gives the
// double that holds a value exactly.
//
// Operations on these formats convert to double, compute, and round once.
// Additions, subtractions, multiplications, divisions and square roots then
// round correctly: double has more than twice their precision plus two bits,
// which makes rounding twice harmless for those operations. A fused
// multiply-add needs more care (fma_bits).
//

#pragma once

#include <bit>
#include <cstdint>

namespace cuda4cpu::detail {

//! Rounding of a conversion or an operation
enum class float_rounding { nearest, toward_zero, down, up };

//! How a format encodes the values that aren't finite numbers
enum class float_specials {
    ieee,     //!< The largest exponent holds the infinities and the NaNs
    one_nan,  //!< No infinity: the largest exponent with every mantissa bit set is NaN (E4M3)
    fnuz,     //!< Finite and NaN-at-negative-zero: no infinity, no -0, and 0x80... is NaN
};

struct float_format {
    int exponent_bits;
    int mantissa_bits;
    int bias;
    float_specials specials;

    constexpr int bits() const { return 1 + exponent_bits + mantissa_bits; }
    constexpr uint32_t sign_bit() const { return uint32_t(1) << (exponent_bits + mantissa_bits); }

    //! The bits of the largest finite value
    constexpr uint32_t max_finite() const
    {
        const uint32_t top = (uint32_t(1) << exponent_bits) - 1, ones = (uint32_t(1) << mantissa_bits) - 1;
        if (specials == float_specials::ieee)
            return ((top - 1) << mantissa_bits) | ones;
        if (specials == float_specials::one_nan)
            return (top << mantissa_bits) | (ones - 1);
        return (top << mantissa_bits) | ones;
    }

    //! The bits of positive infinity (ieee formats only)
    constexpr uint32_t infinity() const { return ((uint32_t(1) << exponent_bits) - 1) << mantissa_bits; }

    //! The bits of the NaN that operations return
    constexpr uint32_t nan() const
    {
        if (specials == float_specials::fnuz)
            return sign_bit();
        return sign_bit() - 1;
    }

    constexpr bool is_nan(uint32_t bits) const
    {
        const uint32_t magnitude = bits & (sign_bit() - 1);
        if (specials == float_specials::ieee)
            return magnitude > infinity();
        if (specials == float_specials::one_nan)
            return magnitude == sign_bit() - 1;
        return bits == sign_bit();
    }

    constexpr bool is_infinity(uint32_t bits) const
    {
        return specials == float_specials::ieee && (bits & (sign_bit() - 1)) == infinity();
    }
};

inline constexpr float_format binary16_format{5, 10, 15, float_specials::ieee};
inline constexpr float_format bfloat16_format{8, 7, 127, float_specials::ieee};
inline constexpr float_format e4m3_format{4, 3, 7, float_specials::one_nan};
inline constexpr float_format e5m2_format{5, 2, 15, float_specials::ieee};
inline constexpr float_format e4m3_fnuz_format{4, 3, 8, float_specials::fnuz};
inline constexpr float_format e5m2_fnuz_format{5, 2, 16, float_specials::fnuz};

//! The bits of x rounded to format F as rounding says. Values beyond the
//! largest finite one become infinity, or NaN in formats without one, unless
//! saturate is set or the magnitude rounds down (toward zero, or away from
//! the sign's infinity): those give the largest finite value. NaN gives F's
//! NaN.
template <float_format F>
constexpr uint32_t to_bits(double x, float_rounding rounding = float_rounding::nearest, bool saturate = false)
{
    constexpr int M = F.mantissa_bits;
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const bool negative = (bits >> 63) != 0;
    const uint32_t sign = negative ? F.sign_bit() : 0;
    const int biased = int((bits >> 52) & 0x7ff);
    const uint64_t fraction = bits & ((uint64_t(1) << 52) - 1);

    // Whether to round the magnitude up (away from zero) or down
    bool magnitude_up = false;
    if (rounding == float_rounding::up)
        magnitude_up = !negative;
    else if (rounding == float_rounding::down)
        magnitude_up = negative;
    auto overflow = [&] {
        if (saturate || (rounding != float_rounding::nearest && !magnitude_up))
            return sign | F.max_finite();
        return F.specials == float_specials::ieee ? sign | F.infinity() : F.nan();
    };

    if (biased == 0x7ff) {
        if (fraction != 0)
            return F.nan();
        return saturate ? sign | F.max_finite() : F.specials == float_specials::ieee ? sign | F.infinity() : F.nan();
    }
    if (biased == 0 && fraction == 0)
        return F.specials == float_specials::fnuz ? 0 : sign;               // zero, which is exact

    // x is significand * 2^(exponent - 52)
    const uint64_t significand = biased != 0 ? fraction | (uint64_t(1) << 52) : fraction;
    const int exponent = biased != 0 ? biased - 1023 : -1022;
    if (exponent + F.bias >= (1 << F.exponent_bits))
        return overflow();                                                  // beyond every exponent of F

    // The value of F below |x| (base, as bits), and the remainder of |x|
    // above it, where half_unit is half a unit in the last place there
    const int min_exponent = 1 - F.bias;
    uint64_t base, remainder, half_unit;
    if (exponent >= min_exponent) {                                         // normal: M + 1 significant bits
        base = (uint64_t(exponent + F.bias) << M) + (significand >> (52 - M)) - (uint64_t(1) << M);
        remainder = significand & ((uint64_t(1) << (52 - M)) - 1);
        half_unit = uint64_t(1) << (51 - M);
    } else {                                                                // subnormal: units of 2^(min_exponent - M)
        const int shift = 52 - M + min_exponent - exponent < 63 ? 52 - M + min_exponent - exponent : 63;
        base = significand >> shift;
        remainder = significand & ((uint64_t(1) << shift) - 1);
        half_unit = uint64_t(1) << (shift - 1);
    }

    uint64_t result = base;
    if (rounding == float_rounding::nearest) {
        if (remainder > half_unit || (remainder == half_unit && (base & 1)))
            result = base + 1;
    } else if (magnitude_up && remainder != 0) {
        result = base + 1;
    }
    if (result > F.max_finite())
        return overflow();
    if (result == 0 && F.specials == float_specials::fnuz)
        return 0;
    return sign | uint32_t(result);
}

//! The value of the F number whose bits are b, which double holds exactly
template <float_format F>
constexpr double from_bits(uint32_t b)
{
    constexpr int M = F.mantissa_bits;
    const uint64_t sign = uint64_t((b & F.sign_bit()) != 0) << 63;
    const unsigned exponent = (b >> M) & ((1u << F.exponent_bits) - 1);
    const uint64_t mantissa = b & ((uint32_t(1) << M) - 1);
    if (F.is_nan(b))
        return std::bit_cast<double>(sign | 0x7ff8000000000000 | (mantissa << (52 - M)));
    if (F.is_infinity(b))
        return std::bit_cast<double>(sign | 0x7ff0000000000000);
    if (exponent == 0) {
        // mantissa units of 2^(1 - bias - M), a power of two double holds
        constexpr double unit = std::bit_cast<double>(uint64_t(1 - F.bias - M + 1023) << 52);
        const double magnitude = double(mantissa) * unit;
        return sign != 0 ? -magnitude : magnitude;
    }
    return std::bit_cast<double>(sign | (uint64_t(int(exponent) - F.bias + 1023) << 52) | (mantissa << (52 - M)));
}

//! a * b + c, rounded once to F, for a, b and c that F holds. The product is
//! exact in double (F has at most 25 significant bits), and so is the error
//! of the double sum (two-sum). Then the sum is rounded to odd: if inexact,
//! it moves to the neighbor whose last bit is 1. Rounding that to F, which
//! has at least two bits fewer, gives the correctly rounded result.
//! Rounding the double sum to nearest instead could land exactly on a tie
//! between two values of F that the exact sum isn't on.
template <float_format F>
constexpr uint32_t fma_bits(double a, double b, double c)
{
    static_assert(F.mantissa_bits <= 24);
    const double p = a * b;
    double s = p + c;
    const uint64_t s_bits = std::bit_cast<uint64_t>(s);
    if ((s_bits & 0x7ff0000000000000) != 0x7ff0000000000000 && (s_bits & 1) == 0) {
        const double pp = s - c, cc = s - pp;
        const double error = (p - pp) + (c - cc);
        if (error != 0)
            s = std::bit_cast<double>((error > 0) == (s > 0) ? s_bits + 1 : s_bits - 1);
    }
    return to_bits<F>(s);
}

}
