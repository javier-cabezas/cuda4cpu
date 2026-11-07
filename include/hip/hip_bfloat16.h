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

// HIP's legacy bfloat16 type, hip_bfloat16, which rocPRIM and other ROCm
// libraries use (hip_bf16.h's __hip_bfloat16 is the newer one): the upper
// half of a float, converted to nearest even, or truncated, with NaNs kept
// NaNs, and arithmetic in float.
#pragma once

#include <bit>
#include <cstdint>
#include <ostream>
#include <type_traits>

struct hip_bfloat16 {
    uint16_t data;

    enum truncate_t { truncate };

    hip_bfloat16() = default;

    //! To nearest even
    explicit hip_bfloat16(float f) : data(float_to_bfloat16(f)) {}

    //! The upper 16 bits
    explicit hip_bfloat16(float f, truncate_t) : data(truncate_float_to_bfloat16(f)) {}

    operator float() const
    {
        return std::bit_cast<float>(uint32_t(data) << 16);
    }

    hip_bfloat16 &operator=(const float &f)
    {
        data = float_to_bfloat16(f);
        return *this;
    }

    static hip_bfloat16 round_to_bfloat16(float f)
    {
        hip_bfloat16 result;
        result.data = float_to_bfloat16(f);
        return result;
    }

    static hip_bfloat16 round_to_bfloat16(float f, truncate_t)
    {
        hip_bfloat16 result;
        result.data = truncate_float_to_bfloat16(f);
        return result;
    }

    //! To nearest even. A NaN whose payload is all in the lower half keeps a
    //! bit of it, so that it stays a NaN.
    static uint16_t float_to_bfloat16(float f)
    {
        uint32_t bits = std::bit_cast<uint32_t>(f);
        if ((bits & 0x7f800000u) != 0x7f800000u)
            bits += 0x7fffu + ((bits >> 16) & 1);
        else if ((bits & 0xffffu) != 0)
            bits |= 0x10000u;
        return uint16_t(bits >> 16);
    }

    static uint16_t truncate_float_to_bfloat16(float f)
    {
        const uint32_t bits = std::bit_cast<uint32_t>(f);
        const bool nan_in_lower_half = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0xffffu) != 0;
        return uint16_t(bits >> 16) | uint16_t(nan_in_lower_half);
    }
};

static_assert(std::is_trivial_v<hip_bfloat16> && std::is_standard_layout_v<hip_bfloat16> &&
              sizeof(hip_bfloat16) == 2);

inline std::ostream &operator<<(std::ostream &os, const hip_bfloat16 &value)
{
    return os << float(value);
}

inline hip_bfloat16 operator+(hip_bfloat16 a) { return a; }
inline hip_bfloat16 operator-(hip_bfloat16 a)
{
    a.data ^= 0x8000;
    return a;
}
inline hip_bfloat16 operator+(hip_bfloat16 a, hip_bfloat16 b) { return hip_bfloat16(float(a) + float(b)); }
inline hip_bfloat16 operator-(hip_bfloat16 a, hip_bfloat16 b) { return hip_bfloat16(float(a) - float(b)); }
inline hip_bfloat16 operator*(hip_bfloat16 a, hip_bfloat16 b) { return hip_bfloat16(float(a) * float(b)); }
inline hip_bfloat16 operator/(hip_bfloat16 a, hip_bfloat16 b) { return hip_bfloat16(float(a) / float(b)); }
inline bool operator<(hip_bfloat16 a, hip_bfloat16 b) { return float(a) < float(b); }
inline bool operator==(hip_bfloat16 a, hip_bfloat16 b) { return float(a) == float(b); }
inline bool operator>(hip_bfloat16 a, hip_bfloat16 b) { return b < a; }
inline bool operator<=(hip_bfloat16 a, hip_bfloat16 b) { return !(a > b); }
inline bool operator!=(hip_bfloat16 a, hip_bfloat16 b) { return !(a == b); }
inline bool operator>=(hip_bfloat16 a, hip_bfloat16 b) { return !(a < b); }
inline hip_bfloat16 &operator+=(hip_bfloat16 &a, hip_bfloat16 b) { return a = a + b; }
inline hip_bfloat16 &operator-=(hip_bfloat16 &a, hip_bfloat16 b) { return a = a - b; }
inline hip_bfloat16 &operator*=(hip_bfloat16 &a, hip_bfloat16 b) { return a = a * b; }
inline hip_bfloat16 &operator/=(hip_bfloat16 &a, hip_bfloat16 b) { return a = a / b; }
inline hip_bfloat16 &operator++(hip_bfloat16 &a) { return a += hip_bfloat16(1.0f); }
inline hip_bfloat16 &operator--(hip_bfloat16 &a) { return a -= hip_bfloat16(1.0f); }
inline hip_bfloat16 operator++(hip_bfloat16 &a, int)
{
    hip_bfloat16 old = a;
    ++a;
    return old;
}
inline hip_bfloat16 operator--(hip_bfloat16 &a, int)
{
    hip_bfloat16 old = a;
    --a;
    return old;
}

// As HIP's header declares them
namespace std {
constexpr bool isinf(hip_bfloat16 a) { return (a.data & 0x7fff) == 0x7f80; }
constexpr bool isnan(hip_bfloat16 a) { return (a.data & 0x7f80) == 0x7f80 && (a.data & 0x7f) != 0; }
constexpr bool iszero(hip_bfloat16 a) { return (a.data & 0x7fff) == 0; }
}
