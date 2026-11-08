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
// HIP's AMD platform as an RDNA4 GPU (built with __HIP_PLATFORM_AMD__ and
// __gfx1201__): its macros and device properties, the split barrier, the
// conversions to and from the OCP 8-bit formats against references written
// apart from amdgcn.h (every code decoded from the format's widths, rounding
// to nearest by searching them, and stochastic rounding's closed form), and
// buffer resources: each out-of-range mode, the scalar offset, dword by
// dword checks, the lane as index, stores and atomics.
//

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>

#if !defined(__HIP_PLATFORM_AMD__) || !defined(__GFX12__) || defined(__GFX11__) || !defined(__gfx1201__) || \
    __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "the AMD platform's macros for RDNA4"
#endif

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

static_assert(__builtin_strcmp(__amdgcn_processor__, "gfx1201") == 0);

//
// The OCP 8-bit formats, decoded from their widths
//

struct format {
    int exponent_bits, mantissa_bits;
    bool has_infinity;                  // E5M2; E4M3 has only NaN, 0x7f
    int bias() const { return (1 << (exponent_bits - 1)) - 1; }
};

constexpr format e4m3{4, 3, false}, e5m2{5, 2, true};

static bool is_nan(format f, unsigned code)
{
    const unsigned magnitude = code & 0x7f;
    if (f.has_infinity)
        return (magnitude >> f.mantissa_bits) == (1u << f.exponent_bits) - 1 && (magnitude & ((1u << f.mantissa_bits) - 1));
    return magnitude == 0x7f;
}

static bool is_infinity(format f, unsigned code)
{
    return f.has_infinity && (code & 0x7f) == ((1u << f.exponent_bits) - 1) << f.mantissa_bits;
}

//! The value of a finite code
static double value(format f, unsigned code)
{
    const unsigned exponent = (code & 0x7f) >> f.mantissa_bits, mantissa = code & ((1u << f.mantissa_bits) - 1);
    const double magnitude = exponent == 0
                                 ? std::ldexp(mantissa, 1 - f.bias() - f.mantissa_bits)
                                 : std::ldexp((1 << f.mantissa_bits) + mantissa, int(exponent) - f.bias() - f.mantissa_bits);
    return code & 0x80 ? -magnitude : magnitude;
}

static double max_finite(format f)
{
    double best = 0;
    for (unsigned c = 0; c < 0x80; ++c)
        if (!is_nan(f, c) && !is_infinity(f, c))
            best = std::max(best, value(f, c));
    return best;
}

//! The code for a NaN, infinity or a magnitude beyond the largest finite one
static unsigned overflow(format f, bool negative)
{
    return (negative ? 0x80 : 0) | (f.has_infinity ? 0x7cu : 0x7fu);
}

//! The positive code whose value is exactly v, or -1
static int code_of(format f, double v)
{
    for (unsigned c = 0; c < 0x80; ++c)
        if (!is_nan(f, c) && !is_infinity(f, c) && value(f, c) == v)
            return int(c);
    return -1;
}

//! To nearest even, by searching every code
static unsigned nearest(format f, float x)
{
    const bool negative = std::signbit(x);
    if (std::isnan(x))
        return (negative ? 0x80 : 0) | 0x7f;
    const double magnitude = std::fabs(double(x));
    if (magnitude > max_finite(f))
        return overflow(f, negative);
    unsigned best = 0;
    for (unsigned c = 1; c < 0x80; ++c) {
        if (is_nan(f, c) || is_infinity(f, c))
            continue;
        const double d = std::fabs(value(f, c) - magnitude), b = std::fabs(value(f, best) - magnitude);
        if (d < b || (d == b && (c & 1) == 0))
            best = c;
    }
    return (negative ? 0x80 : 0) | best;
}

//! Stochastically: |x| in units of 2^-(23 - M) of the format's quantum at
//! |x| (truncated), plus rng's low 23 - M bits, cut to whole quanta
static unsigned stochastic(format f, float x, uint32_t rng)
{
    const bool negative = std::signbit(x);
    if (std::isnan(x))
        return (negative ? 0x80 : 0) | 0x7f;
    const double magnitude = std::fabs(double(x));
    if (magnitude > max_finite(f))
        return overflow(f, negative);
    if (magnitude == 0)
        return negative ? 0x80 : 0;
    const int drop = 23 - f.mantissa_bits;
    int exponent;
    std::frexp(magnitude, &exponent);                   // magnitude in [2^(exponent-1), 2^exponent)
    const int quantum = std::max(exponent - 1, 1 - f.bias()) - f.mantissa_bits;
    const uint64_t units = uint64_t(std::floor(std::ldexp(magnitude, drop - quantum)));
    const uint64_t quanta = (units + (rng & ((1u << drop) - 1))) >> drop;
    const double result = std::ldexp(double(quanta), quantum);
    if (result > max_finite(f))
        return overflow(f, negative);
    const int code = code_of(f, result);
    return code < 0 ? 0xffff : (negative ? 0x80 : 0) | unsigned(code);
}

static void conversions()
{
    // Every code to float, from each byte of the source
    for (unsigned c = 0; c < 256; ++c)
        for (int byte = 0; byte < 4; ++byte) {
            const int src = int(0xa5a5a5a5u & ~(0xffu << (8 * byte))) | int(c << (8 * byte));
            const float fp8 = __builtin_amdgcn_cvt_f32_fp8(src, byte), bf8 = __builtin_amdgcn_cvt_f32_bf8(src, byte);
            EXPECT(is_nan(e4m3, c) ? std::isnan(fp8) : double(fp8) == value(e4m3, c));
            if (is_nan(e5m2, c))
                EXPECT(std::isnan(bf8));
            else if (is_infinity(e5m2, c))
                EXPECT(std::isinf(bf8) && std::signbit(bf8) == bool(c & 0x80));
            else
                EXPECT(double(bf8) == value(e5m2, c));
        }

    // Floats around every code: the code, the midpoints, their neighbors,
    // and the special values, to nearest and stochastically
    std::vector<float> xs = {0.0f, -0.0f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::denorm_min(),
                             std::numeric_limits<float>::max(), 1e-30f, 448.0f, 449.0f, 464.0f, 57344.0f, 61440.0f};
    for (format f : {e4m3, e5m2})
        for (unsigned c = 0; c < 0x80; ++c) {
            if (is_nan(f, c) || is_infinity(f, c))
                continue;
            const float v = float(value(f, c));
            const float mid = float((value(f, c) + value(f, c + 1 == 0x80 ? c : c + 1)) / 2);
            for (float x : {v, mid})
                for (float y : {x, std::nextafter(x, 0.0f), std::nextafter(x, 1e30f)}) {
                    xs.push_back(y);
                    xs.push_back(-y);
                }
        }
    std::mt19937 random(2026);
    for (int i = 0; i < 20000; ++i) {
        const uint32_t sign = random() & 1, exponent = 127 - 24 + random() % 44, mantissa = random() & 0x7fffff;
        xs.push_back(std::bit_cast<float>(sign << 31 | exponent << 23 | mantissa));
    }
    for (float x : xs) {
        const unsigned fp8 = nearest(e4m3, x), bf8 = nearest(e5m2, x);
        EXPECT((__builtin_amdgcn_cvt_pk_fp8_f32(x, 1.0f, 0, false) & 0xffff) == (fp8 | 0x38u << 8));
        EXPECT((__builtin_amdgcn_cvt_pk_bf8_f32(x, 1.0f, 0, false) & 0xffff) == (bf8 | 0x3cu << 8));
        for (uint32_t rng : {0u, 0xffffffffu, uint32_t(random()), uint32_t(random()), 0x00080000u, 0x00100000u}) {
            const unsigned sr8 = stochastic(e4m3, x, rng), sb8 = stochastic(e5m2, x, rng);
            const int got8 = __builtin_amdgcn_cvt_sr_fp8_f32(x, int(rng), 0, 0) & 0xff;
            const int gotb = __builtin_amdgcn_cvt_sr_bf8_f32(x, int(rng), 0, 0) & 0xff;
            if (unsigned(got8) != sr8 || unsigned(gotb) != sb8) {
                std::printf("stochastic(%a, %08x): fp8 %02x, expected %02x; bf8 %02x, expected %02x\n", double(x), rng,
                            got8, sr8, gotb, sb8);
                ++errors;
                return;
            }
        }
    }

    // Stochastic rounding is unbiased: the mean of many roundings of a value
    // between two codes approaches it
    {
        const float x = 1.0f + 0.3f * 0.125f;           // 30% of the way from 1 to 1.125
        double sum = 0;
        for (int i = 0; i < 100000; ++i)
            sum += __builtin_amdgcn_cvt_f32_fp8(__builtin_amdgcn_cvt_sr_fp8_f32(x, int(random()), 0, 0), 0);
        EXPECT(std::fabs(sum / 100000 - double(x)) < 0.001);
    }

    // The rest of the register: the other word, or bytes, is kept
    EXPECT(__builtin_amdgcn_cvt_pk_fp8_f32(1.0f, -2.0f, int(0x12345678), false) == int(0x1234c038));
    EXPECT(__builtin_amdgcn_cvt_pk_fp8_f32(1.0f, -2.0f, int(0x12345678), true) == int(0xc0385678));
    EXPECT(__builtin_amdgcn_cvt_pk_bf8_f32(0.5f, 448.0f, 0, true) == int(0x5f380000));
    EXPECT(__builtin_amdgcn_cvt_sr_fp8_f32(1.0f, 0, int(0x12345678), 2) == int(0x12385678));
    EXPECT(__builtin_amdgcn_cvt_sr_bf8_f32(-1.0f, 0, int(0x12345678), 3) == int(0xbc345678));
#ifdef __clang__
    typedef float float2_t __attribute__((ext_vector_type(2)));
    const float2_t low = __builtin_amdgcn_cvt_pk_f32_fp8(int(0x40c03848), false);
    const float2_t high = __builtin_amdgcn_cvt_pk_f32_bf8(int(0x40c03848), true);
    EXPECT(low.x == 4.0f && low.y == 1.0f && high.x == -2.0f && high.y == 2.0f);
#endif
}

//
// Buffer resources
//

constexpr int out_of_range = 0x30000000, index_and_offset = 0, index_only = 0x10000000, num_records_zero = 0x20000000;
constexpr int add_tid = 1 << 23;

__global__ void buffers(uint32_t *memory, uint32_t *out)
{
    unsigned n = 0;
    // 3: the access must end within NUM_RECORDS minus the scalar offset
    __amdgpu_buffer_rsrc_t r = __builtin_amdgcn_make_buffer_rsrc(memory, 0, 20, out_of_range);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 16, 0, 0);     // [16, 20)
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 17, 0, 0);     // past the end: 0
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 12, 4, 0);     // within 16: memory[4]
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 13, 4, 0);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b8(r, 19, 0, 0);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b8(r, 20, 0, 0);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b16(r, 18, 0, 0);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b16(r, 19, 0, 0);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 1, 0, 0);      // unaligned
    __builtin_amdgcn_raw_buffer_store_b32(0xaaaaaaaa, r, 16, 0, 0);
    __builtin_amdgcn_raw_buffer_store_b32(0xbbbbbbbb, r, 20, 0, 0);   // dropped
    __builtin_amdgcn_raw_buffer_store_b16(0xcccc, r, 19, 0, 0);       // dropped
    __builtin_amdgcn_raw_buffer_store_b8(0xdd, r, 3, 0, 0);
    // 0: NUM_RECORDS nonzero, and the access within the stride
    r = __builtin_amdgcn_make_buffer_rsrc(memory, 8, 1, index_and_offset);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 4, 0, 0);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 8, 0, 0);
    r = __builtin_amdgcn_make_buffer_rsrc(memory, 8, 0, index_and_offset);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 4, 0, 0);
    // 1 and 2: NUM_RECORDS nonzero, anywhere
    r = __builtin_amdgcn_make_buffer_rsrc(memory, 0, 1, index_only);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 40, 0, 0);
    r = __builtin_amdgcn_make_buffer_rsrc(memory, 0, 0, index_only);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 0, 0, 0);
    r = __builtin_amdgcn_make_buffer_rsrc(memory, 0, 1, num_records_zero);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 44, 0, 0);
    r = __builtin_amdgcn_make_buffer_rsrc(memory, 0, 0, num_records_zero);
    out[n++] = __builtin_amdgcn_raw_buffer_load_b32(r, 0, 0, 0);
    // Atomics: the old value in range, else 0 and nothing changes
    r = __builtin_amdgcn_make_buffer_rsrc(memory + 12, 0, 8, out_of_range);
    out[n++] = uint32_t(__builtin_amdgcn_raw_ptr_buffer_atomic_add_i32(5, r, 0, 0, 0));
    out[n++] = uint32_t(__builtin_amdgcn_raw_ptr_buffer_atomic_add_i32(5, r, 8, 0, 0));
    out[n++] = std::bit_cast<uint32_t>(__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32(7.5f, r, 4, 0, 0));
#ifdef __clang__
    typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
    typedef unsigned int u32x3 __attribute__((ext_vector_type(3)));
    typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
    // Wider accesses, dword by dword
    r = __builtin_amdgcn_make_buffer_rsrc(memory + 20, 0, 20, out_of_range);
    const u32x4 four = __builtin_amdgcn_raw_buffer_load_b128(r, 8, 0, 0);
    const u32x2 two = __builtin_amdgcn_raw_buffer_load_b64(r, 16, 0, 0);
    out[n++] = four.x, out[n++] = four.y, out[n++] = four.z, out[n++] = four.w, out[n++] = two.x, out[n++] = two.y;
    __builtin_amdgcn_raw_buffer_store_b128(u32x4{1, 2, 3, 4}, r, 12, 0, 0);    // the last two dropped
    // Three dwords, though the vector is padded to four
    r = __builtin_amdgcn_make_buffer_rsrc(memory + 28, 0, 16, out_of_range);
    __builtin_amdgcn_raw_buffer_store_b96(u32x3{7, 8, 9}, r, 0, 0, 0);
    const u32x3 three = __builtin_amdgcn_raw_buffer_load_b96(r, 4, 0, 0);    // the last one past the end
    out[n++] = three.x, out[n++] = three.y, out[n++] = three.z;
#endif
}

//! With ADD_TID_ENABLE, each lane reads at its index times the stride
__global__ void lanes(uint32_t *memory, uint32_t *out)
{
    const __amdgpu_buffer_rsrc_t r = __builtin_amdgcn_make_buffer_rsrc(memory, 4, 4, out_of_range | add_tid);
    out[threadIdx.x] = __builtin_amdgcn_raw_buffer_load_b32(r, 0, 0, 0);
}

//! The split barrier: every thread's write is seen after the wait
__global__ void split_barrier(int *out)
{
    __shared__ int values[64];
    values[threadIdx.x] = int(threadIdx.x) * 3;
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_sleep_var(1);
    __builtin_amdgcn_s_barrier_wait(-1);
    out[threadIdx.x] = values[63 - threadIdx.x];
}

int main()
{
    hipDeviceProp_t properties;
    EXPECT(hipGetDeviceProperties(&properties, 0) == hipSuccess);
    EXPECT(std::strcmp(properties.gcnArchName, "gfx1201") == 0 && properties.major == 12 && properties.minor == 0);

    conversions();

    {
        uint32_t memory[64], out[64] = {};
        for (uint32_t i = 0; i < 64; ++i)
            memory[i] = 0x01010101u * i + 0x00010203u;
        memory[13] = std::bit_cast<uint32_t>(2.5f);
        uint32_t expected[64];
        std::memcpy(expected, memory, sizeof(memory));
        buffers<<<1, 1>>>(memory, out);
        const auto bytes = reinterpret_cast<const unsigned char *>(expected);
        uint16_t half;
        std::memcpy(&half, bytes + 18, 2);
        uint32_t unaligned;
        std::memcpy(&unaligned, bytes + 1, 4);
        const uint32_t want[] = {expected[4], 0, expected[4], 0, bytes[19], 0, half, 0, unaligned,
                                 expected[1], 0, 0, expected[10], 0, expected[11], 0, expected[12], 0,
                                 std::bit_cast<uint32_t>(2.5f)};
        for (unsigned i = 0; i < sizeof(want) / sizeof(want[0]); ++i)
            if (out[i] != want[i]) {
                std::printf("buffer result %u: %08x, expected %08x\n", i, out[i], want[i]);
                ++errors;
            }
        EXPECT(memory[4] == 0xaaaaaaaa && memory[5] == expected[5]);
        EXPECT((memory[0] >> 24) == 0xdd && (memory[0] & 0xffffff) == (expected[0] & 0xffffff));
        EXPECT(memory[12] == expected[12] + 5 && memory[14] == expected[14]);
        EXPECT(memory[13] == std::bit_cast<uint32_t>(7.5f));
#ifdef __clang__
        const uint32_t wide[] = {expected[22], expected[23], expected[24], 0, expected[24], 0};
        for (unsigned i = 0; i < 6; ++i)
            EXPECT(out[19 + i] == wide[i]);
        EXPECT(memory[23] == 1 && memory[24] == 2 && memory[25] == expected[25] && memory[26] == expected[26]);
        EXPECT(out[25] == 8 && out[26] == 9 && out[27] == expected[31]);
        EXPECT(memory[28] == 7 && memory[29] == 8 && memory[30] == 9 && memory[31] == expected[31]);
#endif
        lanes<<<1, 32>>>(memory, out);
        for (unsigned i = 0; i < 32; ++i)
            EXPECT(out[i] == memory[i]);
    }

    {
        int out[64];
        split_barrier<<<1, 64>>>(out);
        for (int i = 0; i < 64; ++i)
            EXPECT(out[i] == (63 - i) * 3);
    }

    if (errors == 0)
        std::printf("amdgcn_rdna4: all checks passed\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
