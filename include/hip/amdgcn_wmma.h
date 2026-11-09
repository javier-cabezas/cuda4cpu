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

// AMD's wave matrix builtins on 32-lane wavefronts: RDNA3's WMMA
// (__builtin_amdgcn_wmma_*_w32), RDNA4's (*_w32_gfx12) and RDNA4's sparse
// SWMMAC (__builtin_amdgcn_swmmac_*_w32). amdgcn.h includes it with Clang:
// they take and return Clang's ext_vector_type vectors.
//
// Each lane holds part of A, B and C, and gets part of D = A * B + C, a
// 16x16 tile. The layouts are AMD's Matrix Instruction Calculator's
// (InstCalcGfx11, InstCalcGfx12) and AMD's rocjitsu simulator's:
//
// - RDNA3: lane i and its copy i + 16 hold row i of A (column i of B),
//   element k at index k; the hardware reads the copy in the half of the
//   lane of D. Element r of lane l of C and D is (2r + l / 16, l % 16); a
//   16-bit D fills the low or high half (opsel) of each 32-bit element,
//   the other half keeping C's.
// - RDNA4: a lane holds half of a row's K. 16-bit A: lane 16 * (k / 4 % 2)
//   + i, element 4 * (k / 8) + k % 4; 8-bit: lane 16 * (k / 8) + i, element
//   k % 8; 4-bit, K = 32: lane 16 * (k / 16) + i, element k % 16. Element r
//   of lane l of C and D is (8 * (l / 16) + r, l % 16).
// - SWMMAC: A is 4:2 sparse, two of each four values of a row's K = 32 (or
//   64), with an index register whose two bits per value give its place in
//   its group of four. B is dense, and so are C and D, as RDNA4's.
//
// The arithmetic follows the hardware's order, as rocjitsu has it:
// products are exact; float results add them in float in K order (RDNA4's
// 16-bit inputs in its DOT4 order: K 0-3, 8-11, 4-7, 12-15), and 16-bit
// results are rounded at each block of the hardware, every 2 products on
// RDNA3 and 4 on RDNA4, to nearest even for half and by truncation for
// bfloat16. The rounding inside a block isn't the hardware's to the last
// bit. Integer results are exact, wrapped to 32 bits, or with clamp,
// saturated at each of the hardware's stages.
//
// They are lane operations: the lanes that take part are those that reach
// the call together (EXEC), as for amdgcn.h's; the hardware needs all of
// them, and a lane outside EXEC gives zeros here.

#pragma once

namespace cuda4cpu::detail::amdgcn::mm {

typedef _Float16 v8f16 __attribute__((ext_vector_type(8)));
typedef _Float16 v16f16 __attribute__((ext_vector_type(16)));
typedef short v8i16 __attribute__((ext_vector_type(8)));
typedef short v16i16 __attribute__((ext_vector_type(16)));
typedef float v8f32 __attribute__((ext_vector_type(8)));
typedef int v2i32 __attribute__((ext_vector_type(2)));
typedef int v4i32 __attribute__((ext_vector_type(4)));
typedef int v8i32 __attribute__((ext_vector_type(8)));

//! The element types of A and B
enum class kind { f16, bf16, fp8, bf8, int8, int4 };

constexpr unsigned bits_of(kind k)
{
    return k == kind::f16 || k == kind::bf16 ? 16 : k == kind::int4 ? 4 : 8;
}

//! A lane's operands that other lanes read, in each warp's slots (a lane
//! reads only its own C)
struct operands {
    unsigned char a[32], b[32];
    uint32_t index;
};

inline operands *warp_operands()
{
    static thread_local std::unique_ptr<operands[]> slots;
    if (!slots)
        slots.reset(new operands[1024]());
    return slots.get() + warp() * wavefront_size;
}

//! Element index of size bits in bytes
inline uint32_t element(const unsigned char *bytes, unsigned bits, unsigned index)
{
    if (bits == 16) {
        uint16_t value;
        std::memcpy(&value, bytes + 2 * index, 2);
        return value;
    }
    if (bits == 8)
        return bytes[index];
    return (bytes[index / 2] >> (4 * (index % 2))) & 15;
}

//! The warp's operands in one call, and which lanes gave them
struct warp_call {
    const operands *lanes;
    uint32_t exec;

    uint32_t a(unsigned lane, unsigned bits, unsigned index) const
    {
        return exec >> lane & 1 ? element(lanes[lane].a, bits, index) : 0;
    }
    uint32_t b(unsigned lane, unsigned bits, unsigned index) const
    {
        return exec >> lane & 1 ? element(lanes[lane].b, bits, index) : 0;
    }
    uint32_t index(unsigned lane) const { return exec >> lane & 1 ? lanes[lane].index : 0; }
};

inline float to_float(kind k, uint32_t raw)
{
    switch (k) {
    case kind::f16:  return float(::cuda4cpu::detail::from_bits<binary16_format>(raw));
    case kind::bf16: return std::bit_cast<float>(raw << 16);
    case kind::fp8:  return float(::cuda4cpu::detail::from_bits<e4m3_format>(raw));
    default:         return float(::cuda4cpu::detail::from_bits<e5m2_format>(raw));
    }
}

inline int64_t to_int(uint32_t raw, unsigned bits, bool sign)
{
    return sign && (raw >> (bits - 1) & 1) ? int64_t(raw) - (int64_t(1) << bits) : int64_t(raw);
}

//! A float accumulator rounded to a 16-bit result, and back
inline uint16_t narrow(kind out, float acc)
{
    if (out == kind::bf16)
        return uint16_t(std::bit_cast<uint32_t>(acc) >> 16);
    return uint16_t(::cuda4cpu::detail::to_bits<binary16_format>(double(acc)));
}

inline int64_t saturate(int64_t acc)
{
    return std::clamp<int64_t>(acc, INT32_MIN, INT32_MAX);
}

//! Gives the lane's operands to the warp's lanes that reach site with it,
//! and returns compute's D from all of theirs
template <typename D, typename A, typename B, typename F>
inline D run(uint64_t site, const A &a, const B &b, uint32_t index, F compute)
{
    static_assert(sizeof(A) <= 32 && sizeof(B) <= 32);
    const uint32_t ex = exec(site);
    operands *const all = warp_operands();
    operands &mine = all[lane()];
    std::memcpy(mine.a, &a, sizeof a);
    std::memcpy(mine.b, &b, sizeof b);
    mine.index = index;
    __syncwarp(ex);
    const D d = compute(warp_call{all, ex}, lane());
    __syncwarp(ex);
    return d;
}

//
// RDNA3
//

//! A float WMMA of 16-bit A and B: a 32-bit D (v8f32), or a 16-bit one in
//! the low or high halves (opsel), the others kept from C
template <kind In, typename D, typename C>
inline D gfx11_float(const warp_call &w, unsigned l, const C &c, bool packed, kind out, bool opsel)
{
    D d = c;
    const unsigned j = l % 16;
    for (unsigned r = 0; r < 8; ++r) {
        const unsigned i = 2 * r + l / 16, half = 16 * (i % 2);
        float acc;
        if (packed) {
            uint16_t bits;
            std::memcpy(&bits, reinterpret_cast<const char *>(&c) + 2 * (2 * r + opsel), 2);
            acc = to_float(out, bits);
        } else
            acc = float(c[r]);
        for (unsigned k = 0; k < 16; ++k) {
            acc += to_float(In, w.a(half + i, 16, k)) * to_float(In, w.b(half + j, 16, k));
            if (packed && k % 2 == 1)
                acc = to_float(out, narrow(out, acc));
        }
        if (packed) {
            const uint16_t bits = narrow(out, acc);
            std::memcpy(reinterpret_cast<char *>(&d) + 2 * (2 * r + opsel), &bits, 2);
        } else
            d[r] = acc;
    }
    return d;
}

//! An integer WMMA of 8- or 4-bit A and B, saturated after every 2 (8-bit)
//! or 4 (4-bit) products with clamp
inline v8i32 gfx11_int(const warp_call &w, unsigned l, const v8i32 &c, unsigned bits, bool sign_a, bool sign_b,
                       bool clamp)
{
    v8i32 d;
    const unsigned j = l % 16, group = bits == 8 ? 2 : 4;
    for (unsigned r = 0; r < 8; ++r) {
        const unsigned i = 2 * r + l / 16, half = 16 * (i % 2);
        int64_t acc = c[r];
        for (unsigned k = 0; k < 16; ++k) {
            acc += to_int(w.a(half + i, bits, k), bits, sign_a) * to_int(w.b(half + j, bits, k), bits, sign_b);
            if (clamp && (k + 1) % group == 0)
                acc = saturate(acc);
        }
        d[r] = int(uint32_t(acc));
    }
    return d;
}

//
// RDNA4
//

//! The lane and element of a dense A (row i) or B (column i) value at k
inline void gfx12_place(unsigned bits, unsigned K, unsigned i, unsigned k, unsigned &lane_of, unsigned &index)
{
    if (bits == 16) {
        lane_of = 16 * (k / 4 % 2) + i;
        index = 4 * (k / 8) + k % 4;
    } else if (bits == 8 || K == 16) {
        lane_of = 16 * (k / 8) + i;
        index = k % 8;
    } else {
        lane_of = 16 * (k / 16) + i;
        index = k % 16;
    }
}

//! RDNA4's DOT4 order of K for 16-bit inputs: 0-3, 8-11, 4-7, 12-15
constexpr unsigned gfx12_step_k(unsigned step)
{
    return (step & 3) | ((step & 4) << 1) | ((step & 8) >> 1);
}

//! A float WMMA: a 32-bit D (v8f32), or a 16-bit one (v8f16, v8i16)
template <kind In, typename D, typename C>
inline D gfx12_float(const warp_call &w, unsigned l, const C &c, bool packed, kind out)
{
    D d;
    constexpr unsigned bits = bits_of(In);
    const unsigned j = l % 16;
    for (unsigned r = 0; r < 8; ++r) {
        const unsigned i = 8 * (l / 16) + r;
        float acc;
        if (packed) {
            uint16_t raw;
            std::memcpy(&raw, reinterpret_cast<const char *>(&c) + 2 * r, 2);
            acc = to_float(out, raw);
        } else
            acc = float(c[r]);
        for (unsigned step = 0; step < 16; ++step) {
            const unsigned k = bits == 16 ? gfx12_step_k(step) : step;
            unsigned la, ia, lb, ib;
            gfx12_place(bits, 16, i, k, la, ia);
            gfx12_place(bits, 16, j, k, lb, ib);
            acc += to_float(In, w.a(la, bits, ia)) * to_float(In, w.b(lb, bits, ib));
            if (packed && step % 4 == 3)
                acc = to_float(out, narrow(out, acc));
        }
        if (packed) {
            const uint16_t raw = narrow(out, acc);
            std::memcpy(reinterpret_cast<char *>(&d) + 2 * r, &raw, 2);
        } else
            d[r] = acc;
    }
    return d;
}

//! FP8/BF8 A and B, which may differ, into a float D
inline v8f32 gfx12_fp8(const warp_call &w, unsigned l, const v8f32 &c, kind ka, kind kb)
{
    v8f32 d;
    const unsigned j = l % 16;
    for (unsigned r = 0; r < 8; ++r) {
        const unsigned i = 8 * (l / 16) + r;
        float acc = c[r];
        for (unsigned k = 0; k < 16; ++k) {
            unsigned la, ia, lb, ib;
            gfx12_place(8, 16, i, k, la, ia);
            gfx12_place(8, 16, j, k, lb, ib);
            acc += to_float(ka, w.a(la, 8, ia)) * to_float(kb, w.b(lb, 8, ib));
        }
        d[r] = acc;
    }
    return d;
}

//! The products of K's stage of two, with clamp: K = 16 in halves, K = 32
//! as blocks 0 and 2, then 1 and 3
constexpr unsigned gfx12_stage_k(unsigned stage, unsigned local)
{
    return 16 * (local / 8) + 8 * stage + local % 8;
}

inline v8i32 gfx12_int(const warp_call &w, unsigned l, const v8i32 &c, unsigned bits, unsigned K, bool sign_a,
                       bool sign_b, bool clamp)
{
    v8i32 d;
    const unsigned j = l % 16;
    for (unsigned r = 0; r < 8; ++r) {
        const unsigned i = 8 * (l / 16) + r;
        int64_t acc = c[r];
        for (unsigned stage = 0; stage < 2; ++stage) {
            for (unsigned local = 0; local < K / 2; ++local) {
                const unsigned k = gfx12_stage_k(stage, local);
                unsigned la, ia, lb, ib;
                gfx12_place(bits, K, i, k, la, ia);
                gfx12_place(bits, K, j, k, lb, ib);
                acc += to_int(w.a(la, bits, ia), bits, sign_a) * to_int(w.b(lb, bits, ib), bits, sign_b);
            }
            if (clamp)
                acc = saturate(acc);
        }
        d[r] = int(uint32_t(acc));
    }
    return d;
}

//
// SWMMAC: A sparse, the rest dense
//

//! Where row i's compressed value ck of A is, and the K it stands for
inline void sparse_a(const warp_call &w, unsigned bits, unsigned i, unsigned ck, unsigned &lane_of, unsigned &index,
                     unsigned &k)
{
    unsigned slot;
    if (bits == 16) {
        lane_of = i + 16 * (ck / 4 % 2);
        index = 4 * (ck / 8) + ck % 4;
        slot = ck % 4 + 4 * (ck / 8);
    } else if (bits == 8) {
        lane_of = i + 16 * (ck / 8);
        index = ck % 8;
        slot = ck % 8;
    } else {
        lane_of = i + 16 * (ck / 8 % 2);
        index = 8 * (ck / 16) + ck % 8;
        slot = index;
    }
    k = 4 * (ck / 2) + ((w.index(lane_of) >> (2 * slot)) & 3);
}

//! Where B's value at k of column j is, for a K of 32 (or 64, 4-bit)
inline void sparse_b(unsigned bits, unsigned j, unsigned k, unsigned &lane_of, unsigned &index)
{
    if (bits == 16) {
        lane_of = j + 16 * (k / 8 % 2);
        index = 8 * (k / 16) + k % 8;
    } else if (bits == 8) {
        lane_of = j + 16 * (k / 16);
        index = k % 16;
    } else {
        lane_of = j + 16 * (k / 16 % 2);
        index = 16 * (k / 32) + k % 16;
    }
}

//! A float SWMMAC: 16 compressed products in 4 stages, after each of which
//! a 16-bit D is rounded
template <typename D, typename C>
inline D sparse_float(const warp_call &w, unsigned l, const C &c, kind ka, kind kb, bool packed, kind out)
{
    D d;
    const unsigned bits = bits_of(ka), j = l % 16;
    for (unsigned r = 0; r < 8; ++r) {
        const unsigned i = 8 * (l / 16) + r;
        float acc;
        if (packed) {
            uint16_t raw;
            std::memcpy(&raw, reinterpret_cast<const char *>(&c) + 2 * r, 2);
            acc = to_float(out, raw);
        } else
            acc = float(c[r]);
        for (unsigned ck = 0; ck < 16; ++ck) {
            unsigned la, ia, k, lb, ib;
            sparse_a(w, bits, i, ck, la, ia, k);
            sparse_b(bits, j, k, lb, ib);
            acc += to_float(ka, w.a(la, bits, ia)) * to_float(kb, w.b(lb, bits, ib));
            if (packed && ck % 4 == 3)
                acc = to_float(out, narrow(out, acc));
        }
        if (packed) {
            const uint16_t raw = narrow(out, acc);
            std::memcpy(reinterpret_cast<char *>(&d) + 2 * r, &raw, 2);
        } else
            d[r] = acc;
    }
    return d;
}

//! An integer SWMMAC of K / 2 compressed products, in the stages of
//! RDNA4's dense ones with clamp
inline v8i32 sparse_int(const warp_call &w, unsigned l, const v8i32 &c, unsigned bits, unsigned K, bool sign_a,
                        bool sign_b, bool clamp)
{
    v8i32 d;
    const unsigned j = l % 16;
    for (unsigned r = 0; r < 8; ++r) {
        const unsigned i = 8 * (l / 16) + r;
        int64_t acc = c[r];
        for (unsigned stage = 0; stage < 2; ++stage) {
            for (unsigned local = 0; local < K / 4; ++local) {
                const unsigned ck = gfx12_stage_k(stage, local);
                unsigned la, ia, k, lb, ib;
                sparse_a(w, bits, i, ck, la, ia, k);
                sparse_b(bits, j, k, lb, ib);
                acc += to_int(w.a(la, bits, ia), bits, sign_a) * to_int(w.b(lb, bits, ib), bits, sign_b);
            }
            if (clamp)
                acc = saturate(acc);
        }
        d[r] = int(uint32_t(acc));
    }
    return d;
}

}

//
// The builtins: cuda4cpu_amdgcn_<name>, which take the call's place first,
// the macros at the end of this header, and the names as functions
//

#define CUDA4CPU_MM_RUN(D, compute) \
    cuda4cpu::detail::amdgcn::mm::run<D>(site, a, b, index, [&](const auto &w, unsigned l) { return compute; })

// RDNA3

#define CUDA4CPU_GFX11_FLOAT(name, In, AB)                                                                       \
    inline cuda4cpu::detail::amdgcn::mm::v8f32 cuda4cpu_amdgcn_##name(                                            \
        uint64_t site, cuda4cpu::detail::amdgcn::mm::AB a, cuda4cpu::detail::amdgcn::mm::AB b,                    \
        cuda4cpu::detail::amdgcn::mm::v8f32 c)                                                                    \
    {                                                                                                             \
        namespace m = cuda4cpu::detail::amdgcn::mm;                                                               \
        const uint32_t index = 0;                                                                                 \
        return CUDA4CPU_MM_RUN(m::v8f32, (m::gfx11_float<m::kind::In, m::v8f32>(w, l, c, false, m::kind::f16,     \
                                                                                false)));                         \
    }                                                                                                             \
    extern "C" inline cuda4cpu::detail::amdgcn::mm::v8f32 __builtin_amdgcn_##name(                                \
        cuda4cpu::detail::amdgcn::mm::AB a, cuda4cpu::detail::amdgcn::mm::AB b,                                   \
        cuda4cpu::detail::amdgcn::mm::v8f32 c)                                                                    \
    {                                                                                                             \
        return cuda4cpu_amdgcn_##name(0, a, b, c);                                                                \
    }
CUDA4CPU_GFX11_FLOAT(wmma_f32_16x16x16_f16_w32, f16, v16f16)
CUDA4CPU_GFX11_FLOAT(wmma_f32_16x16x16_bf16_w32, bf16, v16i16)
#undef CUDA4CPU_GFX11_FLOAT

#define CUDA4CPU_GFX11_PACKED(name, In, V)                                                                       \
    inline cuda4cpu::detail::amdgcn::mm::V cuda4cpu_amdgcn_##name(                                                \
        uint64_t site, cuda4cpu::detail::amdgcn::mm::V a, cuda4cpu::detail::amdgcn::mm::V b,                      \
        cuda4cpu::detail::amdgcn::mm::V c, bool opsel)                                                            \
    {                                                                                                             \
        namespace m = cuda4cpu::detail::amdgcn::mm;                                                               \
        const uint32_t index = 0;                                                                                 \
        return CUDA4CPU_MM_RUN(m::V, (m::gfx11_float<m::kind::In, m::V>(w, l, c, true, m::kind::In, opsel)));     \
    }                                                                                                             \
    extern "C" inline cuda4cpu::detail::amdgcn::mm::V __builtin_amdgcn_##name(                                    \
        cuda4cpu::detail::amdgcn::mm::V a, cuda4cpu::detail::amdgcn::mm::V b, cuda4cpu::detail::amdgcn::mm::V c, \
        bool opsel)                                                                                               \
    {                                                                                                             \
        return cuda4cpu_amdgcn_##name(0, a, b, c, opsel);                                                         \
    }
// The tied forms keep C's other halves, as these do: on a GPU, the untied
// ones leave them undefined
CUDA4CPU_GFX11_PACKED(wmma_f16_16x16x16_f16_w32, f16, v16f16)
CUDA4CPU_GFX11_PACKED(wmma_f16_16x16x16_f16_tied_w32, f16, v16f16)
CUDA4CPU_GFX11_PACKED(wmma_bf16_16x16x16_bf16_w32, bf16, v16i16)
CUDA4CPU_GFX11_PACKED(wmma_bf16_16x16x16_bf16_tied_w32, bf16, v16i16)
#undef CUDA4CPU_GFX11_PACKED

#define CUDA4CPU_GFX11_INT(name, bits, AB)                                                                       \
    inline cuda4cpu::detail::amdgcn::mm::v8i32 cuda4cpu_amdgcn_##name(                                            \
        uint64_t site, bool sign_a, cuda4cpu::detail::amdgcn::mm::AB a, bool sign_b,                              \
        cuda4cpu::detail::amdgcn::mm::AB b, cuda4cpu::detail::amdgcn::mm::v8i32 c, bool clamp)                    \
    {                                                                                                             \
        namespace m = cuda4cpu::detail::amdgcn::mm;                                                               \
        const uint32_t index = 0;                                                                                 \
        return CUDA4CPU_MM_RUN(m::v8i32, m::gfx11_int(w, l, c, bits, sign_a, sign_b, clamp));                     \
    }                                                                                                             \
    extern "C" inline cuda4cpu::detail::amdgcn::mm::v8i32 __builtin_amdgcn_##name(                                \
        bool sign_a, cuda4cpu::detail::amdgcn::mm::AB a, bool sign_b, cuda4cpu::detail::amdgcn::mm::AB b,         \
        cuda4cpu::detail::amdgcn::mm::v8i32 c, bool clamp)                                                        \
    {                                                                                                             \
        return cuda4cpu_amdgcn_##name(0, sign_a, a, sign_b, b, c, clamp);                                         \
    }
CUDA4CPU_GFX11_INT(wmma_i32_16x16x16_iu8_w32, 8, v4i32)
CUDA4CPU_GFX11_INT(wmma_i32_16x16x16_iu4_w32, 4, v2i32)
#undef CUDA4CPU_GFX11_INT

// RDNA4

#define CUDA4CPU_GFX12_FLOAT(name, In, AB, D, packed, out)                                                       \
    inline cuda4cpu::detail::amdgcn::mm::D cuda4cpu_amdgcn_##name(                                               \
        uint64_t site, cuda4cpu::detail::amdgcn::mm::AB a, cuda4cpu::detail::amdgcn::mm::AB b,                    \
        cuda4cpu::detail::amdgcn::mm::D c)                                                                        \
    {                                                                                                             \
        namespace m = cuda4cpu::detail::amdgcn::mm;                                                               \
        const uint32_t index = 0;                                                                                 \
        return CUDA4CPU_MM_RUN(m::D, (m::gfx12_float<m::kind::In, m::D>(w, l, c, packed, m::kind::out)));         \
    }                                                                                                             \
    extern "C" inline cuda4cpu::detail::amdgcn::mm::D __builtin_amdgcn_##name(                                    \
        cuda4cpu::detail::amdgcn::mm::AB a, cuda4cpu::detail::amdgcn::mm::AB b, cuda4cpu::detail::amdgcn::mm::D c) \
    {                                                                                                             \
        return cuda4cpu_amdgcn_##name(0, a, b, c);                                                                \
    }
CUDA4CPU_GFX12_FLOAT(wmma_f32_16x16x16_f16_w32_gfx12, f16, v8f16, v8f32, false, f16)
CUDA4CPU_GFX12_FLOAT(wmma_f32_16x16x16_bf16_w32_gfx12, bf16, v8i16, v8f32, false, f16)
CUDA4CPU_GFX12_FLOAT(wmma_f16_16x16x16_f16_w32_gfx12, f16, v8f16, v8f16, true, f16)
CUDA4CPU_GFX12_FLOAT(wmma_bf16_16x16x16_bf16_w32_gfx12, bf16, v8i16, v8i16, true, bf16)
#undef CUDA4CPU_GFX12_FLOAT

#define CUDA4CPU_GFX12_FP8(name, ka, kb)                                                                         \
    inline cuda4cpu::detail::amdgcn::mm::v8f32 cuda4cpu_amdgcn_##name(                                            \
        uint64_t site, cuda4cpu::detail::amdgcn::mm::v2i32 a, cuda4cpu::detail::amdgcn::mm::v2i32 b,              \
        cuda4cpu::detail::amdgcn::mm::v8f32 c)                                                                    \
    {                                                                                                             \
        namespace m = cuda4cpu::detail::amdgcn::mm;                                                               \
        const uint32_t index = 0;                                                                                 \
        return CUDA4CPU_MM_RUN(m::v8f32, m::gfx12_fp8(w, l, c, m::kind::ka, m::kind::kb));                        \
    }                                                                                                             \
    extern "C" inline cuda4cpu::detail::amdgcn::mm::v8f32 __builtin_amdgcn_##name(                                \
        cuda4cpu::detail::amdgcn::mm::v2i32 a, cuda4cpu::detail::amdgcn::mm::v2i32 b,                             \
        cuda4cpu::detail::amdgcn::mm::v8f32 c)                                                                    \
    {                                                                                                             \
        return cuda4cpu_amdgcn_##name(0, a, b, c);                                                                \
    }
CUDA4CPU_GFX12_FP8(wmma_f32_16x16x16_fp8_fp8_w32_gfx12, fp8, fp8)
CUDA4CPU_GFX12_FP8(wmma_f32_16x16x16_fp8_bf8_w32_gfx12, fp8, bf8)
CUDA4CPU_GFX12_FP8(wmma_f32_16x16x16_bf8_fp8_w32_gfx12, bf8, fp8)
CUDA4CPU_GFX12_FP8(wmma_f32_16x16x16_bf8_bf8_w32_gfx12, bf8, bf8)
#undef CUDA4CPU_GFX12_FP8

#define CUDA4CPU_GFX12_INT(name, bits, K, AB)                                                                    \
    inline cuda4cpu::detail::amdgcn::mm::v8i32 cuda4cpu_amdgcn_##name(uint64_t site, bool sign_a, AB a,           \
                                                                       bool sign_b, AB b,                         \
                                                                       cuda4cpu::detail::amdgcn::mm::v8i32 c,     \
                                                                       bool clamp)                                \
    {                                                                                                             \
        namespace m = cuda4cpu::detail::amdgcn::mm;                                                               \
        const uint32_t index = 0;                                                                                 \
        return CUDA4CPU_MM_RUN(m::v8i32, m::gfx12_int(w, l, c, bits, K, sign_a, sign_b, clamp));                  \
    }                                                                                                             \
    extern "C" inline cuda4cpu::detail::amdgcn::mm::v8i32 __builtin_amdgcn_##name(                                \
        bool sign_a, AB a, bool sign_b, AB b, cuda4cpu::detail::amdgcn::mm::v8i32 c, bool clamp)                  \
    {                                                                                                             \
        return cuda4cpu_amdgcn_##name(0, sign_a, a, sign_b, b, c, clamp);                                         \
    }
CUDA4CPU_GFX12_INT(wmma_i32_16x16x16_iu8_w32_gfx12, 8, 16, cuda4cpu::detail::amdgcn::mm::v2i32)
CUDA4CPU_GFX12_INT(wmma_i32_16x16x16_iu4_w32_gfx12, 4, 16, int)
CUDA4CPU_GFX12_INT(wmma_i32_16x16x32_iu4_w32_gfx12, 4, 32, cuda4cpu::detail::amdgcn::mm::v2i32)
#undef CUDA4CPU_GFX12_INT

// SWMMAC: C is also the result's register on a GPU

#define CUDA4CPU_SWMMAC_FLOAT(name, ka, kb, A, B, D, packed, out)                                                \
    inline cuda4cpu::detail::amdgcn::mm::D cuda4cpu_amdgcn_##name(                                               \
        uint64_t site, cuda4cpu::detail::amdgcn::mm::A a, cuda4cpu::detail::amdgcn::mm::B b,                      \
        cuda4cpu::detail::amdgcn::mm::D c, int index)                                                             \
    {                                                                                                             \
        namespace m = cuda4cpu::detail::amdgcn::mm;                                                               \
        return CUDA4CPU_MM_RUN(m::D, (m::sparse_float<m::D>(w, l, c, m::kind::ka, m::kind::kb, packed,            \
                                                            m::kind::out)));                                      \
    }                                                                                                             \
    extern "C" inline cuda4cpu::detail::amdgcn::mm::D __builtin_amdgcn_##name(                                    \
        cuda4cpu::detail::amdgcn::mm::A a, cuda4cpu::detail::amdgcn::mm::B b, cuda4cpu::detail::amdgcn::mm::D c,  \
        int index)                                                                                                \
    {                                                                                                             \
        return cuda4cpu_amdgcn_##name(0, a, b, c, index);                                                         \
    }
CUDA4CPU_SWMMAC_FLOAT(swmmac_f32_16x16x32_f16_w32, f16, f16, v8f16, v16f16, v8f32, false, f16)
CUDA4CPU_SWMMAC_FLOAT(swmmac_f32_16x16x32_bf16_w32, bf16, bf16, v8i16, v16i16, v8f32, false, f16)
CUDA4CPU_SWMMAC_FLOAT(swmmac_f16_16x16x32_f16_w32, f16, f16, v8f16, v16f16, v8f16, true, f16)
CUDA4CPU_SWMMAC_FLOAT(swmmac_bf16_16x16x32_bf16_w32, bf16, bf16, v8i16, v16i16, v8i16, true, bf16)
CUDA4CPU_SWMMAC_FLOAT(swmmac_f32_16x16x32_fp8_fp8_w32, fp8, fp8, v2i32, v4i32, v8f32, false, f16)
CUDA4CPU_SWMMAC_FLOAT(swmmac_f32_16x16x32_fp8_bf8_w32, fp8, bf8, v2i32, v4i32, v8f32, false, f16)
CUDA4CPU_SWMMAC_FLOAT(swmmac_f32_16x16x32_bf8_fp8_w32, bf8, fp8, v2i32, v4i32, v8f32, false, f16)
CUDA4CPU_SWMMAC_FLOAT(swmmac_f32_16x16x32_bf8_bf8_w32, bf8, bf8, v2i32, v4i32, v8f32, false, f16)
#undef CUDA4CPU_SWMMAC_FLOAT

#define CUDA4CPU_SWMMAC_INT(name, bits, K, A, B)                                                                 \
    inline cuda4cpu::detail::amdgcn::mm::v8i32 cuda4cpu_amdgcn_##name(uint64_t site, bool sign_a, A a,            \
                                                                       bool sign_b, B b,                          \
                                                                       cuda4cpu::detail::amdgcn::mm::v8i32 c,     \
                                                                       int index, bool clamp)                     \
    {                                                                                                             \
        namespace m = cuda4cpu::detail::amdgcn::mm;                                                               \
        return CUDA4CPU_MM_RUN(m::v8i32, m::sparse_int(w, l, c, bits, K, sign_a, sign_b, clamp));                 \
    }                                                                                                             \
    extern "C" inline cuda4cpu::detail::amdgcn::mm::v8i32 __builtin_amdgcn_##name(                                \
        bool sign_a, A a, bool sign_b, B b, cuda4cpu::detail::amdgcn::mm::v8i32 c, int index, bool clamp)         \
    {                                                                                                             \
        return cuda4cpu_amdgcn_##name(0, sign_a, a, sign_b, b, c, index, clamp);                                  \
    }
CUDA4CPU_SWMMAC_INT(swmmac_i32_16x16x32_iu8_w32, 8, 32, cuda4cpu::detail::amdgcn::mm::v2i32,
                    cuda4cpu::detail::amdgcn::mm::v4i32)
CUDA4CPU_SWMMAC_INT(swmmac_i32_16x16x32_iu4_w32, 4, 32, int, cuda4cpu::detail::amdgcn::mm::v2i32)
CUDA4CPU_SWMMAC_INT(swmmac_i32_16x16x64_iu4_w32, 4, 64, cuda4cpu::detail::amdgcn::mm::v2i32,
                    cuda4cpu::detail::amdgcn::mm::v4i32)
#undef CUDA4CPU_SWMMAC_INT

#undef CUDA4CPU_MM_RUN

// The macros, which pass the call's place
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(...) cuda4cpu_amdgcn_wmma_f32_16x16x16_f16_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(...) cuda4cpu_amdgcn_wmma_f32_16x16x16_bf16_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(...) cuda4cpu_amdgcn_wmma_f16_16x16x16_f16_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32(...) cuda4cpu_amdgcn_wmma_f16_16x16x16_f16_tied_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(...) cuda4cpu_amdgcn_wmma_bf16_16x16x16_bf16_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w32(...) cuda4cpu_amdgcn_wmma_bf16_16x16x16_bf16_tied_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(...) cuda4cpu_amdgcn_wmma_i32_16x16x16_iu8_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(...) cuda4cpu_amdgcn_wmma_i32_16x16x16_iu4_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(...) cuda4cpu_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(...) cuda4cpu_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(...) cuda4cpu_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(...) cuda4cpu_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(...) cuda4cpu_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(...) cuda4cpu_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(...) cuda4cpu_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(...) cuda4cpu_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(...) cuda4cpu_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(...) cuda4cpu_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(...) cuda4cpu_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(...) cuda4cpu_amdgcn_swmmac_f32_16x16x32_f16_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(...) cuda4cpu_amdgcn_swmmac_f32_16x16x32_bf16_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32(...) cuda4cpu_amdgcn_swmmac_f16_16x16x32_f16_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32(...) cuda4cpu_amdgcn_swmmac_bf16_16x16x32_bf16_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(...) cuda4cpu_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(...) cuda4cpu_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(...) cuda4cpu_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(...) cuda4cpu_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(...) cuda4cpu_amdgcn_swmmac_i32_16x16x32_iu8_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32(...) cuda4cpu_amdgcn_swmmac_i32_16x16x32_iu4_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(...) cuda4cpu_amdgcn_swmmac_i32_16x16x64_iu4_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
