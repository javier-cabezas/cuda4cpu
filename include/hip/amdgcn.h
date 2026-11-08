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

// AMD's GPU builtins (__builtin_amdgcn_*), for HIP's AMD platform (see
// hip_common.h), which hip_runtime.h includes this header on. They emulate an
// RDNA3 GPU, whose wavefronts have 32 lanes, as cuda4cpu's warps.
//
// - The lane operations exchange the values of the lanes that take part:
//   EXEC, the lanes of the warp that reach the call together, as
//   __activemask() finds them (see the macros at the end). A lane gets
//   nothing from a lane outside EXEC
//   (zero, or its old value where the instruction keeps it), where a GPU may
//   give a stale value. DPP, ds_swizzle and permlane follow the encodings of
//   LLVM's AMDGPU backend and AMD's ISA manuals: ds_swizzle's FFT and rotate
//   modes aren't supported.
// - Builtins of any AMD GPU compile, not only RDNA3's: as with the compiler
//   for a GPU without them, code is expected to check the gfx macros.
// - They are inline functions, not the compiler's builtins, so
//   __has_builtin(__builtin_amdgcn_...) is false. Those that aren't templates
//   have C linkage, as builtins do, for code that declares them
//   (extern "C" void __builtin_amdgcn_s_sleep(int);, in rocPRIM).
// - Those that take or return Clang's ext_vector_type vectors (the matrix
//   operations, packed conversions) aren't here yet.
#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <type_traits>

#include "../cuda_runtime.h"

namespace cuda4cpu::detail::amdgcn {

constexpr unsigned wavefront_size = 32;

template <typename T>
inline uint64_t to_bits(T value)
{
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) <= 8, "a lane exchanges up to 8 bytes");
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

template <typename T>
inline T from_bits(uint64_t bits)
{
    T value;
    std::memcpy(&value, &bits, sizeof(T));
    return value;
}

inline unsigned lane()
{
    return thread_block::lane_id();
}

//! The calling CUDA thread's warp in its block
inline unsigned warp()
{
    const auto &t = thread_block::get_thread(), &d = thread_block::get_block_dim();
    return ((t.z * d.y + t.y) * d.x + t.x) / wavefront_size;
}

[[noreturn]] inline void unsupported(const char *what)
{
    std::fprintf(stderr, "cuda4cpu: %s\n", what);
    std::abort();
}

//! The values of the warp's lanes, and which lanes gave theirs (EXEC)
struct lanes {
    uint64_t value[wavefront_size];
    uint32_t exec;

    bool active(unsigned l) const
    {
        return l < wavefront_size && (exec >> l & 1) != 0;
    }
};

//! A slot per lane of each warp of the block. A block runs on one OS thread,
//! so a thread-local buffer is the block's, like __shared__.
inline uint64_t *warp_slots()
{
    static thread_local std::unique_ptr<uint64_t[]> slots;
    if (!slots)
        slots.reset(new uint64_t[(1024 / wavefront_size) * wavefront_size]());
    return slots.get() + warp() * wavefront_size;
}

//! The lanes of the calling warp that reach the call at site together,
//! which a GPU runs the instruction for: lanes in another branch are at
//! another call
inline uint32_t exec(uint64_t site)
{
    return thread_block::warp_active_mask(site);
}

//! Gives the calling lane's value to the lanes that reach the call at site
//! with it (EXEC), and gets theirs. The first sync waits for them to write
//! their slots, the second keeps the slots from being overwritten before
//! everyone has read them.
inline lanes exchange(uint64_t site, uint64_t value)
{
    lanes all;
    all.exec = exec(site);
    uint64_t *const slots = warp_slots();
    slots[lane()] = value;
    __syncwarp(all.exec);
    std::memcpy(all.value, slots, sizeof all.value);
    __syncwarp(all.exec);
    return all;
}

//! The lane that a lane reads with DPP control ctrl, or -1 out of bounds
inline int dpp_source(unsigned l, int ctrl)
{
    const unsigned row = l & ~15u, i = l & 15u, n = unsigned(ctrl) & 15u;
    if (ctrl >= 0 && ctrl <= 0xff)              // quad_perm
        return int((l & ~3u) | ((unsigned(ctrl) >> (2 * (l & 3))) & 3));
    if (ctrl >= 0x101 && ctrl <= 0x10f)         // row_shl
        return i + n < 16 ? int(l + n) : -1;
    if (ctrl >= 0x111 && ctrl <= 0x11f)         // row_shr
        return i >= n ? int(l - n) : -1;
    if (ctrl >= 0x121 && ctrl <= 0x12f)         // row_ror
        return int(row | ((i - n) & 15));
    if (ctrl >= 0x150 && ctrl <= 0x15f)         // row_share
        return int(row | n);
    if (ctrl >= 0x160 && ctrl <= 0x16f)         // row_xmask
        return int(row | (i ^ n));
    switch (ctrl) {
    case 0x130:                                 // wave_shl:1
        return l + 1 < wavefront_size ? int(l + 1) : -1;
    case 0x134:                                 // wave_rol:1
        return int((l + 1) % wavefront_size);
    case 0x138:                                 // wave_shr:1
        return l >= 1 ? int(l - 1) : -1;
    case 0x13c:                                 // wave_ror:1
        return int((l + wavefront_size - 1) % wavefront_size);
    case 0x140:                                 // row_mirror
        return int(row | (15 - i));
    case 0x141:                                 // row_half_mirror
        return int((l & ~7u) | (7 - (l & 7)));
    case 0x142:                                 // row_bcast:15, from the previous row's last lane
        return row >= 16 ? int(row - 1) : -1;
    case 0x143:                                 // row_bcast:31, to rows 2 and 3
        return row >= 32 ? 31 : -1;
    }
    unsupported("invalid DPP control");
}

template <typename T>
inline T update_dpp(uint64_t site, T old, T src, int ctrl, int row_mask, int bank_mask, bool bound_ctrl)
{
    const lanes all = exchange(site, to_bits(src));
    const unsigned l = lane();
    const int from = dpp_source(l, ctrl);
    // Lanes of the rows and banks that the masks leave out keep their old value
    if (((row_mask >> (l >> 4)) & 1) == 0 || ((bank_mask >> ((l >> 2) & 3)) & 1) == 0)
        return old;
    // Out of bounds or inactive: zero with bound_ctrl, else the old value
    if (from < 0 || !all.active(unsigned(from)))
        return bound_ctrl ? from_bits<T>(0) : old;
    return from_bits<T>(all.value[from]);
}

//! The lane that a lane reads with ds_swizzle's offset
inline unsigned swizzle_source(unsigned l, int offset)
{
    const unsigned p = unsigned(offset) & 0xffff;
    if ((p & 0x8000) == 0) {
        // Bit masks: and, or, xor, within groups of 32 lanes
        const unsigned and_mask = p & 31, or_mask = (p >> 5) & 31, xor_mask = (p >> 10) & 31;
        return (l & ~31u) | ((((l & 31) & and_mask) | or_mask) ^ xor_mask);
    }
    if ((p & 0xff00) == 0x8000)                 // a permutation of each quad
        return (l & ~3u) | ((p >> (2 * (l & 3))) & 3);
    unsupported("ds_swizzle's FFT and rotate modes (offsets from 0xC000) aren't supported");
}

template <typename T>
inline T permlane16(uint64_t site, T old, T src, unsigned lo, unsigned hi, bool fi, bool bound_ctrl, bool other_row)
{
    const lanes all = exchange(site, to_bits(src));
    const unsigned l = lane(), i = l & 15;
    const unsigned select = (i < 8 ? lo >> (4 * i) : hi >> (4 * (i - 8))) & 15;
    const unsigned from = ((l & ~15u) ^ (other_row ? 16u : 0u)) | select;
    if (!all.active(from))
        return fi || bound_ctrl ? from_bits<T>(0) : old;
    return from_bits<T>(all.value[from]);
}

//! The reduction of the active lanes' values, in their order
template <typename T, typename Op>
inline T reduce(uint64_t site, T value, Op op)
{
    const lanes all = exchange(site, to_bits(value));
    bool first = true;
    T result{};
    for (unsigned l = 0; l < wavefront_size; ++l) {
        if (!all.active(l))
            continue;
        const T v = from_bits<T>(all.value[l]);
        result = first ? v : op(result, v);
        first = false;
    }
    return result;
}

//! The same, from an identity: for subtractions, 0 - v0 - v1 ...
template <typename T, typename Op>
inline T fold(uint64_t site, T value, T identity, Op op)
{
    const lanes all = exchange(site, to_bits(value));
    T result = identity;
    for (unsigned l = 0; l < wavefront_size; ++l)
        if (all.active(l))
            result = op(result, from_bits<T>(all.value[l]));
    return result;
}

//! The lowest bits of a dot product's terms, signed or not
inline int64_t field(uint32_t x, unsigned index, unsigned bits, bool is_signed)
{
    const uint32_t v = (x >> (index * bits)) & ((1u << bits) - 1);
    return is_signed && (v >> (bits - 1)) != 0 ? int64_t(v) - (int64_t(1) << bits) : int64_t(v);
}

inline int64_t dot(uint32_t a, bool a_signed, uint32_t b, bool b_signed, unsigned bits)
{
    int64_t sum = 0;
    for (unsigned k = 0; k < 32 / bits; ++k)
        sum += field(a, k, bits, a_signed) * field(b, k, bits, b_signed);
    return sum;
}

inline int32_t clamp_int(int64_t v, bool clamp)
{
    if (clamp)
        return int32_t(std::clamp<int64_t>(v, INT32_MIN, INT32_MAX));
    return int32_t(uint32_t(uint64_t(v)));
}

inline uint32_t clamp_uint(int64_t v, bool clamp)
{
    if (clamp)
        return uint32_t(std::clamp<int64_t>(v, 0, UINT32_MAX));
    return uint32_t(uint64_t(v));
}

//! sin(2 pi x) and cos(2 pi x), as the hardware's: x in turns. The
//! reduction to [-0.5, 0.5] makes -0 a +0, so sin keeps a zero as it is.
inline double turns(double x)
{
    return 2 * std::numbers::pi * (x - std::nearbyint(x));
}

template <typename T>
inline T fract(T x)
{
    const T f = x - std::floor(x);
    return std::min(f, std::nextafter(T(1), T(0)));
}

template <typename T>
inline bool fp_class(T x, int mask)
{
    int bit;
    const bool negative = std::signbit(x);
    switch (std::fpclassify(x)) {
    case FP_NAN: {
        // Signaling: the most significant bit of the mantissa is clear
        using U = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
        const U bits = std::bit_cast<U>(x);
        bit = (bits >> (std::numeric_limits<T>::digits - 2) & 1) != 0 ? 1 : 0;
        break;
    }
    case FP_INFINITE:  bit = negative ? 2 : 9; break;
    case FP_NORMAL:    bit = negative ? 3 : 8; break;
    case FP_SUBNORMAL: bit = negative ? 4 : 7; break;
    default:           bit = negative ? 5 : 6; break;
    }
    return (mask >> bit & 1) != 0;
}

template <typename T>
inline T med3(T a, T b, T c)
{
    if (std::isnan(a) || std::isnan(b) || std::isnan(c))
        return std::fmin(std::fmin(a, b), c);
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

template <typename T, typename Update>
inline T atomic_update(T *address, Update update)
{
    std::atomic_ref<T> ref(*address);
    T old = ref.load();
    while (!ref.compare_exchange_weak(old, update(old))) {
    }
    return old;
}

inline std::memory_order memory_order(unsigned order)
{
    switch (order) {
    case __ATOMIC_RELAXED: return std::memory_order_relaxed;
    case __ATOMIC_CONSUME:
    case __ATOMIC_ACQUIRE: return std::memory_order_acquire;
    case __ATOMIC_RELEASE: return std::memory_order_release;
    case __ATOMIC_ACQ_REL: return std::memory_order_acq_rel;
    default:               return std::memory_order_seq_cst;
    }
}

//! The name that the divergent barrier check gives s_barrier
extern "C" inline constexpr char s_barrier_name[] = "__builtin_amdgcn_s_barrier()";

//! HSA's kernel dispatch packet, which dispatch_ptr points to
struct dispatch_packet {
    uint16_t header;
    uint16_t setup;
    uint16_t workgroup_size_x, workgroup_size_y, workgroup_size_z;
    uint16_t reserved0;
    uint32_t grid_size_x, grid_size_y, grid_size_z;
    uint32_t private_segment_size;
    uint32_t group_segment_size;
    uint64_t kernel_object;
    void *kernarg_address;
    uint64_t reserved2;
    uint64_t completion_signal;
};
static_assert(sizeof(dispatch_packet) == 64);

}

//
// Dispatch: the IDs and sizes of work-items (threads), work-groups (blocks)
// and the grid, which HSA measures in work-items
//

extern "C" inline unsigned int __builtin_amdgcn_workitem_id_x() { return threadIdx.x; }
extern "C" inline unsigned int __builtin_amdgcn_workitem_id_y() { return threadIdx.y; }
extern "C" inline unsigned int __builtin_amdgcn_workitem_id_z() { return threadIdx.z; }
extern "C" inline unsigned int __builtin_amdgcn_workgroup_id_x() { return blockIdx.x; }
extern "C" inline unsigned int __builtin_amdgcn_workgroup_id_y() { return blockIdx.y; }
extern "C" inline unsigned int __builtin_amdgcn_workgroup_id_z() { return blockIdx.z; }
extern "C" inline unsigned short __builtin_amdgcn_workgroup_size_x() { return (unsigned short)blockDim.x; }
extern "C" inline unsigned short __builtin_amdgcn_workgroup_size_y() { return (unsigned short)blockDim.y; }
extern "C" inline unsigned short __builtin_amdgcn_workgroup_size_z() { return (unsigned short)blockDim.z; }
extern "C" inline unsigned int __builtin_amdgcn_grid_size_x() { return gridDim.x * blockDim.x; }
extern "C" inline unsigned int __builtin_amdgcn_grid_size_y() { return gridDim.y * blockDim.y; }
extern "C" inline unsigned int __builtin_amdgcn_grid_size_z() { return gridDim.z * blockDim.z; }

extern "C" constexpr unsigned int __builtin_amdgcn_wavefrontsize()
{
    return cuda4cpu::detail::amdgcn::wavefront_size;
}

//! The launch's dispatch packet, filled in for the calling thread's launch
extern "C" inline void *__builtin_amdgcn_dispatch_ptr()
{
    static thread_local cuda4cpu::detail::amdgcn::dispatch_packet packet;
    const auto block = blockDim, grid = gridDim;
    packet = {};
    packet.setup = block.z * grid.z > 1 ? 3 : block.y * grid.y > 1 ? 2 : 1;
    packet.workgroup_size_x = uint16_t(block.x);
    packet.workgroup_size_y = uint16_t(block.y);
    packet.workgroup_size_z = uint16_t(block.z);
    packet.grid_size_x = grid.x * block.x;
    packet.grid_size_y = grid.y * block.y;
    packet.grid_size_z = grid.z * block.z;
    return &packet;
}

//
// Lane operations. Each is a function cuda4cpu_amdgcn_<name> that takes the
// call's place first, which the macro __builtin_amdgcn_<name> passes (at
// the end of this header): the lanes that reach the call together are the
// instruction's EXEC. The functions with the builtins' names, for calls
// that bypass the macros, share one place.
//

//! The place of a call, from its file and line
#define CUDA4CPU_AMDGCN_SITE ::cuda4cpu::detail::call_site(__FILE__, __LINE__)

template <typename T>
inline T cuda4cpu_amdgcn_readfirstlane(uint64_t site, T value)
{
    namespace a = cuda4cpu::detail::amdgcn;
    const a::lanes all = a::exchange(site, a::to_bits(value));
    return a::from_bits<T>(all.value[std::countr_zero(all.exec)]);
}

template <typename T>
inline T cuda4cpu_amdgcn_readlane(uint64_t site, T value, int lane)
{
    namespace a = cuda4cpu::detail::amdgcn;
    const a::lanes all = a::exchange(site, a::to_bits(value));
    const unsigned from = unsigned(lane) % a::wavefront_size;
    return all.active(from) ? a::from_bits<T>(all.value[from]) : a::from_bits<T>(0);
}

//! old, except in the given lane, which gets the first active lane's src
template <typename T>
inline T cuda4cpu_amdgcn_writelane(uint64_t site, T src, int lane, T old)
{
    namespace a = cuda4cpu::detail::amdgcn;
    const T value = cuda4cpu_amdgcn_readfirstlane(site, src);
    return a::lane() == unsigned(lane) % a::wavefront_size ? value : old;
}

inline uint64_t cuda4cpu_amdgcn_read_exec(uint64_t site)
{
    return cuda4cpu::detail::amdgcn::exec(site);
}

inline uint32_t cuda4cpu_amdgcn_read_exec_lo(uint64_t site)
{
    return cuda4cpu::detail::amdgcn::exec(site);
}

extern "C" inline uint32_t __builtin_amdgcn_read_exec_hi()
{
    return 0;
}

inline uint32_t cuda4cpu_amdgcn_ballot_w32(uint64_t site, bool predicate)
{
    namespace a = cuda4cpu::detail::amdgcn;
    const a::lanes all = a::exchange(site, predicate);
    uint32_t mask = 0;
    for (unsigned l = 0; l < a::wavefront_size; ++l)
        mask |= all.active(l) && all.value[l] != 0 ? 1u << l : 0;
    return mask;
}

inline uint64_t cuda4cpu_amdgcn_ballot_w64(uint64_t site, bool predicate)
{
    return cuda4cpu_amdgcn_ballot_w32(site, predicate);
}

extern "C" inline bool __builtin_amdgcn_inverse_ballot_w32(uint32_t mask)
{
    return (mask >> cuda4cpu::detail::amdgcn::lane() & 1) != 0;
}

extern "C" inline bool __builtin_amdgcn_inverse_ballot_w64(uint64_t mask)
{
    return (mask >> cuda4cpu::detail::amdgcn::lane() & 1) != 0;
}

//! add plus the bits of mask below the calling lane, of lanes 0 to 31
extern "C" inline unsigned int __builtin_amdgcn_mbcnt_lo(unsigned int mask, unsigned int add)
{
    const unsigned l = cuda4cpu::detail::amdgcn::lane();
    return add + unsigned(std::popcount(l == 0 ? 0u : mask & ((1u << l) - 1)));
}

//! The same, of lanes 32 to 63, which a 32-lane wavefront doesn't have
extern "C" inline unsigned int __builtin_amdgcn_mbcnt_hi(unsigned int, unsigned int add)
{
    return add;
}

//! src of the lane that addr / 4 names: a backward permutation
inline int cuda4cpu_amdgcn_ds_bpermute(uint64_t site, int addr, int src)
{
    namespace a = cuda4cpu::detail::amdgcn;
    const a::lanes all = a::exchange(site, a::to_bits(src));
    const unsigned from = (unsigned(addr) >> 2) % a::wavefront_size;
    return all.active(from) ? a::from_bits<int>(all.value[from]) : 0;
}

//! The src that a lane sent to the calling lane, whose number is its addr / 4:
//! a forward permutation. The highest such lane wins; zero if none.
inline int cuda4cpu_amdgcn_ds_permute(uint64_t site, int addr, int src)
{
    namespace a = cuda4cpu::detail::amdgcn;
    const unsigned to = (unsigned(addr) >> 2) % a::wavefront_size;
    const a::lanes all = a::exchange(site, uint64_t(to) << 32 | uint32_t(src));
    const unsigned l = a::lane();
    int result = 0;
    for (unsigned from = 0; from < a::wavefront_size; ++from)
        if (all.active(from) && (all.value[from] >> 32) == l)
            result = int(uint32_t(all.value[from]));
    return result;
}

inline int cuda4cpu_amdgcn_ds_swizzle(uint64_t site, int src, int offset)
{
    namespace a = cuda4cpu::detail::amdgcn;
    const a::lanes all = a::exchange(site, a::to_bits(src));
    const unsigned from = a::swizzle_source(a::lane(), offset);
    return all.active(from) ? a::from_bits<int>(all.value[from]) : 0;
}

template <typename T>
inline T cuda4cpu_amdgcn_update_dpp(uint64_t site, T old, T src, int dpp_ctrl, int row_mask, int bank_mask,
                                    bool bound_ctrl)
{
    return cuda4cpu::detail::amdgcn::update_dpp(site, old, src, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
}

//! update_dpp with no old value: lanes that keep theirs get zero here
template <typename T>
inline T cuda4cpu_amdgcn_mov_dpp(uint64_t site, T src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl)
{
    namespace a = cuda4cpu::detail::amdgcn;
    return a::update_dpp(site, a::from_bits<T>(0), src, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
}

//! Within groups of 8 lanes, each lane reads the one that its 3 bits of sel name
template <typename T>
inline T cuda4cpu_amdgcn_mov_dpp8(uint64_t site, T src, unsigned int sel)
{
    namespace a = cuda4cpu::detail::amdgcn;
    const a::lanes all = a::exchange(site, a::to_bits(src));
    const unsigned l = a::lane(), from = (l & ~7u) | ((sel >> (3 * (l & 7))) & 7);
    return all.active(from) ? a::from_bits<T>(all.value[from]) : a::from_bits<T>(0);
}

//! Within rows of 16 lanes, each lane reads the one that its 4 bits of
//! {src2, src1} name. fi would read inactive lanes, whose values aren't
//! known here: they give zero.
template <typename T>
inline T cuda4cpu_amdgcn_permlane16(uint64_t site, T old, T src, unsigned int src1, unsigned int src2, bool fi,
                                    bool bound_ctrl)
{
    return cuda4cpu::detail::amdgcn::permlane16(site, old, src, src1, src2, fi, bound_ctrl, false);
}

//! The same, reading the other row
template <typename T>
inline T cuda4cpu_amdgcn_permlanex16(uint64_t site, T old, T src, unsigned int src1, unsigned int src2, bool fi,
                                     bool bound_ctrl)
{
    return cuda4cpu::detail::amdgcn::permlane16(site, old, src, src1, src2, fi, bound_ctrl, true);
}

//! Swaps the halves of a 64-lane wavefront: nothing with 32 lanes
template <typename T>
inline T __builtin_amdgcn_permlane64(T src)
{
    return src;
}

//! src of lane idx, modulo the wavefront size
inline int cuda4cpu_amdgcn_wave_shuffle(uint64_t site, int src, int idx)
{
    namespace a = cuda4cpu::detail::amdgcn;
    const a::lanes all = a::exchange(site, a::to_bits(src));
    const unsigned from = unsigned(idx) % a::wavefront_size;
    return all.active(from) ? a::from_bits<int>(all.value[from]) : 0;
}

// Reductions over the active lanes, whose strategy argument only matters to
// the compiler
#define CUDA4CPU_WAVE_REDUCE(name, T, ...)                                    \
    inline T cuda4cpu_amdgcn_wave_reduce_##name(uint64_t site, T value, int)   \
    {                                                                          \
        return cuda4cpu::detail::amdgcn::__VA_ARGS__;                          \
    }                                                                          \
    extern "C" inline T __builtin_amdgcn_wave_reduce_##name(T value, int strategy) \
    {                                                                          \
        return cuda4cpu_amdgcn_wave_reduce_##name(0, value, strategy);         \
    }
CUDA4CPU_WAVE_REDUCE(add_u32, uint32_t, fold<uint32_t>(site, value, 0, [](uint32_t a, uint32_t b) { return a + b; }))
CUDA4CPU_WAVE_REDUCE(add_u64, uint64_t, fold<uint64_t>(site, value, 0, [](uint64_t a, uint64_t b) { return a + b; }))
CUDA4CPU_WAVE_REDUCE(sub_u32, uint32_t, fold<uint32_t>(site, value, 0, [](uint32_t a, uint32_t b) { return a - b; }))
CUDA4CPU_WAVE_REDUCE(sub_u64, uint64_t, fold<uint64_t>(site, value, 0, [](uint64_t a, uint64_t b) { return a - b; }))
CUDA4CPU_WAVE_REDUCE(min_i32, int32_t, reduce(site, value, [](int32_t a, int32_t b) { return std::min(a, b); }))
CUDA4CPU_WAVE_REDUCE(min_i64, int64_t, reduce(site, value, [](int64_t a, int64_t b) { return std::min(a, b); }))
CUDA4CPU_WAVE_REDUCE(min_u32, uint32_t, reduce(site, value, [](uint32_t a, uint32_t b) { return std::min(a, b); }))
CUDA4CPU_WAVE_REDUCE(min_u64, uint64_t, reduce(site, value, [](uint64_t a, uint64_t b) { return std::min(a, b); }))
CUDA4CPU_WAVE_REDUCE(max_i32, int32_t, reduce(site, value, [](int32_t a, int32_t b) { return std::max(a, b); }))
CUDA4CPU_WAVE_REDUCE(max_i64, int64_t, reduce(site, value, [](int64_t a, int64_t b) { return std::max(a, b); }))
CUDA4CPU_WAVE_REDUCE(max_u32, uint32_t, reduce(site, value, [](uint32_t a, uint32_t b) { return std::max(a, b); }))
CUDA4CPU_WAVE_REDUCE(max_u64, uint64_t, reduce(site, value, [](uint64_t a, uint64_t b) { return std::max(a, b); }))
CUDA4CPU_WAVE_REDUCE(and_b32, int32_t, fold<int32_t>(site, value, -1, [](int32_t a, int32_t b) { return a & b; }))
CUDA4CPU_WAVE_REDUCE(and_b64, int64_t, fold<int64_t>(site, value, -1, [](int64_t a, int64_t b) { return a & b; }))
CUDA4CPU_WAVE_REDUCE(or_b32, int32_t, fold<int32_t>(site, value, 0, [](int32_t a, int32_t b) { return a | b; }))
CUDA4CPU_WAVE_REDUCE(or_b64, int64_t, fold<int64_t>(site, value, 0, [](int64_t a, int64_t b) { return a | b; }))
CUDA4CPU_WAVE_REDUCE(xor_b32, int32_t, fold<int32_t>(site, value, 0, [](int32_t a, int32_t b) { return a ^ b; }))
CUDA4CPU_WAVE_REDUCE(xor_b64, int64_t, fold<int64_t>(site, value, 0, [](int64_t a, int64_t b) { return a ^ b; }))
CUDA4CPU_WAVE_REDUCE(fadd_f32, float, fold<float>(site, value, -0.0f, [](float a, float b) { return a + b; }))
CUDA4CPU_WAVE_REDUCE(fadd_f64, double, fold<double>(site, value, -0.0, [](double a, double b) { return a + b; }))
CUDA4CPU_WAVE_REDUCE(fsub_f32, float, fold<float>(site, value, 0.0f, [](float a, float b) { return a - b; }))
CUDA4CPU_WAVE_REDUCE(fsub_f64, double, fold<double>(site, value, 0.0, [](double a, double b) { return a - b; }))
CUDA4CPU_WAVE_REDUCE(fmin_f32, float, reduce(site, value, [](float a, float b) { return std::fmin(a, b); }))
CUDA4CPU_WAVE_REDUCE(fmin_f64, double, reduce(site, value, [](double a, double b) { return std::fmin(a, b); }))
CUDA4CPU_WAVE_REDUCE(fmax_f32, float, reduce(site, value, [](float a, float b) { return std::fmax(a, b); }))
CUDA4CPU_WAVE_REDUCE(fmax_f64, double, reduce(site, value, [](double a, double b) { return std::fmax(a, b); }))
#undef CUDA4CPU_WAVE_REDUCE

// The builtins' names as functions, for calls that bypass the macros
// ((__builtin_amdgcn_readlane)(...)), which share one place
template <typename T>
inline T __builtin_amdgcn_readfirstlane(T value) { return cuda4cpu_amdgcn_readfirstlane(0, value); }
template <typename T>
inline T __builtin_amdgcn_readlane(T value, int lane) { return cuda4cpu_amdgcn_readlane(0, value, lane); }
template <typename T>
inline T __builtin_amdgcn_writelane(T src, int lane, T old) { return cuda4cpu_amdgcn_writelane(0, src, lane, old); }
extern "C" inline uint64_t __builtin_amdgcn_read_exec() { return cuda4cpu_amdgcn_read_exec(0); }
extern "C" inline uint32_t __builtin_amdgcn_read_exec_lo() { return cuda4cpu_amdgcn_read_exec_lo(0); }
extern "C" inline uint32_t __builtin_amdgcn_ballot_w32(bool predicate) { return cuda4cpu_amdgcn_ballot_w32(0, predicate); }
extern "C" inline uint64_t __builtin_amdgcn_ballot_w64(bool predicate) { return cuda4cpu_amdgcn_ballot_w64(0, predicate); }
extern "C" inline int __builtin_amdgcn_ds_bpermute(int addr, int src) { return cuda4cpu_amdgcn_ds_bpermute(0, addr, src); }
extern "C" inline int __builtin_amdgcn_ds_permute(int addr, int src) { return cuda4cpu_amdgcn_ds_permute(0, addr, src); }
extern "C" inline int __builtin_amdgcn_ds_swizzle(int src, int offset) { return cuda4cpu_amdgcn_ds_swizzle(0, src, offset); }
template <typename T>
inline T __builtin_amdgcn_update_dpp(T old, T src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl)
{
    return cuda4cpu_amdgcn_update_dpp(0, old, src, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
}
template <typename T>
inline T __builtin_amdgcn_mov_dpp(T src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl)
{
    return cuda4cpu_amdgcn_mov_dpp(0, src, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
}
template <typename T>
inline T __builtin_amdgcn_mov_dpp8(T src, unsigned int sel) { return cuda4cpu_amdgcn_mov_dpp8(0, src, sel); }
template <typename T>
inline T __builtin_amdgcn_permlane16(T old, T src, unsigned int src1, unsigned int src2, bool fi, bool bound_ctrl)
{
    return cuda4cpu_amdgcn_permlane16(0, old, src, src1, src2, fi, bound_ctrl);
}
template <typename T>
inline T __builtin_amdgcn_permlanex16(T old, T src, unsigned int src1, unsigned int src2, bool fi, bool bound_ctrl)
{
    return cuda4cpu_amdgcn_permlanex16(0, old, src, src1, src2, fi, bound_ctrl);
}
extern "C" inline int __builtin_amdgcn_wave_shuffle(int src, int idx) { return cuda4cpu_amdgcn_wave_shuffle(0, src, idx); }

//
// Clang's atomics for HIP (__hip_atomic_*), with a memory order
// (__ATOMIC_*) and a scope (__HIP_MEMORY_SCOPE_*). Every scope is the
// system's here, so the scope argument only matters to the compiler.
//

// Clang defines these only when it compiles HIP
#ifndef __HIP_MEMORY_SCOPE_SINGLETHREAD
#define __HIP_MEMORY_SCOPE_SINGLETHREAD 1
#define __HIP_MEMORY_SCOPE_WAVEFRONT 2
#define __HIP_MEMORY_SCOPE_WORKGROUP 3
#define __HIP_MEMORY_SCOPE_AGENT 4
#define __HIP_MEMORY_SCOPE_SYSTEM 5
#endif

// Clang has the functions for every target: they are only defined for GCC
#if !__has_builtin(__hip_atomic_load)

template <typename T>
inline T __hip_atomic_load(T *ptr, int order, int)
{
    return std::atomic_ref<T>(*ptr).load(cuda4cpu::detail::amdgcn::memory_order(order));
}

template <typename T>
inline void __hip_atomic_store(T *ptr, std::type_identity_t<T> value, int order, int)
{
    std::atomic_ref<T>(*ptr).store(value, cuda4cpu::detail::amdgcn::memory_order(order));
}

template <typename T>
inline T __hip_atomic_exchange(T *ptr, std::type_identity_t<T> value, int order, int)
{
    return std::atomic_ref<T>(*ptr).exchange(value, cuda4cpu::detail::amdgcn::memory_order(order));
}

//! The failure order can't be stronger than the success one in C++; it is
//! the success order here, which is as strong
template <typename T>
inline bool __hip_atomic_compare_exchange_strong(T *ptr, T *expected, std::type_identity_t<T> desired, int success,
                                                 int, int)
{
    return std::atomic_ref<T>(*ptr).compare_exchange_strong(*expected, desired,
                                                            cuda4cpu::detail::amdgcn::memory_order(success));
}

template <typename T>
inline bool __hip_atomic_compare_exchange_weak(T *ptr, T *expected, std::type_identity_t<T> desired, int success,
                                               int failure, int scope)
{
    return __hip_atomic_compare_exchange_strong(ptr, expected, desired, success, failure, scope);
}

template <typename T>
inline T __hip_atomic_fetch_add(T *ptr, std::type_identity_t<T> value, int order, int)
{
    return std::atomic_ref<T>(*ptr).fetch_add(value, cuda4cpu::detail::amdgcn::memory_order(order));
}

template <typename T>
inline T __hip_atomic_fetch_sub(T *ptr, std::type_identity_t<T> value, int order, int)
{
    return std::atomic_ref<T>(*ptr).fetch_sub(value, cuda4cpu::detail::amdgcn::memory_order(order));
}

template <typename T>
inline T __hip_atomic_fetch_and(T *ptr, std::type_identity_t<T> value, int order, int)
{
    return std::atomic_ref<T>(*ptr).fetch_and(value, cuda4cpu::detail::amdgcn::memory_order(order));
}

template <typename T>
inline T __hip_atomic_fetch_or(T *ptr, std::type_identity_t<T> value, int order, int)
{
    return std::atomic_ref<T>(*ptr).fetch_or(value, cuda4cpu::detail::amdgcn::memory_order(order));
}

template <typename T>
inline T __hip_atomic_fetch_xor(T *ptr, std::type_identity_t<T> value, int order, int)
{
    return std::atomic_ref<T>(*ptr).fetch_xor(value, cuda4cpu::detail::amdgcn::memory_order(order));
}

template <typename T>
inline T __hip_atomic_fetch_min(T *ptr, std::type_identity_t<T> value, int, int)
{
    return cuda4cpu::detail::amdgcn::atomic_update(ptr, [&](T old) { return value < old ? value : old; });
}

template <typename T>
inline T __hip_atomic_fetch_max(T *ptr, std::type_identity_t<T> value, int, int)
{
    return cuda4cpu::detail::amdgcn::atomic_update(ptr, [&](T old) { return old < value ? value : old; });
}

#endif

//
// Synchronization, clocks, and hints to the compiler and the hardware,
// which do nothing here
//

//! __syncthreads(), which the divergent barrier check names by this name
extern "C" inline void __builtin_amdgcn_s_barrier()
{
    cuda4cpu::thread_block::syncthreads(cuda4cpu::detail::amdgcn::s_barrier_name, 0);
}

//! On a GPU, it only keeps the compiler from moving code across it, since
//! a wavefront's lanes run together; here they don't, so it waits for the
//! lanes that reach it together (EXEC), as a call in a branch must
inline void cuda4cpu_amdgcn_wave_barrier(uint64_t site)
{
    (void)cuda4cpu::detail::amdgcn::exec(site);
}

extern "C" inline void __builtin_amdgcn_wave_barrier()
{
    cuda4cpu_amdgcn_wave_barrier(0);
}

//! A fence with an order (__ATOMIC_*) for a scope ("workgroup", "agent",
//! ...) and optionally some address spaces: one OS thread runs a block, so
//! a fence for the system serves them all
template <typename... AddressSpaces>
inline void __builtin_amdgcn_fence(unsigned int order, const char *, AddressSpaces...)
{
    std::atomic_thread_fence(cuda4cpu::detail::amdgcn::memory_order(order));
}

//! Lets the other threads of the block run, as a waiting wavefront lets others
extern "C" inline void __builtin_amdgcn_s_sleep(int)
{
    cuda4cpu::thread_block::interleave();
}

//! The shader clock, as clock64()
extern "C" inline uint64_t __builtin_amdgcn_s_memtime()
{
    return uint64_t(clock64());
}

//! The constant 100 MHz clock
extern "C" inline uint64_t __builtin_amdgcn_s_memrealtime()
{
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count()) / 10;
}

extern "C" inline void __builtin_amdgcn_s_waitcnt(int) {}
extern "C" inline void __builtin_amdgcn_sched_barrier(int) {}
extern "C" inline void __builtin_amdgcn_sched_group_barrier(int, int, int) {}
extern "C" inline void __builtin_amdgcn_iglp_opt(int) {}
extern "C" inline void __builtin_amdgcn_s_setprio(short) {}
extern "C" inline void __builtin_amdgcn_s_dcache_inv() {}
extern "C" inline void __builtin_amdgcn_s_dcache_wb() {}
extern "C" inline void __builtin_amdgcn_buffer_wbinvl1() {}

//
// Integer and bit operations
//

//! Bytes of {src0, src1} (selectors 0 to 7), their sign bits (8 to 11), 0x00
//! (12) or 0xff (13 and above), chosen by each byte of sel
extern "C" inline unsigned int __builtin_amdgcn_perm(unsigned int src0, unsigned int src1, unsigned int sel)
{
    const uint64_t both = uint64_t(src0) << 32 | src1;
    unsigned int result = 0;
    for (unsigned k = 0; k < 4; ++k) {
        const unsigned s = (sel >> (8 * k)) & 0xff;
        unsigned byte;
        switch (s) {
        case 8:  byte = (src1 >> 15 & 1) * 0xff; break;
        case 9:  byte = (src1 >> 31 & 1) * 0xff; break;
        case 10: byte = (src0 >> 15 & 1) * 0xff; break;
        case 11: byte = (src0 >> 31 & 1) * 0xff; break;
        case 12: byte = 0; break;
        default: byte = s < 8 ? unsigned(both >> (8 * s)) & 0xff : 0xff; break;
        }
        result |= byte << (8 * k);
    }
    return result;
}

//! {src0, src1} shifted right by src2 bits (alignbit) or bytes (alignbyte)
extern "C" inline unsigned int __builtin_amdgcn_alignbit(unsigned int src0, unsigned int src1, unsigned int src2)
{
    return unsigned(((uint64_t(src0) << 32) | src1) >> (src2 & 31));
}

extern "C" inline unsigned int __builtin_amdgcn_alignbyte(unsigned int src0, unsigned int src1, unsigned int src2)
{
    return unsigned(((uint64_t(src0) << 32) | src1) >> (8 * (src2 & 3)));
}

//! The width bits of src from bit offset, zero- or sign-extended
extern "C" inline unsigned int __builtin_amdgcn_ubfe(unsigned int src, unsigned int offset, unsigned int width)
{
    width &= 31;
    return width == 0 ? 0 : (src >> (offset & 31)) & ((1u << width) - 1);
}

extern "C" inline int __builtin_amdgcn_sbfe(int src, unsigned int offset, unsigned int width)
{
    width &= 31;
    if (width == 0)
        return 0;
    const unsigned bits = unsigned(src >> (offset & 31)) << (32 - width);
    return int(bits) >> (32 - width);
}

//! The average of each byte of src0 and src1, rounded up when the byte of
//! src2 is odd
extern "C" inline unsigned int __builtin_amdgcn_lerp(unsigned int src0, unsigned int src1, unsigned int src2)
{
    unsigned int result = 0;
    for (unsigned k = 0; k < 32; k += 8)
        result |= (((src0 >> k & 0xff) + (src1 >> k & 0xff) + (src2 >> k & 1)) >> 1) << k;
    return result;
}

//! The sum of absolute differences of bytes (sad_u8, and shifted left 16 in
//! sad_hi_u8) or of 16-bit halves (sad_u16), plus src2
extern "C" inline unsigned int __builtin_amdgcn_sad_u8(unsigned int src0, unsigned int src1, unsigned int src2)
{
    unsigned int sum = src2;
    for (unsigned k = 0; k < 32; k += 8)
        sum += unsigned(std::abs(int(src0 >> k & 0xff) - int(src1 >> k & 0xff)));
    return sum;
}

extern "C" inline unsigned int __builtin_amdgcn_sad_hi_u8(unsigned int src0, unsigned int src1, unsigned int src2)
{
    return (__builtin_amdgcn_sad_u8(src0, src1, 0) << 16) + src2;
}

extern "C" inline unsigned int __builtin_amdgcn_sad_u16(unsigned int src0, unsigned int src1, unsigned int src2)
{
    return src2 + unsigned(std::abs(int(src0 & 0xffff) - int(src1 & 0xffff))) +
           unsigned(std::abs(int(src0 >> 16) - int(src1 >> 16)));
}

// Dot products of the 8-bit (dot4) or 4-bit (dot8) integers packed in a and
// b, plus c; with clamp, saturated rather than wrapped
extern "C" inline int __builtin_amdgcn_sdot4(int a, int b, int c, bool clamp)
{
    namespace d = cuda4cpu::detail::amdgcn;
    return d::clamp_int(c + d::dot(uint32_t(a), true, uint32_t(b), true, 8), clamp);
}

extern "C" inline int __builtin_amdgcn_sdot8(int a, int b, int c, bool clamp)
{
    namespace d = cuda4cpu::detail::amdgcn;
    return d::clamp_int(c + d::dot(uint32_t(a), true, uint32_t(b), true, 4), clamp);
}

extern "C" inline unsigned int __builtin_amdgcn_udot4(unsigned int a, unsigned int b, unsigned int c, bool clamp)
{
    namespace d = cuda4cpu::detail::amdgcn;
    return d::clamp_uint(int64_t(c) + d::dot(a, false, b, false, 8), clamp);
}

extern "C" inline unsigned int __builtin_amdgcn_udot8(unsigned int a, unsigned int b, unsigned int c, bool clamp)
{
    namespace d = cuda4cpu::detail::amdgcn;
    return d::clamp_uint(int64_t(c) + d::dot(a, false, b, false, 4), clamp);
}

//! With the signedness of each operand given
extern "C" inline int __builtin_amdgcn_sudot4(bool a_signed, int a, bool b_signed, int b, int c, bool clamp)
{
    namespace d = cuda4cpu::detail::amdgcn;
    return d::clamp_int(c + d::dot(uint32_t(a), a_signed, uint32_t(b), b_signed, 8), clamp);
}

extern "C" inline int __builtin_amdgcn_sudot8(bool a_signed, int a, bool b_signed, int b, int c, bool clamp)
{
    namespace d = cuda4cpu::detail::amdgcn;
    return d::clamp_int(c + d::dot(uint32_t(a), a_signed, uint32_t(b), b_signed, 4), clamp);
}

//! Each bit twice
extern "C" inline uint64_t __builtin_amdgcn_s_bitreplicate(unsigned int src)
{
    uint64_t result = 0;
    for (unsigned k = 0; k < 32; ++k)
        result |= uint64_t(src >> k & 1) * 3 << (2 * k);
    return result;
}

//! old with byte (sel & 3) replaced by f rounded to an integer in [0, 255]
extern "C" inline unsigned int __builtin_amdgcn_cvt_pk_u8_f32(float f, unsigned int sel, unsigned int old)
{
    const unsigned shift = 8 * (sel & 3);
    const float rounded = std::isnan(f) ? 0.0f : std::clamp(std::nearbyint(f), 0.0f, 255.0f);
    return (old & ~(0xffu << shift)) | (unsigned(rounded) << shift);
}

//
// Floating point: reciprocals and square roots, exact here; sines and
// cosines of x turns (2 pi x radians); exp2 and log, which is log2
//

extern "C" inline double __builtin_amdgcn_rcp(double x) { return 1.0 / x; }
extern "C" inline float __builtin_amdgcn_rcpf(float x) { return 1.0f / x; }
extern "C" inline _Float16 __builtin_amdgcn_rcph(_Float16 x) { return _Float16(1.0f / float(x)); }
extern "C" inline double __builtin_amdgcn_rsq(double x) { return 1.0 / std::sqrt(x); }
extern "C" inline float __builtin_amdgcn_rsqf(float x) { return 1.0f / std::sqrt(x); }
extern "C" inline _Float16 __builtin_amdgcn_rsqh(_Float16 x) { return _Float16(1.0f / std::sqrt(float(x))); }
extern "C" inline double __builtin_amdgcn_sqrt(double x) { return std::sqrt(x); }
extern "C" inline float __builtin_amdgcn_sqrtf(float x) { return std::sqrt(x); }
extern "C" inline _Float16 __builtin_amdgcn_sqrth(_Float16 x) { return _Float16(std::sqrt(float(x))); }
extern "C" inline float __builtin_amdgcn_sinf(float x) { return x == 0 ? x : float(std::sin(cuda4cpu::detail::amdgcn::turns(x))); }
extern "C" inline float __builtin_amdgcn_cosf(float x) { return float(std::cos(cuda4cpu::detail::amdgcn::turns(x))); }
extern "C" inline _Float16 __builtin_amdgcn_sinh(_Float16 x) { return x == 0 ? x : _Float16(std::sin(cuda4cpu::detail::amdgcn::turns(float(x)))); }
extern "C" inline _Float16 __builtin_amdgcn_cosh(_Float16 x) { return _Float16(std::cos(cuda4cpu::detail::amdgcn::turns(float(x)))); }
extern "C" inline float __builtin_amdgcn_exp2f(float x) { return std::exp2(x); }
extern "C" inline float __builtin_amdgcn_logf(float x) { return std::log2(x); }

extern "C" inline double __builtin_amdgcn_ldexp(double x, int e) { return std::ldexp(x, e); }
extern "C" inline float __builtin_amdgcn_ldexpf(float x, int e) { return std::ldexp(x, e); }
extern "C" inline _Float16 __builtin_amdgcn_ldexph(_Float16 x, int e) { return _Float16(std::ldexp(float(x), e)); }

//! frexp's mantissa (the value itself if infinite or NaN) and exponent (0 then)
extern "C" inline double __builtin_amdgcn_frexp_mant(double x) { int e; return std::isfinite(x) ? std::frexp(x, &e) : x; }
extern "C" inline float __builtin_amdgcn_frexp_mantf(float x) { int e; return std::isfinite(x) ? std::frexp(x, &e) : x; }
extern "C" inline _Float16 __builtin_amdgcn_frexp_manth(_Float16 x) { return _Float16(__builtin_amdgcn_frexp_mantf(float(x))); }
extern "C" inline int __builtin_amdgcn_frexp_exp(double x) { int e = 0; if (std::isfinite(x)) std::frexp(x, &e); return e; }
extern "C" inline int __builtin_amdgcn_frexp_expf(float x) { int e = 0; if (std::isfinite(x)) std::frexp(x, &e); return e; }
extern "C" inline short __builtin_amdgcn_frexp_exph(_Float16 x) { return short(__builtin_amdgcn_frexp_expf(float(x))); }

//! x - floor(x), below 1
extern "C" inline double __builtin_amdgcn_fract(double x) { return cuda4cpu::detail::amdgcn::fract(x); }
extern "C" inline float __builtin_amdgcn_fractf(float x) { return cuda4cpu::detail::amdgcn::fract(x); }
extern "C" inline _Float16 __builtin_amdgcn_fracth(_Float16 x) { return _Float16(cuda4cpu::detail::amdgcn::fract(float(x))); }

//! Whether x is of a class that mask has: bit 0 signaling NaN, 1 quiet NaN,
//! 2 -infinity, 3 -normal, 4 -subnormal, 5 -0, 6 +0, 7 +subnormal,
//! 8 +normal, 9 +infinity
extern "C" inline bool __builtin_amdgcn_class(double x, int mask) { return cuda4cpu::detail::amdgcn::fp_class(x, mask); }
extern "C" inline bool __builtin_amdgcn_classf(float x, int mask) { return cuda4cpu::detail::amdgcn::fp_class(x, mask); }
extern "C" inline bool __builtin_amdgcn_classh(_Float16 x, int mask)
{
    // A half's class is its float's, except subnormals and signaling NaNs
    const uint16_t bits = std::bit_cast<uint16_t>(x);
    const unsigned exponent = bits >> 10 & 31, mantissa = bits & 0x3ff;
    const bool negative = (bits >> 15) != 0;
    int bit;
    if (exponent == 31)
        bit = mantissa == 0 ? (negative ? 2 : 9) : (mantissa >> 9 & 1) != 0 ? 1 : 0;
    else if (exponent == 0)
        bit = mantissa == 0 ? (negative ? 5 : 6) : (negative ? 4 : 7);
    else
        bit = negative ? 3 : 8;
    return (mask >> bit & 1) != 0;
}

//! The median, or with a NaN, the minimum of the others
extern "C" inline float __builtin_amdgcn_fmed3f(float a, float b, float c) { return cuda4cpu::detail::amdgcn::med3(a, b, c); }
extern "C" inline _Float16 __builtin_amdgcn_fmed3h(_Float16 a, _Float16 b, _Float16 c)
{
    return _Float16(cuda4cpu::detail::amdgcn::med3(float(a), float(b), float(c)));
}

//
// Atomics: the old value. The order and scope arguments of atomic_inc and
// atomic_dec only matter to the compiler; every atomic is sequentially
// consistent here.
//

//! old >= value ? 0 : old + 1
extern "C" inline uint32_t __builtin_amdgcn_atomic_inc32(volatile uint32_t *p, uint32_t value, unsigned int, const char *)
{
    return cuda4cpu::detail::amdgcn::atomic_update(const_cast<uint32_t *>(p),
                                                   [&](uint32_t old) { return old >= value ? 0 : old + 1; });
}

extern "C" inline uint64_t __builtin_amdgcn_atomic_inc64(volatile uint64_t *p, uint64_t value, unsigned int, const char *)
{
    return cuda4cpu::detail::amdgcn::atomic_update(const_cast<uint64_t *>(p),
                                                   [&](uint64_t old) { return old >= value ? 0 : old + 1; });
}

//! old == 0 || old > value ? value : old - 1
extern "C" inline uint32_t __builtin_amdgcn_atomic_dec32(volatile uint32_t *p, uint32_t value, unsigned int, const char *)
{
    return cuda4cpu::detail::amdgcn::atomic_update(const_cast<uint32_t *>(p),
                                                   [&](uint32_t old) { return old == 0 || old > value ? value : old - 1; });
}

extern "C" inline uint64_t __builtin_amdgcn_atomic_dec64(volatile uint64_t *p, uint64_t value, unsigned int, const char *)
{
    return cuda4cpu::detail::amdgcn::atomic_update(const_cast<uint64_t *>(p),
                                                   [&](uint64_t old) { return old == 0 || old > value ? value : old - 1; });
}

// Floating-point atomics on global, flat (generic) and LDS (shared) memory,
// which are all memory here
#define CUDA4CPU_FLOAT_ATOMIC(name, T, update)                                                    \
    extern "C" inline T __builtin_amdgcn_##name(T *p, T value)                                    \
    {                                                                                             \
        return cuda4cpu::detail::amdgcn::atomic_update(p, [&](T old) { return update; });        \
    }
CUDA4CPU_FLOAT_ATOMIC(global_atomic_fadd_f32, float, old + value)
CUDA4CPU_FLOAT_ATOMIC(global_atomic_fadd_f64, double, old + value)
CUDA4CPU_FLOAT_ATOMIC(global_atomic_fmin_f64, double, std::fmin(old, value))
CUDA4CPU_FLOAT_ATOMIC(global_atomic_fmax_f64, double, std::fmax(old, value))
CUDA4CPU_FLOAT_ATOMIC(flat_atomic_fadd_f32, float, old + value)
CUDA4CPU_FLOAT_ATOMIC(flat_atomic_fadd_f64, double, old + value)
CUDA4CPU_FLOAT_ATOMIC(flat_atomic_fmin_f64, double, std::fmin(old, value))
CUDA4CPU_FLOAT_ATOMIC(flat_atomic_fmax_f64, double, std::fmax(old, value))
CUDA4CPU_FLOAT_ATOMIC(ds_atomic_fadd_f32, float, old + value)
CUDA4CPU_FLOAT_ATOMIC(ds_atomic_fadd_f64, double, old + value)
#undef CUDA4CPU_FLOAT_ATOMIC

// LDS atomics with an order, a scope and volatility, which only matter to
// the compiler
extern "C" inline float __builtin_amdgcn_ds_faddf(float *p, float value, int, int, bool)
{
    return cuda4cpu::detail::amdgcn::atomic_update(p, [&](float old) { return old + value; });
}

extern "C" inline float __builtin_amdgcn_ds_fminf(float *p, float value, int, int, bool)
{
    return cuda4cpu::detail::amdgcn::atomic_update(p, [&](float old) { return std::fmin(old, value); });
}

extern "C" inline float __builtin_amdgcn_ds_fmaxf(float *p, float value, int, int, bool)
{
    return cuda4cpu::detail::amdgcn::atomic_update(p, [&](float old) { return std::fmax(old, value); });
}

// The lane operations' macros, which pass the call's place: defined last,
// so that they don't rename the functions above
#define __builtin_amdgcn_read_exec() cuda4cpu_amdgcn_read_exec(CUDA4CPU_AMDGCN_SITE)
#define __builtin_amdgcn_wave_barrier() cuda4cpu_amdgcn_wave_barrier(CUDA4CPU_AMDGCN_SITE)
#define __builtin_amdgcn_read_exec_lo() cuda4cpu_amdgcn_read_exec_lo(CUDA4CPU_AMDGCN_SITE)
#define __builtin_amdgcn_readfirstlane(...) cuda4cpu_amdgcn_readfirstlane(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_readlane(...) cuda4cpu_amdgcn_readlane(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_writelane(...) cuda4cpu_amdgcn_writelane(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_ballot_w32(...) cuda4cpu_amdgcn_ballot_w32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_ballot_w64(...) cuda4cpu_amdgcn_ballot_w64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_ds_bpermute(...) cuda4cpu_amdgcn_ds_bpermute(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_ds_permute(...) cuda4cpu_amdgcn_ds_permute(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_ds_swizzle(...) cuda4cpu_amdgcn_ds_swizzle(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_update_dpp(...) cuda4cpu_amdgcn_update_dpp(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_mov_dpp(...) cuda4cpu_amdgcn_mov_dpp(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_mov_dpp8(...) cuda4cpu_amdgcn_mov_dpp8(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_permlane16(...) cuda4cpu_amdgcn_permlane16(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_permlanex16(...) cuda4cpu_amdgcn_permlanex16(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_shuffle(...) cuda4cpu_amdgcn_wave_shuffle(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_add_u32(...) cuda4cpu_amdgcn_wave_reduce_add_u32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_add_u64(...) cuda4cpu_amdgcn_wave_reduce_add_u64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_sub_u32(...) cuda4cpu_amdgcn_wave_reduce_sub_u32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_sub_u64(...) cuda4cpu_amdgcn_wave_reduce_sub_u64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_min_i32(...) cuda4cpu_amdgcn_wave_reduce_min_i32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_min_i64(...) cuda4cpu_amdgcn_wave_reduce_min_i64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_min_u32(...) cuda4cpu_amdgcn_wave_reduce_min_u32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_min_u64(...) cuda4cpu_amdgcn_wave_reduce_min_u64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_max_i32(...) cuda4cpu_amdgcn_wave_reduce_max_i32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_max_i64(...) cuda4cpu_amdgcn_wave_reduce_max_i64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_max_u32(...) cuda4cpu_amdgcn_wave_reduce_max_u32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_max_u64(...) cuda4cpu_amdgcn_wave_reduce_max_u64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_and_b32(...) cuda4cpu_amdgcn_wave_reduce_and_b32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_and_b64(...) cuda4cpu_amdgcn_wave_reduce_and_b64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_or_b32(...) cuda4cpu_amdgcn_wave_reduce_or_b32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_or_b64(...) cuda4cpu_amdgcn_wave_reduce_or_b64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_xor_b32(...) cuda4cpu_amdgcn_wave_reduce_xor_b32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_xor_b64(...) cuda4cpu_amdgcn_wave_reduce_xor_b64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_fadd_f32(...) cuda4cpu_amdgcn_wave_reduce_fadd_f32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_fadd_f64(...) cuda4cpu_amdgcn_wave_reduce_fadd_f64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_fsub_f32(...) cuda4cpu_amdgcn_wave_reduce_fsub_f32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_fsub_f64(...) cuda4cpu_amdgcn_wave_reduce_fsub_f64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_fmin_f32(...) cuda4cpu_amdgcn_wave_reduce_fmin_f32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_fmin_f64(...) cuda4cpu_amdgcn_wave_reduce_fmin_f64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_fmax_f32(...) cuda4cpu_amdgcn_wave_reduce_fmax_f32(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
#define __builtin_amdgcn_wave_reduce_fmax_f64(...) cuda4cpu_amdgcn_wave_reduce_fmax_f64(CUDA4CPU_AMDGCN_SITE, __VA_ARGS__)
