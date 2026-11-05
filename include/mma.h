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

// CUDA's warp matrix functions (nvcuda::wmma), which run on tensor cores on
// a GPU and here on the CPU:
// - A fragment holds a tile of a matrix spread across the 32 lanes of a warp.
//   CUDA leaves which lane holds which element unspecified, so programs only
//   apply the same operation to every element. Here lane l holds the
//   elements l * num_elements to l * num_elements + num_elements - 1 of the
//   tile, modulo its size, in the order of the fragment's layout (row-major
//   for accumulators), so that each lane loads consecutive addresses. half's
//   A and B fragments have 16 elements per lane, as in CUDA, so lanes hold
//   some elements twice.
// - load_matrix_sync, store_matrix_sync and mma_sync wait for the whole warp,
//   as PTX's .sync instructions do. mma_sync gathers A and B in a buffer of
//   the warp and each lane computes its elements of D: products are exact,
//   and sums are float (double for double, 64-bit for integers) in order of
//   k, rounded once to D's type. Tensor cores add in an unspecified order.
// - Types: half and bfloat16 (m16n16k16, m32n8k16, m8n32k16, into float, or
//   half for half), tf32 (m16n16k8, into float), double (m8n8k4) and 8-bit
//   integers (m16n16k16, m32n8k16, m8n32k16, into int). The sub-byte types
//   of wmma::experimental aren't implemented.
#pragma once

#if !defined(__CUDA_MMA_H__)
#define __CUDA_MMA_H__
#endif

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include "cuda_runtime.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"

namespace nvcuda::wmma {

struct matrix_a {};
struct matrix_b {};
struct accumulator {};

struct row_major {};
struct col_major {};

enum layout_t { mem_row_major, mem_col_major };

namespace precision {
struct tf32 {};
}

}

namespace cuda4cpu::detail::wmma {

namespace wmma = nvcuda::wmma;

//! The types of an element type: what a fragment stores, what fill_fragment
//! takes, and what mma_sync computes in
template <typename T>
struct element {
    using storage = T;
    using compute = float;
};

template <>
struct element<wmma::precision::tf32> {
    using storage = float;
    using compute = float;
};

template <>
struct element<double> {
    using storage = double;
    using compute = double;
};

template <>
struct element<signed char> {
    using storage = signed char;
    using compute = int64_t;
};

template <>
struct element<unsigned char> {
    using storage = unsigned char;
    using compute = int64_t;
};

template <>
struct element<int> {
    using storage = int;
    using compute = int64_t;
};

template <typename T>
inline constexpr bool is_input =
    std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, wmma::precision::tf32> ||
    std::is_same_v<T, double> || std::is_same_v<T, signed char> || std::is_same_v<T, unsigned char>;

template <typename T>
inline constexpr bool is_accumulator =
    std::is_same_v<T, float> || std::is_same_v<T, __half> || std::is_same_v<T, double> || std::is_same_v<T, int>;

//! Whether CUDA has the shape m x n x k for inputs of type T
template <int m, int n, int k, typename T>
inline constexpr bool valid_shape =
    std::is_same_v<T, wmma::precision::tf32> ? m == 16 && n == 16 && k == 8
    : std::is_same_v<T, double>              ? m == 8 && n == 8 && k == 4
                                             : k == 16 && ((m == 16 && n == 16) || (m == 32 && n == 8) ||
                                                           (m == 8 && n == 32));

//! The rows and columns of a fragment's tile
template <typename Use, int m, int n, int k>
struct tile;

template <int m, int n, int k>
struct tile<wmma::matrix_a, m, n, k> {
    static constexpr int rows = m, cols = k;
};

template <int m, int n, int k>
struct tile<wmma::matrix_b, m, n, k> {
    static constexpr int rows = k, cols = n;
};

template <int m, int n, int k>
struct tile<wmma::accumulator, m, n, k> {
    static constexpr int rows = m, cols = n;
};

//! Elements per lane: CUDA's counts, which are the tile's share of each lane,
//! except half's A and B fragments, which always have 16
template <typename Use, int m, int n, int k, typename T>
inline constexpr int elements_per_lane =
    !std::is_same_v<Use, wmma::accumulator> && std::is_same_v<T, __half>
        ? 16
        : tile<Use, m, n, k>::rows * tile<Use, m, n, k>::cols / 32;

template <typename T, int size>
struct fragment_base {
    using element_type         = T;
    using storage_element_type = typename element<T>::storage;
    enum { num_elements = size, num_storage_elements = size };
    storage_element_type x[size];
};

//! Lane of the calling CUDA thread in its warp, and its warp in the block
inline unsigned lane_id()
{
    return thread_block::lane_id();
}

inline unsigned warp_id()
{
    const auto &t = thread_block::get_thread(), &d = thread_block::get_block_dim();
    return ((t.z * d.y + t.y) * d.x + t.x) / 32;
}

inline void sync_warp()
{
    thread_block::syncwarp(0xffffffffu);
}

//! Where the calling lane's first element is in a tile of rows x cols,
//! row-major. Its other elements follow it, and lanes past the tile's size
//! hold the same elements again.
template <int rows, int cols, int per_lane>
inline int first_index(unsigned lane)
{
    static_assert((rows * cols) % per_lane == 0);
    return int(lane * per_lane % unsigned(rows * cols));
}

//! Whether the calling lane's elements aren't copies of another lane's
template <int rows, int cols, int per_lane>
inline bool holds_first_copy(unsigned lane)
{
    return lane * per_lane < unsigned(rows * cols);
}

//! Buffer of each warp of the block, for mma_sync. A block runs on one OS
//! thread, so a thread-local buffer is the block's, like __shared__.
inline constexpr size_t buffer_bytes_per_warp = (32 * 16 + 16 * 8) * sizeof(int64_t);

inline std::byte *warp_buffer()
{
    static thread_local std::unique_ptr<std::byte[]> buffer;
    if (!buffer)
        buffer.reset(new std::byte[32 * buffer_bytes_per_warp]);
    return buffer.get() + warp_id() * buffer_bytes_per_warp;
}

//! __half2float, with integer operations and no branches, so that it
//! vectorizes: mma_sync converts every element of A and B. Normal numbers
//! change their exponent's bias, subnormals are their bits times 2^-24 (exact
//! in float), and infinities and NaNs keep their payload.
inline float half_to_float(__half value)
{
    const uint32_t bits = __half_raw(value).x, sign = (bits & 0x8000u) << 16, magnitude = bits & 0x7fffu;
    const uint32_t normal = (magnitude + ((127 - 15) << 10)) << 13;
    const uint32_t subnormal = std::bit_cast<uint32_t>(float(int(magnitude)) * 0x1p-24f);
    const uint32_t special = 0x7f800000u | (magnitude & 0x3ffu) << 13;
    // Masks rather than conditions, which GCC's vectorizer at -O2 rejects
    const uint32_t is_special = 0u - uint32_t(magnitude >= 0x7c00u), is_subnormal = 0u - uint32_t(magnitude < 0x400u);
    return std::bit_cast<float>(sign | (special & is_special) | (subnormal & is_subnormal) |
                                (normal & ~(is_special | is_subnormal)));
}

template <typename C, typename T>
inline C to_compute(T value)
{
    if constexpr (std::is_same_v<T, __half>)
        return C(half_to_float(value));
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return C(std::bit_cast<float>(uint32_t(__nv_bfloat16_raw(value).x) << 16));
    else
        return C(value);
}

//! A tf32 input as the tensor cores read it: without the 13 low bits
inline float tf32_input(float value)
{
    if (std::isnan(value))
        return value;
    return std::bit_cast<float>(std::bit_cast<uint32_t>(value) & 0xffffe000u);
}

//! Writes the calling lane's elements of an A or B fragment, a tile of rows
//! x cols, to mma_sync's buffer, row-major, as mma_sync computes with them.
//! Lanes that hold copies of others' elements have nothing to add.
template <typename T, typename Layout, int rows, int cols, int size>
inline void to_buffer(typename element<T>::compute *buffer, unsigned lane,
                      const typename element<T>::storage (&x)[size])
{
    using C = typename element<T>::compute;
    if (!holds_first_copy<rows, cols, size>(lane))
        return;
    // Converted apart from the stores, which are scattered for col_major, so
    // that the conversion vectorizes
    C values[size];
    for (int i = 0; i < size; ++i) {
        values[i] = to_compute<C>(x[i]);
        if constexpr (std::is_same_v<T, wmma::precision::tf32>)
            values[i] = tf32_input(values[i]);
    }
    const int first = first_index<rows, cols, size>(lane);
    for (int i = 0; i < size; ++i) {
        const int e = first + i;
        buffer[std::is_same_v<Layout, wmma::row_major> ? e : e % rows * cols + e / rows] = values[i];
    }
}

//! A result in the accumulator's type. With satf, infinities become the
//! largest finite value and NaN zero, and integers saturate.
template <typename T, typename C>
inline T from_compute(C value, bool satf)
{
    if constexpr (std::is_same_v<T, int>) {
        if (satf)
            return int(std::clamp<int64_t>(value, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()));
        return int(uint32_t(uint64_t(value)));
    } else {
        T result;
        if constexpr (std::is_same_v<T, __half>)
            result = __float2half(float(value));
        else
            result = T(value);
        if (satf) {
            const C rounded = to_compute<C>(result);
            if (std::isnan(rounded))
                return T(0.0f);
            if (std::isinf(rounded)) {
                if constexpr (std::is_same_v<T, __half>)
                    return __float2half(rounded > 0 ? 65504.0f : -65504.0f);
                else
                    return rounded > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::lowest();
            }
        }
        return result;
    }
}

}

namespace nvcuda::wmma {

template <typename Use, int m, int n, int k, typename T, typename Layout = void>
class fragment
    : public cuda4cpu::detail::wmma::fragment_base<T, cuda4cpu::detail::wmma::elements_per_lane<Use, m, n, k, T>> {
    static_assert(std::is_same_v<Use, matrix_a> || std::is_same_v<Use, matrix_b> ||
                      std::is_same_v<Use, accumulator>,
                  "a fragment is a matrix_a, a matrix_b or an accumulator");
    static_assert(std::is_same_v<Use, accumulator> ? cuda4cpu::detail::wmma::is_accumulator<T>
                                                   : cuda4cpu::detail::wmma::is_input<T>,
                  "unsupported element type for this fragment");
    static_assert(std::is_same_v<Use, accumulator> ? std::is_void_v<Layout>
                                                   : std::is_same_v<Layout, row_major> ||
                                                         std::is_same_v<Layout, col_major>,
                  "matrix_a and matrix_b fragments are row_major or col_major, accumulators have no layout");
    static_assert(std::is_same_v<Use, accumulator> || cuda4cpu::detail::wmma::valid_shape<m, n, k, T>,
                  "unsupported shape for this element type");
};

//! Rounds a float to tf32, to nearest with ties away from zero, as
//! cvt.rna.tf32.f32: the 13 low bits of the result are zero
inline float __float_to_tf32(float in)
{
    if (!std::isfinite(in))
        return in;
    return std::bit_cast<float>((std::bit_cast<uint32_t>(in) + 0x1000u) & 0xffffe000u);
}

template <typename Use, int m, int n, int k, typename T, typename Layout>
inline void fill_fragment(fragment<Use, m, n, k, T, Layout> &a,
                          const typename cuda4cpu::detail::wmma::element<T>::storage &v)
{
    for (auto &e : a.x)
        e = v;
}

//! Loads a tile of A or B, whose element (row, col) is at mptr[row * ldm + col]
//! (row_major) or mptr[col * ldm + row] (col_major)
template <typename Use, int m, int n, int k, typename T, typename Layout>
    requires(!std::is_same_v<Use, accumulator>)
inline void load_matrix_sync(fragment<Use, m, n, k, T, Layout> &a,
                             const typename cuda4cpu::detail::wmma::element<T>::storage *mptr, unsigned ldm)
{
    namespace d = cuda4cpu::detail::wmma;
    using tile = d::tile<Use, m, n, k>;
    using frag = fragment<Use, m, n, k, T, Layout>;
    // In the layout's order, the lane's elements are consecutive in memory
    // within each row (row_major) or column (col_major)
    constexpr int extent = std::is_same_v<Layout, row_major> ? tile::cols : tile::rows;
    d::sync_warp();
    const int first = d::first_index<tile::rows, tile::cols, frag::num_elements>(d::lane_id());
    for (int i = 0; i < frag::num_elements; ++i)
        a.x[i] = mptr[size_t((first + i) / extent) * ldm + size_t((first + i) % extent)];
}

//! Loads a tile of C, laid out in memory as layout says
template <int m, int n, int k, typename T>
inline void load_matrix_sync(fragment<accumulator, m, n, k, T> &a, const T *mptr, unsigned ldm, layout_t layout)
{
    namespace d = cuda4cpu::detail::wmma;
    using frag = fragment<accumulator, m, n, k, T>;
    d::sync_warp();
    const int first = d::first_index<m, n, frag::num_elements>(d::lane_id());
    for (int i = 0; i < frag::num_elements; ++i) {
        const size_t row = size_t((first + i) / n), col = size_t((first + i) % n);
        a.x[i] = layout == mem_row_major ? mptr[row * ldm + col] : mptr[col * ldm + row];
    }
}

template <int m, int n, int k, typename T>
inline void store_matrix_sync(T *mptr, const fragment<accumulator, m, n, k, T> &a, unsigned ldm, layout_t layout)
{
    namespace d = cuda4cpu::detail::wmma;
    using frag = fragment<accumulator, m, n, k, T>;
    d::sync_warp();
    const int first = d::first_index<m, n, frag::num_elements>(d::lane_id());
    for (int i = 0; i < frag::num_elements; ++i) {
        const size_t row = size_t((first + i) / n), col = size_t((first + i) % n);
        (layout == mem_row_major ? mptr[row * ldm + col] : mptr[col * ldm + row]) = a.x[i];
    }
}

//! D = A * B + C, for the whole warp. d may be c.
template <int m, int n, int k, typename Ta, typename La, typename Tb, typename Lb, typename Tc, typename Td>
inline void mma_sync(fragment<accumulator, m, n, k, Td> &d, const fragment<matrix_a, m, n, k, Ta, La> &a,
                     const fragment<matrix_b, m, n, k, Tb, Lb> &b, const fragment<accumulator, m, n, k, Tc> &c,
                     bool satf = false)
{
    namespace w = cuda4cpu::detail::wmma;
    using C = typename w::element<Ta>::compute;
    constexpr bool integers = std::is_same_v<C, int64_t>;
    static_assert(std::is_same_v<C, typename w::element<Tb>::compute> &&
                      (std::is_same_v<Ta, Tb> || integers),
                  "A and B must have the same type");
    static_assert(integers ? std::is_same_v<Tc, int> && std::is_same_v<Td, int>
                  : std::is_same_v<Ta, double> ? std::is_same_v<Tc, double> && std::is_same_v<Td, double>
                  : std::is_same_v<Ta, __half>
                      ? (std::is_same_v<Tc, float> || std::is_same_v<Tc, __half>) &&
                            (std::is_same_v<Td, float> || std::is_same_v<Td, __half>)
                      : std::is_same_v<Tc, float> && std::is_same_v<Td, float>,
                  "unsupported accumulator type for these inputs");
    using fa = fragment<matrix_a, m, n, k, Ta, La>;
    using fb = fragment<matrix_b, m, n, k, Tb, Lb>;
    using fd = fragment<accumulator, m, n, k, Td>;
    static_assert((m * k + k * n) * sizeof(C) <= w::buffer_bytes_per_warp);

    // Every lane gives its elements of A and B, once the previous mma_sync of
    // the warp is done with the buffer
    w::sync_warp();
    C *const buffer_a = reinterpret_cast<C *>(w::warp_buffer());
    C *const buffer_b = buffer_a + m * k;
    const unsigned lane = w::lane_id();
    w::to_buffer<Ta, La, m, k>(buffer_a, lane, a.x);
    w::to_buffer<Tb, Lb, k, n>(buffer_b, lane, b.x);
    w::sync_warp();

    // The lane's elements of D are consecutive in a row, so that the inner
    // loop vectorizes. Products of half, bfloat16 and tf32 are exact in float,
    // where an fma would be a call without -mfma; double's need one.
    constexpr int per_lane = fd::num_elements;
    static_assert(per_lane * 32 == m * n && n % per_lane == 0);
    const int first = w::first_index<m, n, per_lane>(lane), row = first / n, col = first % n;
    C sums[per_lane];
    for (int i = 0; i < per_lane; ++i)
        sums[i] = w::to_compute<C>(c.x[i]);
    for (int j = 0; j < k; ++j) {
        const C x = buffer_a[row * k + j];
        const C *const y = buffer_b + j * n + col;
        for (int i = 0; i < per_lane; ++i) {
            if constexpr (std::is_same_v<C, double>)
                sums[i] = std::fma(x, y[i], sums[i]);
            else
                sums[i] += x * y[i];
        }
    }
    for (int i = 0; i < per_lane; ++i)
        d.x[i] = w::from_compute<Td>(sums[i], satf);
}

}
