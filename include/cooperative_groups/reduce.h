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


// CUDA's cooperative_groups/reduce.h: reduce() over a tile or a coalesced
// group, and the operators it takes.

#pragma once

#include <cstring>
#include <type_traits>

#include "../cooperative_groups.h"

namespace cooperative_groups {

template <typename T>
struct plus {
    T operator()(const T &a, const T &b) const { return a + b; }
};

//! The smaller value
template <typename T>
struct less {
    T operator()(const T &a, const T &b) const { return b < a ? b : a; }
};

//! The larger value
template <typename T>
struct greater {
    T operator()(const T &a, const T &b) const { return a < b ? b : a; }
};

template <typename T>
struct bit_and {
    T operator()(const T &a, const T &b) const { return a & b; }
};

template <typename T>
struct bit_or {
    T operator()(const T &a, const T &b) const { return a | b; }
};

template <typename T>
struct bit_xor {
    T operator()(const T &a, const T &b) const { return a ^ b; }
};

}

namespace cuda4cpu::detail::cg {

//! Reduces value over the first count ranks of a group of up to 32 threads
//! (a tile or a coalesced group); every thread gets the result
template <typename Group, typename T, typename Op>
inline T reduce_warp(const Group &group, T value, Op &&op, unsigned count)
{
    const unsigned rank = unsigned(group.thread_rank());
    for (unsigned offset = std::bit_ceil(count) / 2; offset > 0; offset /= 2) {
        T other = group.shfl_down(value, offset);
        if (rank + offset < count)
            value = op(value, other);
    }
    return group.shfl(value, 0);
}

//! Reduces value over a tile of Size threads, larger than a warp: each warp
//! reduces its part, and every thread combines the parts. The parts are in
//! storage that is per block, like __shared__ memory.
template <unsigned Size, typename T, typename Op>
inline T reduce_tile(T value, Op &&op)
{
    static_assert(std::is_trivially_copyable_v<T>, "reduce() over tiles larger than a warp needs a trivially "
                                                   "copyable type");
    alignas(T) static thread_local unsigned char parts[32 * sizeof(T)];

    const unsigned rank = block_rank();
    const cooperative_groups::thread_block_tile<32> warp{tile_tag{}, 0, 1};
    value = reduce_warp(warp, value, op, tile_extent(32));
    if (rank % 32 == 0)
        std::memcpy(parts + rank / 32 * sizeof(T), &value, sizeof(T));
    tile_sync(Size);

    const unsigned first = (rank & ~(Size - 1)) / 32, nwarps = (tile_extent(Size) + 31) / 32;
    T result;
    std::memcpy(&result, parts + first * sizeof(T), sizeof(T));
    for (unsigned w = 1; w < nwarps; ++w) {
        T part;
        std::memcpy(&part, parts + (first + w) * sizeof(T), sizeof(T));
        result = op(result, part);
    }
    tile_sync(Size);   // before the parts are written again
    return result;
}

}

namespace cooperative_groups {

template <unsigned Size, typename Parent, typename T, typename Op>
inline T reduce(const thread_block_tile<Size, Parent> &group, T value, Op &&op)
{
    namespace cg = cuda4cpu::detail::cg;
    if constexpr (Size <= 32)
        return cg::reduce_warp(group, value, op, cg::tile_extent(Size));
    else
        return cg::reduce_tile<Size>(value, op);
}

template <typename T, typename Op>
inline T reduce(const coalesced_group &group, T value, Op &&op)
{
    return cuda4cpu::detail::cg::reduce_warp(group, value, op, unsigned(group.num_threads()));
}

}
