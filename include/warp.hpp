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

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <source_location>
#include <type_traits>

#include "launch.hpp"

namespace cuda4cpu {

//
// Warp-level primitives. A warp is 32 consecutive threads of a block, in the
// same linear order as CUDA (x fastest). An operation completes when every
// lane in mask has reached it, except lanes that have returned or are waiting
// at __syncthreads() (they took a different branch). A lane whose source lane
// does not take part gets its own value back.
//
// Lanes in different branches that run warp operations at the same time must
// use disjoint masks (as CUDA requires); otherwise the operations are mixed.
//

inline namespace cuda_api {

inline constexpr int warpSize = 32;

}

namespace detail {

template <typename T>
inline uint64_t
to_lane_value(T value)
{
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) <= sizeof(uint64_t),
                  "warp shuffles support types of up to 8 bytes");
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

template <typename T>
inline T
from_lane_value(uint64_t bits)
{
    T value;
    std::memcpy(&value, &bits, sizeof(T));
    return value;
}

template <typename T>
inline T
shuffle(unsigned mask, T var, unsigned src_lane)
{
    if constexpr (sizeof(T) <= sizeof(uint64_t)) {
        return from_lane_value<T>(thread_block::warp_shuffle(mask, to_lane_value(var), src_lane));
    } else {
        // Larger types move in 8-byte pieces
        static_assert(std::is_trivially_copyable_v<T>, "warp shuffles need trivially copyable types");
        unsigned char bytes[sizeof(T)];
        std::memcpy(bytes, &var, sizeof(T));
        for (size_t i = 0; i < sizeof(T); i += sizeof(uint64_t)) {
            uint64_t piece = 0;
            const size_t n = std::min(sizeof(uint64_t), sizeof(T) - i);
            std::memcpy(&piece, bytes + i, n);
            piece = thread_block::warp_shuffle(mask, piece, src_lane);
            std::memcpy(bytes + i, &piece, n);
        }
        T result;
        std::memcpy(&result, bytes, sizeof(T));
        return result;
    }
}

//! Identifies a call site: user-space addresses fit in 47 bits
inline uint64_t
call_site(const std::source_location &site)
{
    return uint64_t(reinterpret_cast<uintptr_t>(site.file_name())) ^ (uint64_t(site.line()) << 47);
}

}

inline namespace cuda_api {

inline void __syncwarp(unsigned mask = 0xffffffff)
{
    thread_block::syncwarp(mask);
}

//! Lanes of the calling warp that call __activemask() at the same place
//! together. It waits until every lane of the warp that hasn't returned is
//! blocked (at a warp operation or a barrier) and returns the lanes waiting
//! here: lanes in another branch don't come here, so they are left out.
inline unsigned __activemask(std::source_location site = std::source_location::current())
{
    return thread_block::warp_active_mask(detail::call_site(site));
}

//! Value of var in lane srcLane of the caller's group of width lanes
template <typename T>
inline T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize)
{
    unsigned lane = thread_block::lane_id();
    unsigned src  = (lane & ~unsigned(width - 1)) + (unsigned(srcLane) & unsigned(width - 1));
    return detail::shuffle(mask, var, src);
}

//! Value of var in the lane delta positions below, or the caller's own value
//! if that lane is outside its group of width lanes
template <typename T>
inline T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = warpSize)
{
    unsigned lane = thread_block::lane_id();
    unsigned src  = (lane & unsigned(width - 1)) < delta ? lane : lane - delta;
    return detail::shuffle(mask, var, src);
}

//! Value of var in the lane delta positions above, or the caller's own value
//! if that lane is outside its group of width lanes
template <typename T>
inline T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = warpSize)
{
    unsigned lane = thread_block::lane_id();
    unsigned src  = (lane & unsigned(width - 1)) + delta >= unsigned(width) ? lane : lane + delta;
    return detail::shuffle(mask, var, src);
}

//! Value of var in lane (lane ^ laneMask), or the caller's own value if that
//! lane is in a later group of width lanes
template <typename T>
inline T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize)
{
    unsigned lane = thread_block::lane_id();
    unsigned src  = lane ^ unsigned(laneMask);
    if (src / unsigned(width) > lane / unsigned(width))
        src = lane;
    return detail::shuffle(mask, var, src);
}

//! Mask of the participating lanes whose predicate is non-zero
inline unsigned __ballot_sync(unsigned mask, int predicate)
{
    return unsigned(thread_block::warp_ballot(mask, predicate != 0));
}

inline int __any_sync(unsigned mask, int predicate)
{
    return unsigned(thread_block::warp_ballot(mask, predicate != 0)) != 0;
}

inline int __all_sync(unsigned mask, int predicate)
{
    uint64_t result = thread_block::warp_ballot(mask, predicate != 0);
    return unsigned(result) == unsigned(result >> 32);
}

//! Mask of the participating lanes whose value equals the caller's
template <typename T>
inline unsigned __match_any_sync(unsigned mask, T value)
{
    return thread_block::warp_match(mask, detail::to_lane_value(value), false);
}

//! Mask of the participating lanes if all of them have the same value (and
//! *pred = 1), or 0 (and *pred = 0) otherwise
template <typename T>
inline unsigned __match_all_sync(unsigned mask, T value, int *pred)
{
    unsigned result = thread_block::warp_match(mask, detail::to_lane_value(value), true);
    *pred = result != 0;
    return result;
}

}

}
