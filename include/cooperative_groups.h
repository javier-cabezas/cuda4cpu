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


// CUDA's cooperative groups: thread_block, grid_group, thread_block_tile,
// coalesced_group and the partitions that create them.
//
// - Tiles of up to 32 threads are built on the warp functions. Larger tiles
//   (64 to 1024 threads) synchronize with a barrier over their warps.
// - coalesced_threads() is __activemask(): the lanes that reach the call
//   together, leaving out lanes in other branches.
// - grid_group::sync() requires cudaLaunchCooperativeKernel, which runs every
//   block at the same time, each on its own OS thread.

#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <source_location>

#include "cuda_runtime.h"

namespace cuda4cpu::detail::cg {

//! Rank of the calling thread in its block, x fastest
inline unsigned block_rank()
{
    const dim3 t = thread_block::get_thread(), d = thread_block::get_block_dim();
    return (t.z * d.y + t.y) * d.x + t.x;
}

inline unsigned block_size()
{
    const dim3 d = thread_block::get_block_dim();
    return d.x * d.y * d.z;
}

//! Rank of the calling block in the grid, x fastest
inline unsigned long long grid_block_rank()
{
    const dim3 b = thread_block::get_block(), g = thread_block::get_grid_dim();
    return (static_cast<unsigned long long>(b.z) * g.y + b.y) * g.x + b.x;
}

inline unsigned long long grid_blocks()
{
    const dim3 g = thread_block::get_grid_dim();
    return static_cast<unsigned long long>(g.x) * g.y * g.z;
}

inline unsigned lane()
{
    return thread_block::lane_id();
}

//! Lanes of the calling warp below the caller
inline unsigned lanes_below()
{
    return (1u << lane()) - 1;
}

//! The lane of mask whose rank (number of lanes of mask below it) is rank
inline unsigned nth_lane(unsigned mask, unsigned rank)
{
    for (; rank > 0; --rank)
        mask &= mask - 1;
    return unsigned(std::countr_zero(mask));
}

//! Moves the bits of value at the positions of mask to the low bits, in order
inline unsigned compress(unsigned value, unsigned mask)
{
    unsigned result = 0, bit = 0;
    for (; mask != 0; mask &= mask - 1, ++bit) {
        if (value & mask & -mask)
            result |= 1u << bit;
    }
    return result;
}

//! Lanes of the tile of size lanes (a power of two up to 32) that holds the caller
inline unsigned tile_lanes(unsigned size)
{
    return size >= 32 ? ~0u : ((1u << size) - 1) << (lane() & ~(size - 1));
}

//! Barrier of the tile of size threads (a power of two) that holds the caller
inline void tile_sync(unsigned size)
{
    if (size <= 32)
        __syncwarp(tile_lanes(size));
    else
        thread_block::sync_warps(block_rank() / size * (size / 32), size / 32);
}

//! Threads of the block from the first thread of the caller's tile of size
//! threads to the end of the tile: fewer than size in a partial last tile
inline unsigned tile_extent(unsigned size)
{
    const unsigned first = block_rank() & ~(size - 1);
    return std::min(size, block_size() - first);
}

inline void block_sync(const std::source_location &site)
{
    thread_block::syncthreads(site.file_name(), int(site.line()));
}

struct tile_tag {};

}

namespace cooperative_groups {

class coalesced_group;

//! Any group: the type that the specific groups convert to
class thread_group {
public:
    unsigned long long num_threads() const
    {
        namespace cg = cuda4cpu::detail::cg;
        switch (kind_) {
        case kind::block:     return cg::block_size();
        case kind::grid:      return cg::grid_blocks() * cg::block_size();
        case kind::tile:      return size_;
        case kind::coalesced: return unsigned(std::popcount(mask_));
        }
        return 0;
    }

    unsigned long long size() const
    {
        return num_threads();
    }

    unsigned long long thread_rank() const
    {
        namespace cg = cuda4cpu::detail::cg;
        switch (kind_) {
        case kind::block:     return cg::block_rank();
        case kind::grid:      return cg::grid_block_rank() * cg::block_size() + cg::block_rank();
        case kind::tile:      return cg::block_rank() & (size_ - 1);
        case kind::coalesced: return unsigned(std::popcount(mask_ & cg::lanes_below()));
        }
        return 0;
    }

    void sync(std::source_location site = std::source_location::current()) const
    {
        namespace cg = cuda4cpu::detail::cg;
        switch (kind_) {
        case kind::block:     cg::block_sync(site); break;
        case kind::grid:      cuda4cpu::thread_block::sync_grid(site.file_name(), int(site.line())); break;
        case kind::tile:      cg::tile_sync(size_); break;
        case kind::coalesced: __syncwarp(mask_); break;
        }
    }

protected:
    enum class kind : unsigned char { block, grid, tile, coalesced };

    explicit thread_group(kind k, unsigned size = 0, unsigned mask = 0, unsigned meta_rank = 0,
                          unsigned meta_size = 1) :
        kind_{k}, size_{size}, mask_{mask}, meta_rank_{meta_rank}, meta_size_{meta_size}
    {
    }

    kind kind_;
    unsigned size_;        //!< Threads in a tile
    unsigned mask_;        //!< Lanes of a coalesced group
    unsigned meta_rank_;   //!< Rank of this group among the partitions of its parent
    unsigned meta_size_;   //!< Number of partitions of its parent

    friend thread_group tiled_partition(const thread_group &parent, unsigned tilesz);
};

//! The calling thread's block
class thread_block : public thread_group {
public:
    static void sync(std::source_location site = std::source_location::current())
    {
        cuda4cpu::detail::cg::block_sync(site);
    }

    static unsigned thread_rank()
    {
        return cuda4cpu::detail::cg::block_rank();
    }

    static unsigned num_threads()
    {
        return cuda4cpu::detail::cg::block_size();
    }

    static unsigned size()
    {
        return num_threads();
    }

    static dim3 group_index()
    {
        return cuda4cpu::thread_block::get_block();
    }

    static dim3 thread_index()
    {
        return cuda4cpu::thread_block::get_thread();
    }

    static dim3 dim_threads()
    {
        return cuda4cpu::thread_block::get_block_dim();
    }

    static dim3 group_dim()
    {
        return dim_threads();
    }

private:
    thread_block() : thread_group(kind::block) {}

    friend thread_block this_thread_block();
};

inline thread_block this_thread_block()
{
    return {};
}

//! Scratch memory for tiles larger than a warp on GPUs before compute
//! capability 8.0. cuda4cpu doesn't need it.
template <unsigned MaxBlockSize = 1024>
struct block_tile_memory {};

template <unsigned MaxBlockSize>
inline thread_block this_thread_block(block_tile_memory<MaxBlockSize> & /* scratch */)
{
    return this_thread_block();
}

//! All the threads of the grid. sync() requires a cooperative launch.
class grid_group : public thread_group {
public:
    static void sync(std::source_location site = std::source_location::current())
    {
        cuda4cpu::thread_block::sync_grid(site.file_name(), int(site.line()));
    }

    //! Whether the kernel was launched with cudaLaunchCooperativeKernel
    static bool is_valid()
    {
        return cuda4cpu::thread_block::cooperative_launch();
    }

    static unsigned long long thread_rank()
    {
        namespace cg = cuda4cpu::detail::cg;
        return cg::grid_block_rank() * cg::block_size() + cg::block_rank();
    }

    static unsigned long long num_threads()
    {
        namespace cg = cuda4cpu::detail::cg;
        return cg::grid_blocks() * cg::block_size();
    }

    static unsigned long long size()
    {
        return num_threads();
    }

    static unsigned long long block_rank()
    {
        return cuda4cpu::detail::cg::grid_block_rank();
    }

    static unsigned long long num_blocks()
    {
        return cuda4cpu::detail::cg::grid_blocks();
    }

    static dim3 block_index()
    {
        return cuda4cpu::thread_block::get_block();
    }

    static dim3 dim_blocks()
    {
        return cuda4cpu::thread_block::get_grid_dim();
    }

    static dim3 group_dim()
    {
        return dim_blocks();
    }

private:
    grid_group() : thread_group(kind::grid) {}

    friend grid_group this_grid();
};

inline grid_group this_grid()
{
    return {};
}

template <unsigned Size, typename Parent = void>
class thread_block_tile;

//! Size consecutive threads of a block (Size is a power of two up to 1024).
//! Tiles of up to 32 threads also have the warp functions.
template <unsigned Size>
class thread_block_tile<Size, void> : public thread_group {
    static_assert(Size > 0 && Size <= 1024 && (Size & (Size - 1)) == 0,
                  "a tile's size must be a power of two up to 1024");

public:
    thread_block_tile(cuda4cpu::detail::cg::tile_tag, unsigned meta_rank, unsigned meta_size) :
        thread_group(kind::tile, Size, 0, meta_rank, meta_size)
    {
    }

    static constexpr unsigned long long num_threads()
    {
        return Size;
    }

    static constexpr unsigned long long size()
    {
        return Size;
    }

    static unsigned thread_rank()
    {
        return cuda4cpu::detail::cg::block_rank() & (Size - 1);
    }

    //! Rank of this tile among the tiles of its parent
    unsigned meta_group_rank() const
    {
        return meta_rank_;
    }

    //! Number of tiles in the parent
    unsigned meta_group_size() const
    {
        return meta_size_;
    }

    static void sync()
    {
        cuda4cpu::detail::cg::tile_sync(Size);
    }

    template <typename T>
        requires(Size <= 32)
    T shfl(T var, int srcRank) const
    {
        return __shfl_sync(lanes(), var, srcRank, int(Size));
    }

    template <typename T>
        requires(Size <= 32)
    T shfl_up(T var, unsigned delta) const
    {
        return __shfl_up_sync(lanes(), var, delta, int(Size));
    }

    template <typename T>
        requires(Size <= 32)
    T shfl_down(T var, unsigned delta) const
    {
        return __shfl_down_sync(lanes(), var, delta, int(Size));
    }

    template <typename T>
        requires(Size <= 32)
    T shfl_xor(T var, unsigned laneMask) const
    {
        return __shfl_xor_sync(lanes(), var, int(laneMask), int(Size));
    }

    int any(int predicate) const
        requires(Size <= 32)
    {
        return __any_sync(lanes(), predicate);
    }

    int all(int predicate) const
        requires(Size <= 32)
    {
        return __all_sync(lanes(), predicate);
    }

    //! Bit i is the predicate of the thread of rank i
    unsigned ballot(int predicate) const
        requires(Size <= 32)
    {
        return __ballot_sync(lanes(), predicate) >> first_lane();
    }

    template <typename T>
        requires(Size <= 32)
    unsigned match_any(T value) const
    {
        return __match_any_sync(lanes(), value) >> first_lane();
    }

    template <typename T>
        requires(Size <= 32)
    unsigned match_all(T value, int &pred) const
    {
        return __match_all_sync(lanes(), value, &pred) >> first_lane();
    }

private:
    static unsigned lanes()
    {
        return cuda4cpu::detail::cg::tile_lanes(Size);
    }

    static unsigned first_lane()
    {
        return cuda4cpu::detail::cg::lane() & ~(Size - 1);
    }
};

//! A tile that remembers its parent's type; it converts to thread_block_tile<Size>
template <unsigned Size, typename Parent>
class thread_block_tile : public thread_block_tile<Size, void> {
public:
    using thread_block_tile<Size, void>::thread_block_tile;
};

template <unsigned Size>
inline thread_block_tile<Size, thread_block> tiled_partition(const thread_block & /* parent */)
{
    namespace cg = cuda4cpu::detail::cg;
    return {cg::tile_tag{}, cg::block_rank() / Size, (cg::block_size() + Size - 1) / Size};
}

template <unsigned Size, unsigned ParentSize, typename Parent>
inline thread_block_tile<Size, thread_block_tile<ParentSize, Parent>>
tiled_partition(const thread_block_tile<ParentSize, Parent> &parent)
{
    static_assert(Size <= ParentSize, "a tile can only be split into smaller tiles");
    return {cuda4cpu::detail::cg::tile_tag{}, unsigned(parent.thread_rank()) / Size, ParentSize / Size};
}

//! The calling thread alone
inline thread_block_tile<1, void> this_thread()
{
    return {cuda4cpu::detail::cg::tile_tag{}, cuda4cpu::detail::cg::block_rank(),
            cuda4cpu::detail::cg::block_size()};
}

//! Lanes of a warp, in rank order: the lanes that called coalesced_threads()
//! together, or a partition of a tile or of another coalesced group
class coalesced_group : public thread_group {
public:
    explicit coalesced_group(unsigned mask, unsigned meta_rank = 0, unsigned meta_size = 1) :
        thread_group(kind::coalesced, 0, mask, meta_rank, meta_size)
    {
    }

    unsigned long long num_threads() const
    {
        return unsigned(std::popcount(mask_));
    }

    unsigned long long size() const
    {
        return num_threads();
    }

    unsigned thread_rank() const
    {
        return unsigned(std::popcount(mask_ & cuda4cpu::detail::cg::lanes_below()));
    }

    unsigned meta_group_rank() const
    {
        return meta_rank_;
    }

    unsigned meta_group_size() const
    {
        return meta_size_;
    }

    void sync() const
    {
        __syncwarp(mask_);
    }

    template <typename T>
    T shfl(T var, unsigned srcRank) const
    {
        namespace cg = cuda4cpu::detail::cg;
        return cuda4cpu::detail::shuffle(mask_, var, cg::nth_lane(mask_, srcRank % unsigned(size())));
    }

    template <typename T>
    T shfl_down(T var, unsigned delta) const
    {
        namespace cg = cuda4cpu::detail::cg;
        const unsigned rank = thread_rank() + delta;
        return cuda4cpu::detail::shuffle(mask_, var, rank < size() ? cg::nth_lane(mask_, rank) : cg::lane());
    }

    template <typename T>
    T shfl_up(T var, unsigned delta) const
    {
        namespace cg = cuda4cpu::detail::cg;
        const unsigned rank = thread_rank();
        return cuda4cpu::detail::shuffle(mask_, var, rank >= delta ? cg::nth_lane(mask_, rank - delta) : cg::lane());
    }

    int any(int predicate) const
    {
        return __any_sync(mask_, predicate);
    }

    int all(int predicate) const
    {
        return __all_sync(mask_, predicate);
    }

    //! Bit i is the predicate of the thread of rank i
    unsigned ballot(int predicate) const
    {
        return cuda4cpu::detail::cg::compress(__ballot_sync(mask_, predicate), mask_);
    }

    template <typename T>
    unsigned match_any(T value) const
    {
        return cuda4cpu::detail::cg::compress(__match_any_sync(mask_, value), mask_);
    }

    template <typename T>
    unsigned match_all(T value, int &pred) const
    {
        return cuda4cpu::detail::cg::compress(__match_all_sync(mask_, value, &pred), mask_);
    }

    //! Lanes of the warp in this group
    unsigned lanes() const
    {
        return mask_;
    }
};

//! The lanes of the calling warp that call it together: lanes in other
//! branches are left out
inline coalesced_group coalesced_threads(std::source_location site = std::source_location::current())
{
    return coalesced_group(__activemask(site));
}

//! Splits a coalesced group into groups of tilesz consecutive ranks
inline coalesced_group tiled_partition(const coalesced_group &parent, unsigned tilesz)
{
    namespace cg = cuda4cpu::detail::cg;
    const unsigned rank = parent.thread_rank(), first = rank / tilesz * tilesz;
    const unsigned n = unsigned(parent.num_threads());
    unsigned mask = 0;
    for (unsigned r = first; r < first + tilesz && r < n; ++r)
        mask |= 1u << cg::nth_lane(parent.lanes(), r);
    return coalesced_group(mask, rank / tilesz, (n + tilesz - 1) / tilesz);
}

//! Splits a group into tiles of tilesz threads (a power of two)
inline thread_group tiled_partition(const thread_group &parent, unsigned tilesz)
{
    if (parent.kind_ == thread_group::kind::coalesced)
        return tiled_partition(coalesced_group(parent.mask_), tilesz);
    const unsigned rank = unsigned(parent.thread_rank());
    return thread_group(thread_group::kind::tile, tilesz, 0, rank / tilesz,
                        unsigned((parent.num_threads() + tilesz - 1) / tilesz));
}

//! Splits a group into the threads with the same label
template <typename Label>
inline coalesced_group labeled_partition(const coalesced_group &parent, Label label)
{
    return coalesced_group(__match_any_sync(parent.lanes(), label));
}

template <typename Label, unsigned Size, typename Parent>
    requires(Size <= 32)
inline coalesced_group labeled_partition(const thread_block_tile<Size, Parent> & /* parent */, Label label)
{
    return coalesced_group(__match_any_sync(cuda4cpu::detail::cg::tile_lanes(Size), label));
}

//! Splits a group into the threads whose predicate is true and those whose predicate is false
inline coalesced_group binary_partition(const coalesced_group &parent, bool predicate)
{
    return labeled_partition(parent, predicate);
}

template <unsigned Size, typename Parent>
    requires(Size <= 32)
inline coalesced_group binary_partition(const thread_block_tile<Size, Parent> &parent, bool predicate)
{
    return labeled_partition(parent, predicate);
}

template <typename Group>
inline void sync(const Group &group, std::source_location site = std::source_location::current())
{
    if constexpr (std::is_same_v<Group, thread_block> || std::is_same_v<Group, grid_group> ||
                  std::is_same_v<Group, thread_group>)
        group.sync(site);
    else
        group.sync();
}

template <typename Group>
inline unsigned long long thread_rank(const Group &group)
{
    return group.thread_rank();
}

template <typename Group>
inline unsigned long long group_size(const Group &group)
{
    return group.num_threads();
}

}
