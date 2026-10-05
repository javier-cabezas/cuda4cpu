/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
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
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "cuda/mycuda.hpp"

namespace cuda4cpu {

template <typename... Args>
struct grid_launcher;

namespace detail {

struct fiber_access;

template <typename... Args>
struct kernel_call;

//!
//! Type-erased kernel launch: the loops that run the CUDA threads of a block,
//! instantiated for the kernel's signature so that each thread is a direct
//! call through a function pointer
//!
struct kernel_closure {
    //! Runs the CUDA threads of the block back to back, starting with the
    //! current one, until all have returned
    void (*run_direct)(const kernel_closure &);
    //! Runs the current CUDA thread
    void (*run_one)(const kernel_closure &);
};

}

//!
//! Kernel call configuration
//!
struct launch_conf {
    dim3 grid;
    dim3 block;
    size_t shared_mem = 0;   //!< Bytes of dynamic shared memory per block

    //! Returns the number of threads within a thread block
    //! \return The number of threads within a thread block
    size_t nthreads() const
    {
        return size_t(block.x) * size_t(block.y) * size_t(block.z);
    }

    //! Returns the number of blocks within the computation grid
    //! \return The number of blocks within the computation grid
    size_t nblocks() const
    {
        return size_t(grid.x) * size_t(grid.y) * size_t(grid.z);
    }

    //! Returns the identifier of the i-th thread block of the grid
    //! \param i Linear index of the thread block, in [0, nblocks())
    //! \return The identifier of the thread block
    dim3 block_id(size_t i) const
    {
        return dim3(unsigned(i % grid.x),
                    unsigned((i / grid.x) % grid.y),
                    unsigned(i / (size_t(grid.x) * size_t(grid.y))));
    }
};

//!
//! Provides information about the system
//!
class system {
public:
    inline static const system &
    get_system()
    {
        return sys_;
    }

    inline int
    get_num_procs() const
    {
        return cpus_;
    }

private:
    system();

    //! Fallback topology used when libnuma is unavailable: all CPUs in node 0
    void init_single_node();

    static system sys_;
    int cpus_;
    int nodes_;

    std::map<int, int> cpu2node_;
    std::map<int, std::vector<int>> node2cpus_;
};

//!
//! Executes the thread blocks of a grid on one OS thread.
//!
//! Each CUDA thread of the block is a fiber with its own guarded stack. A block
//! starts in "direct" mode, where its CUDA threads run back to back as plain
//! calls on the stack of fiber 0. When a thread first calls __syncthreads(), the
//! block switches to fiber mode: the threads that already returned are done,
//! and the rest are scheduled round-robin, switching whenever a thread waits
//! at a barrier (__syncthreads) or at a warp-synchronous operation (shuffles,
//! votes, __syncwarp). Threads that return early leave the barriers, like on
//! GPUs. Fibers are created once and reused by every block and launch with the
//! same block size on the same OS thread.
//!
class thread_block {
public:
    //! Returns the thread block of the calling OS thread for the given launch
    //! configuration, creating its fibers on first use
    //! \param conf Launch configuration
    //! \param stack_size Size of the stack of each CUDA thread, in bytes
    static thread_block &acquire(const launch_conf &conf, size_t stack_size);

    //! Runs all the CUDA threads of one thread block to completion
    //! \param kernel Kernel and arguments of the launch
    //! \param block_id Identifier of the thread block to be executed
    void execute(const detail::kernel_closure &kernel, dim3 block_id);

    ~thread_block();

    thread_block(const thread_block &) = delete;
    thread_block & operator=(const thread_block &) = delete;

    //! Implementation of the __syncthreads intrinsic.
    //! Switches to the next CUDA thread until all the live threads in the block have reached the barrier.
    static void syncthreads();

    //! Implementation of __syncwarp: waits for the live lanes of the calling warp in mask
    static void syncwarp(unsigned mask);

    //! Exchanges a value among the live lanes of the calling warp in mask. Each
    //! lane gets the value of lane src_lane (0-31), or its own value if that
    //! lane does not take part.
    static uint64_t warp_shuffle(unsigned mask, uint64_t value, unsigned src_lane);

    //! Implementation of the warp votes. Returns the ballot of predicate over
    //! the live lanes in mask (low 32 bits) and the mask of those lanes (high 32 bits).
    static uint64_t warp_ballot(unsigned mask, bool predicate);

    //! Implementation of __activemask: the lanes of the calling warp that have not returned
    static unsigned warp_active_mask();

    //! Lane of the calling CUDA thread within its warp
    static inline unsigned
    lane_id()
    {
        return Vars_.lane;
    }

    //! Dynamic shared memory of the calling thread block
    static inline void *
    get_dynamic_shared()
    {
        return Current_->shared_mem_;
    }

    //! Implementation of the threadIdx special variable.
    //! Returns a dim3 object with the identifier of the calling thread within the thread block.
    static inline dim3
    get_thread()
    {
        return Vars_.thread_idx.get();
    }

    //! Implementation of the blockIdx special variable.
    //! Returns a dim3 object with the identifier of the calling thread block within the grid.
    static inline dim3
    get_block()
    {
        return Vars_.block_idx.get();
    }

    //! Implementation of the blockDim special variable.
    //! Returns a dim3 object with the number of threads in each dimension of the thread block.
    static inline dim3
    get_block_dim()
    {
        return Vars_.block_dim.get();
    }

    //! Implementation of the gridDim special variable.
    //! Returns a dim3 object with the number of blocks in each dimension of the computation grid.
    static inline dim3
    get_grid_dim()
    {
        return Vars_.grid_dim.get();
    }

private:
    struct fibers;
    friend struct detail::fiber_access;
    template <typename... Args>
    friend struct detail::kernel_call;

    thread_block(dim3 block, size_t stack_size);

    static void fiber_entry(unsigned ptr_high, unsigned ptr_low);

    [[noreturn]] void fiber_main();
    [[noreturn]] void finish_current();
    void switch_to_thread(size_t from, size_t to);
    [[noreturn]] void deadlock() const;
    void promote();
    void switch_from_current();
    size_t next_runnable(size_t from) const;
    void release_block();
    void try_release_warp(size_t tid);
    void try_release_warp_waiters(size_t w);
    unsigned warp_live_mask(size_t base) const;
    uint64_t warp_op(unsigned char kind, unsigned mask, uint64_t value, unsigned src_lane);
    void reserve_shared(size_t bytes);

    launch_conf conf_;
    const detail::kernel_closure *kernel_;
    dim3 block_id_;
    size_t nthreads_;
    size_t stack_size_;
    size_t cur_;    //!< CUDA thread being executed
    bool direct_;   //!< No CUDA thread has reached a barrier in this block yet
    void *shared_mem_;
    std::vector<dim3> ids_;
    std::unique_ptr<fibers> fibers_;

    //! A dim3 as a trivial type, which __thread requires
    struct raw_dim3 {
        unsigned x, y, z;

        dim3 get() const { return dim3(x, y, z); }
        void set(const dim3 &d) { x = d.x; y = d.y; z = d.z; }
    };

    //! Built-in variables of the CUDA thread running on this OS thread, kept
    //! up to date by the scheduler so that reading one is a single load
    struct builtin_vars {
        raw_dim3 thread_idx;
        raw_dim3 block_idx;
        raw_dim3 block_dim;
        raw_dim3 grid_dim;
        unsigned lane;
    };

    //! Makes tid the running CUDA thread
    void set_current_thread(size_t tid)
    {
        cur_ = tid;
        Vars_.thread_idx.set(ids_[tid]);
        Vars_.lane = unsigned(tid % 32);
    }

    //! Called in direct mode when a CUDA thread returns. Moves to the next
    //! thread, or returns false if it was the last one. If the thread had
    //! waited at a barrier, the block is in fiber mode now and the thread
    //! finishes as a fiber instead.
    static bool next_direct_thread()
    {
        thread_block &block = *Current_;
        if (!block.direct_) [[unlikely]]
            block.finish_current();
        if (block.cur_ + 1 == block.nthreads_)
            return false;
        block.set_current_thread(block.cur_ + 1);
        return true;
    }

    // __thread instead of thread_local: these are constant-initialized, and
    // thread_local would make every access from other files call a TLS
    // initialization check. initial-exec avoids a __tls_get_addr call on every
    // access from the library.
    [[gnu::tls_model("initial-exec")]] static __thread thread_block *Current_;
    [[gnu::tls_model("initial-exec")]] static __thread builtin_vars Vars_;
};

//! Replacement for `extern __shared__ T name[];`, which can't be expressed
//! in C++: returns the dynamic shared memory of the calling thread block, whose
//! size is the shared_mem argument of launch().
template <typename T>
inline T *
dynamic_shared()
{
    return static_cast<T *>(thread_block::get_dynamic_shared());
}

namespace detail {

template <typename... Args>
struct kernel_call : kernel_closure {
    template <typename... Args2>
    kernel_call(void (*f)(Args...), Args2 &&...a) :
        kernel_closure{&run_direct_impl, &run_one_impl},
        func(f),
        args(std::forward<Args2>(a)...)
    {
    }

    static void run_direct_impl(const kernel_closure &closure)
    {
        const auto &call = static_cast<const kernel_call &>(closure);
        do {
            std::apply(call.func, call.args);
        } while (thread_block::next_direct_thread());
    }

    static void run_one_impl(const kernel_closure &closure)
    {
        const auto &call = static_cast<const kernel_call &>(closure);
        std::apply(call.func, call.args);
    }

    void (*func)(Args...);
    std::tuple<std::decay_t<Args>...> args;
};

}

template <typename... Args>
struct grid_launcher {
    grid_launcher(void (&func)(Args...), dim3 conf_grid, dim3 conf_block, size_t shared_mem) :
                  func_{func},
                  conf_{conf_grid,
                        conf_block,
                        shared_mem}
    {
    }

    template <typename... Args2>
    void call(Args2 &&...args)
    {
        const size_t nblocks = conf_.nblocks();
        if (nblocks == 0 || conf_.nthreads() == 0)
            return;

        const detail::kernel_call<Args...> kernel(&func_, std::forward<Args2>(args)...);
        const size_t stack_size = detail::stack_size.load(std::memory_order_relaxed);

        // Contiguous chunks of blocks, handed out dynamically so that blocks of
        // uneven cost balance across OS threads. About 8 chunks per OS thread
        // keeps neighboring blocks together and the scheduling overhead low.
        // Only the OS threads that get blocks acquire (and, the first time,
        // create) fibers.
        const size_t chunk = std::max<size_t>(1, nblocks / (size_t(omp_get_max_threads()) * 8));

        #pragma omp parallel
        {
            thread_block *block = nullptr;

            #pragma omp for schedule(dynamic, chunk)
            for (size_t i = 0; i < nblocks; ++i) {
                if (block == nullptr)
                    block = &thread_block::acquire(conf_, stack_size);
                block->execute(kernel, conf_.block_id(i));
            }
        }
    }

private:
    void (&func_)(Args...);
    launch_conf conf_;
};

//! Equivalent to func<<<grid, block, shared_mem, stream>>>. Kernels always run
//! synchronously, so the stream is ignored.
template <typename... Args>
grid_launcher<Args...>
launch(void (&func)(Args...), dim3 grid, dim3 block,
       size_t shared_mem = 0, cudaStream_t /* stream */ = nullptr)
{
    return grid_launcher<Args...>(func, grid, block, shared_mem);
}

} // namespace cuda4cpu
