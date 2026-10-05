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

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <omp.h>

#include "cuda/mycuda.hpp"

namespace cuda4cpu {

template <typename... Args>
struct grid_launcher;

//!
//! Kernel call configuration
//!
struct launch_conf {
    dim3 grid;
    dim3 block;

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
//! and the rest are scheduled round-robin, switching at every barrier. Threads
//! that return early leave the barrier, like on GPUs. Fibers are created once
//! and reused by every block and launch with the same block size on the same OS
//! thread.
//!
class thread_block {
public:
    //! Returns the thread block of the calling OS thread for the given launch
    //! configuration, creating its fibers on first use
    //! \param conf Launch configuration
    //! \param stack_size Size of the stack of each CUDA thread, in bytes
    static thread_block &acquire(const launch_conf &conf, size_t stack_size);

    //! Runs all the CUDA threads of one thread block to completion
    //! \param func Kernel invocation for one CUDA thread
    //! \param block_id Identifier of the thread block to be executed
    void execute(const std::function<void ()> &func, dim3 block_id);

    ~thread_block();

    thread_block(const thread_block &) = delete;
    thread_block & operator=(const thread_block &) = delete;

    //! Implementation of the __syncthreads intrinsic.
    //! Switches to the next CUDA thread until all the live threads in the block have reached the barrier.
    static void syncthreads();

    //! Implementation of the threadIdx special variable.
    //! Returns a dim3 object with the identifier of the calling thread within the thread block.
    static inline const dim3 &
    get_thread()
    {
        return Current_->ids_[Current_->cur_];
    }

    //! Implementation of the blockIdx special variable.
    //! Returns a dim3 object with the identifier of the calling thread block within the grid.
    static inline const dim3 &
    get_block()
    {
        return Current_->block_id_;
    }

    //! Implementation of the blockDim special variable.
    //! Returns a dim3 object with the number of threads in each dimension of the thread block.
    static inline const dim3 &
    get_block_dim()
    {
        return Current_->conf_.block;
    }

    //! Implementation of the gridDim special variable.
    //! Returns a dim3 object with the number of blocks in each dimension of the computation grid.
    static inline const dim3 &
    get_grid_dim()
    {
        return Current_->conf_.grid;
    }

private:
    struct fibers;

    thread_block(dim3 block, size_t stack_size);

    static void fiber_entry(unsigned ptr_high, unsigned ptr_low);

    [[noreturn]] void fiber_main();
    [[noreturn]] void finish_current();
    [[noreturn]] void jump_to(size_t tid);
    void promote();
    size_t next_live(size_t from) const;

    launch_conf conf_;
    const std::function<void ()> *func_;
    dim3 block_id_;
    size_t nthreads_;
    size_t stack_size_;
    size_t cur_;    //!< CUDA thread being executed
    size_t live_;   //!< CUDA threads that have not returned yet (fiber mode only)
    bool direct_;   //!< No CUDA thread has reached a barrier in this block yet
    std::vector<dim3> ids_;
    std::unique_ptr<fibers> fibers_;

    // initial-exec avoids a __tls_get_addr call on every access from the library
    [[gnu::tls_model("initial-exec")]] static thread_local thread_block *Current_;
};

template <typename... Args>
struct grid_launcher {
    grid_launcher(void (&func)(Args...), dim3 conf_grid, dim3 conf_block) :
                  func_{func},
                  conf_{conf_grid,
                        conf_block}
    {
    }

    template <typename... Args2>
    void call(Args2 &&...args)
    {
        const size_t nblocks = conf_.nblocks();
        if (nblocks == 0 || conf_.nthreads() == 0)
            return;

        const std::function<void ()> closure = std::bind(func_, args...);
        const size_t stack_size = detail::stack_size.load(std::memory_order_relaxed);

        // Contiguous chunks of blocks per OS thread. Only the OS threads that
        // get blocks acquire (and, the first time, create) fibers.
        #pragma omp parallel
        {
            thread_block *block = nullptr;

            #pragma omp for schedule(static)
            for (size_t i = 0; i < nblocks; ++i) {
                if (block == nullptr)
                    block = &thread_block::acquire(conf_, stack_size);
                block->execute(closure, conf_.block_id(i));
            }
        }
    }

private:
    void (&func_)(Args...);
    launch_conf conf_;
};

template <typename... Args>
grid_launcher<Args...>
launch(void (&func)(Args...), dim3 grid, dim3 block)
{
    return grid_launcher<Args...>(func, grid, block);
}

} // namespace cuda4cpu
