/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
 */

#ifndef CUDA4CPU_LAUNCH_HPP_
#define CUDA4CPU_LAUNCH_HPP_

#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

#include <omp.h>
#include <numa.h>

#include <setjmp.h>
#include <ucontext.h>

#include "types.hpp"

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

    static system sys_;
    int cpus_;
    int nodes_;

    std::map<int, int> cpu2node_;
    std::map<int, std::vector<int>> node2cpus_;
};

//!
//! Proxy object to execute thread blocks
//!
class thread_block
{
    struct fiber_t
    {
        ucontext_t fib;
        jmp_buf    jmp;
    };

    struct fiber_ctx_t
    {
        unsigned    tid;
        jmp_buf    *cur;
        ucontext_t *prv;
    };

    thread_block(const thread_block &) = delete;
    thread_block & operator=(const thread_block &) = delete;
    thread_block(thread_block &&) = default;
    thread_block & operator=(thread_block &&) = default;

public:
    thread_block(const std::function<void ()> &func,
                 const launch_conf &conf) :
        func_{func},
        conf_{conf},
        nthreads_{conf.nthreads()},
        thread_id_barrier_{0},
        callees_{nthreads_},
        callees_base_{nthreads_},
        stacks_{nthreads_}
    {
        ids.reserve(nthreads_);

        // Fill initial contexts with current context
        for (size_t i = 0; i < nthreads_; ++i) {
            // Allocate stack for the thrad
            auto stack = new unsigned char[SIGSTKSZ];

            // Precompute thread ids
            dim3 id;

            id.x = i % conf.block.x;
            id.y = (i / conf.block.x) % conf.block.y;
            id.z = i / (conf.block.x * conf.block.y);

            stacks_[i] = stack;
            ids[i]     = id;

            create_fiber(callees_base_[i], stacks_[i], i);
        }
    }

    ~thread_block()
    {
        for (auto stack : stacks_) delete []stack;
    }

    //! Execute an instance of the thread block for the given id
    //! \param block_id Identifier of the thread block to be executed
    void execute(dim3 block_id)
    {
        block_id_ = block_id;
        thread_id_barrier_ = 0;

        std::chrono::time_point<std::chrono::system_clock> start, end;
        static std::chrono::duration<double> elapsed;

        start = std::chrono::system_clock::now();
        std::copy(callees_base_.begin(), callees_base_.end(), callees_.begin());
        end = std::chrono::system_clock::now();
        elapsed += end - start;
        // std::cout << "CUDA-style init: " << elapsed.count() << "s\n";

        Current_ = this;
        switch_fiber(caller_, callees_[0]);
        Current_ = nullptr;
        // std::cout << "CUDA-style init: " << elapsed.count() << "s\n";
    }

    //! Implementation of the __syncthreads intrinsic.
    //! Switches to a different thread until all threads in the block have reached this barrier.
    static void syncthreads()
    {
        size_t old_id = Current_->thread_id_barrier_;
        size_t new_id;
        if (old_id == Current_->nthreads_ - 1) {
            new_id = 0;
        } else {
            new_id = old_id + 1;
        }
        Current_->thread_id_barrier_ = new_id;
        // std::cout << "S: " << old_id << "->" << new_id << "(" << Current_->ids[new_id].x << "," << Current_->ids[new_id].y << ")\n";
        switch_fiber(Current_->callees_[old_id],
                     Current_->callees_[new_id]);
    }

    //! Implementation of the threadIdx special variable.
    //! Returns a dim3 object with the identifier of the calling thread within the thread block.
    static inline const dim3 &
    get_thread()
    {
        return Current_->ids[Current_->thread_id_barrier_];
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

    //! Implementation of the gridDim special variable....
    //! Returns a dim3 object with the number of blocks in each dimension of the computation grid.
    static inline const dim3 &
    get_grid_dim()
    {
        return Current_->conf_.grid;
    }

private:
    //! Starts execution of a fiber on the CPU.
    //! @param ptr_high Holds the high 32 bits of the address of the context of the fiber to be used.
    //! @param ptr_low Holds the low 32 bits of the address of the context of the fiber to be used.
    static void start_fiber(uint32_t ptr_high, uint32_t ptr_low)
    {
        fiber_ctx_t* ctx = reinterpret_cast<fiber_ctx_t *>(
                            (static_cast<size_t>(ptr_high) << 32) + ptr_low);
        int tid = ctx->tid;
        if (_setjmp(*ctx->cur) == 0) {
            ucontext_t tmp;
            swapcontext(&tmp, ctx->prv);
        }
        thread_block_call(tid);
    }

    static void create_fiber(fiber_t &ctx, unsigned char *stack, unsigned tid)
    {
        getcontext(&ctx.fib);

        ctx.fib.uc_link          = 0;
        ctx.fib.uc_stack.ss_sp   = stack;
        ctx.fib.uc_stack.ss_size = SIGSTKSZ;
        ucontext_t tmp;
        fiber_ctx_t fib_ctx{tid, &ctx.jmp, &tmp};
        makecontext(&ctx.fib, (void(*)())start_fiber, 2,
                    reinterpret_cast<size_t>(&fib_ctx) >> 32, &fib_ctx);
        swapcontext(&tmp, &ctx.fib);
    }

    //! Switches execution between fibers.
    //! @param from Handler of the fiber being preempted.
    //! @param to Handler of the next fiber to be resumed.
    static inline void
    switch_fiber(fiber_t &from, fiber_t &to)
    {
        if (_setjmp(from.jmp) == 0)
            _longjmp(to.jmp, 1);
    }

    //! Starts execution of a thread within the thread block.
    //! @param tid Identifier of the thread to be executed.
    static void thread_block_call(unsigned tid)
    {
        Current_->func_();
        if (tid == Current_->nthreads_ - 1) {
            //std::cout << "F: " << tid << "->Caller\n";
            switch_fiber(Current_->callees_[tid],
                         Current_->caller_);
        } else {
            //std::cout << "F: " << tid << "->" << tid + 1 << "(" << Current->ids[tid + 1].x << "," << Current->ids[tid + 1].y << ")\n";
            Current_->thread_id_barrier_ += 1;
            switch_fiber(Current_->callees_[tid],
                         Current_->callees_[tid + 1]);
        }
    }

    const std::function<void ()> &func_;
    const launch_conf &conf_;
    dim3 block_id_;
    size_t nthreads_;
    size_t thread_id_barrier_;

    fiber_t caller_;
    std::vector<fiber_t> callees_;
    std::vector<fiber_t> callees_base_;
    std::vector<unsigned char *> stacks_;
    std::vector<dim3> ids;

    static thread_local thread_block *Current_;
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
        closure_ = std::bind(func_, args...);

        const system &sys = system::get_system();

        int procs = sys.get_num_procs();
        unsigned partial = conf_.nblocks() / procs;
        if (conf_.nblocks() % procs)
            ++partial;

        #pragma omp parallel for schedule(static, 1)
        for (int p = 0; p < procs; ++p) {
            //std::cout << p << ": " << partial << "\n";
            thread_block block{closure_, conf_};

            for (size_t i = p * partial; i < conf_.nblocks() && i < (p + 1) * partial; ++i) {
                dim3 id;

                id.x = i % conf_.grid.x;
                id.y = (i / conf_.grid.x) % conf_.grid.y;
                id.z = i / (conf_.grid.x * conf_.grid.y);

#if 1
                block.execute(id);
#else
                blocks.push_back({closure, conf_block, id});
                blocks[i]();
#endif
            }
        }
    }

private:
    void (&func_)(Args...);
    launch_conf conf_;

    std::function<void ()> closure_;
    std::vector<thread_block> blocks_;
};

template <typename... Args>
grid_launcher<Args...>
launch(void (&func)(Args...), dim3 grid, dim3 block)
{
    return grid_launcher<Args...>(func, grid, block);
}

} // namespace cuda4cpu



#endif // CUDA4CPU_LAUNCH_HPP_
