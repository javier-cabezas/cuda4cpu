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

template <typename F, typename... Stored>
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
    template <typename F, typename... Stored>
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

//! Replacement for `extern __shared__ T name[];` at namespace scope, where a
//! pointer variable would be initialized only once: every use refers to the
//! dynamic shared memory of the calling thread block. Converts to T * and
//! supports subscripts.
template <typename T>
struct dynamic_shared_array {
    operator T *() const
    {
        return dynamic_shared<T>();
    }

    template <typename I>
    T &operator[](I i) const
    {
        return dynamic_shared<T>()[i];
    }
};

namespace detail {

//! One launch: a callable (the kernel, or a lambda that calls it) and the
//! arguments, copied once when the kernel is launched
template <typename F, typename... Stored>
struct kernel_call : kernel_closure {
    template <typename... Args>
    kernel_call(F f, Args &&...a) :
        kernel_closure{&run_direct_impl, &run_one_impl},
        func(std::move(f)),
        args(std::forward<Args>(a)...)
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

    F func;
    std::tuple<Stored...> args;
};

//! CUDA's limits: launches outside them fail on a GPU, so they fail here too
inline bool valid_launch(const launch_conf &conf)
{
    const dim3 &b = conf.block, &g = conf.grid;
    return b.x > 0 && b.y > 0 && b.z > 0 && b.x <= 1024 && b.y <= 1024 && b.z <= 64 &&
           conf.nthreads() <= 1024 &&
           g.x > 0 && g.y > 0 && g.z > 0 && g.x <= 2147483647u && g.y <= 65535 && g.z <= 65535;
}

//! Runs every block of a launch whose configuration is valid
inline void run_grid(const launch_conf &conf, const kernel_closure &kernel)
{
    const size_t nblocks = conf.nblocks();
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
                block = &thread_block::acquire(conf, stack_size);
            block->execute(kernel, conf.block_id(i));
        }
    }
}

}

//! Launch of a kernel given as a function: call(args...) converts the
//! arguments to the kernel's parameter types and runs the grid
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
        if (!detail::valid_launch(conf_)) {
            detail::record_error(cudaErrorInvalidConfiguration);
            return;
        }
        const detail::kernel_call<void (*)(Args...), std::decay_t<Args>...>
            kernel(&func_, std::forward<Args2>(args)...);
        detail::run_grid(conf_, kernel);
    }

private:
    void (&func_)(Args...);
    launch_conf conf_;
};

//! Launch of a kernel given as a callable, such as the lambda that
//! cuda4cpu-rewrite generates for kernel<<<...>>>(args...): call(args...)
//! copies the arguments and calls the callable with them in every CUDA thread
template <typename F>
struct callable_launcher {
    callable_launcher(F func, dim3 conf_grid, dim3 conf_block, size_t shared_mem) :
                      func_{std::move(func)},
                      conf_{conf_grid,
                            conf_block,
                            shared_mem}
    {
    }

    template <typename... Args>
    void call(Args &&...args)
    {
        if (!detail::valid_launch(conf_)) {
            detail::record_error(cudaErrorInvalidConfiguration);
            return;
        }
        const detail::kernel_call<F, std::decay_t<Args>...> kernel(func_, std::forward<Args>(args)...);
        detail::run_grid(conf_, kernel);
    }

private:
    F func_;
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

//! Launches a callable as a kernel: every CUDA thread calls it with the
//! arguments given to call(). This is what cuda4cpu-rewrite generates, with a
//! lambda that calls the kernel, so that template arguments are deduced and
//! default arguments apply as in CUDA, and the compiler can inline the kernel.
template <typename F>
    requires std::is_class_v<std::remove_cvref_t<F>>
callable_launcher<std::remove_cvref_t<F>>
launch(F &&func, dim3 grid, dim3 block,
       size_t shared_mem = 0, cudaStream_t /* stream */ = nullptr)
{
    return callable_launcher<std::remove_cvref_t<F>>(std::forward<F>(func), grid, block, shared_mem);
}

namespace detail {

template <typename... Args, size_t... I>
cudaError_t launch_with_argument_array(void (*func)(Args...), dim3 grid, dim3 block, void **args,
                                       size_t shared_mem, std::index_sequence<I...>)
{
    if (!valid_launch(launch_conf{grid, block, shared_mem}))
        return record_error(cudaErrorInvalidConfiguration);

    launch(*func, grid, block, shared_mem).call(*static_cast<std::remove_reference_t<Args> *>(args[I])...);
    return cudaSuccess;
}

}

inline namespace cuda_api {

//! Launches func with the arguments that args points to, one pointer per
//! parameter, as in CUDA
template <typename... Args>
cudaError_t cudaLaunchKernel(void (*func)(Args...), dim3 gridDim, dim3 blockDim, void **args,
                             size_t sharedMem = 0, cudaStream_t /* stream */ = nullptr)
{
    return detail::launch_with_argument_array(func, gridDim, blockDim, args, sharedMem,
                                              std::index_sequence_for<Args...>{});
}

template <typename T>
    requires std::is_void_v<T>
cudaError_t cudaLaunchKernel(const T * /* func */, dim3, dim3, void **, size_t = 0, cudaStream_t = nullptr)
{
    static_assert(!std::is_void_v<T>,
                  "cuda4cpu needs the kernel's type to unpack its arguments: pass the kernel itself "
                  "to cudaLaunchKernel, not a void pointer");
    return cudaErrorNotSupported;
}

}

} // namespace cuda4cpu
