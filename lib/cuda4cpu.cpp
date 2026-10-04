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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <new>

#include <setjmp.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>

#ifdef CUDA4CPU_HAVE_NUMA
#include <numa.h>
#endif

#ifdef CUDA4CPU_HANDLE_VALGRIND
#include <valgrind/valgrind.h>
#endif

#include "cuda4cpu.hpp"

namespace cuda4cpu {

// static variables
system system::sys_;
[[gnu::tls_model("initial-exec")]] thread_local thread_block *
thread_block::Current_ = nullptr;

namespace {

//! Maximum number of thread blocks (one per block size) cached per OS thread
constexpr size_t max_cached_blocks = 4;

constexpr size_t no_thread = size_t(-1);

//! Stacks are page aligned, so the hot top of every fiber's stack would map to
//! the same L1 cache sets and evict each other on every switch. Shift each top
//! by a different number of cache lines to spread them over all the sets.
constexpr size_t cache_line = 64;
constexpr size_t stack_offset_lines = 64;

constexpr size_t stack_offset(size_t i)
{
    return (i % stack_offset_lines) * cache_line;
}

struct jump_buffer {
    jmp_buf buf;
};

}

//!
//! Fiber stacks and contexts of a thread block
//!
struct thread_block::fibers {
    enum class state : unsigned char {
        not_started,    //!< Starts from the beginning of the kernel when scheduled
        started,        //!< Running, or suspended at a barrier
        finished        //!< Returned from the kernel
    };

    //! Allocates one stack per fiber in a single mapping, each one with a
    //! guard page below it so that a stack overflow faults instead of silently
    //! corrupting the neighboring stack.
    fibers(size_t n, size_t stack_size) :
        page{size_t(sysconf(_SC_PAGESIZE))},
        stack_size{(stack_size + page - 1) / page * page},
        stride{this->stack_size + page},
        map_size{n * stride},
        start(n),
        ctx(n),
        states(n, state::not_started)
    {
        void *map = mmap(nullptr, map_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_STACK, -1, 0);
        if (map == MAP_FAILED)
            throw std::bad_alloc();
        base = static_cast<unsigned char *>(map);

        for (size_t i = 0; i < n; ++i) {
            if (mprotect(base + i * stride, page, PROT_NONE) != 0) {
                munmap(base, map_size);
                throw std::bad_alloc();
            }
#ifdef CUDA4CPU_HANDLE_VALGRIND
            valgrind_ids.push_back(VALGRIND_STACK_REGISTER(stack(i), stack(i) + this->stack_size));
#endif
        }
    }

    ~fibers()
    {
#ifdef CUDA4CPU_HANDLE_VALGRIND
        for (auto id : valgrind_ids) VALGRIND_STACK_DEREGISTER(id);
#endif
        munmap(base, map_size);
    }

    //! Lowest address of the usable stack of the i-th fiber
    unsigned char *stack(size_t i) const
    {
        return base + i * stride + page;
    }

    const size_t page;
    const size_t stack_size;
    const size_t stride;
    const size_t map_size;
    unsigned char *base;

    std::vector<jump_buffer> start;   //!< Entry point of each fiber, captured once at creation
    std::vector<jump_buffer> ctx;     //!< Where each started fiber resumes
    std::vector<state> states;
    jump_buffer caller;               //!< Where execute() resumes when the block completes
#ifdef CUDA4CPU_HANDLE_VALGRIND
    std::vector<unsigned> valgrind_ids;
#endif
};

namespace {

struct fiber_init {
    jump_buffer *start;
    ucontext_t  *creator;
};

}

thread_block::thread_block(dim3 block, size_t stack_size) :
    conf_{dim3(), block},
    func_{nullptr},
    nthreads_{conf_.nthreads()},
    stack_size_{stack_size},
    cur_{0},
    live_{0},
    direct_{true},
    ids_(nthreads_),
    fibers_{std::make_unique<fibers>(nthreads_, stack_size)}
{
    for (size_t i = 0; i < nthreads_; ++i) {
        // Precompute thread ids
        ids_[i] = dim3(unsigned(i % block.x),
                       unsigned((i / block.x) % block.y),
                       unsigned(i / (size_t(block.x) * size_t(block.y))));

        // Enter the fiber once so that it captures its entry point in start[i]
        ucontext_t fiber, creator;
        getcontext(&fiber);
        fiber.uc_link          = nullptr;
        fiber.uc_stack.ss_sp   = fibers_->stack(i);
        fiber.uc_stack.ss_size = fibers_->stack_size - stack_offset(i);

        fiber_init init{&fibers_->start[i], &creator};
        auto ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&init));
        makecontext(&fiber, reinterpret_cast<void (*)()>(fiber_entry), 2,
                    unsigned(ptr >> 32), unsigned(ptr & 0xffffffffu));
        swapcontext(&creator, &fiber);
    }
}

thread_block::~thread_block() = default;

thread_block &
thread_block::acquire(const launch_conf &conf, size_t stack_size)
{
    // Most recently used last
    static thread_local std::vector<std::unique_ptr<thread_block>> cache;

    auto it = std::find_if(cache.begin(), cache.end(), [&](const auto &block) {
        const dim3 &b = block->conf_.block;
        return b.x == conf.block.x && b.y == conf.block.y && b.z == conf.block.z &&
               block->stack_size_ == stack_size;
    });

    std::unique_ptr<thread_block> block;
    if (it != cache.end()) {
        block = std::move(*it);
        cache.erase(it);
    } else {
        if (cache.size() == max_cached_blocks)
            cache.erase(cache.begin());
        block.reset(new thread_block(conf.block, stack_size));
    }

    block->conf_ = conf;
    cache.push_back(std::move(block));
    return *cache.back();
}

void thread_block::execute(const std::function<void ()> &func, dim3 block_id)
{
    func_     = &func;
    block_id_ = block_id;
    cur_      = 0;
    direct_   = true;

    thread_block *prev = Current_;
    Current_ = this;
    if (_setjmp(fibers_->caller.buf) == 0)
        _longjmp(fibers_->start[0].buf, 1);
    Current_ = prev;
}

//! Entry point of every fiber.
//! @param ptr_high Holds the high 32 bits of the address of the fiber_init object.
//! @param ptr_low Holds the low 32 bits of the address of the fiber_init object.
void thread_block::fiber_entry(unsigned ptr_high, unsigned ptr_low)
{
    auto *init = reinterpret_cast<fiber_init *>(
                    static_cast<uintptr_t>((uint64_t(ptr_high) << 32) | ptr_low));

    // On creation, record the entry point and go back to the constructor. The
    // scheduler jumps here every time the fiber has to start a CUDA thread.
    if (_setjmp(init->start->buf) == 0)
        setcontext(init->creator);

    Current_->fiber_main();
}

void thread_block::fiber_main()
{
    for (;;) {
        (*func_)();

        if (!direct_)
            finish_current();

        // Direct mode: the CUDA thread returned without reaching a barrier
        if (++cur_ == nthreads_)
            _longjmp(fibers_->caller.buf, 1);
    }
}

//! Switches the block to fiber mode when the current CUDA thread reaches the
//! first barrier. Threads before it already returned, and threads after it
//! have not started yet.
void thread_block::promote()
{
    using state = fibers::state;

    auto &states = fibers_->states;
    std::fill(states.begin(), states.begin() + cur_, state::finished);
    std::fill(states.begin() + cur_, states.end(), state::not_started);
    states[cur_] = state::started;

    live_   = nthreads_ - cur_;
    direct_ = false;
}

void thread_block::syncthreads()
{
    thread_block &block = *Current_;
    if (block.direct_)
        block.promote();

    // Threads run in index order between barriers, so when no thread after
    // this one is still alive, every live thread has reached the barrier.
    const size_t self = block.cur_;
    size_t next = block.next_live(self + 1);
    if (next == no_thread)
        next = block.next_live(0);
    if (next == self)
        return;

    if (_setjmp(block.fibers_->ctx[self].buf) == 0)
        block.jump_to(next);
}

void thread_block::finish_current()
{
    fibers_->states[cur_] = fibers::state::finished;
    if (--live_ == 0)
        _longjmp(fibers_->caller.buf, 1);

    size_t next = next_live(cur_ + 1);
    if (next == no_thread)
        next = next_live(0);
    jump_to(next);
}

void thread_block::jump_to(size_t tid)
{
    auto &f = *fibers_;
    cur_ = tid;
    if (f.states[tid] == fibers::state::not_started) {
        f.states[tid] = fibers::state::started;
        _longjmp(f.start[tid].buf, 1);
    }
    _longjmp(f.ctx[tid].buf, 1);
}

//! Returns the first CUDA thread at or after from that has not returned, or
//! no_thread if there is none
size_t thread_block::next_live(size_t from) const
{
    const auto &states = fibers_->states;
    for (size_t i = from; i < nthreads_; ++i) {
        if (states[i] != fibers::state::finished)
            return i;
    }
    return no_thread;
}

system::system()
{
    cpus_ = omp_get_num_procs();
    omp_set_num_threads(cpus_);

#ifdef CUDA4CPU_HAVE_NUMA
    if (numa_available() < 0) {
        init_single_node();
        return;
    }

    nodes_ = numa_num_configured_nodes();

    for (int i = 0; i < nodes_; ++i) {
        bitmask *mask = numa_allocate_cpumask();

        numa_node_to_cpus(i, mask);
        std::vector<int> node2cpus;
        for (int j = 0; j < cpus_; ++j) {
            if (numa_bitmask_isbitset(mask, j)) {
                cpu2node_[j] = i;
                node2cpus.push_back(j);
            }
        }
        node2cpus_[i] = node2cpus;

        numa_free_cpumask(mask);
    }
#else
    init_single_node();
#endif
}

void system::init_single_node()
{
    nodes_ = 1;

    std::vector<int> node2cpus;
    for (int j = 0; j < cpus_; ++j) {
        cpu2node_[j] = 0;
        node2cpus.push_back(j);
    }
    node2cpus_[0] = node2cpus;
}

}
