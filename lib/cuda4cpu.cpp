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
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdio>
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

// Fibers switch with a hand-written context switch on x86-64, and with
// _setjmp/_longjmp elsewhere (or when CUDA4CPU_PORTABLE_CONTEXT is defined)
#if defined(__x86_64__) && !defined(CUDA4CPU_PORTABLE_CONTEXT)
#define CUDA4CPU_ASM_CONTEXT 1
#else
#define CUDA4CPU_ASM_CONTEXT 0
#endif

#if CUDA4CPU_ASM_CONTEXT
extern "C" {

//! Pushes the callee-saved registers of the System V x86-64 ABI on the
//! current stack, stores the stack pointer in *save, switches to the stack
//! load and pops the registers saved there. The x87 control word and MXCSR
//! are not switched: CUDA has no per-thread floating-point environment, so
//! all the fibers of a block share the OS thread's.
void cuda4cpu_switch_context(void **save, void *load);

//! First code run on a fresh fiber stack: aligns the stack, marks the end of
//! the call stack for debuggers, and calls cuda4cpu_fiber_start
void cuda4cpu_fiber_trampoline();

[[gnu::visibility("hidden")]] void cuda4cpu_fiber_start();

}

asm(R"(
    .text
    .p2align 4
    .globl  cuda4cpu_switch_context
    .hidden cuda4cpu_switch_context
    .type   cuda4cpu_switch_context, @function
cuda4cpu_switch_context:
    pushq   %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15
    movq    %rsp, (%rdi)
    movq    %rsi, %rsp
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    ret
    .size   cuda4cpu_switch_context, .-cuda4cpu_switch_context

    .p2align 4
    .globl  cuda4cpu_fiber_trampoline
    .hidden cuda4cpu_fiber_trampoline
    .type   cuda4cpu_fiber_trampoline, @function
cuda4cpu_fiber_trampoline:
    .cfi_startproc
    .cfi_undefined rip
    xorl    %ebp, %ebp
    andq    $-16, %rsp
    call    cuda4cpu_fiber_start@PLT
    ud2
    .cfi_endproc
    .size   cuda4cpu_fiber_trampoline, .-cuda4cpu_fiber_trampoline
)");
#endif

namespace cuda4cpu {

// static variables
system system::sys_;
[[gnu::tls_model("initial-exec")]] __thread thread_block *
thread_block::Current_ = nullptr;
[[gnu::tls_model("initial-exec")]] __thread thread_block::builtin_vars
thread_block::Vars_;

namespace {

//! Maximum number of thread blocks (one per block size) cached per OS thread
constexpr size_t max_cached_blocks = 4;

constexpr size_t no_thread = size_t(-1);

constexpr size_t warp_size = 32;

//! Alignment of the dynamic shared memory of a block
constexpr size_t shared_alignment = 64;

constexpr uint32_t lane_bit(size_t tid)
{
    return 1u << (tid % warp_size);
}

//! Makes [addr, addr + size) fault on access. Guard regions (Linux 6.13+) take
//! the address-space lock only for reading and don't split the mapping, so
//! OS threads can create fibers in parallel without hitting vm.max_map_count;
//! mprotect is the fallback on older kernels.
bool install_guard(void *addr, size_t size)
{
#ifdef MADV_GUARD_INSTALL
    static std::atomic<bool> guard_regions{true};
    if (guard_regions.load(std::memory_order_relaxed)) {
        if (madvise(addr, size, MADV_GUARD_INSTALL) == 0)
            return true;
        if (errno != EINVAL)
            return false;
        guard_regions.store(false, std::memory_order_relaxed);
    }
#endif
    return mprotect(addr, size, PROT_NONE) == 0;
}

struct free_deleter {
    void operator()(void *ptr) const { std::free(ptr); }
};

//! Stacks are page aligned, so the hot top of every fiber's stack would map to
//! the same L1 cache sets and evict each other on every switch. Shift each top
//! by a different number of cache lines to spread them over all the sets.
constexpr size_t cache_line = 64;
constexpr size_t stack_offset_lines = 64;

constexpr size_t stack_offset(size_t i)
{
    return (i % stack_offset_lines) * cache_line;
}

#if CUDA4CPU_ASM_CONTEXT
//! A suspended fiber: its registers are saved on its own stack
struct context {
    void *sp;
};
#else
struct context {
    jmp_buf buf;
};
#endif

}

//!
//! Fiber stacks and contexts of a thread block
//!
struct thread_block::fibers {

    //! Warp-synchronous operations
    enum op : unsigned char {
        op_sync,
        op_shuffle,
        op_ballot
    };

    //! Allocates one stack per fiber in a single mapping, each one with a
    //! guard page below it so that a stack overflow faults instead of silently
    //! corrupting the neighboring stack.
    fibers(size_t n, size_t stack_size) :
        page{size_t(sysconf(_SC_PAGESIZE))},
        stack_size{(stack_size + page - 1) / page * page},
        stride{this->stack_size + page},
        map_size{n * stride},
        ctx(n),
#if !CUDA4CPU_ASM_CONTEXT
        start(n),
#endif
        live((n + warp_size - 1) / warp_size),
        started(live.size()),
        block_wait(live.size()),
        warp_wait(live.size()),
        warp_mask(n),
        warp_kind(n),
        warp_src(n),
        warp_value(n),
        warp_result(n)
    {
        void *map = mmap(nullptr, map_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_STACK, -1, 0);
        if (map == MAP_FAILED)
            throw std::bad_alloc();
        base = static_cast<unsigned char *>(map);

        for (size_t i = 0; i < n; ++i) {
            if (!install_guard(base + i * stride, page)) {
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

    std::vector<context> ctx;         //!< Where each started fiber resumes
#if !CUDA4CPU_ASM_CONTEXT
    std::vector<context> start;       //!< Entry point of each fiber, captured once at creation
#endif
    context caller;                   //!< Where execute() resumes when the block completes
    context discarded;                //!< Saved context of fibers that never resume

    //! Saves the running context in save and resumes load
    void switch_to(context &save, context &load)
    {
#if CUDA4CPU_ASM_CONTEXT
        cuda4cpu_switch_context(&save.sp, load.sp);
#else
        if (_setjmp(save.buf) == 0)
            _longjmp(load.buf, 1);
#endif
    }

    //! Saves the running context in save and starts fiber i from its entry point
    void start_fiber(context &save, size_t i)
    {
#if CUDA4CPU_ASM_CONTEXT
        // The frame that cuda4cpu_switch_context pops: r15, r14, r13, r12, rbx,
        // rbp, and the return address
        void **sp = reinterpret_cast<void **>(stack(i) + stack_size - stack_offset(i)) - 7;
        std::fill(sp, sp + 6, nullptr);
        sp[6] = reinterpret_cast<void *>(cuda4cpu_fiber_trampoline);
        cuda4cpu_switch_context(&save.sp, sp);
#else
        switch_to(save, start[i]);
#endif
    }

    // Scheduling state in fiber mode, as one bit per lane and one word per
    // warp. A thread can run when it is live and not waiting.
    std::vector<uint32_t> live;         //!< Has not returned from the kernel
    std::vector<uint32_t> started;      //!< Has started, so it resumes from ctx
    std::vector<uint32_t> block_wait;   //!< Waiting at __syncthreads()
    std::vector<uint32_t> warp_wait;    //!< Waiting at a warp-synchronous operation

    size_t live_count    = 0;         //!< Threads that have not returned
    size_t block_arrived = 0;         //!< Threads waiting at __syncthreads()

    uint32_t runnable(size_t w) const
    {
        return live[w] & ~block_wait[w] & ~warp_wait[w];
    }

    // Pending warp-synchronous operation of each thread
    std::vector<unsigned>      warp_mask;
    std::vector<unsigned char> warp_kind;
    std::vector<unsigned>      warp_src;
    std::vector<uint64_t>      warp_value;
    std::vector<uint64_t>      warp_result;

    std::unique_ptr<void, free_deleter> shared;   //!< Dynamic shared memory
    size_t shared_capacity = 0;
#ifdef CUDA4CPU_HANDLE_VALGRIND
    std::vector<unsigned> valgrind_ids;
#endif
};

namespace {

#if !CUDA4CPU_ASM_CONTEXT
struct fiber_init {
    context    *start;
    ucontext_t *creator;
};
#endif

}

thread_block::thread_block(dim3 block, size_t stack_size) :
    conf_{dim3(), block},
    kernel_{nullptr},
    nthreads_{conf_.nthreads()},
    stack_size_{stack_size},
    cur_{0},
    direct_{true},
    shared_mem_{nullptr},
    ids_(nthreads_),
    fibers_{std::make_unique<fibers>(nthreads_, stack_size)}
{
    for (size_t i = 0; i < nthreads_; ++i) {
        // Precompute thread ids
        ids_[i] = dim3(unsigned(i % block.x),
                       unsigned((i / block.x) % block.y),
                       unsigned(i / (size_t(block.x) * size_t(block.y))));

#if !CUDA4CPU_ASM_CONTEXT
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
#endif
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
    block->reserve_shared(conf.shared_mem);
    cache.push_back(std::move(block));
    return *cache.back();
}

void thread_block::reserve_shared(size_t bytes)
{
    auto &f = *fibers_;
    if (bytes <= f.shared_capacity)
        return;

    size_t size = (bytes + shared_alignment - 1) / shared_alignment * shared_alignment;
    void *ptr = std::aligned_alloc(shared_alignment, size);
    if (ptr == nullptr)
        throw std::bad_alloc();

    f.shared.reset(ptr);
    f.shared_capacity = size;
    shared_mem_ = ptr;
}

void thread_block::execute(const detail::kernel_closure &kernel, dim3 block_id)
{
    kernel_   = &kernel;
    block_id_ = block_id;
    direct_   = true;

    thread_block *prev = Current_;
    builtin_vars prev_vars = Vars_;
    Current_ = this;
    Vars_.block_idx.set(block_id);
    Vars_.block_dim.set(conf_.block);
    Vars_.grid_dim.set(conf_.grid);
    set_current_thread(0);
    fibers_->start_fiber(fibers_->caller, 0);
    Current_ = prev;
    Vars_    = prev_vars;
}

#if CUDA4CPU_ASM_CONTEXT
namespace detail {

struct fiber_access {
    [[noreturn]] static void start()
    {
        thread_block::Current_->fiber_main();
    }
};

}
#else
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
#endif

void thread_block::fiber_main()
{
    if (direct_) {
        // Fiber 0 at the start of a block. Returns when every CUDA thread has
        // returned without reaching a barrier.
        kernel_->run_direct(*kernel_);
        fibers_->switch_to(fibers_->discarded, fibers_->caller);
    } else {
        kernel_->run_one(*kernel_);
        finish_current();
    }
    __builtin_unreachable();
}

//! Switches the block to fiber mode when the current CUDA thread reaches the
//! first barrier or warp-synchronous operation. Threads before it already
//! returned, and threads after it have not started yet.
void thread_block::promote()
{
    auto &f = *fibers_;
    for (size_t w = 0; w < f.live.size(); ++w) {
        const size_t base = w * warp_size;
        uint32_t exists = nthreads_ - base >= warp_size ? ~0u : (1u << (nthreads_ - base)) - 1;
        uint32_t after  = base >= cur_ ? ~0u : (cur_ - base >= warp_size ? 0u : ~0u << (cur_ - base));
        f.live[w]       = exists & after;
        f.started[w]    = 0;
        f.block_wait[w] = 0;
        f.warp_wait[w]  = 0;
    }
    f.started[cur_ / warp_size] = lane_bit(cur_);

    f.live_count    = nthreads_ - cur_;
    f.block_arrived = 0;
    direct_         = false;
}

void thread_block::syncthreads()
{
    thread_block &block = *Current_;
    if (block.direct_)
        block.promote();

    auto &f = *block.fibers_;
    const size_t w = block.cur_ / warp_size;
    f.block_wait[w] |= lane_bit(block.cur_);
    if (++f.block_arrived == f.live_count)
        block.release_block();
    else
        block.try_release_warp_waiters(w);   // they no longer wait for this lane

    block.switch_from_current();
}

void thread_block::syncwarp(unsigned mask)
{
    Current_->warp_op(fibers::op_sync, mask, 0, 0);
}

uint64_t thread_block::warp_shuffle(unsigned mask, uint64_t value, unsigned src_lane)
{
    return Current_->warp_op(fibers::op_shuffle, mask, value, src_lane);
}

uint64_t thread_block::warp_ballot(unsigned mask, bool predicate)
{
    return Current_->warp_op(fibers::op_ballot, mask, predicate ? 1 : 0, 0);
}

unsigned thread_block::warp_active_mask()
{
    const thread_block &block = *Current_;
    return block.warp_live_mask(block.cur_ / warp_size * warp_size);
}

//! Publishes the calling thread's part of a warp-synchronous operation, waits
//! until every participating lane has done the same, and returns its result.
uint64_t thread_block::warp_op(unsigned char kind, unsigned mask, uint64_t value, unsigned src_lane)
{
    if (direct_)
        promote();

    auto &f = *fibers_;
    const size_t self = cur_;

    f.warp_mask[self]  = mask | lane_bit(self);
    f.warp_kind[self]  = kind;
    f.warp_src[self]   = src_lane % warp_size;
    f.warp_value[self] = value;
    f.warp_wait[self / warp_size] |= lane_bit(self);
    try_release_warp(self);

    switch_from_current();
    return f.warp_result[self];
}

//! Returns the lanes of the warp starting at thread base that take part in
//! warp-synchronous operations: those that have not returned and are not
//! waiting at __syncthreads(), which means they are in a different branch
unsigned thread_block::warp_live_mask(size_t base) const
{
    if (!direct_) {
        const size_t w = base / warp_size;
        return fibers_->live[w] & ~fibers_->block_wait[w];
    }

    // In direct mode, threads before the current one have returned and the
    // rest have not started
    const size_t end = std::min(base + warp_size, nthreads_);
    unsigned mask = 0;
    for (size_t t = std::max(base, cur_); t < end; ++t)
        mask |= lane_bit(t);
    return mask;
}

void thread_block::release_block()
{
    auto &f = *fibers_;
    std::fill(f.block_wait.begin(), f.block_wait.end(), 0u);
    f.block_arrived = 0;
}

//! Completes the warp-synchronous operation that thread tid waits at, if all
//! the live lanes in its mask have reached it, and makes them runnable
void thread_block::try_release_warp(size_t tid)
{
    auto &f = *fibers_;
    const size_t w    = tid / warp_size;
    const size_t base = w * warp_size;
    const uint32_t lanes = warp_live_mask(base) & f.warp_mask[tid];
    if ((lanes & ~f.warp_wait[w]) != 0)
        return;

    switch (f.warp_kind[tid]) {
    case fibers::op_shuffle:
        for (uint32_t m = lanes; m != 0; m &= m - 1) {
            unsigned l   = unsigned(std::countr_zero(m));
            unsigned src = f.warp_src[base + l];
            f.warp_result[base + l] = f.warp_value[base + ((lanes >> src & 1) ? src : l)];
        }
        break;
    case fibers::op_ballot: {
        uint64_t ballot = 0;
        for (uint32_t m = lanes; m != 0; m &= m - 1) {
            unsigned l = unsigned(std::countr_zero(m));
            if (f.warp_value[base + l] != 0)
                ballot |= uint64_t(1) << l;
        }
        for (uint32_t m = lanes; m != 0; m &= m - 1)
            f.warp_result[base + unsigned(std::countr_zero(m))] = ballot | uint64_t(lanes) << 32;
        break;
    }
    default:
        break;
    }

    f.warp_wait[w] &= ~lanes;
}

//! Suspends the current thread, which just started waiting or was released,
//! and resumes the next runnable thread in round-robin order
void thread_block::switch_from_current()
{
    const size_t self = cur_;
    const size_t next = next_runnable(self + 1);
    if (next == self)
        return;

    switch_to_thread(self, next);
}

void thread_block::finish_current()
{
    auto &f = *fibers_;
    const size_t self = cur_;
    const size_t w    = self / warp_size;
    f.live[w] &= ~lane_bit(self);
    if (--f.live_count == 0) {
        f.switch_to(f.discarded, f.caller);
        __builtin_unreachable();
    }

    // Threads that return early leave the barriers that wait for them
    if (f.block_arrived > 0 && f.block_arrived == f.live_count)
        release_block();

    try_release_warp_waiters(w);

    switch_to_thread(no_thread, next_runnable(self + 1));
    __builtin_unreachable();
}

//! Completes the warp-synchronous operations of warp w that no longer wait for
//! any lane, after one of its lanes returned or started waiting at __syncthreads()
void thread_block::try_release_warp_waiters(size_t w)
{
    auto &f = *fibers_;
    for (uint32_t pending = f.warp_wait[w]; pending != 0; ) {
        unsigned l = unsigned(std::countr_zero(pending));
        try_release_warp(w * warp_size + l);
        pending &= f.warp_wait[w] & ~(1u << l);
    }
}

//! Suspends thread from (unless it is no_thread, which never resumes) and runs
//! thread to, from the beginning of the kernel if it has not started yet.
//! Returns when from is resumed.
void thread_block::switch_to_thread(size_t from, size_t to)
{
    auto &f = *fibers_;
    context &save = from == no_thread ? f.discarded : f.ctx[from];
    set_current_thread(to);

    uint32_t &started = f.started[to / warp_size];
    if ((started & lane_bit(to)) == 0) {
        started |= lane_bit(to);
        f.start_fiber(save, to);
    } else {
        f.switch_to(save, f.ctx[to]);
    }
}

//! Returns the first thread at or after from, wrapping around, that can run.
//! There always is one: barriers are released as soon as nothing else can
//! arrive at them.
size_t thread_block::next_runnable(size_t from) const
{
    const auto &f = *fibers_;
    const size_t nwarps = f.live.size();
    const size_t first  = from % nthreads_;
    const size_t w0     = first / warp_size;
    const uint32_t from_lane = ~0u << (first % warp_size);

    if (uint32_t m = f.runnable(w0) & from_lane)
        return w0 * warp_size + size_t(std::countr_zero(m));
    for (size_t k = 1; k < nwarps; ++k) {
        size_t w = (w0 + k) % nwarps;
        if (uint32_t m = f.runnable(w))
            return w * warp_size + size_t(std::countr_zero(m));
    }
    if (uint32_t m = f.runnable(w0) & ~from_lane)
        return w0 * warp_size + size_t(std::countr_zero(m));

    deadlock();
}

void thread_block::deadlock() const
{
    std::fprintf(stderr,
                 "cuda4cpu: internal error: no runnable thread in block (%u, %u, %u), but some "
                 "threads are still waiting at barriers. Please report this.\n",
                 block_id_.x, block_id_.y, block_id_.z);
    std::abort();
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

#if CUDA4CPU_ASM_CONTEXT
void cuda4cpu_fiber_start()
{
    cuda4cpu::detail::fiber_access::start();
}
#endif
