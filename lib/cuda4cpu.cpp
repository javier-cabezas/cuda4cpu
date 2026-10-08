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

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <new>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>

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

// The sanitizer fiber API (sanitizer/common_interface_defs.h), referenced
// weakly: the functions exist only when the program runs with a sanitizer
// runtime, so the same build of the library works with and without one.
extern "C" {
[[gnu::weak]] void __sanitizer_start_switch_fiber(void **fake_stack_save, const void *bottom, size_t size);
[[gnu::weak]] void __sanitizer_finish_switch_fiber(void *fake_stack_save, const void **bottom_old,
                                                   size_t *size_old);
}

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

//! A suspended fiber
struct context {
#if CUDA4CPU_ASM_CONTEXT
    void *sp;                         //!< Its registers are saved on its own stack
#else
    jmp_buf buf;
#endif
};

//! What the sanitizer fiber API needs about a context: the stack it runs on
//! and AddressSanitizer's fake stack. Kept apart from context, which every
//! switch touches.
struct sanitizer_context {
    const void *stack_bottom = nullptr;
    size_t stack_size = 0;
    size_t stack_index = size_t(-1);    //!< Fiber stack it runs on; -1 for the OS thread's
    void *fake_stack = nullptr;
};

//!
//! Debugging settings, read once from the environment
//!
struct settings {
    enum class order { forward, reverse, random };
    enum class divergence { warn, error, ignore };

    order schedule = order::forward;            //!< CUDA4CPU_SCHEDULE
    uint64_t seed = 0;
    divergence barriers = divergence::warn;     //!< CUDA4CPU_DIVERGENT_BARRIERS

    static const settings &get()
    {
        static const settings s = read();
        return s;
    }

private:
    static settings read()
    {
        settings s;
        if (const char *v = std::getenv("CUDA4CPU_SCHEDULE"); v != nullptr && *v != '\0') {
            std::string value = v;
            if (value == "forward") {
                s.schedule = order::forward;
            } else if (value == "reverse") {
                s.schedule = order::reverse;
            } else if (value == "random" || value.rfind("random:", 0) == 0) {
                s.schedule = order::random;
                s.seed = value.size() > 7 ? std::strtoull(value.c_str() + 7, nullptr, 0)
                                          : std::random_device{}();
                std::fprintf(stderr, "cuda4cpu: running threads in random order, seed %llu "
                                     "(CUDA4CPU_SCHEDULE=random:%llu reproduces it)\n",
                             (unsigned long long)s.seed, (unsigned long long)s.seed);
            } else {
                std::fprintf(stderr, "cuda4cpu: ignoring CUDA4CPU_SCHEDULE=%s: expected forward, "
                                     "reverse, random or random:<seed>\n", v);
            }
        }
        if (const char *v = std::getenv("CUDA4CPU_DIVERGENT_BARRIERS"); v != nullptr && *v != '\0') {
            std::string value = v;
            if (value == "warn")
                s.barriers = divergence::warn;
            else if (value == "error")
                s.barriers = divergence::error;
            else if (value == "ignore")
                s.barriers = divergence::ignore;
            else
                std::fprintf(stderr, "cuda4cpu: ignoring CUDA4CPU_DIVERGENT_BARRIERS=%s: expected "
                                     "warn, error or ignore\n", v);
        }
        return s;
    }
};

//! Reports threads of a block that reached different __syncthreads() calls,
//! once for each pair of calls
void report_divergent_barrier(const char *file1, int line1, const char *file2, int line2, dim3 block)
{
    static std::mutex lock;
    static std::set<std::pair<std::string, std::string>> reported;

    auto where = [](const char *file, int line) {
        return file == nullptr ? std::string("unknown location") : std::string(file) + ":" + std::to_string(line);
    };
    const std::string first = where(file1, line1), other = where(file2, line2);

    const bool error = settings::get().barriers == settings::divergence::error;

    // Held until the report is out, and through the abort in error mode, so
    // that blocks running in parallel report one at a time
    std::lock_guard<std::mutex> guard(lock);
    if (!reported.insert(std::minmax(first, other)).second && !error)
        return;
    std::fprintf(stderr,
                 "cuda4cpu: %s: threads of block (%u, %u, %u) reached different __syncthreads() calls, "
                 "at %s and %s. CUDA requires every thread of a block to reach the same one.\n",
                 error ? "error" : "warning", block.x, block.y, block.z, first.c_str(), other.c_str());
    if (error)
        std::abort();
}

}

//!
//! Fiber stacks and contexts of a thread block
//!
struct thread_block::fibers {

    //! Warp-synchronous operations
    enum op : unsigned char {
        op_sync,
        op_shuffle,
        op_ballot,
        op_match_any,
        op_match_all,
        op_converge
    };

    //! Allocates one stack per fiber in a single mapping, each one with a
    //! guard page below it so that a stack overflow faults instead of silently
    //! corrupting the neighboring stack.
    fibers(size_t n, size_t stack_size) :
        sanitize{__sanitizer_start_switch_fiber != nullptr && __sanitizer_finish_switch_fiber != nullptr},
        page{size_t(sysconf(_SC_PAGESIZE))},
        stack_size{(stack_size + page - 1) / page * page},
        stride{this->stack_size + page},
        map_size{n * stride},
        ctx(n),
#if !CUDA4CPU_ASM_CONTEXT
        start(n),
#endif
        warps((n + warp_size - 1) / warp_size),
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

    //! The program runs with a sanitizer that needs to know about fiber switches
    const bool sanitize;

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
        if (sanitize) [[unlikely]]
            sanitizer_leave(save, load);
#if CUDA4CPU_ASM_CONTEXT
        cuda4cpu_switch_context(&save.sp, load.sp);
#else
        if (_setjmp(save.buf) == 0)
            _longjmp(load.buf, 1);
#endif
        if (sanitize) [[unlikely]]
            sanitizer_resumed(save);
    }

    //! Saves the running context in save and starts fiber i from its entry
    //! point, on stack i
    void start_fiber(context &save, size_t i)
    {
        set_stack(i, i);
        starting_stack = i;
#if CUDA4CPU_ASM_CONTEXT
        // The frame that cuda4cpu_switch_context pops: r15, r14, r13, r12, rbx,
        // rbp, and the return address
        void **sp = reinterpret_cast<void **>(stack(i) + stack_size - stack_offset(i)) - 7;
        for (int r = 0; r < 6; ++r)
            sp[r] = nullptr;
        sp[6] = reinterpret_cast<void *>(cuda4cpu_fiber_trampoline);
        if (sanitize) [[unlikely]]
            sanitizer_leave(save, ctx[i]);
        cuda4cpu_switch_context(&save.sp, sp);
#else
        if (sanitize) [[unlikely]]
            sanitizer_leave(save, ctx[i]);
        if (_setjmp(save.buf) == 0)
            _longjmp(start[i].buf, 1);
#endif
        if (sanitize) [[unlikely]]
            sanitizer_resumed(save);
    }

    //! Tells AddressSanitizer that the running context switches to the stack
    //! that load runs on. A context saved in discarded never resumes: its fake
    //! stack goes back to the stack it ran on. Call this last before the
    //! switch, after the caller's last use of its addressable locals.
    [[gnu::noinline]] void sanitizer_leave(context &save, const context &load)
    {
        void **fake_stack = &sanitizer_of(save).fake_stack;
        if (&save == &discarded) {
            if (stack_fake_stacks.empty())
                stack_fake_stacks.resize(ctx.size());
            fake_stack = &stack_fake_stacks[running_stack];
        }
        const auto &target = sanitizer_of(load);
        __sanitizer_start_switch_fiber(fake_stack, target.stack_bottom, target.stack_size);
    }

    //! Tells AddressSanitizer that self runs again
    [[gnu::noinline]] void sanitizer_resumed(context &self)
    {
        auto &sc = sanitizer_of(self);
        __sanitizer_finish_switch_fiber(sc.fake_stack, nullptr, nullptr);
        running_stack = sc.stack_index;
    }

    //! Tells AddressSanitizer that a fiber started, with the fake stack the
    //! last thread on its stack left. When execute() starts it, the stack it
    //! comes from is the OS thread's, where the block returns.
    void sanitizer_started(bool from_caller)
    {
        if (!sanitize) [[likely]]
            return;
        if (stack_fake_stacks.empty())
            stack_fake_stacks.resize(ctx.size());
        running_stack = starting_stack;
        void *fake_stack = std::exchange(stack_fake_stacks[running_stack], nullptr);

        const void *bottom = nullptr;
        size_t size = 0;
        __sanitizer_finish_switch_fiber(fake_stack, &bottom, &size);
        if (from_caller) {
            sanitizer_caller.stack_bottom = bottom;
            sanitizer_caller.stack_size   = size;
        }
    }

    // Scheduling state in fiber mode, as one bit per lane, together for each
    // warp so that a scheduling decision reads a single cache line. A thread
    // can run when it is live and not waiting.
    struct alignas(32) warp_state {
        uint32_t live          = 0;   //!< Has not returned from the kernel
        uint32_t started       = 0;   //!< Has started, so it resumes from ctx
        uint32_t block_wait    = 0;   //!< Waiting at __syncthreads()
        uint32_t warp_wait     = 0;   //!< Waiting at a warp-synchronous operation
        uint32_t tile_wait     = 0;   //!< Waiting at the barrier of a tile larger than a warp
        uint32_t converge_wait = 0;   //!< Waiting in __activemask() (also in warp_wait)

        // The warps of the tile barrier pending in this warp, [tile_begin,
        // tile_end) (empty if none). Tiles are aligned groups of whole warps,
        // so a warp is in at most one pending tile barrier.
        unsigned tile_begin = 0;
        unsigned tile_end   = 0;
    };
    std::vector<warp_state> warps;

    size_t live_count    = 0;         //!< Threads that have not returned
    size_t block_arrived = 0;         //!< Threads waiting at __syncthreads()
    const char *barrier_file = nullptr;   //!< Where the first thread called the pending __syncthreads()
    int barrier_line = 0;
    bool grid_pending = false;        //!< The pending __syncthreads() is a grid_group::sync()

    // AddressSanitizer fake stacks of the fiber stacks. Like the stacks, they
    // are reused: a thread that returns leaves its fake stack here, and the
    // next thread that starts on that stack takes it (creating and destroying
    // one per CUDA thread costs two mmap calls).
    std::vector<void *> stack_fake_stacks;
    std::vector<sanitizer_context> sanitizer_ctx;     //!< One for each context in ctx
    sanitizer_context sanitizer_caller;
    sanitizer_context sanitizer_discarded;

    sanitizer_context &sanitizer_of(const context &c)
    {
        if (&c == &caller)
            return sanitizer_caller;
        if (&c == &discarded)
            return sanitizer_discarded;
        if (sanitizer_ctx.empty())
            sanitizer_ctx.resize(ctx.size());
        return sanitizer_ctx[size_t(&c - ctx.data())];
    }

    //! Records that the thread whose context is ctx[t] runs on stack s
    void set_stack(size_t t, size_t s)
    {
        if (!sanitize) [[likely]]
            return;
        auto &sc = sanitizer_of(ctx[t]);
        sc.stack_bottom = stack(s);
        sc.stack_size   = stack_size;
        sc.stack_index  = s;
    }
    size_t running_stack  = size_t(-1);
    size_t starting_stack = size_t(-1);

    // Running order of the threads when CUDA4CPU_SCHEDULE is not forward
    std::vector<unsigned> order;
    std::vector<unsigned> position;   //!< Inverse of order
    size_t direct_stack = 0;          //!< Stack of the fiber that runs direct mode

    // The rarer warp operations, kept out of try_release_warp so that the
    // path of shuffles and votes stays short

    //! Results of the __match_any_sync or __match_all_sync that thread tid
    //! waits at, over lanes
    [[gnu::noinline]] void compute_match(size_t tid, uint32_t lanes)
    {
        const size_t base = tid / warp_size * warp_size;
        if (warp_kind[tid] == op_match_any) {
            for (uint32_t m = lanes; m != 0; m &= m - 1) {
                unsigned l = unsigned(std::countr_zero(m));
                uint32_t same = 0;
                for (uint32_t k = lanes; k != 0; k &= k - 1) {
                    unsigned o = unsigned(std::countr_zero(k));
                    if (warp_value[base + o] == warp_value[base + l])
                        same |= 1u << o;
                }
                warp_result[base + l] = same;
            }
        } else {
            bool equal = true;
            for (uint32_t m = lanes; m != 0; m &= m - 1)
                equal = equal && warp_value[base + unsigned(std::countr_zero(m))] == warp_value[tid];
            for (uint32_t m = lanes; m != 0; m &= m - 1)
                warp_result[base + unsigned(std::countr_zero(m))] = equal ? lanes : 0;
        }
    }

    //! Completes the __activemask() call that thread tid waits at, once every
    //! lane that can run is waiting: the lanes waiting at the same call are the
    //! ones executing it together
    [[gnu::noinline]] void release_converge(size_t tid)
    {
        const size_t base = tid / warp_size * warp_size;
        auto &warp = warps[tid / warp_size];
        uint32_t group = 0;
        for (uint32_t m = warp.converge_wait; m != 0; m &= m - 1) {
            unsigned l = unsigned(std::countr_zero(m));
            if (warp_value[base + l] == warp_value[tid])
                group |= 1u << l;
        }
        for (uint32_t m = group; m != 0; m &= m - 1)
            warp_result[base + unsigned(std::countr_zero(m))] = group;
        warp.warp_wait     &= ~group;
        warp.converge_wait &= ~group;
    }

    uint32_t runnable(size_t w) const
    {
        return warps[w].live & ~warps[w].block_wait & ~warps[w].warp_wait & ~warps[w].tile_wait;
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
    pos_{0},
    order_{nullptr},
    direct_{true},
    shared_mem_{nullptr},
    ids_(nthreads_),
    fibers_{std::make_unique<fibers>(nthreads_, stack_size)}
{
    if (settings::get().schedule != settings::order::forward) {
        auto &f = *fibers_;
        f.order.resize(nthreads_);
        f.position.resize(nthreads_);
        std::iota(f.order.begin(), f.order.end(), 0u);
        if (settings::get().schedule == settings::order::reverse)
            std::reverse(f.order.begin(), f.order.end());
        for (size_t p = 0; p < nthreads_; ++p)
            f.position[f.order[p]] = unsigned(p);
        order_ = f.order.data();
    }

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
    pos_      = 0;

    auto &f = *fibers_;
    if (settings::get().schedule == settings::order::random) {
        // A different order for every block, reproducible from the seed
        uint64_t linear = (uint64_t(block_id.z) * conf_.grid.y + block_id.y) * conf_.grid.x + block_id.x;
        std::mt19937_64 random(settings::get().seed ^ (linear * 0x9e3779b97f4a7c15ull));
        std::shuffle(f.order.begin(), f.order.end(), random);
        for (size_t p = 0; p < nthreads_; ++p)
            f.position[f.order[p]] = unsigned(p);
    }

    // Direct mode runs on the stack of the first thread in the order: when the
    // block switches to fiber mode, that thread has returned or is the one
    // that keeps running there, so no other thread starts on that stack.
    const size_t first = order_ ? order_[0] : 0;
    f.direct_stack = first;

    thread_block *prev = Current_;
    builtin_vars prev_vars = Vars_;
    Current_ = this;
    Vars_.block_idx.set(block_id);
    Vars_.block_dim.set(conf_.block);
    Vars_.grid_dim.set(conf_.grid);
    set_current_thread(first);
    f.start_fiber(f.caller, first);
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
    fibers_->sanitizer_started(direct_);
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
    for (size_t w = 0; w < f.warps.size(); ++w) {
        const size_t base = w * warp_size;
        uint32_t exists = nthreads_ - base >= warp_size ? ~0u : (1u << (nthreads_ - base)) - 1;
        // In the default order, the threads before cur_ have returned
        uint32_t after  = order_ || base >= cur_ ? ~0u
                                                 : (cur_ - base >= warp_size ? 0u : ~0u << (cur_ - base));
        f.warps[w].live       = exists & after;
        f.warps[w].started    = 0;
        f.warps[w].block_wait = 0;
        f.warps[w].warp_wait  = 0;
        f.warps[w].tile_wait  = 0;
        f.warps[w].tile_begin = 0;
        f.warps[w].tile_end   = 0;
        f.warps[w].converge_wait = 0;
    }
    if (order_) {
        for (size_t p = 0; p < pos_; ++p)
            f.warps[order_[p] / warp_size].live &= ~lane_bit(order_[p]);
    }
    f.warps[cur_ / warp_size].started = lane_bit(cur_);

    // The current thread keeps running on the stack of direct mode
    f.set_stack(cur_, f.direct_stack);

    f.live_count    = nthreads_ - pos_;
    f.block_arrived = 0;
    f.grid_pending  = false;
    direct_         = false;
}

void thread_block::syncthreads(const char *file, int line)
{
    thread_block &block = *Current_;
    if (block.direct_)
        block.promote();

    auto &f = *block.fibers_;
    if (settings::get().barriers != settings::divergence::ignore)
        block.check_barrier_site(file, line);

    const size_t w = block.cur_ / warp_size;
    f.warps[w].block_wait |= lane_bit(block.cur_);
    if (++f.block_arrived == f.live_count) {
        block.release_block();
    } else {
        // They no longer wait for this lane
        block.try_release_warp_waiters(w);
        block.try_release_tile(w);
    }

    block.switch_from_current();
}

void thread_block::sync_grid(const char *file, int line)
{
    thread_block &block = *Current_;
    if (block.kernel_->grid == nullptr) {
        std::fprintf(stderr, "cuda4cpu: grid_group::sync() in a kernel not launched with "
                             "cudaLaunchCooperativeKernel, in block (%u, %u, %u)\n",
                     block.block_id_.x, block.block_id_.y, block.block_id_.z);
        std::abort();
    }
    if (block.direct_)
        block.promote();
    block.fibers_->grid_pending = true;
    syncthreads(file, line);
}

void thread_block::sync_warps(size_t first_warp, size_t nwarps)
{
    thread_block &block = *Current_;
    if (block.direct_)
        block.promote();

    auto &f = *block.fibers_;
    const size_t w   = block.cur_ / warp_size;
    const size_t end = std::min(first_warp + nwarps, f.warps.size());
    f.warps[w].tile_wait |= lane_bit(block.cur_);
    for (size_t t = first_warp; t < end; ++t) {
        f.warps[t].tile_begin = unsigned(first_warp);
        f.warps[t].tile_end   = unsigned(end);
    }
    block.try_release_tile(w);
    block.try_release_warp_waiters(w);   // warp operations leave this lane out now

    block.switch_from_current();
}

unsigned thread_block::warp_match(unsigned mask, uint64_t value, bool all)
{
    return unsigned(Current_->warp_op(all ? fibers::op_match_all : fibers::op_match_any, mask, value, 0));
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

unsigned thread_block::warp_active_mask(uint64_t site)
{
    return unsigned(Current_->warp_op(fibers::op_converge, ~0u, site, 0));
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
    f.warps[self / warp_size].warp_wait |= lane_bit(self);
    if (kind == fibers::op_converge)
        f.warps[self / warp_size].converge_wait |= lane_bit(self);
    try_release_warp(self);
    if (f.warps[self / warp_size].converge_wait != 0)
        try_release_converge(self / warp_size);   // they may have waited for this lane

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
        return fibers_->warps[w].live & ~fibers_->warps[w].block_wait & ~fibers_->warps[w].tile_wait;
    }

    // In direct mode, the threads before the current one in the running order
    // have returned and the rest have not started
    const size_t end = std::min(base + warp_size, nthreads_);
    unsigned mask = 0;
    if (order_ == nullptr) {
        for (size_t t = std::max(base, cur_); t < end; ++t)
            mask |= lane_bit(t);
    } else {
        for (size_t t = base; t < end; ++t)
            mask |= lane_bit(t);
        for (size_t p = 0; p < pos_; ++p) {
            if (order_[p] >= base && order_[p] < end)
                mask &= ~lane_bit(order_[p]);
        }
    }
    return mask;
}

void thread_block::release_block()
{
    auto &f = *fibers_;
    if (f.grid_pending) {
        // Every thread of the block is at grid_group::sync(): the last one to
        // arrive waits for the other blocks, which run on other OS threads
        f.grid_pending = false;
        kernel_->grid->arrive_and_wait();
    }
    for (auto &warp : f.warps)
        warp.block_wait = 0;
    f.block_arrived = 0;
}

//! Completes the warp-synchronous operation that thread tid waits at, if all
//! the live lanes in its mask have reached it, and makes them runnable
void thread_block::try_release_warp(size_t tid)
{
    auto &f = *fibers_;
    const size_t w    = tid / warp_size;
    const size_t base = w * warp_size;
    const auto &warp = f.warps[w];
    const uint32_t lanes = warp_live_mask(base) & f.warp_mask[tid];
    const bool converge = f.warp_kind[tid] == fibers::op_converge;
    // For other operations, lanes in __activemask() haven't arrived: they'll
    // come once it returns
    const uint32_t arrived = converge ? warp.warp_wait : warp.warp_wait & ~warp.converge_wait;
    if ((lanes & ~arrived) != 0)
        return;

    if (converge) {
        f.release_converge(tid);
        return;
    }
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
    case fibers::op_match_any:
    case fibers::op_match_all:
        f.compute_match(tid, lanes);
        break;
    default:
        break;
    }

    f.warps[w].warp_wait &= ~lanes;
}

//! Suspends the current thread, which just started waiting or was released,
//! and resumes the next runnable thread in round-robin order
void thread_block::switch_from_current()
{
    const size_t self = cur_;
    const size_t next = order_ ? next_runnable_after(self) : next_runnable(self + 1);
    if (next == self)
        return;

    switch_to_thread(self, next);
}

void thread_block::finish_current()
{
    auto &f = *fibers_;
    const size_t self = cur_;
    const size_t w    = self / warp_size;
    f.warps[w].live &= ~lane_bit(self);
    if (--f.live_count == 0) {
        f.switch_to(f.discarded, f.caller);
        __builtin_unreachable();
    }

    // Threads that return early leave the barriers that wait for them
    if (f.block_arrived > 0 && f.block_arrived == f.live_count)
        release_block();

    try_release_warp_waiters(w);
    try_release_tile(w);

    switch_to_thread(no_thread, order_ ? next_runnable_after(self) : next_runnable(self + 1));
    __builtin_unreachable();
}

//! Completes the warp-synchronous operations of warp w that no longer wait for
//! any lane, after one of its lanes returned or started waiting at __syncthreads()
void thread_block::try_release_warp_waiters(size_t w)
{
    auto &f = *fibers_;
    for (uint32_t pending = f.warps[w].warp_wait; pending != 0; ) {
        unsigned l = unsigned(std::countr_zero(pending));
        try_release_warp(w * warp_size + l);
        pending &= f.warps[w].warp_wait & ~(1u << l);
    }
}

//! Completes the __activemask() calls of warp w that no longer wait for any lane
void thread_block::try_release_converge(size_t w)
{
    auto &f = *fibers_;
    for (uint32_t pending = f.warps[w].converge_wait; pending != 0; ) {
        unsigned l = unsigned(std::countr_zero(pending));
        try_release_warp(w * warp_size + l);
        pending &= f.warps[w].converge_wait & ~(1u << l);
    }
}

//! Completes the tile barrier pending in warp w, if every thread of the tile
//! that has not returned and is not at __syncthreads() has reached it
void thread_block::try_release_tile(size_t w)
{
    auto &f = *fibers_;
    const size_t begin = f.warps[w].tile_begin, end = f.warps[w].tile_end;
    for (size_t t = begin; t < end; ++t) {
        if ((f.warps[t].live & ~f.warps[t].block_wait & ~f.warps[t].tile_wait) != 0)
            return;
    }
    for (size_t t = begin; t < end; ++t) {
        f.warps[t].tile_wait  = 0;
        f.warps[t].tile_begin = 0;
        f.warps[t].tile_end   = 0;
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

    uint32_t &started = f.warps[to / warp_size].started;
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
    const size_t nwarps = f.warps.size();
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

//! Returns the thread after tid in the running order, wrapping around, that
//! can run
size_t thread_block::next_runnable_after(size_t tid) const
{
    if (order_ == nullptr)
        return next_runnable(tid + 1);

    const auto &f = *fibers_;
    const size_t p = f.position[tid];
    for (size_t k = 1; k <= nthreads_; ++k) {
        const unsigned t = order_[(p + k) % nthreads_];
        if (f.runnable(t / warp_size) & lane_bit(t))
            return t;
    }
    deadlock();
}

//! Checks that the current thread waits at the same __syncthreads() call as
//! the threads that arrived before it
void thread_block::check_barrier_site(const char *file, int line)
{
    auto &f = *fibers_;
    if (f.block_arrived == 0) {
        f.barrier_file = file;
        f.barrier_line = line;
    } else if (line != f.barrier_line ||
               (file != f.barrier_file && (file == nullptr || f.barrier_file == nullptr ||
                                           std::strcmp(file, f.barrier_file) != 0))) {
        report_divergent_barrier(f.barrier_file, f.barrier_line, file, line, block_id_);
    }
}

void thread_block::deadlock() const
{
    std::fprintf(stderr,
                 "cuda4cpu: internal error: no runnable thread in block (%u, %u, %u), but some "
                 "threads are still waiting at barriers. Please report this.\n",
                 block_id_.x, block_id_.y, block_id_.z);
    std::abort();
}

void *detail::allocate_aligned(size_t size)
{
    // Rounding up to the alignment must not wrap around to a small size
    if (size > SIZE_MAX - allocation_alignment)
        return nullptr;

    return std::aligned_alloc(allocation_alignment,
                              (size + allocation_alignment - 1) / allocation_alignment * allocation_alignment);
}

void detail::memory_info(size_t &free, size_t &total)
{
    const long page = sysconf(_SC_PAGESIZE);
    const long available = sysconf(_SC_AVPHYS_PAGES), physical = sysconf(_SC_PHYS_PAGES);
    free  = page > 0 && available > 0 ? size_t(available) * size_t(page) : 0;
    total = page > 0 && physical > 0 ? size_t(physical) * size_t(page) : 0;
}

void detail::get_device_properties(cudaDeviceProp &prop)
{
    // Computed once: none of this changes while the process runs
    static const cudaDeviceProp props = [] {
        cudaDeviceProp p{};

        // The first CPU's model name and current clock, from /proc/cpuinfo
        std::string name = "cuda4cpu";
        double mhz = 0.0;
        std::ifstream cpuinfo("/proc/cpuinfo");
        for (std::string line; std::getline(cpuinfo, line) && line.find_first_not_of(" \t") != std::string::npos; ) {
            size_t colon = line.find(':');
            if (colon == std::string::npos)
                continue;
            size_t start = line.find_first_not_of(" \t", colon + 1);
            std::string value = start == std::string::npos ? "" : line.substr(start);
            if (line.rfind("model name", 0) == 0 && !value.empty())
                name += ": " + value;
            else if (line.rfind("cpu MHz", 0) == 0)
                mhz = std::atof(value.c_str());
        }
        std::snprintf(p.name, sizeof(p.name), "%s", name.c_str());

        long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
        p.totalGlobalMem = pages > 0 && page > 0 ? size_t(pages) * size_t(page) : 0;

        // Maximum clock in kHz, or the current one if cpufreq is not available
        long khz = 0;
        std::ifstream freq("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
        freq >> khz;
        p.clockRate = khz > 0 ? int(khz) : int(mhz * 1000.0);

        long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
        p.l2CacheSize = l2 > 0 ? int(l2) : 0;

        // CUDA's limits for threads, blocks and grids
        p.warpSize           = 32;
        p.maxThreadsPerBlock = 1024;
        p.maxThreadsDim[0]   = 1024;
        p.maxThreadsDim[1]   = 1024;
        p.maxThreadsDim[2]   = 64;
        p.maxGridSize[0]     = INT_MAX;
        p.maxGridSize[1]     = 65535;
        p.maxGridSize[2]     = 65535;
        p.sharedMemPerBlock          = 48 * 1024;
        p.sharedMemPerBlockOptin     = 48 * 1024;
        p.sharedMemPerMultiprocessor = 48 * 1024;
        p.totalConstMem         = 64 * 1024;
        p.regsPerBlock          = 65536;
        p.regsPerMultiprocessor = 65536;
        p.memPitch              = INT_MAX;
        p.textureAlignment      = 512;
        p.texturePitchAlignment = 32;

        // The newest compute capability whose features cuda4cpu implements
        // (no tensor cores), so that code that checks it picks supported paths
        p.major = 6;
        p.minor = 0;

        // Each OS thread is a multiprocessor that runs one block at a time
        p.multiProcessorCount         = omp_get_max_threads();
        p.maxThreadsPerMultiProcessor = 1024;
        p.maxBlocksPerMultiProcessor  = 1;
        // All the blocks of a cooperative launch run at once, one per OS thread
        p.cooperativeLaunch           = 1;

        // Device memory is host memory
        p.integrated                        = 1;
        p.canMapHostMemory                  = 1;
        p.unifiedAddressing                 = 1;
        p.managedMemory                     = 1;
        p.concurrentManagedAccess           = 1;
        p.pageableMemoryAccess              = 1;
        p.directManagedMemAccessFromHost    = 1;
        p.hostNativeAtomicSupported         = 1;
        p.canUseHostPointerForRegisteredMem = 1;
        p.hostRegisterSupported             = 1;
        p.hostRegisterReadOnlySupported     = 1;
        p.memoryPoolsSupported              = 1;
        p.globalL1CacheSupported            = 1;
        p.localL1CacheSupported             = 1;
        p.singleToDoublePrecisionPerfRatio  = 2;

        return p;
    }();

    prop = props;
}

system::system()
{
    cpus_ = omp_get_num_procs();

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
