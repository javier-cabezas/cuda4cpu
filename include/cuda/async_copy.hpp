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

// The part of libcu++ (CUDA's C++ library) for asynchronous copies:
// cuda::memcpy_async with thread-scope pipelines and with barriers, and
// cuda::barrier, for <cuda/pipeline> and <cuda/barrier>. On the CPU a copy
// is done when memcpy_async returns, so waiting for one only needs the
// synchronization that CUDA already requires between the threads that copy
// and those that read. A pipeline that a group shares
// (make_pipeline(group, shared_state)) waits as CUDA's does: a stage's
// consumers for every producer's commit, and its producers, before reusing
// it, for every consumer's release.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#include "../cuda_runtime.h"

namespace cuda {

enum thread_scope {
    thread_scope_system = 0,
    thread_scope_device = 1,
    thread_scope_block  = 2,
    thread_scope_thread = 3,
};

//! A size in bytes that is a multiple of Alignment, as are the addresses it
//! goes with
template <::std::size_t Alignment>
struct aligned_size_t {
    static constexpr ::std::size_t align = Alignment;
    ::std::size_t value;
    constexpr explicit aligned_size_t(::std::size_t size) : value(size) {}
    constexpr operator ::std::size_t() const { return value; }
};

}

namespace cuda4cpu::detail {

struct empty_completion {
    void operator()() noexcept {}
};

//! Lets the other CUDA threads of the block run while the caller waits
inline void yield_to_block()
{
    thread_block::interleave();
}

//! A copy that a group's threads share: its first thread makes all of it.
//! The block's threads run on one OS thread, so splitting it would gain
//! nothing, and a tile of a partial warp has fewer threads than its size.
inline void group_copy(void *dst, const void *src, ::std::size_t size, unsigned long long rank)
{
    if (rank == 0)
        ::std::memcpy(dst, src, size);
}

}

namespace cuda {

//! A barrier that expected arrivals complete, after which completion runs
//! and the next phase starts. Waiting lets the block's other threads run, so
//! a barrier of the block's threads works; a barrier of several blocks needs
//! them to run at once, as in CUDA.
template <thread_scope Scope, typename CompletionFunction = cuda4cpu::detail::empty_completion>
class barrier {
public:
    using arrival_token = ::std::uint64_t;

    barrier() = default;
    constexpr explicit barrier(::std::ptrdiff_t expected, CompletionFunction completion = CompletionFunction())
        : expected_(expected), pending_(expected), phase_(0), completion_(completion)
    {
    }
    barrier(const barrier &) = delete;
    barrier &operator=(const barrier &) = delete;

    [[nodiscard]] arrival_token arrive(::std::ptrdiff_t update = 1)
    {
        // Arrivals beyond the expected ones are undefined; they complete the
        // phase rather than hang
        const arrival_token phase = phase_.load();
        if (pending_.fetch_sub(update) <= update) {
            completion_();
            pending_.store(expected_.load());
            phase_.store(phase + 1);
        }
        return phase;
    }

    void wait(arrival_token &&token) const
    {
        while (phase_.load() == token)
            cuda4cpu::detail::yield_to_block();
    }

    void arrive_and_wait()
    {
        wait(arrive());
    }

    //! Arrives, and leaves the barrier for the next phases
    void arrive_and_drop()
    {
        expected_.fetch_sub(1);
        (void)arrive();
    }

    static constexpr ::std::ptrdiff_t max() noexcept
    {
        return ::std::numeric_limits<::std::ptrdiff_t>::max();
    }

private:
    ::std::atomic<::std::ptrdiff_t> expected_, pending_;
    ::std::atomic<arrival_token> phase_;
    CompletionFunction completion_;
};

//! Initializes a barrier declared without arguments, as __shared__ ones are
template <thread_scope Scope, typename CompletionFunction>
inline void init(barrier<Scope, CompletionFunction> *bar, ::std::ptrdiff_t expected)
{
    ::new (bar) barrier<Scope, CompletionFunction>(expected);
}

template <thread_scope Scope, typename CompletionFunction>
inline void init(barrier<Scope, CompletionFunction> *bar, ::std::ptrdiff_t expected, CompletionFunction completion)
{
    ::new (bar) barrier<Scope, CompletionFunction>(expected, completion);
}

template <thread_scope Scope>
class pipeline;

}

namespace cuda4cpu::detail {

//! The arrivals at one stage of a shared pipeline: a phase ends when the
//! expected threads have arrived, and waits are for a phase to end, letting
//! the block's other threads run
class stage_barrier {
public:
    void init(::std::ptrdiff_t expected)
    {
        expected_.store(expected);
        pending_.store(expected);
        phase_.store(0);
    }

    void arrive()
    {
        if (pending_.fetch_sub(1) <= 1) {
            pending_.store(expected_.load());
            phase_.fetch_add(1);
        }
    }

    //! A thread that has arrived at phases [0, arrived) leaves: it no longer
    //! takes part in the phases after them, including the current one if
    //! it hasn't arrived at it yet. A block's threads run on one OS thread,
    //! so nothing else changes the barrier meanwhile.
    void leave(::std::uint64_t arrived)
    {
        expected_.fetch_sub(1);
        if (done(arrived))
            arrive();
    }

    //! Whether phases [0, phase) have ended
    bool done(::std::uint64_t phase) const
    {
        return phase_.load() >= phase;
    }

    void wait(::std::uint64_t phase) const
    {
        while (!done(phase))
            yield_to_block();
    }

    template <typename Clock, typename Duration>
    bool wait_until(::std::uint64_t phase, const ::std::chrono::time_point<Clock, Duration> &deadline) const
    {
        while (!done(phase)) {
            if (Clock::now() >= deadline)
                return false;
            yield_to_block();
        }
        return true;
    }

private:
    ::std::atomic<::std::ptrdiff_t> expected_{0}, pending_{0};
    ::std::atomic<::std::uint64_t> phase_{0};
};

}

namespace cuda4cpu::detail {

struct pipeline_maker;

}

namespace cuda {

enum class pipeline_role {
    producer,
    consumer,
};

//! The state that a group's threads share for a pipeline of Stages stages,
//! in shared memory: make_pipeline initializes it
template <thread_scope Scope, ::std::uint8_t Stages>
class pipeline_shared_state {
public:
    pipeline_shared_state() = default;
    pipeline_shared_state(const pipeline_shared_state &) = delete;
    pipeline_shared_state &operator=(const pipeline_shared_state &) = delete;

private:
    friend struct cuda4cpu::detail::pipeline_maker;

    cuda4cpu::detail::stage_barrier produced_[Stages], consumed_[Stages];
    ::std::atomic<::std::size_t> producers_{0}, consumers_{0}, refcount_{0};
};

//! A pipeline that the threads of a group of a block share
//! (thread_scope_block). Each thread has its own, made by make_pipeline from
//! the shared state, and its stages go round: the producers acquire one,
//! copy into it and commit it, and the consumers wait for it, read it and
//! release it.
template <thread_scope Scope>
class pipeline {
public:
    pipeline(pipeline &&other) noexcept { *this = ::std::move(other); }
    pipeline(const pipeline &) = delete;
    pipeline &operator=(const pipeline &) = delete;
    pipeline &operator=(pipeline &&other) noexcept
    {
        if (this != &other) {
            if (refcount_ != nullptr)
                quit();
            produced_ = other.produced_;
            consumed_ = other.consumed_;
            refcount_ = other.refcount_;
            stages_   = other.stages_;
            head_     = other.head_;
            tail_     = other.tail_;
            producer_ = other.producer_;
            consumer_ = other.consumer_;
            other.refcount_ = nullptr;
        }
        return *this;
    }
    ~pipeline()
    {
        if (refcount_ != nullptr)
            quit();
    }

    //! Waits until the next stage's consumers have released its last use
    void producer_acquire()
    {
        producer_ = true;
        consumed_[head_ % stages_].wait(head_ / stages_);
    }

    void producer_commit()
    {
        producer_ = true;
        produced_[head_ % stages_].arrive();
        ++head_;
    }

    //! Waits until every producer has committed the oldest stage
    void consumer_wait()
    {
        consumer_ = true;
        produced_[tail_ % stages_].wait(tail_ / stages_ + 1);
    }

    template <typename Rep, typename Period>
    bool consumer_wait_for(const ::std::chrono::duration<Rep, Period> &duration)
    {
        return consumer_wait_until(::std::chrono::steady_clock::now() + duration);
    }

    template <typename Clock, typename Duration>
    bool consumer_wait_until(const ::std::chrono::time_point<Clock, Duration> &deadline)
    {
        consumer_ = true;
        return produced_[tail_ % stages_].wait_until(tail_ / stages_ + 1, deadline);
    }

    void consumer_release()
    {
        consumer_ = true;
        consumed_[tail_ % stages_].arrive();
        ++tail_;
    }

    //! Leaves the pipeline: the others no longer wait for this thread, in
    //! the roles it took. Returns whether it was the last thread, which may
    //! destroy the state; false if it had quit already, or was moved from.
    bool quit()
    {
        if (refcount_ == nullptr)
            return false;
        for (::std::uint8_t s = 0; s < stages_; ++s) {
            // How many times this thread committed and released stage s
            if (producer_)
                produced_[s].leave((head_ + stages_ - 1 - s) / stages_);
            if (consumer_)
                consumed_[s].leave((tail_ + stages_ - 1 - s) / stages_);
        }
        const bool last = refcount_->fetch_sub(1) == 1;
        refcount_ = nullptr;
        return last;
    }

private:
    pipeline() = default;

    friend struct cuda4cpu::detail::pipeline_maker;

    cuda4cpu::detail::stage_barrier *produced_ = nullptr, *consumed_ = nullptr;
    ::std::atomic<::std::size_t> *refcount_ = nullptr;
    ::std::uint8_t stages_ = 0;
    ::std::uint64_t head_ = 0, tail_ = 0;      // stages committed and released, so far
    // Its roles: given to make_pipeline, or (with a count of producers)
    // those of the calls it makes
    bool producer_ = false, consumer_ = false;
};

}

namespace cuda4cpu::detail {

//! Initializes the shared state, with the group's counts of producers and
//! consumers, and makes the calling thread's pipeline. With producer_count,
//! the group has that many producers, any of its threads, and the others
//! consume; otherwise each thread says what it is, and the group syncs
//! twice more to count them: before the counts start from zero, and after
//! they are taken. The stages' barriers count arrivals of threads that run
//! one at a time, as a block's do on its OS thread.
struct pipeline_maker {
    static constexpr ::std::size_t counted = ~::std::size_t(0);

    template <typename Group, ::cuda::thread_scope Scope, ::std::uint8_t Stages>
    static ::cuda::pipeline<Scope> make(const Group &group, ::cuda::pipeline_shared_state<Scope, Stages> *state,
                                        bool producer, bool consumer, ::std::size_t producer_count = counted)
    {
        static_assert(Scope == ::cuda::thread_scope_block,
                      "cuda4cpu: only pipelines of a block's threads (thread_scope_block) can be shared");
        if (producer_count != counted) {
            if (group.thread_rank() == 0) {
                state->producers_.store(producer_count);
                state->consumers_.store(group.size() - producer_count);
                state->refcount_.store(group.size());
            }
        } else {
            if (group.thread_rank() == 0) {
                state->producers_.store(0);
                state->consumers_.store(0);
                state->refcount_.store(group.size());
            }
            group.sync();
            if (producer)
                state->producers_.fetch_add(1);
            if (consumer)
                state->consumers_.fetch_add(1);
            group.sync();
        }
        if (group.thread_rank() == 0) {
            for (::std::uint8_t s = 0; s < Stages; ++s) {
                state->produced_[s].init(::std::ptrdiff_t(state->producers_.load()));
                state->consumed_[s].init(::std::ptrdiff_t(state->consumers_.load()));
            }
        }
        group.sync();
        ::cuda::pipeline<Scope> p;
        p.produced_ = state->produced_;
        p.consumed_ = state->consumed_;
        p.refcount_ = &state->refcount_;
        p.stages_   = Stages;
        p.producer_ = producer;
        p.consumer_ = consumer;
        return p;
    }
};

}

namespace cuda {

//! Makes the pipeline of the calling thread from the state that the group's
//! threads share, all of which must call it: every thread produces and
//! consumes
template <typename Group, thread_scope Scope, ::std::uint8_t Stages>
pipeline<Scope> make_pipeline(const Group &group, pipeline_shared_state<Scope, Stages> *state)
{
    return cuda4cpu::detail::pipeline_maker::make(group, state, true, true);
}

//! producer_count of the group's threads produce, and the others consume:
//! which ones, the calls they make say
template <typename Group, thread_scope Scope, ::std::uint8_t Stages>
pipeline<Scope> make_pipeline(const Group &group, pipeline_shared_state<Scope, Stages> *state,
                              ::std::size_t producer_count)
{
    return cuda4cpu::detail::pipeline_maker::make(group, state, false, false, producer_count);
}

//! The calling thread produces or consumes, as role says
template <typename Group, thread_scope Scope, ::std::uint8_t Stages>
pipeline<Scope> make_pipeline(const Group &group, pipeline_shared_state<Scope, Stages> *state, pipeline_role role)
{
    const bool producer = role == pipeline_role::producer;
    return cuda4cpu::detail::pipeline_maker::make(group, state, producer, !producer);
}

//! A pipeline of the calling thread: its stages are done when they are
//! committed
template <>
class pipeline<thread_scope_thread> {
public:
    pipeline() = default;
    pipeline(pipeline &&) = default;
    pipeline(const pipeline &) = delete;
    pipeline &operator=(pipeline &&) = default;
    pipeline &operator=(const pipeline &) = delete;

    void producer_acquire() {}
    void producer_commit() {}
    void consumer_wait() {}
    template <typename Rep, typename Period>
    bool consumer_wait_for(const ::std::chrono::duration<Rep, Period> &)
    {
        return true;
    }
    template <typename Clock, typename Duration>
    bool consumer_wait_until(const ::std::chrono::time_point<Clock, Duration> &)
    {
        return true;
    }
    void consumer_release() {}
    bool quit()
    {
        return true;
    }
};

inline pipeline<thread_scope_thread> make_pipeline()
{
    return {};
}

template <::std::uint8_t Prior>
inline void pipeline_consumer_wait_prior(pipeline<thread_scope_thread> &)
{
}

template <typename Shape, thread_scope Scope>
inline void memcpy_async(void *destination, const void *source, Shape size, pipeline<Scope> &)
{
    ::std::memcpy(destination, source, ::std::size_t(size));
}

//! A copy that the threads of group share, into a shared pipeline's stage
template <typename Group, typename Shape, thread_scope Scope>
inline void memcpy_async(const Group &group, void *destination, const void *source, Shape size, pipeline<Scope> &)
{
    cuda4cpu::detail::group_copy(destination, source, ::std::size_t(size), group.thread_rank());
}

template <typename Shape, thread_scope Scope, typename CompletionFunction>
inline void memcpy_async(void *destination, const void *source, Shape size, barrier<Scope, CompletionFunction> &)
{
    ::std::memcpy(destination, source, ::std::size_t(size));
}

//! A copy that the threads of group share
template <typename Group, typename Shape, thread_scope Scope, typename CompletionFunction>
inline void memcpy_async(const Group &group, void *destination, const void *source, Shape size,
                         barrier<Scope, CompletionFunction> &)
{
    cuda4cpu::detail::group_copy(destination, source, ::std::size_t(size), group.thread_rank());
}

}
