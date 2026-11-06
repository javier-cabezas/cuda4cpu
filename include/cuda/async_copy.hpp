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
// and those that read. Pipelines that a group shares
// (pipeline<thread_scope_block>, make_pipeline(group, shared_state)) aren't
// implemented.
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

//! The bytes of [0, size) that the thread of rank rank copies, when count
//! threads share a copy
inline void copy_share(void *dst, const void *src, ::std::size_t size, unsigned long long rank,
                       unsigned long long count)
{
    const ::std::size_t chunk = (size + count - 1) / count, first = ::std::size_t(rank) * chunk;
    if (first < size)
        ::std::memcpy(static_cast<char *>(dst) + first, static_cast<const char *>(src) + first,
                      ::std::min(chunk, size - first));
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

    friend void init(barrier *bar, ::std::ptrdiff_t expected, CompletionFunction completion = CompletionFunction())
    {
        ::new (bar) barrier(expected, completion);
    }

    [[nodiscard]] arrival_token arrive(::std::ptrdiff_t update = 1)
    {
        const arrival_token phase = phase_.load();
        if (pending_.fetch_sub(update) == update) {
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

template <thread_scope Scope>
class pipeline;

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

template <typename Shape>
inline void memcpy_async(void *destination, const void *source, Shape size, pipeline<thread_scope_thread> &)
{
    ::std::memcpy(destination, source, ::std::size_t(size));
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
    cuda4cpu::detail::copy_share(destination, source, ::std::size_t(size), group.thread_rank(), group.num_threads());
}

}
