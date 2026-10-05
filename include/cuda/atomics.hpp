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

#include <atomic>
#include <type_traits>

namespace cuda4cpu {

//
// Atomic functions. Like CUDA's, they are relaxed: they don't order other
// memory accesses (use the __threadfence functions for that). Thread blocks run
// in parallel on different cores, so they are real atomic operations.
//
// The value parameter is not deduced, so that calls like atomicAdd(&u, 1) on
// an unsigned int pick the unsigned overload, as in CUDA.
//

namespace detail {

template <typename T>
concept atomic_arith = std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
                       std::is_same_v<T, unsigned long long int>;

template <typename T>
concept atomic_minmax = atomic_arith<T> || std::is_same_v<T, long long int>;

template <typename T>
concept atomic_add = atomic_arith<T> || std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept atomic_exch = atomic_arith<T> || std::is_same_v<T, float>;

template <typename T>
inline std::atomic_ref<T>
atomic(T *address)
{
    return std::atomic_ref<T>(*address);
}

template <typename T, typename F>
inline T
atomic_update(T *address, F &&update)
{
    auto ref = atomic(address);
    T old = ref.load(std::memory_order_relaxed);
    while (!ref.compare_exchange_weak(old, update(old), std::memory_order_relaxed)) {
    }
    return old;
}

}

inline namespace cuda_api {

template <detail::atomic_add T>
inline T atomicAdd(T *address, std::type_identity_t<T> val)
{
    return detail::atomic(address).fetch_add(val, std::memory_order_relaxed);
}

template <typename T> requires std::is_same_v<T, int> || std::is_same_v<T, unsigned int>
inline T atomicSub(T *address, std::type_identity_t<T> val)
{
    return detail::atomic(address).fetch_sub(val, std::memory_order_relaxed);
}

template <detail::atomic_exch T>
inline T atomicExch(T *address, std::type_identity_t<T> val)
{
    return detail::atomic(address).exchange(val, std::memory_order_relaxed);
}

template <detail::atomic_minmax T>
inline T atomicMin(T *address, std::type_identity_t<T> val)
{
    return detail::atomic_update(address, [val](T old) { return val < old ? val : old; });
}

template <detail::atomic_minmax T>
inline T atomicMax(T *address, std::type_identity_t<T> val)
{
    return detail::atomic_update(address, [val](T old) { return val > old ? val : old; });
}

//! Stores ((old >= val) ? 0 : (old + 1)) and returns old
inline unsigned int atomicInc(unsigned int *address, unsigned int val)
{
    return detail::atomic_update(address, [val](unsigned int old) { return old >= val ? 0u : old + 1; });
}

//! Stores (((old == 0) || (old > val)) ? val : (old - 1)) and returns old
inline unsigned int atomicDec(unsigned int *address, unsigned int val)
{
    return detail::atomic_update(address, [val](unsigned int old) {
        return (old == 0 || old > val) ? val : old - 1;
    });
}

//! Stores (old == compare ? val : old) and returns old
template <detail::atomic_arith T>
inline T atomicCAS(T *address, std::type_identity_t<T> compare, std::type_identity_t<T> val)
{
    detail::atomic(address).compare_exchange_strong(compare, val, std::memory_order_relaxed);
    return compare;
}

template <detail::atomic_arith T>
inline T atomicAnd(T *address, std::type_identity_t<T> val)
{
    return detail::atomic(address).fetch_and(val, std::memory_order_relaxed);
}

template <detail::atomic_arith T>
inline T atomicOr(T *address, std::type_identity_t<T> val)
{
    return detail::atomic(address).fetch_or(val, std::memory_order_relaxed);
}

template <detail::atomic_arith T>
inline T atomicXor(T *address, std::type_identity_t<T> val)
{
    return detail::atomic(address).fetch_xor(val, std::memory_order_relaxed);
}

//
// Memory fences. Threads of a block share one OS thread, so a block fence only
// needs to stop the compiler from reordering accesses.
//

inline void __threadfence_block()
{
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

inline void __threadfence()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

inline void __threadfence_system()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

}

}
