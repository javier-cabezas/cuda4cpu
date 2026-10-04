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
#include <atomic>
#include <cstddef>

#include "types.hpp"

namespace cuda4cpu {

enum cudaLimit {
    cudaLimitStackSize      = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02
};

namespace detail {

//! Smallest stack given to each CUDA thread. CPU code (libc calls, the
//! launcher) needs much more stack than GPU code, so smaller requests are
//! rounded up to this size.
inline constexpr size_t min_stack_size = 64 * 1024;

//! Stack size of each CUDA thread for subsequent launches
inline std::atomic<size_t> stack_size{min_stack_size};

}

static inline
cudaError_t cudaDeviceSynchronize()
{
    return 0;
}

static inline
cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value)
{
    if (limit == cudaLimitStackSize)
        detail::stack_size = std::max(value, detail::min_stack_size);

    return 0;
}

static inline
cudaError_t cudaDeviceGetLimit(size_t *value, cudaLimit limit)
{
    *value = (limit == cudaLimitStackSize) ? detail::stack_size.load() : 0;

    return 0;
}

}
