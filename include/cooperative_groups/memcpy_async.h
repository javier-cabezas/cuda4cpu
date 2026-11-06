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

// Cooperative groups' memcpy_async: a copy that the threads of a group share,
// and wait, which waits for the group's copies and syncs it. On the CPU each
// thread copies its share when it calls memcpy_async.
#pragma once

#include "../cooperative_groups.h"
#include "../cuda/async_copy.hpp"

namespace cooperative_groups {

//! Copies size bytes (a size_t or a cuda::aligned_size_t) from src to dst
template <typename Group, typename T, typename Shape>
inline void memcpy_async(const Group &group, T *dst, const T *src, const Shape &size)
{
    cuda4cpu::detail::copy_share(dst, src, ::std::size_t(size), group.thread_rank(), group.num_threads());
}

//! Copies src_count elements from src to dst, which holds dst_count
template <typename Group, typename T>
inline void memcpy_async(const Group &group, T *dst, ::std::size_t dst_count, const T *src, ::std::size_t src_count)
{
    cuda4cpu::detail::copy_share(dst, src, ::std::min(dst_count, src_count) * sizeof(T), group.thread_rank(),
                                 group.num_threads());
}

template <typename Group>
inline void wait(const Group &group)
{
    group.sync();
}

template <unsigned NumStages, typename Group>
inline void wait_prior(const Group &group)
{
    group.sync();
}

}
