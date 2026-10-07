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


// CUDA's cooperative_groups/scan.h: inclusive_scan() and exclusive_scan()
// over a tile of up to 32 threads or a coalesced group.

#pragma once

#include "reduce.h"

namespace cuda4cpu::detail::cg {

template <typename Group, typename T, typename Op>
inline T inclusive_scan_warp(const Group &group, T value, Op &&op)
{
    const unsigned rank = unsigned(group.thread_rank());
    const unsigned count = unsigned(group.num_threads());
    for (unsigned offset = 1; offset < count; offset *= 2) {
        T other = group.shfl_up(value, offset);
        if (rank >= offset)
            value = op(other, value);
    }
    return value;
}

}

namespace cooperative_groups {

//! Thread i gets op applied to the values of ranks 0 to i
template <typename Group, typename T, typename Op>
inline T inclusive_scan(const Group &group, T value, Op &&op)
{
    return cuda4cpu::detail::cg::inclusive_scan_warp(group, value, op);
}

template <typename Group, typename T>
inline T inclusive_scan(const Group &group, T value)
{
    return inclusive_scan(group, value, plus<T>());
}

//! Thread i gets op applied to the values of ranks 0 to i - 1; rank 0 gets T()
template <typename Group, typename T, typename Op>
inline T exclusive_scan(const Group &group, T value, Op &&op)
{
    T inclusive = cuda4cpu::detail::cg::inclusive_scan_warp(group, value, op);
    T previous  = group.shfl_up(inclusive, 1);
    return group.thread_rank() == 0 ? T() : previous;
}

template <typename Group, typename T>
inline T exclusive_scan(const Group &group, T value)
{
    return exclusive_scan(group, value, plus<T>());
}

}
