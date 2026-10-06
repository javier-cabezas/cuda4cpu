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

#include <cstddef>
#include <cstdlib>
#include <cstring>

//
// CUDA keyword overrides
//
#ifdef __global__
#undef __global__
#endif

#ifdef __device__
#undef __device__
#endif

#ifdef __host__
#undef __host__
#endif

#ifdef __shared__
#undef __shared__
#endif

#ifdef __constant__
#undef __constant__
#endif

#define __global__
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(...)
#define __grid_constant__
#define __managed__
#define __align__(n) __attribute__((aligned(n)))
#define __device__
#define __host__
#define __shared__ static thread_local
#define __constant__

// The file and line identify each __syncthreads() for the divergent barrier
// check, and are the same in every copy the compiler inlines
#define __syncthreads() cuda4cpu::thread_block::syncthreads(__FILE__, __LINE__)

#define threadIdx cuda4cpu::thread_block::get_thread()
#define blockIdx  cuda4cpu::thread_block::get_block()

#define blockDim cuda4cpu::thread_block::get_block_dim()
#define gridDim  cuda4cpu::thread_block::get_grid_dim()


