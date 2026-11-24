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

// cuda4cpu's overlay of NVIDIA's Thrust (CCCL) on its OpenMP backend: the
// names of Thrust's CUDA system (thrust::cuda::par and par_nosync, with
// .on(stream), vector, pointer, malloc, free, ...) on OpenMP's, and what
// the CUDA system's headers bring along (thrust::system_error), for programs
// written for it. With another device system, nothing.
#pragma once

#include <thrust/detail/config.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP && !defined(ROCTHRUST_VERSION)

#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/memory.h>
#include <thrust/system/omp/vector.h>
#include <thrust/system_error.h>

#include <cuda_runtime.h>

THRUST_NAMESPACE_BEGIN

namespace cuda4cpu_cuda_system
{

//! OpenMP's parallel policy, which a stream doesn't change: work runs when
//! it's issued
struct par_t : thrust::system::omp::detail::par_t
{
  constexpr par_t() = default;

  par_t on(cudaStream_t) const
  {
    return *this;
  }
};

inline constexpr par_t par{};
inline constexpr par_t par_nosync{};

using thrust::system::omp::tag;
using thrust::system::omp::execution_policy;
using thrust::system::omp::pointer;
using thrust::system::omp::reference;
using thrust::system::omp::vector;
using thrust::system::omp::universal_vector;
using thrust::system::omp::universal_pointer;
using thrust::system::omp::malloc;
using thrust::system::omp::free;

//! CUDA's error category: the system's, since errors are cudaError_t
inline const thrust::system::error_category& cuda_category()
{
  return thrust::system::system_category();
}

} // namespace cuda4cpu_cuda_system

namespace cuda     = cuda4cpu_cuda_system;
namespace cuda_cub = cuda4cpu_cuda_system;

namespace system
{
namespace cuda = thrust::cuda4cpu_cuda_system;
using thrust::cuda4cpu_cuda_system::cuda_category;
} // namespace system

THRUST_NAMESPACE_END

#endif
