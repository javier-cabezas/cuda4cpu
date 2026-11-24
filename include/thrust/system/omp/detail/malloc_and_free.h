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

// cuda4cpu's overlay of NVIDIA's Thrust (CCCL) on its OpenMP backend: device
// allocations (thrust::malloc, thrust::device_malloc, temporary buffers) go
// through cudaMalloc and cudaFree, so that cuda4cpu's checks know them as
// device memory. They are OpenMP's own overloads, which win over the
// sequential ones they'd use otherwise. rocThrust's pass through.
#pragma once

#include_next <thrust/system/omp/detail/malloc_and_free.h>

#if !defined(ROCTHRUST_VERSION)

#include <cstddef>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/omp/detail/execution_policy.h>

#include <cuda_runtime.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{

template <typename DerivedPolicy>
inline void* malloc(execution_policy<DerivedPolicy>&, std::size_t n)
{
  void* p = nullptr;
  return cudaMalloc(&p, n) == cudaSuccess ? p : nullptr;
}

template <typename DerivedPolicy, typename Pointer>
inline void free(execution_policy<DerivedPolicy>&, Pointer ptr)
{
  cudaFree(thrust::raw_pointer_cast(ptr));
}

} // namespace detail
} // namespace omp
} // namespace system
THRUST_NAMESPACE_END

#endif
