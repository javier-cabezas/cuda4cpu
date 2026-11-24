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

// cuda4cpu's overlay of NVIDIA's Thrust (CCCL) on its OpenMP backend, which
// cuda4cpu builds CUDA programs' Thrust with: the device memory resource
// allocates with cudaMalloc, and the universal one with cudaMallocManaged,
// instead of operator new, so that the memory of a device_vector is device
// memory to cuda4cpu's checks (copies in the wrong direction, leaks, ...).
// The rest is CCCL's, as in its own memory_resource.h. rocThrust's passes
// through.
#pragma once

#include <thrust/detail/config.h>

#if defined(ROCTHRUST_VERSION)
#include_next <thrust/system/omp/memory_resource.h>
#else

#include <new>

#include <thrust/mr/fancy_pointer_resource.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/system/omp/pointer.h>

#include <cuda_runtime.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{

//! Memory from cudaMalloc (or cudaMallocManaged), whose 256-byte alignment
//! covers Thrust's
template <bool Managed>
class cuda4cpu_resource final : public thrust::mr::memory_resource<>
{
public:
  void* do_allocate(std::size_t bytes, std::size_t /* alignment */) override
  {
    void* p = nullptr;
    const cudaError_t status = Managed ? cudaMallocManaged(&p, bytes) : cudaMalloc(&p, bytes);
    if (status != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    return p;
  }

  void do_deallocate(void* p, std::size_t, std::size_t) override
  {
    cudaFree(p);
  }
};

using native_resource = thrust::mr::fancy_pointer_resource<cuda4cpu_resource<false>, thrust::omp::pointer<void>>;
using universal_native_resource =
  thrust::mr::fancy_pointer_resource<cuda4cpu_resource<true>, thrust::omp::universal_pointer<void>>;

} // namespace detail

using memory_resource                       = detail::native_resource;
using universal_memory_resource             = detail::universal_native_resource;
using universal_host_pinned_memory_resource = universal_memory_resource;

} // namespace omp
} // namespace system
THRUST_NAMESPACE_END

#endif
