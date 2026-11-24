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

// Thrust in a C++ source that includes it before cuda_runtime.h, as host
// code built apart from the kernels does: Thrust still takes the OpenMP
// backend and cuda4cpu's device memory.

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>

#include <cstdio>

int main()
{
    thrust::device_vector<int> v(1000);
    thrust::sequence(thrust::cuda::par, v.begin(), v.end(), 0);
    thrust::sort(v.begin(), v.end(), thrust::greater<int>());
    const int sum = thrust::reduce(v.begin(), v.end());
    cudaPointerAttributes attributes{};
    const bool device = cudaPointerGetAttributes(&attributes, thrust::raw_pointer_cast(v.data())) == cudaSuccess &&
                        attributes.type == cudaMemoryTypeDevice;
    const bool ok = v[0] == 999 && v[999] == 0 && sum == 999 * 1000 / 2 && device;
    std::printf("thrust_host: %s\n", ok ? "all passed" : "FAILED");
    return ok ? 0 : 1;
}
