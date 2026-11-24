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

// NVIDIA's Thrust (CCCL) on cuda4cpu, on its OpenMP backend: algorithms on
// device_vector, Thrust's CUDA-system names (thrust::cuda::par.on(stream),
// par_nosync, thrust::cuda::vector), device_vector's memory as device memory
// to cuda4cpu (cudaPointerGetAttributes, and copies that
// CUDA4CPU_MEMCPY_DIRECTION=error accepts), universal_vector's as managed
// memory, device_malloc, thrust::system_error from an algorithm's header,
// and a kernel on a device_vector's data. Built with the CCCL that
// CUDA4CPU_CCCL_DIR names.

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/transform.h>
#include <thrust/universal_vector.h>

#include <cstdio>

static int failures = 0;

#define EXPECT(cond)                                                                 \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond);          \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

__global__ void triple(int *v, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        v[i] *= 3;
}

static cudaMemoryType memory_type(const void *p)
{
    cudaPointerAttributes attributes{};
    return cudaPointerGetAttributes(&attributes, p) == cudaSuccess ? attributes.type : cudaMemoryTypeUnregistered;
}

int main()
{
    const int n = 100000;
    thrust::device_vector<int> v(n);
    thrust::sequence(v.begin(), v.end());
    thrust::sort(v.begin(), v.end(), thrust::greater<int>());
    EXPECT(v[0] == n - 1 && v[n - 1] == 0);
    EXPECT(thrust::reduce(v.begin(), v.end(), 0ll, thrust::plus<long long>()) == (long long)n * (n - 1) / 2);

    // The CUDA system's policies, on OpenMP
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    thrust::sort(thrust::cuda::par.on(stream), v.begin(), v.end());
    EXPECT(v[0] == 0 && v[n - 1] == n - 1);
    thrust::device_vector<int> scanned(n);
    thrust::inclusive_scan(thrust::cuda::par_nosync, v.begin(), v.end(), scanned.begin());
    EXPECT(scanned[3] == 0 + 1 + 2 + 3);
    thrust::cuda::vector<float> f(4, 1.5f);
    thrust::transform(thrust::cuda::par, f.begin(), f.end(), f.begin(), thrust::negate<float>());
    EXPECT(f[2] == -1.5f);
    cudaStreamDestroy(stream);

    // A device_vector's memory is device memory: copies into it from the
    // host are host-to-device ones, and kernels take it
    int *raw = thrust::raw_pointer_cast(v.data());
    EXPECT(memory_type(raw) == cudaMemoryTypeDevice);
    thrust::host_vector<int> host(n, 7);
    EXPECT(cudaMemcpy(raw, host.data(), n * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    triple<<<(n + 255) / 256, 256>>>(raw, n);
    EXPECT(cudaDeviceSynchronize() == cudaSuccess);
    EXPECT(v[12345] == 21);
    thrust::universal_vector<int> u(10, 2);
    EXPECT(memory_type(thrust::raw_pointer_cast(u.data())) == cudaMemoryTypeManaged);
    thrust::device_ptr<double> d = thrust::device_malloc<double>(16);
    EXPECT(memory_type(thrust::raw_pointer_cast(d)) == cudaMemoryTypeDevice);
    thrust::device_free(d);

    // thrust::system_error, which the CUDA system's headers bring along
    try {
        throw thrust::system_error(1, thrust::system::cuda_category(), "test");
    } catch (const thrust::system_error &e) {
        EXPECT(e.code().value() == 1);
    }

    if (failures == 0)
        std::printf("thrust: all passed\n");
    return failures == 0 ? 0 : 1;
}
