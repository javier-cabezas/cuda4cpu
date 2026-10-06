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


//
// The CUDA syntax that cuda4cpu-rewrite handles, checked at run time: every
// form of kernel launch, extern __shared__ in functions, at namespace scope
// and in a local header, and CUDA syntax in comments and strings, which must
// stay as it is.
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>

#include "features.cuh"

static int errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

// A launch in a comment stays as it is: fill<<<1, 1>>>(nullptr, 0);
static const char *launch_text = "fill<<<1, 1>>>(nullptr, 0)";
static const char *raw_text = R"(fill<<<2, 2>>>(nullptr, 0))";
static const int million = 1'000'000;

__global__ void fill(int *out, int value)
{
    out[threadIdx.x] = value;
}

// Template arguments deduced from the launch arguments, as in CUDA
template <typename T>
__global__ void scale(T *data, T factor)
{
    data[threadIdx.x] *= factor;
}

// Default arguments apply
__global__ void with_default(int *out, int value = 42)
{
    out[threadIdx.x] = value;
}

// Template arguments with commas and nested templates
template <typename Pair, int N>
__global__ void pair_kernel(int *out)
{
    out[threadIdx.x] = int(sizeof(Pair)) + N;
}

namespace outer {
namespace inner {

__global__ void qualified(int *out)
{
    out[threadIdx.x] = 7;
}

}
}

// Dynamic shared memory declared at namespace scope and used by two kernels
extern __shared__ int ns_shared[];

__global__ void reverse_ns(int *data)
{
    ns_shared[threadIdx.x] = data[threadIdx.x];
    __syncthreads();
    data[threadIdx.x] = ns_shared[blockDim.x - 1 - threadIdx.x];
}

__global__ void sum_ns(int *data, int *sum)
{
    int *s = ns_shared;
    s[threadIdx.x] = data[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (unsigned i = 0; i < blockDim.x; ++i)
            total += s[i];
        *sum = total;
    }
}

// Aligned raw bytes, reinterpreted, as CUDA code often does
__global__ void aligned_bytes(float *out)
{
    extern __shared__ __align__(16) unsigned char bytes[];
    float *f = reinterpret_cast<float *>(bytes);
    f[threadIdx.x] = float(threadIdx.x) * 0.5f;
    __syncthreads();
    out[threadIdx.x] = f[blockDim.x - 1 - threadIdx.x];
}

#define LAUNCH_FILL(out, v) fill<<<1, 4>>>(out, v)

int main()
{
    int out[32] = {};

    fill<<<1, 4>>>(out, 1);
    EXPECT(out[0] == 1 && out[3] == 1);

    float f[4] = {1, 2, 3, 4};
    scale<<<1, 4>>>(f, 2.0f);
    EXPECT(f[3] == 8.0f);

    with_default<<<1, 2>>>(out);
    EXPECT(out[1] == 42);

    pair_kernel<std::pair<int, std::map<int, int>>, 3><<<dim3(1), dim3(2)>>>(out);
    EXPECT(out[1] == int(sizeof(std::pair<int, std::map<int, int>>)) + 3);

    outer::inner::qualified<<<1, 2>>>(out);
    EXPECT(out[0] == 7);

    void (*fp)(int *, int) = fill;
    fp<<<1, 4>>>(out, 9);
    EXPECT(out[2] == 9);

    // A launch over several lines, with a shift in the configuration
    int n = 8;
    fill<<<n >> 3,
           n>>>(out,
                5);
    EXPECT(out[7] == 5);

    LAUNCH_FILL(out, 6);
    EXPECT(out[3] == 6);

    if (n > 100) fill<<<1, 1>>>(out, 0); else fill<<<1, 1>>>(out, 11);
    EXPECT(out[0] == 11);

    int data[64];
    for (int i = 0; i < 64; ++i)
        data[i] = i;
    reverse_ns<<<1, 64, 64 * sizeof(int)>>>(data);
    EXPECT(data[0] == 63 && data[63] == 0);
    int sum = 0;
    sum_ns<<<1, 64, 64 * sizeof(int)>>>(data, &sum);
    EXPECT(sum == 63 * 64 / 2);

    float back[16];
    aligned_bytes<<<1, 16, 16 * sizeof(float)>>>(back);
    EXPECT(back[0] == 7.5f && back[15] == 0.0f);

    double values[32];
    for (int i = 0; i < 32; ++i)
        values[i] = i;
    EXPECT(header_sum(values, 32) == 496.0);

    // cudaLaunchKernel with an array of pointers to the arguments
    int value = 3;
    int *ptr = out;
    void *args[] = {&ptr, &value};
    EXPECT(cudaLaunchKernel(fill, dim3(1), dim3(4), args) == cudaSuccess);
    EXPECT(out[3] == 3);
    EXPECT(cudaLaunchKernel(fill, dim3(1), dim3(2048), args) == cudaErrorInvalidConfiguration);
    cudaGetLastError();

    // Strings keep their CUDA syntax, and diagnostics refer to this file
    EXPECT(std::strcmp(launch_text, "fill<<<1, 1>>>(nullptr, 0)") == 0);
    EXPECT(std::strcmp(raw_text, "fill<<<2, 2>>>(nullptr, 0)") == 0);
    EXPECT(million == 1000000);
    EXPECT(std::strstr(__FILE__, "features.cu") != nullptr && std::strstr(__FILE__, ".cpp") == nullptr);
    const int line = __LINE__;
    EXPECT(line == 192);   // line numbers match the original file

    if (errors == 0)
        std::printf("rewrite features: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
