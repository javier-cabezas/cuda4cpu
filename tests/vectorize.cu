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

// Kernels without barriers, run with and without CUDA4CPU_VECTORIZE_THREADS:
// block sizes that aren't multiples of the vector width, 2D and 3D blocks,
// bounds checks, early returns, loops in each thread, and arguments too large
// to copy per block. Checked against a host reference. With the macro and
// AVX2, Clang 22 vectorizes the loops over the threads of saxpy and stencil.

#include <cstdio>
#include <vector>

static int failures = 0;

static void check(bool ok, const char *what, int a, int b)
{
    if (!ok && ++failures <= 10)
        std::printf("FAILED: %s (%d, %d)\n", what, a, b);
}

__global__ void saxpy(float a, const float *x, float *y, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

// Every thread writes its built-in variables
__global__ void ids(int *out, int w, int h, int d)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < w && y < h && z < d)
        out[(z * h + y) * w + x] = threadIdx.x + 100 * threadIdx.y + 10000 * threadIdx.z +
                                   1000000 * (blockIdx.x + blockIdx.y + blockIdx.z);
}

__global__ void stencil(float *out, const float *in, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && y > 0 && x < w - 1 && y < h - 1)
        out[y * w + x] = in[y * w + x] + in[y * w + x - 1] + in[y * w + x + 1] + in[(y - 1) * w + x] +
                         in[(y + 1) * w + x];
}

// Odd threads return early; a loop whose trip count depends on the thread
__global__ void uneven(int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || (i & 1))
        return;
    int acc = 0;
    for (int k = 0; k < i % 7; ++k)
        acc += k * i;
    out[i] = acc;
}

struct big_args {
    float coeff[96];
};

// Arguments larger than run_direct_impl copies
__global__ void with_big_args(big_args args, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = args.coeff[i % 96] * 2.f;
}

int main()
{
    for (unsigned threads : {1u, 7u, 33u, 64u, 100u, 256u, 1000u}) {
        const unsigned n = 5003;
        std::vector<float> x(n), y(n), ref(n);
        for (unsigned i = 0; i < n; ++i) {
            x[i] = float(i % 13);
            y[i] = ref[i] = float(i % 5);
            ref[i] = 3.f * x[i] + ref[i];
        }
        saxpy<<<(n + threads - 1) / threads, threads>>>(3.f, x.data(), y.data(), n);
        for (unsigned i = 0; i < n; ++i)
            check(y[i] == ref[i], "saxpy", threads, i);
    }

    for (dim3 block : {dim3(8, 4, 2), dim3(3, 5, 7), dim3(32, 1, 1), dim3(17, 3, 1), dim3(1, 1, 64)}) {
        const int w = 45, h = 23, d = 9;
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (d + block.z - 1) / block.z);
        std::vector<int> out(w * h * d, -1);
        ids<<<grid, block>>>(out.data(), w, h, d);
        for (int z = 0; z < d; ++z)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    int expected = x % block.x + 100 * (y % block.y) + 10000 * (z % block.z) +
                                   1000000 * (x / block.x + y / block.y + z / block.z);
                    check(out[(z * h + y) * w + x] == expected, "ids", block.x * 100 + block.y, x + 1000 * y);
                }
    }

    for (dim3 block : {dim3(64, 4), dim3(5, 3), dim3(16, 16)}) {
        const int w = 131, h = 77;
        std::vector<float> in(w * h), out(w * h, -1.f);
        for (int i = 0; i < w * h; ++i)
            in[i] = float(i % 17);
        stencil<<<dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y), block>>>(out.data(), in.data(), w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float expected = -1.f;
                if (x > 0 && y > 0 && x < w - 1 && y < h - 1)
                    expected = in[y * w + x] + in[y * w + x - 1] + in[y * w + x + 1] + in[(y - 1) * w + x] +
                               in[(y + 1) * w + x];
                check(out[y * w + x] == expected, "stencil", x, y);
            }
    }

    {
        const int n = 3001;
        std::vector<int> out(n, -1);
        uneven<<<(n + 127) / 128, 128>>>(out.data(), n);
        for (int i = 0; i < n; ++i) {
            int expected = -1;
            if (!(i & 1)) {
                expected = 0;
                for (int k = 0; k < i % 7; ++k)
                    expected += k * i;
            }
            check(out[i] == expected, "uneven", i, out[i]);
        }
    }

    {
        const int n = 1000;
        big_args args;
        for (int k = 0; k < 96; ++k)
            args.coeff[k] = float(k);
        std::vector<float> out(n);
        with_big_args<<<(n + 63) / 64, 64>>>(args, out.data(), n);
        for (int i = 0; i < n; ++i)
            check(out[i] == float(i % 96) * 2.f, "big args", i, int(out[i]));
    }

    if (failures)
        std::printf("%d failures\n", failures);
    else
        std::printf("PASSED\n");
    return failures != 0;
}
