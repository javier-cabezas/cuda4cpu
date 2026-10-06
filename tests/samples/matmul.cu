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
// Tiled single-precision matrix multiplication, as in the CUDA samples'
// "matrixMul": each block computes a BLOCK_SIZE x BLOCK_SIZE tile of C,
// staging tiles of A and B through static shared memory.
//

#include <random>
#include <vector>

#include <cuda_runtime.h>
#include "sample.hpp"

template <int BLOCK_SIZE>
__global__ void matrixMulCUDA(float *C, const float *A, const float *B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

int main()
{
    constexpr int block_size = 16;
    const int hA = 512, wA = 512, wB = 512;

    std::vector<float> A(size_t(hA) * wA), B(size_t(wA) * wB);
    std::vector<float> C(size_t(hA) * wB), C_gold(size_t(hA) * wB);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &v : A) v = dist(gen);
    for (auto &v : B) v = dist(gen);

    double ms = sample::time_ms([&] {
        matrixMulCUDA<block_size><<<dim3(wB / block_size, hA / block_size),
                                    dim3(block_size, block_size)>>>(C.data(), A.data(), B.data(), wA, wB);
    });

    double host_ms = sample::time_ms([&] {
        #pragma omp parallel for
        for (int i = 0; i < hA; ++i) {
            for (int j = 0; j < wB; ++j) {
                float sum = 0;
                for (int k = 0; k < wA; ++k)
                    sum += A[size_t(i) * wA + k] * B[size_t(k) * wB + j];
                C_gold[size_t(i) * wB + j] = sum;
            }
        }
    });

    sample::report("matrixMul", ms, host_ms);
    return sample::finish(sample::count_mismatches(C.data(), C_gold.data(), C.size(), 1e-4));
}
