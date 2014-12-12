/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
 */

#undef _FORTIFY_SOURCE

#include <iostream>
#include <chrono>
#include <cstring>

#include "cuda4cpu.hpp"

using namespace cuda4cpu;

template <unsigned Halo>
__global__
void
stencil2D_cuda(float *B,
               const float *A,
               unsigned A_size_0)
{
    __shared__ float tile[4 + 2 * Halo][32 + 2 * Halo];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int sx = tx + Halo;
    int sy = ty + Halo;

    int i = ty + by * blockDim.y + Halo;
    int j = tx + bx * blockDim.x + Halo;

    unsigned off = i * A_size_0 + j;

    float val = A[off];

    tile[sy][sx] = val;

    if (tx < Halo) {
        tile[sy][tx                    ] = A[off - Halo];
        tile[sy][tx + blockDim.x + Halo] = A[off + blockDim.x];
    }
    if (Halo <= 4) {
        if (ty < Halo) {
            tile[ty                    ][sx] = A[off - Halo * A_size_0];
            tile[ty + blockDim.y + Halo][sx] = A[off + blockDim.y * A_size_0];
        }
    } else {
        for (int i2 = 0; i2 < Halo; i2 += 4) {
            tile[ty                     + i2][sx] = A[off - Halo * A_size_0 + i2];
            tile[ty + blockDim.y + Halo + i2][sx] = A[off + blockDim.y * A_size_0 + i2];
        }
    }

    __syncthreads();

    for (int k = 1; k <= Halo; ++k) {
        val += 3.f * (tile[sy][sx - k] + tile[sy][sx + k]) +
               2.f * (tile[sy - k][sx] + tile[sy + k][sx]);
    }

    B[off] = val;
}

static const unsigned Halo = 4;

void stencil2D_host(float *B, const float *A, unsigned DimX, unsigned DimY)
{
    unsigned TotalDimX = DimX + 2 * Halo;

    #pragma omp parallel for
    for (unsigned i = Halo; i < DimY + Halo; ++i) {
        for (unsigned j = Halo; j < DimX + Halo; ++j) {
            float val = A[i * TotalDimX + j];
            for (unsigned k = 1; k <= Halo; ++k) {
                val += 3.f * (A[i *TotalDimX + j - k] + A[i * TotalDimX + j + k]) +
                       2.f * (A[(i - k) * TotalDimX + j] + A[(i + k) * TotalDimX + j]);
            }
            B[i * TotalDimX + j] = val;
        }
    }
}

static const unsigned DimX = 2048;
static const unsigned DimY = 2048;

static const unsigned TotalDimX = DimX + 2 * Halo;
static const unsigned TotalDimY = DimY + 2 * Halo;

using array_type = float[TotalDimY][TotalDimX];
using array_type_ptr = array_type *;
using array_type_ref = array_type &;

int main()
{
    array_type_ref A = *(array_type_ptr)new float[TotalDimY * TotalDimX * sizeof(float)];
    array_type_ref B = *(array_type_ptr)new float[TotalDimY * TotalDimX * sizeof(float)];
    array_type_ref B_gold = *(array_type_ptr)new float[TotalDimY * TotalDimX * sizeof(float)];

    std::chrono::time_point<std::chrono::system_clock> start, end;

    for (unsigned i = 0; i < TotalDimY; ++i) {
        for (unsigned j = 0; j < TotalDimX; ++j) {
            A[i][j] = float(i + j);

            B[i][j]      = 0.f;
            B_gold[i][j] = 0.f;
        }
    }

    std::chrono::duration<double> elapsed;

    start = std::chrono::system_clock::now();
    auto test = launch(stencil2D_cuda<Halo>, dim3(DimX/32, DimY/4, 1), dim3(32, 4));
    test.call((float *)B, (float *)A, TotalDimX);
    end = std::chrono::system_clock::now();

    elapsed = end - start;
    std::cout << "CUDA-style: " << elapsed.count() << "s\n";

    start = std::chrono::system_clock::now();
    stencil2D_host((float *) &B_gold, (float *)&A, DimX, DimY);
    end = std::chrono::system_clock::now();

    elapsed = end - start;
    std::cout << "CPU-style: " << elapsed.count() << "s\n";

    for (unsigned i = Halo; i < DimY + Halo; ++i) {
        for (unsigned j = Halo; j < DimX + Halo; ++j) {
            if (B[i][j] != B_gold[i][j]) {
                std::cout << i << "," << j << "\n";
                std::cout << B[i][j] << " vs " << B_gold[i][j] << "\n";
            }
        }
    }

    delete [](float *)A;
    delete [](float *)B;
    delete [](float *)B_gold;

    return 0;
}
