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
stencil3D(float *B,
          const float *A,
          unsigned cols,
          unsigned rows,
          unsigned planes)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int sx = tx + Halo;
    int sy = ty + Halo;

    int k = tx + bx * blockDim.x + Halo;
    int j = ty + by * blockDim.y + Halo;

    register float val;
    register float pre[Halo];
    register float post[Halo];

    __shared__ float tile[4 + 2 * Halo][32 + 2 * Halo];

    unsigned planeSize = cols * rows;

    int idx = j * cols + k;

    for (int i = 1; i < Halo; ++i) {
        pre[i]  = A[idx + (i - 1) * planeSize];
    }

    val = A[idx + (Halo - 1) * planeSize];

    for (int i = 0; i < Halo; ++i) {
        post[i] = A[idx + (i + Halo) * planeSize];
    }

    idx += Halo * planeSize;

    for (int i = Halo; i < planes - Halo; ++i) {
        for (int i2 = 0; i2 < Halo - 1; ++i2) {
            pre[i2] = pre[i2 + 1];
        }

        pre[Halo - 1] = val;
        val           = post[0];

        for (int i2 = 0; i2 < Halo - 1; ++i2) {
            post[i2] = post[i2 + 1];
        }

        post[Halo - 1] = A[idx + Halo * planeSize];

        __syncthreads();

        if (tx < Halo) {
            tile[sy][tx                    ] = A[idx - Halo];
            tile[sy][tx + blockDim.x + Halo] = A[idx + blockDim.x];
        }
        if (Halo <= 4) {
            if (ty < Halo) {
                tile[ty                    ][sx] = A[idx - cols * Halo];
                tile[ty + blockDim.y + Halo][sx] = A[idx + cols * blockDim.y];
            }
        } else {
            for (unsigned j2 = 0; j2 < Halo; j2 += 4) {
                tile[ty                     + j2][sx] = A[idx - cols * Halo + j2];
                tile[ty + blockDim.y + Halo + j2][sx] = A[idx + cols * blockDim.y + j2];
            }
        }

        tile[sy][sx] = val;

        __syncthreads();

        float c = val;
        for (int s = 1; s <= Halo; ++s) {
            c += 3.f * (tile[sy][sx - s] + tile[sy][sx + s]) +
                 2.f * (tile[sy - s][sx] + tile[sy + s][sx]) +
                 1.f * (pre[Halo - s] + post[s - 1]);
        }

        B[idx] = c;

        idx += planeSize;
    }
}

static const unsigned Halo = 4;

void stencil3D_host(float *B, const float *A, unsigned DimX, unsigned DimY, unsigned DimZ)
{
    const unsigned TotalDimX = DimX + 2 * Halo;
    const unsigned TotalDimY = DimY + 2 * Halo;
    const unsigned TotalDimXY = TotalDimX * TotalDimY;

    #pragma omp parallel for
    for (unsigned i = Halo; i < DimZ + Halo; ++i) {
        for (unsigned j = Halo; j < DimY + Halo; ++j) {
            for (unsigned k = Halo; k < DimX + Halo; ++k) {
                float val = A[i * TotalDimXY + j * TotalDimX + k];
                for (unsigned l = 1; l <= Halo; ++l) {
                    val += 3.f * (A[i * TotalDimXY + j * TotalDimX + (k - l)] + A[i * TotalDimXY + j * TotalDimX + (k + l)]) +
                           2.f * (A[i * TotalDimXY + (j - l) * TotalDimX + k] + A[i * TotalDimXY + (j + l) * TotalDimX + k]) +
                           1.f * (A[(i - l) * TotalDimXY + j * TotalDimX + k] + A[(i + l) * TotalDimXY + j * TotalDimX + k]);
                }
                B[i * TotalDimXY + j * TotalDimX + k] = val;
            }
        }
    }
}

static const unsigned DimX = 256;
static const unsigned DimY = 256;
static const unsigned DimZ = 256;

static const unsigned TotalDimX = DimX + 2 * Halo;
static const unsigned TotalDimY = DimY + 2 * Halo;
static const unsigned TotalDimZ = DimZ + 2 * Halo;

static const unsigned TotalDimXYZ = TotalDimX * TotalDimY * TotalDimZ;

using array_type = float[TotalDimZ][TotalDimY][TotalDimX];
using array_type_ptr = array_type *;
using array_type_ref = array_type &;

int main()
{
    array_type_ref A      = *(array_type_ptr)new float[TotalDimXYZ * sizeof(float)];
    array_type_ref B      = *(array_type_ptr)new float[TotalDimXYZ * sizeof(float)];
    array_type_ref B_gold = *(array_type_ptr)new float[TotalDimXYZ * sizeof(float)];

    std::chrono::time_point<std::chrono::system_clock> start, end;

    for (unsigned i = 0; i < TotalDimZ; ++i) {
        for (unsigned j = 0; j < TotalDimY; ++j) {
            for (unsigned k = 0; k < TotalDimX; ++k) {
                A[i][j][k] = float(i + j + k);

                B[i][j][k]      = 0.f;
                B_gold[i][j][k] = 0.f;
            }
        }
    }

    std::chrono::duration<double> elapsed;

    start = std::chrono::system_clock::now();
    auto test = launch(stencil3D<Halo>, dim3(DimX/32, DimY/4, 1), dim3(32, 4));
    test.call((float *)B, (float *)A, TotalDimX, TotalDimY, TotalDimZ);
    end = std::chrono::system_clock::now();

    elapsed = end - start;
    std::cout << "CUDA-style: " << elapsed.count() << "s\n";

    start = std::chrono::system_clock::now();
    stencil3D_host((float *) &B_gold, (float *)&A, DimX, DimY, DimZ);
    end = std::chrono::system_clock::now();

    elapsed = end - start;
    std::cout << "CPU-style: " << elapsed.count() << "s\n";

    for (unsigned i = Halo; i < DimZ + Halo; ++i) {
        for (unsigned j = Halo; j < DimY + Halo; ++j) {
            for (unsigned k = Halo; k < DimX + Halo; ++k) {
                if (B[i][j][k] != B_gold[i][j][k]) {
                    std::cout << i << "," << j << "," << k << "\n";
                }
            }
        }
    }

    delete [](float *)A;
    delete [](float *)B;
    delete [](float *)B_gold;

    return 0;
}
