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

// hipBLAS and rocBLAS, through their own names: each takes its own
// enumerations (AMD's values, 111 and up) and complex types, and gives its
// own statuses. The routines are cuBLAS's (tests/cublas.cu), so a few of each
// kind, against references, are enough: level 1 to 3, packed and band
// storage, a hermitian update, batches, the device's pointers, copies, and
// the errors.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char *what, int line)
{
    if (!ok) {
        ++failures;
        std::printf("line %d: %s\n", line, what);
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define HIPBLAS(call) check((call) == HIPBLAS_STATUS_SUCCESS, #call, __LINE__)
#define ROCBLAS(call) check((call) == rocblas_status_success, #call, __LINE__)

bool near(double a, double b)
{
    return std::abs(a - b) <= 1e-4 * (1 + std::abs(b));
}

// A = [1 2 3; 4 5 6] (2 x 3, column-major), and its products
const float A[6] = {1, 4, 2, 5, 3, 6};

void hipblas()
{
    hipblasHandle_t h;
    HIPBLAS(hipblasCreate(&h));

    // gemm with transposes: C = A^T A (3 x 3)
    float C[9] = {};
    const float one = 1, zero = 0;
    HIPBLAS(hipblasSgemm(h, HIPBLAS_OP_T, HIPBLAS_OP_N, 3, 3, 2, &one, A, 2, A, 2, &zero, C, 3));
    const float AtA[9] = {17, 22, 27, 22, 29, 36, 27, 36, 45};
    CHECK(std::memcmp(C, AtA, sizeof(C)) == 0);

    // A batch of two, strided
    float As[12], Cs[18] = {};
    std::memcpy(As, A, sizeof(A));
    for (int i = 0; i < 6; ++i)
        As[6 + i] = 2 * A[i];
    HIPBLAS(hipblasSgemmStridedBatched(h, HIPBLAS_OP_T, HIPBLAS_OP_N, 3, 3, 2, &one, As, 2, 6, As, 2, 6, &zero, Cs, 3,
                                       9, 2));
    CHECK(std::memcmp(Cs, AtA, sizeof(AtA)) == 0 && Cs[9 + 4] == 4 * 29);

    // scal and her2, with hipComplex
    float x[4] = {1, 2, 3, 4};
    const float three = 3;
    HIPBLAS(hipblasSscal(h, 4, &three, x, 1));
    CHECK(x[0] == 3 && x[3] == 12);
    hipComplex a[4] = {}, v[2] = {make_hipComplex(1, 1), make_hipComplex(0, 2)};
    const hipComplex alpha = make_hipComplex(1, 0);
    HIPBLAS(hipblasCher2(h, HIPBLAS_FILL_MODE_LOWER, 2, &alpha, v, 1, v, 1, a, 2));
    // A += 2 v v^H: (0,0) = 2|v0|^2 = 4, (1,0) = 2 v1 conj(v0) = 2 (0+2i)(1-i) = 4 + 4i, (1,1) = 8
    CHECK(near(a[0].x, 4) && near(a[0].y, 0) && near(a[1].x, 4) && near(a[1].y, 4) && near(a[3].x, 8));
    CHECK(a[2].x == 0 && a[2].y == 0); // the upper triangle isn't touched

    // Copies of vectors and matrices
    float h1[4] = {1, 2, 3, 4}, d1[8] = {};
    HIPBLAS(hipblasSetVector(4, sizeof(float), h1, 1, d1, 2));
    CHECK(d1[0] == 1 && d1[6] == 4 && d1[1] == 0);
    float m[6] = {};
    HIPBLAS(hipblasSetMatrix(2, 3, sizeof(float), A, 2, m, 2));
    CHECK(std::memcmp(m, A, sizeof(A)) == 0);
    float back[6] = {};
    HIPBLAS(hipblasGetMatrix(2, 3, sizeof(float), m, 2, back, 2));
    CHECK(std::memcmp(back, A, sizeof(A)) == 0);

    // The device's pointers
    HIPBLAS(hipblasSetPointerMode(h, HIPBLAS_POINTER_MODE_DEVICE));
    hipblasPointerMode_t mode;
    HIPBLAS(hipblasGetPointerMode(h, &mode));
    CHECK(mode == HIPBLAS_POINTER_MODE_DEVICE);
    float *d_result;
    CHECK(hipMalloc(&d_result, sizeof(float)) == hipSuccess);
    HIPBLAS(hipblasSdot(h, 4, x, 1, x, 1, d_result));
    CHECK(*d_result == 9 + 36 + 81 + 144);
    CHECK(hipFree(d_result) == hipSuccess);

    // Errors and their names
    CHECK(hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, 2, 2, 2, &one, A, 1, A, 2, &zero, C, 2) ==
          HIPBLAS_STATUS_INVALID_VALUE);
    CHECK(hipblasSgemm(h, hipblasOperation_t(0), HIPBLAS_OP_N, 2, 2, 2, &one, A, 2, A, 2, &zero, C, 2) ==
          HIPBLAS_STATUS_INVALID_VALUE);
    CHECK(hipblasSscal(nullptr, 4, &three, x, 1) == HIPBLAS_STATUS_NOT_INITIALIZED);
    CHECK(std::strcmp(hipblasStatusToString(HIPBLAS_STATUS_INVALID_VALUE), "HIPBLAS_STATUS_INVALID_VALUE") == 0);
    static_assert(hipblasVersionMajor >= 3);
    HIPBLAS(hipblasDestroy(h));
}

void rocblas()
{
    rocblas_handle h;
    ROCBLAS(rocblas_create_handle(&h));
    hipStream_t stream;
    CHECK(hipStreamCreate(&stream) == hipSuccess);
    ROCBLAS(rocblas_set_stream(h, stream));
    hipStream_t got;
    ROCBLAS(rocblas_get_stream(h, &got));
    CHECK(got == stream);

    // Level 1
    float x[4] = {1, -5, 3, 2}, y[4] = {1, 1, 1, 1};
    const float two = 2;
    ROCBLAS(rocblas_saxpy(h, 4, &two, x, 1, y, 1));
    CHECK(y[1] == -9 && y[3] == 5);
    float dot = 0, norm = 0;
    ROCBLAS(rocblas_sdot(h, 4, x, 1, y, 1, &dot));
    CHECK(dot == 3 + 45 + 21 + 10);
    ROCBLAS(rocblas_snrm2(h, 2, x + 2, 1, &norm));
    CHECK(near(norm, std::sqrt(13.0)));
    rocblas_int index = 0;
    ROCBLAS(rocblas_isamax(h, 4, x, 1, &index));
    CHECK(index == 2);

    // Level 2: y = A x, and a packed symmetric product
    const float v3[3] = {1, 1, 1};
    float r[2] = {};
    const float one = 1, zero = 0;
    ROCBLAS(rocblas_sgemv(h, rocblas_operation_none, 2, 3, &one, A, 2, v3, 1, &zero, r, 1));
    CHECK(r[0] == 6 && r[1] == 15);
    const double ap[3] = {2, 1, 3}; // upper: [2 1; 1 3]
    const double xv[2] = {1, 2};
    double yv[2] = {};
    const double dzero = 0, done = 1;
    ROCBLAS(rocblas_dspmv(h, rocblas_fill_upper, 2, &done, ap, xv, 1, &dzero, yv, 1));
    CHECK(yv[0] == 4 && yv[1] == 7);

    // her with rocblas_float_complex, as rocm-examples' her: A += alpha x x^H
    rocblas_float_complex a[4] = {}, xc[2] = {{1, 2}, {3, 0}};
    const float alpha = 2;
    ROCBLAS(rocblas_cher(h, rocblas_fill_upper, 2, &alpha, xc, 1, a, 2));
    // (0,0) = 2 * 5, (0,1) = 2 x0 conj(x1) = (6, 12), (1,1) = 18
    CHECK(a[0] == rocblas_float_complex(10, 0) && a[2] == rocblas_float_complex(6, 12) && a[3].real() == 18);
    CHECK(a[1] == rocblas_float_complex(0, 0)); // the lower triangle isn't touched

    // Level 3: gemm, and a triangular solve
    float C[9] = {};
    ROCBLAS(rocblas_sgemm(h, rocblas_operation_transpose, rocblas_operation_none, 3, 3, 2, &one, A, 2, A, 2, &zero, C,
                          3));
    CHECK(C[4] == 29 && C[8] == 45);
    const double tri[4] = {2, 0, 1, 4}; // upper [2 1; 0 4]
    double b[2] = {4, 8};
    ROCBLAS(rocblas_dtrsm(h, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit,
                          2, 1, &done, tri, 2, b, 2));
    CHECK(b[0] == 1 && b[1] == 2);

    // Copies
    float host[3] = {7, 8, 9}, dev[3] = {};
    ROCBLAS(rocblas_set_vector(3, sizeof(float), host, 1, dev, 1));
    CHECK(dev[2] == 9);

    // Errors: sizes, enumerations, the handle
    CHECK(rocblas_sgemv(h, rocblas_operation_none, 2, 3, &one, A, 1, v3, 1, &zero, r, 1) ==
          rocblas_status_invalid_size);
    CHECK(rocblas_sgemv(h, rocblas_operation(5), 2, 3, &one, A, 2, v3, 1, &zero, r, 1) ==
          rocblas_status_invalid_value);
    CHECK(rocblas_saxpy(nullptr, 4, &two, x, 1, y, 1) == rocblas_status_invalid_handle);
    CHECK(std::strcmp(rocblas_status_to_string(rocblas_status_invalid_size), "rocblas_status_invalid_size") == 0);
    std::size_t size = 0;
    ROCBLAS(rocblas_get_version_string_size(&size));
    std::vector<char> version(size);
    ROCBLAS(rocblas_get_version_string(version.data(), size));
    CHECK(std::strncmp(version.data(), "5.", 2) == 0);

    ROCBLAS(rocblas_destroy_handle(h));
    CHECK(hipStreamDestroy(stream) == hipSuccess);
}

} // namespace

int main()
{
    hipblas();
    rocblas();
    if (failures != 0) {
        std::printf("hip_blas: %d failures\n", failures);
        return 1;
    }
    std::printf("hip_blas: all passed\n");
    return 0;
}
