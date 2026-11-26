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

// rocBLAS's API on cuda4cpu's BLAS (cublas_api.h's): the same routines,
// with rocBLAS's names, enumerations and complex types. The functions are
// in the library, with C linkage. The 32-bit interface (rocblas_int is
// int32_t), as rocBLAS's default build.
#pragma once

#include <cstddef>
#include <cstdint>

#include "../hip/hip_runtime_api.h"

#define ROCBLAS_VERSION_MAJOR 5
#define ROCBLAS_VERSION_MINOR 2
#define ROCBLAS_VERSION_PATCH 0
#define ROCBLAS_VERSION_TWEAK 0

typedef int32_t rocblas_int;
typedef int64_t rocblas_stride;
typedef float rocblas_float;
typedef double rocblas_double;

//! rocBLAS's complex numbers: two numbers, laid out as std::complex (and
//! hipComplex, cuComplex)
template <typename T>
struct rocblas_complex_num {
    T x{};
    T y{};

    constexpr rocblas_complex_num() = default;
    constexpr rocblas_complex_num(T real, T imag = T()) : x(real), y(imag) {}
    template <typename U>
    constexpr explicit rocblas_complex_num(const rocblas_complex_num<U> &other) : x(T(other.x)), y(T(other.y))
    {
    }

    constexpr T real() const { return x; }
    constexpr T imag() const { return y; }
    constexpr void real(T value) { x = value; }
    constexpr void imag(T value) { y = value; }

    friend constexpr rocblas_complex_num operator+(rocblas_complex_num a, rocblas_complex_num b)
    {
        return {a.x + b.x, a.y + b.y};
    }
    friend constexpr rocblas_complex_num operator-(rocblas_complex_num a, rocblas_complex_num b)
    {
        return {a.x - b.x, a.y - b.y};
    }
    friend constexpr rocblas_complex_num operator-(rocblas_complex_num a) { return {-a.x, -a.y}; }
    friend constexpr rocblas_complex_num operator*(rocblas_complex_num a, rocblas_complex_num b)
    {
        return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    }
    friend constexpr rocblas_complex_num operator/(rocblas_complex_num a, rocblas_complex_num b)
    {
        const T d = b.x * b.x + b.y * b.y;
        return {(a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d};
    }
    constexpr rocblas_complex_num &operator+=(rocblas_complex_num b) { return *this = *this + b; }
    constexpr rocblas_complex_num &operator-=(rocblas_complex_num b) { return *this = *this - b; }
    constexpr rocblas_complex_num &operator*=(rocblas_complex_num b) { return *this = *this * b; }
    constexpr rocblas_complex_num &operator/=(rocblas_complex_num b) { return *this = *this / b; }
    friend constexpr bool operator==(rocblas_complex_num a, rocblas_complex_num b) { return a.x == b.x && a.y == b.y; }
    friend constexpr bool operator!=(rocblas_complex_num a, rocblas_complex_num b) { return !(a == b); }
};

typedef rocblas_complex_num<float> rocblas_float_complex;
typedef rocblas_complex_num<double> rocblas_double_complex;

extern "C" {

struct _rocblas_handle;
typedef struct _rocblas_handle *rocblas_handle;

typedef enum rocblas_status_ {
    rocblas_status_success             = 0,
    rocblas_status_invalid_handle      = 1,
    rocblas_status_not_implemented     = 2,
    rocblas_status_invalid_pointer     = 3,
    rocblas_status_invalid_size        = 4,
    rocblas_status_memory_error        = 5,
    rocblas_status_internal_error      = 6,
    rocblas_status_perf_degraded       = 7,
    rocblas_status_size_query_mismatch = 8,
    rocblas_status_size_increased      = 9,
    rocblas_status_size_unchanged      = 10,
    rocblas_status_invalid_value       = 11,
    rocblas_status_continue            = 12,
    rocblas_status_check_numerics_fail = 13,
    rocblas_status_excluded_from_build = 14,
    rocblas_status_arch_mismatch       = 15
} rocblas_status;

typedef enum rocblas_operation_ {
    rocblas_operation_none                = 111,
    rocblas_operation_transpose           = 112,
    rocblas_operation_conjugate_transpose = 113
} rocblas_operation;
typedef enum rocblas_fill_ { rocblas_fill_upper = 121, rocblas_fill_lower = 122, rocblas_fill_full = 123 } rocblas_fill;
typedef enum rocblas_diagonal_ { rocblas_diagonal_non_unit = 131, rocblas_diagonal_unit = 132 } rocblas_diagonal;
typedef enum rocblas_side_ { rocblas_side_left = 141, rocblas_side_right = 142, rocblas_side_both = 143 } rocblas_side;
typedef enum rocblas_pointer_mode_ { rocblas_pointer_mode_host = 0, rocblas_pointer_mode_device = 1 } rocblas_pointer_mode;
typedef enum rocblas_atomics_mode_ { rocblas_atomics_not_allowed = 0, rocblas_atomics_allowed = 1 } rocblas_atomics_mode;

// Management
rocblas_status rocblas_create_handle(rocblas_handle *handle);
rocblas_status rocblas_destroy_handle(rocblas_handle handle);
rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream);
rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t *stream);
rocblas_status rocblas_set_pointer_mode(rocblas_handle handle, rocblas_pointer_mode pointer_mode);
rocblas_status rocblas_get_pointer_mode(rocblas_handle handle, rocblas_pointer_mode *pointer_mode);
rocblas_status rocblas_set_atomics_mode(rocblas_handle handle, rocblas_atomics_mode atomics_mode);
rocblas_status rocblas_get_atomics_mode(rocblas_handle handle, rocblas_atomics_mode *atomics_mode);
rocblas_status rocblas_set_workspace(rocblas_handle handle, void *addr, size_t size);
const char *rocblas_status_to_string(rocblas_status status);
void rocblas_initialize(void);
rocblas_status rocblas_get_version_string(char *buf, size_t len);
rocblas_status rocblas_get_version_string_size(size_t *len);

rocblas_status rocblas_set_vector(rocblas_int n, rocblas_int elem_size, const void *x, rocblas_int incx, void *y,
                                  rocblas_int incy);
rocblas_status rocblas_get_vector(rocblas_int n, rocblas_int elem_size, const void *x, rocblas_int incx, void *y,
                                  rocblas_int incy);
rocblas_status rocblas_set_matrix(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, const void *a,
                                  rocblas_int lda, void *b, rocblas_int ldb);
rocblas_status rocblas_get_matrix(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, const void *a,
                                  rocblas_int lda, void *b, rocblas_int ldb);
rocblas_status rocblas_set_vector_async(rocblas_int n, rocblas_int elem_size, const void *x, rocblas_int incx,
                                        void *y, rocblas_int incy, hipStream_t stream);
rocblas_status rocblas_get_vector_async(rocblas_int n, rocblas_int elem_size, const void *x, rocblas_int incx,
                                        void *y, rocblas_int incy, hipStream_t stream);
rocblas_status rocblas_set_matrix_async(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, const void *a,
                                        rocblas_int lda, void *b, rocblas_int ldb, hipStream_t stream);
rocblas_status rocblas_get_matrix_async(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, const void *a,
                                        rocblas_int lda, void *b, rocblas_int ldb, hipStream_t stream);

// cuda4cpu-blas: begin rocBLAS's routines (generated by lib/make-blas-api.py)
rocblas_status rocblas_isamax(rocblas_handle handle, rocblas_int n, const float *x, rocblas_int incx,
                              rocblas_int *result);
rocblas_status rocblas_idamax(rocblas_handle handle, rocblas_int n, const double *x, rocblas_int incx,
                              rocblas_int *result);
rocblas_status rocblas_icamax(rocblas_handle handle, rocblas_int n, const rocblas_float_complex *x, rocblas_int incx,
                              rocblas_int *result);
rocblas_status rocblas_izamax(rocblas_handle handle, rocblas_int n, const rocblas_double_complex *x, rocblas_int incx,
                              rocblas_int *result);
rocblas_status rocblas_isamin(rocblas_handle handle, rocblas_int n, const float *x, rocblas_int incx,
                              rocblas_int *result);
rocblas_status rocblas_idamin(rocblas_handle handle, rocblas_int n, const double *x, rocblas_int incx,
                              rocblas_int *result);
rocblas_status rocblas_icamin(rocblas_handle handle, rocblas_int n, const rocblas_float_complex *x, rocblas_int incx,
                              rocblas_int *result);
rocblas_status rocblas_izamin(rocblas_handle handle, rocblas_int n, const rocblas_double_complex *x, rocblas_int incx,
                              rocblas_int *result);
rocblas_status rocblas_sasum(rocblas_handle handle, rocblas_int n, const float *x, rocblas_int incx, float *result);
rocblas_status rocblas_dasum(rocblas_handle handle, rocblas_int n, const double *x, rocblas_int incx, double *result);
rocblas_status rocblas_scasum(rocblas_handle handle, rocblas_int n, const rocblas_float_complex *x, rocblas_int incx,
                              float *result);
rocblas_status rocblas_dzasum(rocblas_handle handle, rocblas_int n, const rocblas_double_complex *x, rocblas_int incx,
                              double *result);
rocblas_status rocblas_saxpy(rocblas_handle handle, rocblas_int n, const float *alpha, const float *x,
                             rocblas_int incx, float *y, rocblas_int incy);
rocblas_status rocblas_daxpy(rocblas_handle handle, rocblas_int n, const double *alpha, const double *x,
                             rocblas_int incx, double *y, rocblas_int incy);
rocblas_status rocblas_caxpy(rocblas_handle handle, rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *x, rocblas_int incx, rocblas_float_complex *y,
                             rocblas_int incy);
rocblas_status rocblas_zaxpy(rocblas_handle handle, rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *x, rocblas_int incx, rocblas_double_complex *y,
                             rocblas_int incy);
rocblas_status rocblas_scopy(rocblas_handle handle, rocblas_int n, const float *x, rocblas_int incx, float *y,
                             rocblas_int incy);
rocblas_status rocblas_dcopy(rocblas_handle handle, rocblas_int n, const double *x, rocblas_int incx, double *y,
                             rocblas_int incy);
rocblas_status rocblas_ccopy(rocblas_handle handle, rocblas_int n, const rocblas_float_complex *x, rocblas_int incx,
                             rocblas_float_complex *y, rocblas_int incy);
rocblas_status rocblas_zcopy(rocblas_handle handle, rocblas_int n, const rocblas_double_complex *x, rocblas_int incx,
                             rocblas_double_complex *y, rocblas_int incy);
rocblas_status rocblas_sdot(rocblas_handle handle, rocblas_int n, const float *x, rocblas_int incx, const float *y,
                            rocblas_int incy, float *result);
rocblas_status rocblas_ddot(rocblas_handle handle, rocblas_int n, const double *x, rocblas_int incx, const double *y,
                            rocblas_int incy, double *result);
rocblas_status rocblas_cdotu(rocblas_handle handle, rocblas_int n, const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *y, rocblas_int incy, rocblas_float_complex *result);
rocblas_status rocblas_zdotu(rocblas_handle handle, rocblas_int n, const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *y, rocblas_int incy, rocblas_double_complex *result);
rocblas_status rocblas_cdotc(rocblas_handle handle, rocblas_int n, const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *y, rocblas_int incy, rocblas_float_complex *result);
rocblas_status rocblas_zdotc(rocblas_handle handle, rocblas_int n, const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *y, rocblas_int incy, rocblas_double_complex *result);
rocblas_status rocblas_snrm2(rocblas_handle handle, rocblas_int n, const float *x, rocblas_int incx, float *result);
rocblas_status rocblas_dnrm2(rocblas_handle handle, rocblas_int n, const double *x, rocblas_int incx, double *result);
rocblas_status rocblas_scnrm2(rocblas_handle handle, rocblas_int n, const rocblas_float_complex *x, rocblas_int incx,
                              float *result);
rocblas_status rocblas_dznrm2(rocblas_handle handle, rocblas_int n, const rocblas_double_complex *x, rocblas_int incx,
                              double *result);
rocblas_status rocblas_srot(rocblas_handle handle, rocblas_int n, float *x, rocblas_int incx, float *y,
                            rocblas_int incy, const float *c, const float *s);
rocblas_status rocblas_drot(rocblas_handle handle, rocblas_int n, double *x, rocblas_int incx, double *y,
                            rocblas_int incy, const double *c, const double *s);
rocblas_status rocblas_crot(rocblas_handle handle, rocblas_int n, rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex *y, rocblas_int incy, const float *c,
                            const rocblas_float_complex *s);
rocblas_status rocblas_zrot(rocblas_handle handle, rocblas_int n, rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex *y, rocblas_int incy, const double *c,
                            const rocblas_double_complex *s);
rocblas_status rocblas_csrot(rocblas_handle handle, rocblas_int n, rocblas_float_complex *x, rocblas_int incx,
                             rocblas_float_complex *y, rocblas_int incy, const float *c, const float *s);
rocblas_status rocblas_zdrot(rocblas_handle handle, rocblas_int n, rocblas_double_complex *x, rocblas_int incx,
                             rocblas_double_complex *y, rocblas_int incy, const double *c, const double *s);
rocblas_status rocblas_srotg(rocblas_handle handle, float *a, float *b, float *c, float *s);
rocblas_status rocblas_drotg(rocblas_handle handle, double *a, double *b, double *c, double *s);
rocblas_status rocblas_crotg(rocblas_handle handle, rocblas_float_complex *a, rocblas_float_complex *b, float *c,
                             rocblas_float_complex *s);
rocblas_status rocblas_zrotg(rocblas_handle handle, rocblas_double_complex *a, rocblas_double_complex *b, double *c,
                             rocblas_double_complex *s);
rocblas_status rocblas_srotm(rocblas_handle handle, rocblas_int n, float *x, rocblas_int incx, float *y,
                             rocblas_int incy, const float *param);
rocblas_status rocblas_drotm(rocblas_handle handle, rocblas_int n, double *x, rocblas_int incx, double *y,
                             rocblas_int incy, const double *param);
rocblas_status rocblas_srotmg(rocblas_handle handle, float *d1, float *d2, float *x1, const float *y1, float *param);
rocblas_status rocblas_drotmg(rocblas_handle handle, double *d1, double *d2, double *x1, const double *y1,
                              double *param);
rocblas_status rocblas_sscal(rocblas_handle handle, rocblas_int n, const float *alpha, float *x, rocblas_int incx);
rocblas_status rocblas_dscal(rocblas_handle handle, rocblas_int n, const double *alpha, double *x, rocblas_int incx);
rocblas_status rocblas_cscal(rocblas_handle handle, rocblas_int n, const rocblas_float_complex *alpha,
                             rocblas_float_complex *x, rocblas_int incx);
rocblas_status rocblas_zscal(rocblas_handle handle, rocblas_int n, const rocblas_double_complex *alpha,
                             rocblas_double_complex *x, rocblas_int incx);
rocblas_status rocblas_csscal(rocblas_handle handle, rocblas_int n, const float *alpha, rocblas_float_complex *x,
                              rocblas_int incx);
rocblas_status rocblas_zdscal(rocblas_handle handle, rocblas_int n, const double *alpha, rocblas_double_complex *x,
                              rocblas_int incx);
rocblas_status rocblas_sswap(rocblas_handle handle, rocblas_int n, float *x, rocblas_int incx, float *y,
                             rocblas_int incy);
rocblas_status rocblas_dswap(rocblas_handle handle, rocblas_int n, double *x, rocblas_int incx, double *y,
                             rocblas_int incy);
rocblas_status rocblas_cswap(rocblas_handle handle, rocblas_int n, rocblas_float_complex *x, rocblas_int incx,
                             rocblas_float_complex *y, rocblas_int incy);
rocblas_status rocblas_zswap(rocblas_handle handle, rocblas_int n, rocblas_double_complex *x, rocblas_int incx,
                             rocblas_double_complex *y, rocblas_int incy);
rocblas_status rocblas_sgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                             rocblas_int kl, rocblas_int ku, const float *alpha, const float *A, rocblas_int lda,
                             const float *x, rocblas_int incx, const float *beta, float *y, rocblas_int incy);
rocblas_status rocblas_dgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                             rocblas_int kl, rocblas_int ku, const double *alpha, const double *A, rocblas_int lda,
                             const double *x, rocblas_int incx, const double *beta, double *y, rocblas_int incy);
rocblas_status rocblas_cgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                             rocblas_int kl, rocblas_int ku, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda, const rocblas_float_complex *x,
                             rocblas_int incx, const rocblas_float_complex *beta, rocblas_float_complex *y,
                             rocblas_int incy);
rocblas_status rocblas_zgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                             rocblas_int kl, rocblas_int ku, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda, const rocblas_double_complex *x,
                             rocblas_int incx, const rocblas_double_complex *beta, rocblas_double_complex *y,
                             rocblas_int incy);
rocblas_status rocblas_sgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                             const float *alpha, const float *A, rocblas_int lda, const float *x, rocblas_int incx,
                             const float *beta, float *y, rocblas_int incy);
rocblas_status rocblas_dgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                             const double *alpha, const double *A, rocblas_int lda, const double *x, rocblas_int incx,
                             const double *beta, double *y, rocblas_int incy);
rocblas_status rocblas_cgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                             const rocblas_float_complex *alpha, const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *x, rocblas_int incx, const rocblas_float_complex *beta,
                             rocblas_float_complex *y, rocblas_int incy);
rocblas_status rocblas_zgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                             const rocblas_double_complex *alpha, const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *x, rocblas_int incx, const rocblas_double_complex *beta,
                             rocblas_double_complex *y, rocblas_int incy);
rocblas_status rocblas_sgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                                     const float *alpha, const float *const Aarray[], rocblas_int lda,
                                     const float *const xarray[], rocblas_int incx, const float *beta,
                                     float *const yarray[], rocblas_int incy, rocblas_int batchCount);
rocblas_status rocblas_dgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                                     const double *alpha, const double *const Aarray[], rocblas_int lda,
                                     const double *const xarray[], rocblas_int incx, const double *beta,
                                     double *const yarray[], rocblas_int incy, rocblas_int batchCount);
rocblas_status rocblas_cgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                                     const rocblas_float_complex *alpha, const rocblas_float_complex *const Aarray[],
                                     rocblas_int lda, const rocblas_float_complex *const xarray[], rocblas_int incx,
                                     const rocblas_float_complex *beta, rocblas_float_complex *const yarray[],
                                     rocblas_int incy, rocblas_int batchCount);
rocblas_status rocblas_zgemv_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n,
                                     const rocblas_double_complex *alpha,
                                     const rocblas_double_complex *const Aarray[], rocblas_int lda,
                                     const rocblas_double_complex *const xarray[], rocblas_int incx,
                                     const rocblas_double_complex *beta, rocblas_double_complex *const yarray[],
                                     rocblas_int incy, rocblas_int batchCount);
rocblas_status rocblas_sgemv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m,
                                             rocblas_int n, const float *alpha, const float *A, rocblas_int lda,
                                             rocblas_stride strideA, const float *x, rocblas_int incx,
                                             rocblas_stride stridex, const float *beta, float *y, rocblas_int incy,
                                             rocblas_stride stridey, rocblas_int batchCount);
rocblas_status rocblas_dgemv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m,
                                             rocblas_int n, const double *alpha, const double *A, rocblas_int lda,
                                             rocblas_stride strideA, const double *x, rocblas_int incx,
                                             rocblas_stride stridex, const double *beta, double *y, rocblas_int incy,
                                             rocblas_stride stridey, rocblas_int batchCount);
rocblas_status rocblas_cgemv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m,
                                             rocblas_int n, const rocblas_float_complex *alpha,
                                             const rocblas_float_complex *A, rocblas_int lda, rocblas_stride strideA,
                                             const rocblas_float_complex *x, rocblas_int incx, rocblas_stride stridex,
                                             const rocblas_float_complex *beta, rocblas_float_complex *y,
                                             rocblas_int incy, rocblas_stride stridey, rocblas_int batchCount);
rocblas_status rocblas_zgemv_strided_batched(rocblas_handle handle, rocblas_operation trans, rocblas_int m,
                                             rocblas_int n, const rocblas_double_complex *alpha,
                                             const rocblas_double_complex *A, rocblas_int lda, rocblas_stride strideA,
                                             const rocblas_double_complex *x, rocblas_int incx,
                                             rocblas_stride stridex, const rocblas_double_complex *beta,
                                             rocblas_double_complex *y, rocblas_int incy, rocblas_stride stridey,
                                             rocblas_int batchCount);
rocblas_status rocblas_sger(rocblas_handle handle, rocblas_int m, rocblas_int n, const float *alpha, const float *x,
                            rocblas_int incx, const float *y, rocblas_int incy, float *A, rocblas_int lda);
rocblas_status rocblas_dger(rocblas_handle handle, rocblas_int m, rocblas_int n, const double *alpha, const double *x,
                            rocblas_int incx, const double *y, rocblas_int incy, double *A, rocblas_int lda);
rocblas_status rocblas_cgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *x, rocblas_int incx, const rocblas_float_complex *y,
                             rocblas_int incy, rocblas_float_complex *A, rocblas_int lda);
rocblas_status rocblas_zgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *x, rocblas_int incx, const rocblas_double_complex *y,
                             rocblas_int incy, rocblas_double_complex *A, rocblas_int lda);
rocblas_status rocblas_cgerc(rocblas_handle handle, rocblas_int m, rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *x, rocblas_int incx, const rocblas_float_complex *y,
                             rocblas_int incy, rocblas_float_complex *A, rocblas_int lda);
rocblas_status rocblas_zgerc(rocblas_handle handle, rocblas_int m, rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *x, rocblas_int incx, const rocblas_double_complex *y,
                             rocblas_int incy, rocblas_double_complex *A, rocblas_int lda);
rocblas_status rocblas_ssbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k,
                             const float *alpha, const float *A, rocblas_int lda, const float *x, rocblas_int incx,
                             const float *beta, float *y, rocblas_int incy);
rocblas_status rocblas_dsbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k,
                             const double *alpha, const double *A, rocblas_int lda, const double *x, rocblas_int incx,
                             const double *beta, double *y, rocblas_int incy);
rocblas_status rocblas_sspmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float *alpha,
                             const float *AP, const float *x, rocblas_int incx, const float *beta, float *y,
                             rocblas_int incy);
rocblas_status rocblas_dspmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double *alpha,
                             const double *AP, const double *x, rocblas_int incx, const double *beta, double *y,
                             rocblas_int incy);
rocblas_status rocblas_sspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float *alpha,
                            const float *x, rocblas_int incx, float *AP);
rocblas_status rocblas_dspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double *alpha,
                            const double *x, rocblas_int incx, double *AP);
rocblas_status rocblas_sspr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float *alpha,
                             const float *x, rocblas_int incx, const float *y, rocblas_int incy, float *AP);
rocblas_status rocblas_dspr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double *alpha,
                             const double *x, rocblas_int incx, const double *y, rocblas_int incy, double *AP);
rocblas_status rocblas_ssymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float *alpha,
                             const float *A, rocblas_int lda, const float *x, rocblas_int incx, const float *beta,
                             float *y, rocblas_int incy);
rocblas_status rocblas_dsymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double *alpha,
                             const double *A, rocblas_int lda, const double *x, rocblas_int incx, const double *beta,
                             double *y, rocblas_int incy);
rocblas_status rocblas_csymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_float_complex *alpha, const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *x, rocblas_int incx, const rocblas_float_complex *beta,
                             rocblas_float_complex *y, rocblas_int incy);
rocblas_status rocblas_zsymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_double_complex *alpha, const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *x, rocblas_int incx, const rocblas_double_complex *beta,
                             rocblas_double_complex *y, rocblas_int incy);
rocblas_status rocblas_ssyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float *alpha,
                            const float *x, rocblas_int incx, float *A, rocblas_int lda);
rocblas_status rocblas_dsyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double *alpha,
                            const double *x, rocblas_int incx, double *A, rocblas_int lda);
rocblas_status rocblas_csyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                            const rocblas_float_complex *alpha, const rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex *A, rocblas_int lda);
rocblas_status rocblas_zsyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                            const rocblas_double_complex *alpha, const rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex *A, rocblas_int lda);
rocblas_status rocblas_ssyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float *alpha,
                             const float *x, rocblas_int incx, const float *y, rocblas_int incy, float *A,
                             rocblas_int lda);
rocblas_status rocblas_dsyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double *alpha,
                             const double *x, rocblas_int incx, const double *y, rocblas_int incy, double *A,
                             rocblas_int lda);
rocblas_status rocblas_csyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_float_complex *alpha, const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *y, rocblas_int incy, rocblas_float_complex *A,
                             rocblas_int lda);
rocblas_status rocblas_zsyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_double_complex *alpha, const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *y, rocblas_int incy, rocblas_double_complex *A,
                             rocblas_int lda);
rocblas_status rocblas_stbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, rocblas_int k, const float *A, rocblas_int lda, float *x,
                             rocblas_int incx);
rocblas_status rocblas_dtbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, rocblas_int k, const double *A, rocblas_int lda, double *x,
                             rocblas_int incx);
rocblas_status rocblas_ctbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, rocblas_int k, const rocblas_float_complex *A, rocblas_int lda,
                             rocblas_float_complex *x, rocblas_int incx);
rocblas_status rocblas_ztbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, rocblas_int k, const rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *x, rocblas_int incx);
rocblas_status rocblas_stbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, rocblas_int k, const float *A, rocblas_int lda, float *x,
                             rocblas_int incx);
rocblas_status rocblas_dtbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, rocblas_int k, const double *A, rocblas_int lda, double *x,
                             rocblas_int incx);
rocblas_status rocblas_ctbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, rocblas_int k, const rocblas_float_complex *A, rocblas_int lda,
                             rocblas_float_complex *x, rocblas_int incx);
rocblas_status rocblas_ztbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, rocblas_int k, const rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *x, rocblas_int incx);
rocblas_status rocblas_stpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const float *AP, float *x, rocblas_int incx);
rocblas_status rocblas_dtpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const double *AP, double *x, rocblas_int incx);
rocblas_status rocblas_ctpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const rocblas_float_complex *AP, rocblas_float_complex *x,
                             rocblas_int incx);
rocblas_status rocblas_ztpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const rocblas_double_complex *AP, rocblas_double_complex *x,
                             rocblas_int incx);
rocblas_status rocblas_stpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const float *AP, float *x, rocblas_int incx);
rocblas_status rocblas_dtpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const double *AP, double *x, rocblas_int incx);
rocblas_status rocblas_ctpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const rocblas_float_complex *AP, rocblas_float_complex *x,
                             rocblas_int incx);
rocblas_status rocblas_ztpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const rocblas_double_complex *AP, rocblas_double_complex *x,
                             rocblas_int incx);
rocblas_status rocblas_strmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const float *A, rocblas_int lda, float *x, rocblas_int incx);
rocblas_status rocblas_dtrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const double *A, rocblas_int lda, double *x, rocblas_int incx);
rocblas_status rocblas_ctrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *x,
                             rocblas_int incx);
rocblas_status rocblas_ztrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *x, rocblas_int incx);
rocblas_status rocblas_strsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const float *A, rocblas_int lda, float *x, rocblas_int incx);
rocblas_status rocblas_dtrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const double *A, rocblas_int lda, double *x, rocblas_int incx);
rocblas_status rocblas_ctrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *x,
                             rocblas_int incx);
rocblas_status rocblas_ztrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int n, const rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *x, rocblas_int incx);
rocblas_status rocblas_chemv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_float_complex *alpha, const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *x, rocblas_int incx, const rocblas_float_complex *beta,
                             rocblas_float_complex *y, rocblas_int incy);
rocblas_status rocblas_zhemv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_double_complex *alpha, const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *x, rocblas_int incx, const rocblas_double_complex *beta,
                             rocblas_double_complex *y, rocblas_int incy);
rocblas_status rocblas_chbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k,
                             const rocblas_float_complex *alpha, const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *x, rocblas_int incx, const rocblas_float_complex *beta,
                             rocblas_float_complex *y, rocblas_int incy);
rocblas_status rocblas_zhbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k,
                             const rocblas_double_complex *alpha, const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *x, rocblas_int incx, const rocblas_double_complex *beta,
                             rocblas_double_complex *y, rocblas_int incy);
rocblas_status rocblas_chpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_float_complex *alpha, const rocblas_float_complex *AP,
                             const rocblas_float_complex *x, rocblas_int incx, const rocblas_float_complex *beta,
                             rocblas_float_complex *y, rocblas_int incy);
rocblas_status rocblas_zhpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_double_complex *alpha, const rocblas_double_complex *AP,
                             const rocblas_double_complex *x, rocblas_int incx, const rocblas_double_complex *beta,
                             rocblas_double_complex *y, rocblas_int incy);
rocblas_status rocblas_cher(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float *alpha,
                            const rocblas_float_complex *x, rocblas_int incx, rocblas_float_complex *A,
                            rocblas_int lda);
rocblas_status rocblas_zher(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double *alpha,
                            const rocblas_double_complex *x, rocblas_int incx, rocblas_double_complex *A,
                            rocblas_int lda);
rocblas_status rocblas_cher2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_float_complex *alpha, const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *y, rocblas_int incy, rocblas_float_complex *A,
                             rocblas_int lda);
rocblas_status rocblas_zher2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_double_complex *alpha, const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *y, rocblas_int incy, rocblas_double_complex *A,
                             rocblas_int lda);
rocblas_status rocblas_chpr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float *alpha,
                            const rocblas_float_complex *x, rocblas_int incx, rocblas_float_complex *AP);
rocblas_status rocblas_zhpr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double *alpha,
                            const rocblas_double_complex *x, rocblas_int incx, rocblas_double_complex *AP);
rocblas_status rocblas_chpr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_float_complex *alpha, const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *y, rocblas_int incy, rocblas_float_complex *AP);
rocblas_status rocblas_zhpr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
                             const rocblas_double_complex *alpha, const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *y, rocblas_int incy, rocblas_double_complex *AP);
rocblas_status rocblas_sgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb, rocblas_int m,
                             rocblas_int n, rocblas_int k, const float *alpha, const float *A, rocblas_int lda,
                             const float *B, rocblas_int ldb, const float *beta, float *C, rocblas_int ldc);
rocblas_status rocblas_dgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb, rocblas_int m,
                             rocblas_int n, rocblas_int k, const double *alpha, const double *A, rocblas_int lda,
                             const double *B, rocblas_int ldb, const double *beta, double *C, rocblas_int ldc);
rocblas_status rocblas_cgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb, rocblas_int m,
                             rocblas_int n, rocblas_int k, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda, const rocblas_float_complex *B,
                             rocblas_int ldb, const rocblas_float_complex *beta, rocblas_float_complex *C,
                             rocblas_int ldc);
rocblas_status rocblas_zgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb, rocblas_int m,
                             rocblas_int n, rocblas_int k, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda, const rocblas_double_complex *B,
                             rocblas_int ldb, const rocblas_double_complex *beta, rocblas_double_complex *C,
                             rocblas_int ldc);
rocblas_status rocblas_sgemm_batched(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                     rocblas_int m, rocblas_int n, rocblas_int k, const float *alpha,
                                     const float *const Aarray[], rocblas_int lda, const float *const Barray[],
                                     rocblas_int ldb, const float *beta, float *const Carray[], rocblas_int ldc,
                                     rocblas_int batchCount);
rocblas_status rocblas_dgemm_batched(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                     rocblas_int m, rocblas_int n, rocblas_int k, const double *alpha,
                                     const double *const Aarray[], rocblas_int lda, const double *const Barray[],
                                     rocblas_int ldb, const double *beta, double *const Carray[], rocblas_int ldc,
                                     rocblas_int batchCount);
rocblas_status rocblas_cgemm_batched(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                     rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_float_complex *alpha,
                                     const rocblas_float_complex *const Aarray[], rocblas_int lda,
                                     const rocblas_float_complex *const Barray[], rocblas_int ldb,
                                     const rocblas_float_complex *beta, rocblas_float_complex *const Carray[],
                                     rocblas_int ldc, rocblas_int batchCount);
rocblas_status rocblas_zgemm_batched(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                     rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_double_complex *alpha,
                                     const rocblas_double_complex *const Aarray[], rocblas_int lda,
                                     const rocblas_double_complex *const Barray[], rocblas_int ldb,
                                     const rocblas_double_complex *beta, rocblas_double_complex *const Carray[],
                                     rocblas_int ldc, rocblas_int batchCount);
rocblas_status rocblas_sgemm_strided_batched(rocblas_handle handle, rocblas_operation transa,
                                             rocblas_operation transb, rocblas_int m, rocblas_int n, rocblas_int k,
                                             const float *alpha, const float *A, rocblas_int lda,
                                             rocblas_stride strideA, const float *B, rocblas_int ldb,
                                             rocblas_stride strideB, const float *beta, float *C, rocblas_int ldc,
                                             rocblas_stride strideC, rocblas_int batchCount);
rocblas_status rocblas_dgemm_strided_batched(rocblas_handle handle, rocblas_operation transa,
                                             rocblas_operation transb, rocblas_int m, rocblas_int n, rocblas_int k,
                                             const double *alpha, const double *A, rocblas_int lda,
                                             rocblas_stride strideA, const double *B, rocblas_int ldb,
                                             rocblas_stride strideB, const double *beta, double *C, rocblas_int ldc,
                                             rocblas_stride strideC, rocblas_int batchCount);
rocblas_status rocblas_cgemm_strided_batched(rocblas_handle handle, rocblas_operation transa,
                                             rocblas_operation transb, rocblas_int m, rocblas_int n, rocblas_int k,
                                             const rocblas_float_complex *alpha, const rocblas_float_complex *A,
                                             rocblas_int lda, rocblas_stride strideA, const rocblas_float_complex *B,
                                             rocblas_int ldb, rocblas_stride strideB,
                                             const rocblas_float_complex *beta, rocblas_float_complex *C,
                                             rocblas_int ldc, rocblas_stride strideC, rocblas_int batchCount);
rocblas_status rocblas_zgemm_strided_batched(rocblas_handle handle, rocblas_operation transa,
                                             rocblas_operation transb, rocblas_int m, rocblas_int n, rocblas_int k,
                                             const rocblas_double_complex *alpha, const rocblas_double_complex *A,
                                             rocblas_int lda, rocblas_stride strideA, const rocblas_double_complex *B,
                                             rocblas_int ldb, rocblas_stride strideB,
                                             const rocblas_double_complex *beta, rocblas_double_complex *C,
                                             rocblas_int ldc, rocblas_stride strideC, rocblas_int batchCount);
rocblas_status rocblas_ssymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m,
                             rocblas_int n, const float *alpha, const float *A, rocblas_int lda, const float *B,
                             rocblas_int ldb, const float *beta, float *C, rocblas_int ldc);
rocblas_status rocblas_dsymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m,
                             rocblas_int n, const double *alpha, const double *A, rocblas_int lda, const double *B,
                             rocblas_int ldb, const double *beta, double *C, rocblas_int ldc);
rocblas_status rocblas_csymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m,
                             rocblas_int n, const rocblas_float_complex *alpha, const rocblas_float_complex *A,
                             rocblas_int lda, const rocblas_float_complex *B, rocblas_int ldb,
                             const rocblas_float_complex *beta, rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_zsymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m,
                             rocblas_int n, const rocblas_double_complex *alpha, const rocblas_double_complex *A,
                             rocblas_int lda, const rocblas_double_complex *B, rocblas_int ldb,
                             const rocblas_double_complex *beta, rocblas_double_complex *C, rocblas_int ldc);
rocblas_status rocblas_chemm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m,
                             rocblas_int n, const rocblas_float_complex *alpha, const rocblas_float_complex *A,
                             rocblas_int lda, const rocblas_float_complex *B, rocblas_int ldb,
                             const rocblas_float_complex *beta, rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_zhemm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m,
                             rocblas_int n, const rocblas_double_complex *alpha, const rocblas_double_complex *A,
                             rocblas_int lda, const rocblas_double_complex *B, rocblas_int ldb,
                             const rocblas_double_complex *beta, rocblas_double_complex *C, rocblas_int ldc);
rocblas_status rocblas_ssyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                             rocblas_int k, const float *alpha, const float *A, rocblas_int lda, const float *beta,
                             float *C, rocblas_int ldc);
rocblas_status rocblas_dsyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                             rocblas_int k, const double *alpha, const double *A, rocblas_int lda, const double *beta,
                             double *C, rocblas_int ldc);
rocblas_status rocblas_csyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                             rocblas_int k, const rocblas_float_complex *alpha, const rocblas_float_complex *A,
                             rocblas_int lda, const rocblas_float_complex *beta, rocblas_float_complex *C,
                             rocblas_int ldc);
rocblas_status rocblas_zsyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                             rocblas_int k, const rocblas_double_complex *alpha, const rocblas_double_complex *A,
                             rocblas_int lda, const rocblas_double_complex *beta, rocblas_double_complex *C,
                             rocblas_int ldc);
rocblas_status rocblas_cherk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                             rocblas_int k, const float *alpha, const rocblas_float_complex *A, rocblas_int lda,
                             const float *beta, rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_zherk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                             rocblas_int k, const double *alpha, const rocblas_double_complex *A, rocblas_int lda,
                             const double *beta, rocblas_double_complex *C, rocblas_int ldc);
rocblas_status rocblas_ssyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const float *alpha, const float *A, rocblas_int lda, const float *B,
                              rocblas_int ldb, const float *beta, float *C, rocblas_int ldc);
rocblas_status rocblas_dsyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const double *alpha, const double *A, rocblas_int lda, const double *B,
                              rocblas_int ldb, const double *beta, double *C, rocblas_int ldc);
rocblas_status rocblas_csyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const rocblas_float_complex *alpha, const rocblas_float_complex *A,
                              rocblas_int lda, const rocblas_float_complex *B, rocblas_int ldb,
                              const rocblas_float_complex *beta, rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_zsyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const rocblas_double_complex *alpha, const rocblas_double_complex *A,
                              rocblas_int lda, const rocblas_double_complex *B, rocblas_int ldb,
                              const rocblas_double_complex *beta, rocblas_double_complex *C, rocblas_int ldc);
rocblas_status rocblas_cher2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const rocblas_float_complex *alpha, const rocblas_float_complex *A,
                              rocblas_int lda, const rocblas_float_complex *B, rocblas_int ldb, const float *beta,
                              rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_zher2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const rocblas_double_complex *alpha, const rocblas_double_complex *A,
                              rocblas_int lda, const rocblas_double_complex *B, rocblas_int ldb, const double *beta,
                              rocblas_double_complex *C, rocblas_int ldc);
rocblas_status rocblas_ssyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const float *alpha, const float *A, rocblas_int lda, const float *B,
                              rocblas_int ldb, const float *beta, float *C, rocblas_int ldc);
rocblas_status rocblas_dsyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const double *alpha, const double *A, rocblas_int lda, const double *B,
                              rocblas_int ldb, const double *beta, double *C, rocblas_int ldc);
rocblas_status rocblas_csyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const rocblas_float_complex *alpha, const rocblas_float_complex *A,
                              rocblas_int lda, const rocblas_float_complex *B, rocblas_int ldb,
                              const rocblas_float_complex *beta, rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_zsyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const rocblas_double_complex *alpha, const rocblas_double_complex *A,
                              rocblas_int lda, const rocblas_double_complex *B, rocblas_int ldb,
                              const rocblas_double_complex *beta, rocblas_double_complex *C, rocblas_int ldc);
rocblas_status rocblas_cherkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const rocblas_float_complex *alpha, const rocblas_float_complex *A,
                              rocblas_int lda, const rocblas_float_complex *B, rocblas_int ldb, const float *beta,
                              rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_zherkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const rocblas_double_complex *alpha, const rocblas_double_complex *A,
                              rocblas_int lda, const rocblas_double_complex *B, rocblas_int ldb, const double *beta,
                              rocblas_double_complex *C, rocblas_int ldc);
rocblas_status rocblas_strmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                             rocblas_diagonal diag, rocblas_int m, rocblas_int n, const float *alpha, const float *A,
                             rocblas_int lda, const float *B, rocblas_int ldb, float *C, rocblas_int ldc);
rocblas_status rocblas_dtrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                             rocblas_diagonal diag, rocblas_int m, rocblas_int n, const double *alpha,
                             const double *A, rocblas_int lda, const double *B, rocblas_int ldb, double *C,
                             rocblas_int ldc);
rocblas_status rocblas_ctrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                             rocblas_diagonal diag, rocblas_int m, rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda, const rocblas_float_complex *B,
                             rocblas_int ldb, rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_ztrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                             rocblas_diagonal diag, rocblas_int m, rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda, const rocblas_double_complex *B,
                             rocblas_int ldb, rocblas_double_complex *C, rocblas_int ldc);
rocblas_status rocblas_strsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                             rocblas_diagonal diag, rocblas_int m, rocblas_int n, const float *alpha, const float *A,
                             rocblas_int lda, float *B, rocblas_int ldb);
rocblas_status rocblas_dtrsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                             rocblas_diagonal diag, rocblas_int m, rocblas_int n, const double *alpha,
                             const double *A, rocblas_int lda, double *B, rocblas_int ldb);
rocblas_status rocblas_ctrsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                             rocblas_diagonal diag, rocblas_int m, rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *B,
                             rocblas_int ldb);
rocblas_status rocblas_ztrsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                             rocblas_diagonal diag, rocblas_int m, rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda, rocblas_double_complex *B,
                             rocblas_int ldb);
rocblas_status rocblas_strsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
                                     rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                                     const float *alpha, const float *const Aarray[], rocblas_int lda,
                                     float *const Barray[], rocblas_int ldb, rocblas_int batchCount);
rocblas_status rocblas_dtrsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
                                     rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                                     const double *alpha, const double *const Aarray[], rocblas_int lda,
                                     double *const Barray[], rocblas_int ldb, rocblas_int batchCount);
rocblas_status rocblas_ctrsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
                                     rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                                     const rocblas_float_complex *alpha, const rocblas_float_complex *const Aarray[],
                                     rocblas_int lda, rocblas_float_complex *const Barray[], rocblas_int ldb,
                                     rocblas_int batchCount);
rocblas_status rocblas_ztrsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
                                     rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                                     const rocblas_double_complex *alpha,
                                     const rocblas_double_complex *const Aarray[], rocblas_int lda,
                                     rocblas_double_complex *const Barray[], rocblas_int ldb, rocblas_int batchCount);
rocblas_status rocblas_sgeam(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb, rocblas_int m,
                             rocblas_int n, const float *alpha, const float *A, rocblas_int lda, const float *beta,
                             const float *B, rocblas_int ldb, float *C, rocblas_int ldc);
rocblas_status rocblas_dgeam(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb, rocblas_int m,
                             rocblas_int n, const double *alpha, const double *A, rocblas_int lda, const double *beta,
                             const double *B, rocblas_int ldb, double *C, rocblas_int ldc);
rocblas_status rocblas_cgeam(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb, rocblas_int m,
                             rocblas_int n, const rocblas_float_complex *alpha, const rocblas_float_complex *A,
                             rocblas_int lda, const rocblas_float_complex *beta, const rocblas_float_complex *B,
                             rocblas_int ldb, rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_zgeam(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb, rocblas_int m,
                             rocblas_int n, const rocblas_double_complex *alpha, const rocblas_double_complex *A,
                             rocblas_int lda, const rocblas_double_complex *beta, const rocblas_double_complex *B,
                             rocblas_int ldb, rocblas_double_complex *C, rocblas_int ldc);
rocblas_status rocblas_sdgmm(rocblas_handle handle, rocblas_side mode, rocblas_int m, rocblas_int n, const float *A,
                             rocblas_int lda, const float *x, rocblas_int incx, float *C, rocblas_int ldc);
rocblas_status rocblas_ddgmm(rocblas_handle handle, rocblas_side mode, rocblas_int m, rocblas_int n, const double *A,
                             rocblas_int lda, const double *x, rocblas_int incx, double *C, rocblas_int ldc);
rocblas_status rocblas_cdgmm(rocblas_handle handle, rocblas_side mode, rocblas_int m, rocblas_int n,
                             const rocblas_float_complex *A, rocblas_int lda, const rocblas_float_complex *x,
                             rocblas_int incx, rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_zdgmm(rocblas_handle handle, rocblas_side mode, rocblas_int m, rocblas_int n,
                             const rocblas_double_complex *A, rocblas_int lda, const rocblas_double_complex *x,
                             rocblas_int incx, rocblas_double_complex *C, rocblas_int ldc);
// cuda4cpu-blas: end rocBLAS's routines

} // extern "C"
