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

// cuBLAS's API (cublas_v2.h's): the BLAS routines and cuBLAS's extensions,
// on the CPU, with C linkage as cuBLAS's. The routines are in the library:
// on a CBLAS (OpenBLAS, BLIS, MKL, ...) that it finds at run time, or on its
// own implementation (docs/features.md). Matrices are column-major, as in
// cuBLAS. Every pointer, device's or host's, is the CPU's, and calls run
// when they're made, or when the graph runs that a stream captured them in.
#pragma once

// cuBLAS's include guards, which code tests before using its types
#ifndef CUBLAS_API_H_
#define CUBLAS_API_H_
#endif

#include <cstddef>
#include <cstdint>

#include "cuComplex.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "library_types.h"

#define CUBLAS_VER_MAJOR 12
#define CUBLAS_VER_MINOR 0
#define CUBLAS_VER_PATCH 1
#define CUBLAS_VER_BUILD 189
#define CUBLAS_VERSION (CUBLAS_VER_MAJOR * 10000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH)

extern "C" {

typedef enum {
    CUBLAS_STATUS_SUCCESS          = 0,
    CUBLAS_STATUS_NOT_INITIALIZED  = 1,
    CUBLAS_STATUS_ALLOC_FAILED     = 3,
    CUBLAS_STATUS_INVALID_VALUE    = 7,
    CUBLAS_STATUS_ARCH_MISMATCH    = 8,
    CUBLAS_STATUS_MAPPING_ERROR    = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR   = 14,
    CUBLAS_STATUS_NOT_SUPPORTED    = 15,
    CUBLAS_STATUS_LICENSE_ERROR    = 16
} cublasStatus_t;

typedef enum { CUBLAS_FILL_MODE_LOWER = 0, CUBLAS_FILL_MODE_UPPER = 1, CUBLAS_FILL_MODE_FULL = 2 } cublasFillMode_t;
typedef enum { CUBLAS_DIAG_NON_UNIT = 0, CUBLAS_DIAG_UNIT = 1 } cublasDiagType_t;
typedef enum { CUBLAS_SIDE_LEFT = 0, CUBLAS_SIDE_RIGHT = 1 } cublasSideMode_t;
typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_HERMITAN = 2,
    CUBLAS_OP_CONJG = 3
} cublasOperation_t;
typedef enum { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 } cublasPointerMode_t;
typedef enum { CUBLAS_ATOMICS_NOT_ALLOWED = 0, CUBLAS_ATOMICS_ALLOWED = 1 } cublasAtomicsMode_t;

typedef enum {
    CUBLAS_GEMM_DFALT             = -1,
    CUBLAS_GEMM_DEFAULT           = -1,
    CUBLAS_GEMM_ALGO0             = 0,
    CUBLAS_GEMM_ALGO1             = 1,
    CUBLAS_GEMM_ALGO2             = 2,
    CUBLAS_GEMM_ALGO3             = 3,
    CUBLAS_GEMM_ALGO4             = 4,
    CUBLAS_GEMM_ALGO5             = 5,
    CUBLAS_GEMM_ALGO6             = 6,
    CUBLAS_GEMM_ALGO7             = 7,
    CUBLAS_GEMM_ALGO8             = 8,
    CUBLAS_GEMM_ALGO9             = 9,
    CUBLAS_GEMM_ALGO10            = 10,
    CUBLAS_GEMM_ALGO11            = 11,
    CUBLAS_GEMM_ALGO12            = 12,
    CUBLAS_GEMM_ALGO13            = 13,
    CUBLAS_GEMM_ALGO14            = 14,
    CUBLAS_GEMM_ALGO15            = 15,
    CUBLAS_GEMM_ALGO16            = 16,
    CUBLAS_GEMM_ALGO17            = 17,
    CUBLAS_GEMM_ALGO18            = 18,
    CUBLAS_GEMM_ALGO19            = 19,
    CUBLAS_GEMM_ALGO20            = 20,
    CUBLAS_GEMM_ALGO21            = 21,
    CUBLAS_GEMM_ALGO22            = 22,
    CUBLAS_GEMM_ALGO23            = 23,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_DFALT_TENSOR_OP   = 99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP   = 100,
    CUBLAS_GEMM_ALGO15_TENSOR_OP  = 115
} cublasGemmAlgo_t;

typedef enum {
    CUBLAS_DEFAULT_MATH                             = 0,
    CUBLAS_TENSOR_OP_MATH                           = 1,
    CUBLAS_PEDANTIC_MATH                            = 2,
    CUBLAS_TF32_TENSOR_OP_MATH                      = 3,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
} cublasMath_t;

typedef enum {
    CUBLAS_COMPUTE_16F           = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC  = 65,
    CUBLAS_COMPUTE_32F           = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC  = 69,
    CUBLAS_COMPUTE_32F_FAST_16F  = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F           = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC  = 71,
    CUBLAS_COMPUTE_32I           = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC  = 73
} cublasComputeType_t;

//! Computing in a data type is cuBLAS 10's form, which cudaDataType still
//! names
typedef cudaDataType cublasDataType_t;

struct cublasContext;
typedef struct cublasContext *cublasHandle_t;

typedef void (*cublasLogCallback)(const char *msg);

// Management
cublasStatus_t cublasCreate(cublasHandle_t *handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version);
cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value);
size_t cublasGetCudartVersion(void);
const char *cublasGetStatusName(cublasStatus_t status);
const char *cublasGetStatusString(cublasStatus_t status);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId);
cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes);
cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode);
cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode);
cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode);
cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, int *smCountTarget);
cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget);
cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char *logFileName);
cublasStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback);
cublasStatus_t cublasGetLoggerCallback(cublasLogCallback *userCallback);

// Copies between the host and the device: strided vectors and matrices
cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy);
cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy);
cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy,
                                    cudaStream_t stream);
cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy,
                                    cudaStream_t stream);
cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb,
                                    cudaStream_t stream);
cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb,
                                    cudaStream_t stream);

// Mixed and reduced precision: matrix products whose types are arguments,
// and half precision's
cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb,
                           const __half *beta, __half *C, int ldc);
cublasStatus_t cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                  int n, int k, const __half *alpha, const __half *const Aarray[], int lda,
                                  const __half *const Barray[], int ldb, const __half *beta, __half *const Carray[],
                                  int ldc, int batchCount);
cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k, const __half *alpha, const __half *A, int lda,
                                         long long int strideA, const __half *B, int ldb, long long int strideB,
                                         const __half *beta, __half *C, int ldc, long long int strideC,
                                         int batchCount);
cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                             int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const void *B,
                             cudaDataType Btype, int ldb, const float *beta, void *C, cudaDataType Ctype, int ldc);
cublasStatus_t cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                             int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda,
                             const void *B, cudaDataType Btype, int ldb, const cuComplex *beta, void *C,
                             cudaDataType Ctype, int ldc);
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                            int k, const void *alpha, const void *A, cudaDataType Atype, int lda, const void *B,
                            cudaDataType Btype, int ldb, const void *beta, void *C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                   int n, int k, const void *alpha, const void *const Aarray[], cudaDataType Atype,
                                   int lda, const void *const Barray[], cudaDataType Btype, int ldb,
                                   const void *beta, void *const Carray[], cudaDataType Ctype, int ldc,
                                   int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype,
                                          int lda, long long int strideA, const void *B, cudaDataType Btype, int ldb,
                                          long long int strideB, const void *beta, void *C, cudaDataType Ctype,
                                          int ldc, long long int strideC, int batchCount,
                                          cublasComputeType_t computeType, cublasGemmAlgo_t algo);

// cuda4cpu-blas: begin cuBLAS's routines (generated by lib/make-blas-api.py)
cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float *x, int incx, int *result);
cublasStatus_t cublasIdamax(cublasHandle_t handle, int n, const double *x, int incx, int *result);
cublasStatus_t cublasIcamax(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result);
cublasStatus_t cublasIzamax(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result);
cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float *x, int incx, int *result);
cublasStatus_t cublasIdamin(cublasHandle_t handle, int n, const double *x, int incx, int *result);
cublasStatus_t cublasIcamin(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result);
cublasStatus_t cublasIzamin(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result);
cublasStatus_t cublasSasum(cublasHandle_t handle, int n, const float *x, int incx, float *result);
cublasStatus_t cublasDasum(cublasHandle_t handle, int n, const double *x, int incx, double *result);
cublasStatus_t cublasScasum(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result);
cublasStatus_t cublasDzasum(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result);
cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y,
                           int incy);
cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y,
                           int incy);
cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx,
                           cuComplex *y, int incy);
cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x,
                           int incx, cuDoubleComplex *y, int incy);
cublasStatus_t cublasScopy(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy);
cublasStatus_t cublasDcopy(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy);
cublasStatus_t cublasCcopy(cublasHandle_t handle, int n, const cuComplex *x, int incx, cuComplex *y, int incy);
cublasStatus_t cublasZcopy(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y,
                           int incy);
cublasStatus_t cublasSdot(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy,
                          float *result);
cublasStatus_t cublasDdot(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy,
                          double *result);
cublasStatus_t cublasCdotu(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy,
                           cuComplex *result);
cublasStatus_t cublasZdotu(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,
                           int incy, cuDoubleComplex *result);
cublasStatus_t cublasCdotc(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy,
                           cuComplex *result);
cublasStatus_t cublasZdotc(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,
                           int incy, cuDoubleComplex *result);
cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, const float *x, int incx, float *result);
cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n, const double *x, int incx, double *result);
cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result);
cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result);
cublasStatus_t cublasSrot(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *c,
                          const float *s);
cublasStatus_t cublasDrot(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *c,
                          const double *s);
cublasStatus_t cublasCrot(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy,
                          const float *c, const cuComplex *s);
cublasStatus_t cublasZrot(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy,
                          const double *c, const cuDoubleComplex *s);
cublasStatus_t cublasCsrot(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy,
                           const float *c, const float *s);
cublasStatus_t cublasZdrot(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy,
                           const double *c, const double *s);
cublasStatus_t cublasSrotg(cublasHandle_t handle, float *a, float *b, float *c, float *s);
cublasStatus_t cublasDrotg(cublasHandle_t handle, double *a, double *b, double *c, double *s);
cublasStatus_t cublasCrotg(cublasHandle_t handle, cuComplex *a, cuComplex *b, float *c, cuComplex *s);
cublasStatus_t cublasZrotg(cublasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b, double *c,
                           cuDoubleComplex *s);
cublasStatus_t cublasSrotm(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *param);
cublasStatus_t cublasDrotm(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy,
                           const double *param);
cublasStatus_t cublasSrotmg(cublasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float *param);
cublasStatus_t cublasDrotmg(cublasHandle_t handle, double *d1, double *d2, double *x1, const double *y1,
                            double *param);
cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx);
cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double *alpha, double *x, int incx);
cublasStatus_t cublasCscal(cublasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx);
cublasStatus_t cublasZscal(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx);
cublasStatus_t cublasCsscal(cublasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx);
cublasStatus_t cublasZdscal(cublasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx);
cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy);
cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy);
cublasStatus_t cublasCswap(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy);
cublasStatus_t cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy);
cublasStatus_t cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku,
                           const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta,
                           float *y, int incy);
cublasStatus_t cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku,
                           const double *alpha, const double *A, int lda, const double *x, int incx,
                           const double *beta, double *y, int incy);
cublasStatus_t cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku,
                           const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx,
                           const cuComplex *beta, cuComplex *y, int incy);
cublasStatus_t cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x,
                           int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha,
                           const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);
cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha,
                           const double *A, int lda, const double *x, int incx, const double *beta, double *y,
                           int incy);
cublasStatus_t cublasCgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha,
                           const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta,
                           cuComplex *y, int incy);
cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
cublasStatus_t cublasSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha,
                                  const float *const Aarray[], int lda, const float *const xarray[], int incx,
                                  const float *beta, float *const yarray[], int incy, int batchCount);
cublasStatus_t cublasDgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha,
                                  const double *const Aarray[], int lda, const double *const xarray[], int incx,
                                  const double *beta, double *const yarray[], int incy, int batchCount);
cublasStatus_t cublasCgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                  const cuComplex *alpha, const cuComplex *const Aarray[], int lda,
                                  const cuComplex *const xarray[], int incx, const cuComplex *beta,
                                  cuComplex *const yarray[], int incy, int batchCount);
cublasStatus_t cublasZgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                  const cuDoubleComplex *alpha, const cuDoubleComplex *const Aarray[], int lda,
                                  const cuDoubleComplex *const xarray[], int incx, const cuDoubleComplex *beta,
                                  cuDoubleComplex *const yarray[], int incy, int batchCount);
cublasStatus_t cublasSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                         const float *alpha, const float *A, int lda, long long int strideA,
                                         const float *x, int incx, long long int stridex, const float *beta, float *y,
                                         int incy, long long int stridey, int batchCount);
cublasStatus_t cublasDgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                         const double *alpha, const double *A, int lda, long long int strideA,
                                         const double *x, int incx, long long int stridex, const double *beta,
                                         double *y, int incy, long long int stridey, int batchCount);
cublasStatus_t cublasCgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                         const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA,
                                         const cuComplex *x, int incx, long long int stridex, const cuComplex *beta,
                                         cuComplex *y, int incy, long long int stridey, int batchCount);
cublasStatus_t cublasZgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                         const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                         long long int strideA, const cuDoubleComplex *x, int incx,
                                         long long int stridex, const cuDoubleComplex *beta, cuDoubleComplex *y,
                                         int incy, long long int stridey, int batchCount);
cublasStatus_t cublasSger(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx,
                          const float *y, int incy, float *A, int lda);
cublasStatus_t cublasDger(cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx,
                          const double *y, int incy, double *A, int lda);
cublasStatus_t cublasCgeru(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx,
                           const cuComplex *y, int incy, cuComplex *A, int lda);
cublasStatus_t cublasZgeru(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A,
                           int lda);
cublasStatus_t cublasCgerc(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx,
                           const cuComplex *y, int incy, cuComplex *A, int lda);
cublasStatus_t cublasZgerc(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A,
                           int lda);
cublasStatus_t cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float *alpha,
                           const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);
cublasStatus_t cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double *alpha,
                           const double *A, int lda, const double *x, int incx, const double *beta, double *y,
                           int incy);
cublasStatus_t cublasSspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *AP,
                           const float *x, int incx, const float *beta, float *y, int incy);
cublasStatus_t cublasDspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *AP,
                           const double *x, int incx, const double *beta, double *y, int incy);
cublasStatus_t cublasSspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x,
                          int incx, float *AP);
cublasStatus_t cublasDspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x,
                          int incx, double *AP);
cublasStatus_t cublasSspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x,
                           int incx, const float *y, int incy, float *AP);
cublasStatus_t cublasDspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x,
                           int incx, const double *y, int incy, double *AP);
cublasStatus_t cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *A,
                           int lda, const float *x, int incx, const float *beta, float *y, int incy);
cublasStatus_t cublasDsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *A,
                           int lda, const double *x, int incx, const double *beta, double *y, int incy);
cublasStatus_t cublasCsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha,
                           const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta,
                           cuComplex *y, int incy);
cublasStatus_t cublasZsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
cublasStatus_t cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x,
                          int incx, float *A, int lda);
cublasStatus_t cublasDsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x,
                          int incx, double *A, int lda);
cublasStatus_t cublasCsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha,
                          const cuComplex *x, int incx, cuComplex *A, int lda);
cublasStatus_t cublasZsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha,
                          const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda);
cublasStatus_t cublasSsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x,
                           int incx, const float *y, int incy, float *A, int lda);
cublasStatus_t cublasDsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x,
                           int incx, const double *y, int incy, double *A, int lda);
cublasStatus_t cublasCsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha,
                           const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda);
cublasStatus_t cublasZsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A,
                           int lda);
cublasStatus_t cublasStbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx);
cublasStatus_t cublasDtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx);
cublasStatus_t cublasCtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx);
cublasStatus_t cublasZtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x,
                           int incx);
cublasStatus_t cublasStbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx);
cublasStatus_t cublasDtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx);
cublasStatus_t cublasCtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx);
cublasStatus_t cublasZtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x,
                           int incx);
cublasStatus_t cublasStpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const float *AP, float *x, int incx);
cublasStatus_t cublasDtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const double *AP, double *x, int incx);
cublasStatus_t cublasCtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx);
cublasStatus_t cublasZtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx);
cublasStatus_t cublasStpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const float *AP, float *x, int incx);
cublasStatus_t cublasDtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const double *AP, double *x, int incx);
cublasStatus_t cublasCtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx);
cublasStatus_t cublasZtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx);
cublasStatus_t cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx);
cublasStatus_t cublasDtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx);
cublasStatus_t cublasCtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx);
cublasStatus_t cublasZtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x,
                           int incx);
cublasStatus_t cublasStrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx);
cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx);
cublasStatus_t cublasCtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx);
cublasStatus_t cublasZtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x,
                           int incx);
cublasStatus_t cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha,
                           const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta,
                           cuComplex *y, int incy);
cublasStatus_t cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
cublasStatus_t cublasChbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex *alpha,
                           const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta,
                           cuComplex *y, int incy);
cublasStatus_t cublasZhbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
cublasStatus_t cublasChpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha,
                           const cuComplex *AP, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y,
                           int incy);
cublasStatus_t cublasZhpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta,
                           cuDoubleComplex *y, int incy);
cublasStatus_t cublasCher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x,
                          int incx, cuComplex *A, int lda);
cublasStatus_t cublasZher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha,
                          const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda);
cublasStatus_t cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha,
                           const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda);
cublasStatus_t cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A,
                           int lda);
cublasStatus_t cublasChpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x,
                          int incx, cuComplex *AP);
cublasStatus_t cublasZhpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha,
                          const cuDoubleComplex *x, int incx, cuDoubleComplex *AP);
cublasStatus_t cublasChpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha,
                           const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP);
cublasStatus_t cublasZhpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *AP);
cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const float *alpha, const float *A, int lda, const float *B, int ldb,
                           const float *beta, float *C, int ldc);
cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const double *alpha, const double *A, int lda, const double *B, int ldb,
                           const double *beta, double *C, int ldc);
cublasStatus_t cublasCgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb,
                           const cuComplex *beta, cuComplex *C, int ldc);
cublasStatus_t cublasZgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
                           int ldc);
cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                             int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb,
                             const cuComplex *beta, cuComplex *C, int ldc);
cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                             int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                             const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
                             int ldc);
cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                  int n, int k, const float *alpha, const float *const Aarray[], int lda,
                                  const float *const Barray[], int ldb, const float *beta, float *const Carray[],
                                  int ldc, int batchCount);
cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                  int n, int k, const double *alpha, const double *const Aarray[], int lda,
                                  const double *const Barray[], int ldb, const double *beta, double *const Carray[],
                                  int ldc, int batchCount);
cublasStatus_t cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                  int n, int k, const cuComplex *alpha, const cuComplex *const Aarray[], int lda,
                                  const cuComplex *const Barray[], int ldb, const cuComplex *beta,
                                  cuComplex *const Carray[], int ldc, int batchCount);
cublasStatus_t cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                  int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *const Aarray[],
                                  int lda, const cuDoubleComplex *const Barray[], int ldb,
                                  const cuDoubleComplex *beta, cuDoubleComplex *const Carray[], int ldc,
                                  int batchCount);
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k, const float *alpha, const float *A, int lda,
                                         long long int strideA, const float *B, int ldb, long long int strideB,
                                         const float *beta, float *C, int ldc, long long int strideC, int batchCount);
cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k, const double *alpha, const double *A, int lda,
                                         long long int strideA, const double *B, int ldb, long long int strideB,
                                         const double *beta, double *C, int ldc, long long int strideC,
                                         int batchCount);
cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
                                         long long int strideA, const cuComplex *B, int ldb, long long int strideB,
                                         const cuComplex *beta, cuComplex *C, int ldc, long long int strideC,
                                         int batchCount);
cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                         int lda, long long int strideA, const cuDoubleComplex *B, int ldb,
                                         long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C,
                                         int ldc, long long int strideC, int batchCount);
cublasStatus_t cublasSsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                           const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta,
                           float *C, int ldc);
cublasStatus_t cublasDsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                           const double *alpha, const double *A, int lda, const double *B, int ldb,
                           const double *beta, double *C, int ldc);
cublasStatus_t cublasCsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb,
                           const cuComplex *beta, cuComplex *C, int ldc);
cublasStatus_t cublasZsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                           int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
cublasStatus_t cublasChemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb,
                           const cuComplex *beta, cuComplex *C, int ldc);
cublasStatus_t cublasZhemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                           int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
cublasStatus_t cublasSsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                           const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc);
cublasStatus_t cublasDsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                           const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc);
cublasStatus_t cublasCsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                           const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, cuComplex *C,
                           int ldc);
cublasStatus_t cublasZsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
cublasStatus_t cublasCherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                           const float *alpha, const cuComplex *A, int lda, const float *beta, cuComplex *C, int ldc);
cublasStatus_t cublasZherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                           const double *alpha, const cuDoubleComplex *A, int lda, const double *beta,
                           cuDoubleComplex *C, int ldc);
cublasStatus_t cublasSsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta,
                            float *C, int ldc);
cublasStatus_t cublasDsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const double *alpha, const double *A, int lda, const double *B, int ldb,
                            const double *beta, double *C, int ldc);
cublasStatus_t cublasCsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb,
                            const cuComplex *beta, cuComplex *C, int ldc);
cublasStatus_t cublasZsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                            int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
cublasStatus_t cublasCher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb,
                            const float *beta, cuComplex *C, int ldc);
cublasStatus_t cublasZher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                            int ldb, const double *beta, cuDoubleComplex *C, int ldc);
cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta,
                            float *C, int ldc);
cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const double *alpha, const double *A, int lda, const double *B, int ldb,
                            const double *beta, double *C, int ldc);
cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb,
                            const cuComplex *beta, cuComplex *C, int ldc);
cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                            int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb,
                            const float *beta, cuComplex *C, int ldc);
cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                            const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                            int ldb, const double *beta, cuDoubleComplex *C, int ldc);
cublasStatus_t cublasStrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha,
                           const float *A, int lda, const float *B, int ldb, float *C, int ldc);
cublasStatus_t cublasDtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha,
                           const double *A, int lda, const double *B, int ldb, double *C, int ldc);
cublasStatus_t cublasCtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha,
                           const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex *C, int ldc);
cublasStatus_t cublasZtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C,
                           int ldc);
cublasStatus_t cublasStrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha,
                           const float *A, int lda, float *B, int ldb);
cublasStatus_t cublasDtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha,
                           const double *A, int lda, double *B, int ldb);
cublasStatus_t cublasCtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha,
                           const cuComplex *A, int lda, cuComplex *B, int ldb);
cublasStatus_t cublasZtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb);
cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha,
                                  const float *const Aarray[], int lda, float *const Barray[], int ldb,
                                  int batchCount);
cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha,
                                  const double *const Aarray[], int lda, double *const Barray[], int ldb,
                                  int batchCount);
cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                  const cuComplex *alpha, const cuComplex *const Aarray[], int lda,
                                  cuComplex *const Barray[], int ldb, int batchCount);
cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                  const cuDoubleComplex *alpha, const cuDoubleComplex *const Aarray[], int lda,
                                  cuDoubleComplex *const Barray[], int ldb, int batchCount);
cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb,
                           float *C, int ldc);
cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           const double *alpha, const double *A, int lda, const double *beta, const double *B,
                           int ldb, double *C, int ldc);
cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta,
                           const cuComplex *B, int ldb, cuComplex *C, int ldc);
cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C,
                           int ldc);
cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float *A, int lda,
                           const float *x, int incx, float *C, int ldc);
cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double *A, int lda,
                           const double *x, int incx, double *C, int ldc);
cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex *A, int lda,
                           const cuComplex *x, int incx, cuComplex *C, int ldc);
cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex *A,
                           int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex *C, int ldc);
// cuda4cpu-blas: end cuBLAS's routines

} // extern "C"

// cuda4cpu-blas: begin cuBLAS's _v2 names (generated by lib/make-blas-api.py)
#define cublasIsamax_v2 cublasIsamax
#define cublasIdamax_v2 cublasIdamax
#define cublasIcamax_v2 cublasIcamax
#define cublasIzamax_v2 cublasIzamax
#define cublasIsamin_v2 cublasIsamin
#define cublasIdamin_v2 cublasIdamin
#define cublasIcamin_v2 cublasIcamin
#define cublasIzamin_v2 cublasIzamin
#define cublasSasum_v2 cublasSasum
#define cublasDasum_v2 cublasDasum
#define cublasScasum_v2 cublasScasum
#define cublasDzasum_v2 cublasDzasum
#define cublasSaxpy_v2 cublasSaxpy
#define cublasDaxpy_v2 cublasDaxpy
#define cublasCaxpy_v2 cublasCaxpy
#define cublasZaxpy_v2 cublasZaxpy
#define cublasScopy_v2 cublasScopy
#define cublasDcopy_v2 cublasDcopy
#define cublasCcopy_v2 cublasCcopy
#define cublasZcopy_v2 cublasZcopy
#define cublasSdot_v2 cublasSdot
#define cublasDdot_v2 cublasDdot
#define cublasCdotu_v2 cublasCdotu
#define cublasZdotu_v2 cublasZdotu
#define cublasCdotc_v2 cublasCdotc
#define cublasZdotc_v2 cublasZdotc
#define cublasSnrm2_v2 cublasSnrm2
#define cublasDnrm2_v2 cublasDnrm2
#define cublasScnrm2_v2 cublasScnrm2
#define cublasDznrm2_v2 cublasDznrm2
#define cublasSrot_v2 cublasSrot
#define cublasDrot_v2 cublasDrot
#define cublasCrot_v2 cublasCrot
#define cublasZrot_v2 cublasZrot
#define cublasCsrot_v2 cublasCsrot
#define cublasZdrot_v2 cublasZdrot
#define cublasSrotg_v2 cublasSrotg
#define cublasDrotg_v2 cublasDrotg
#define cublasCrotg_v2 cublasCrotg
#define cublasZrotg_v2 cublasZrotg
#define cublasSrotm_v2 cublasSrotm
#define cublasDrotm_v2 cublasDrotm
#define cublasSrotmg_v2 cublasSrotmg
#define cublasDrotmg_v2 cublasDrotmg
#define cublasSscal_v2 cublasSscal
#define cublasDscal_v2 cublasDscal
#define cublasCscal_v2 cublasCscal
#define cublasZscal_v2 cublasZscal
#define cublasCsscal_v2 cublasCsscal
#define cublasZdscal_v2 cublasZdscal
#define cublasSswap_v2 cublasSswap
#define cublasDswap_v2 cublasDswap
#define cublasCswap_v2 cublasCswap
#define cublasZswap_v2 cublasZswap
#define cublasSgbmv_v2 cublasSgbmv
#define cublasDgbmv_v2 cublasDgbmv
#define cublasCgbmv_v2 cublasCgbmv
#define cublasZgbmv_v2 cublasZgbmv
#define cublasSgemv_v2 cublasSgemv
#define cublasDgemv_v2 cublasDgemv
#define cublasCgemv_v2 cublasCgemv
#define cublasZgemv_v2 cublasZgemv
#define cublasSger_v2 cublasSger
#define cublasDger_v2 cublasDger
#define cublasCgeru_v2 cublasCgeru
#define cublasZgeru_v2 cublasZgeru
#define cublasCgerc_v2 cublasCgerc
#define cublasZgerc_v2 cublasZgerc
#define cublasSsbmv_v2 cublasSsbmv
#define cublasDsbmv_v2 cublasDsbmv
#define cublasSspmv_v2 cublasSspmv
#define cublasDspmv_v2 cublasDspmv
#define cublasSspr_v2 cublasSspr
#define cublasDspr_v2 cublasDspr
#define cublasSspr2_v2 cublasSspr2
#define cublasDspr2_v2 cublasDspr2
#define cublasSsymv_v2 cublasSsymv
#define cublasDsymv_v2 cublasDsymv
#define cublasCsymv_v2 cublasCsymv
#define cublasZsymv_v2 cublasZsymv
#define cublasSsyr_v2 cublasSsyr
#define cublasDsyr_v2 cublasDsyr
#define cublasCsyr_v2 cublasCsyr
#define cublasZsyr_v2 cublasZsyr
#define cublasSsyr2_v2 cublasSsyr2
#define cublasDsyr2_v2 cublasDsyr2
#define cublasCsyr2_v2 cublasCsyr2
#define cublasZsyr2_v2 cublasZsyr2
#define cublasStbmv_v2 cublasStbmv
#define cublasDtbmv_v2 cublasDtbmv
#define cublasCtbmv_v2 cublasCtbmv
#define cublasZtbmv_v2 cublasZtbmv
#define cublasStbsv_v2 cublasStbsv
#define cublasDtbsv_v2 cublasDtbsv
#define cublasCtbsv_v2 cublasCtbsv
#define cublasZtbsv_v2 cublasZtbsv
#define cublasStpmv_v2 cublasStpmv
#define cublasDtpmv_v2 cublasDtpmv
#define cublasCtpmv_v2 cublasCtpmv
#define cublasZtpmv_v2 cublasZtpmv
#define cublasStpsv_v2 cublasStpsv
#define cublasDtpsv_v2 cublasDtpsv
#define cublasCtpsv_v2 cublasCtpsv
#define cublasZtpsv_v2 cublasZtpsv
#define cublasStrmv_v2 cublasStrmv
#define cublasDtrmv_v2 cublasDtrmv
#define cublasCtrmv_v2 cublasCtrmv
#define cublasZtrmv_v2 cublasZtrmv
#define cublasStrsv_v2 cublasStrsv
#define cublasDtrsv_v2 cublasDtrsv
#define cublasCtrsv_v2 cublasCtrsv
#define cublasZtrsv_v2 cublasZtrsv
#define cublasChemv_v2 cublasChemv
#define cublasZhemv_v2 cublasZhemv
#define cublasChbmv_v2 cublasChbmv
#define cublasZhbmv_v2 cublasZhbmv
#define cublasChpmv_v2 cublasChpmv
#define cublasZhpmv_v2 cublasZhpmv
#define cublasCher_v2 cublasCher
#define cublasZher_v2 cublasZher
#define cublasCher2_v2 cublasCher2
#define cublasZher2_v2 cublasZher2
#define cublasChpr_v2 cublasChpr
#define cublasZhpr_v2 cublasZhpr
#define cublasChpr2_v2 cublasChpr2
#define cublasZhpr2_v2 cublasZhpr2
#define cublasSgemm_v2 cublasSgemm
#define cublasDgemm_v2 cublasDgemm
#define cublasCgemm_v2 cublasCgemm
#define cublasZgemm_v2 cublasZgemm
#define cublasSsymm_v2 cublasSsymm
#define cublasDsymm_v2 cublasDsymm
#define cublasCsymm_v2 cublasCsymm
#define cublasZsymm_v2 cublasZsymm
#define cublasChemm_v2 cublasChemm
#define cublasZhemm_v2 cublasZhemm
#define cublasSsyrk_v2 cublasSsyrk
#define cublasDsyrk_v2 cublasDsyrk
#define cublasCsyrk_v2 cublasCsyrk
#define cublasZsyrk_v2 cublasZsyrk
#define cublasCherk_v2 cublasCherk
#define cublasZherk_v2 cublasZherk
#define cublasSsyr2k_v2 cublasSsyr2k
#define cublasDsyr2k_v2 cublasDsyr2k
#define cublasCsyr2k_v2 cublasCsyr2k
#define cublasZsyr2k_v2 cublasZsyr2k
#define cublasCher2k_v2 cublasCher2k
#define cublasZher2k_v2 cublasZher2k
#define cublasStrmm_v2 cublasStrmm
#define cublasDtrmm_v2 cublasDtrmm
#define cublasCtrmm_v2 cublasCtrmm
#define cublasZtrmm_v2 cublasZtrmm
#define cublasStrsm_v2 cublasStrsm
#define cublasDtrsm_v2 cublasDtrsm
#define cublasCtrsm_v2 cublasCtrsm
#define cublasZtrsm_v2 cublasZtrsm
// cuda4cpu-blas: end cuBLAS's _v2 names

// The management functions' _v2 names
#define cublasCreate_v2 cublasCreate
#define cublasDestroy_v2 cublasDestroy
#define cublasGetVersion_v2 cublasGetVersion
#define cublasSetStream_v2 cublasSetStream
#define cublasGetStream_v2 cublasGetStream
#define cublasGetPointerMode_v2 cublasGetPointerMode
#define cublasSetPointerMode_v2 cublasSetPointerMode

// cuBLAS 10's computeType, a cudaDataType (CUDA_R_32F, ...), which code
// still passes: cuBLAS's own header has these overloads too
inline cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                   int n, int k, const void *alpha, const void *A, cudaDataType Atype, int lda,
                                   const void *B, cudaDataType Btype, int ldb, const void *beta, void *C,
                                   cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo)
{
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc,
                        static_cast<cublasComputeType_t>(computeType), algo);
}

inline cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k, const void *alpha, const void *const Aarray[],
                                          cudaDataType Atype, int lda, const void *const Barray[],
                                          cudaDataType Btype, int ldb, const void *beta, void *const Carray[],
                                          cudaDataType Ctype, int ldc, int batchCount, cudaDataType computeType,
                                          cublasGemmAlgo_t algo)
{
    return cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta,
                               Carray, Ctype, ldc, batchCount, static_cast<cublasComputeType_t>(computeType), algo);
}

inline cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                                 cublasOperation_t transb, int m, int n, int k, const void *alpha,
                                                 const void *A, cudaDataType Atype, int lda, long long int strideA,
                                                 const void *B, cudaDataType Btype, int ldb, long long int strideB,
                                                 const void *beta, void *C, cudaDataType Ctype, int ldc,
                                                 long long int strideC, int batchCount, cudaDataType computeType,
                                                 cublasGemmAlgo_t algo)
{
    return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb,
                                      strideB, beta, C, Ctype, ldc, strideC, batchCount,
                                      static_cast<cublasComputeType_t>(computeType), algo);
}
