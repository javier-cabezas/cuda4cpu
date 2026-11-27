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

// cuFFT's API: plans of fast Fourier transforms in one to three dimensions,
// complex to complex, real to complex and complex to real, single and double
// precision, in batches, with the basic and the advanced data layouts. The
// functions are in the library, with C linkage, as cuFFT's; transforms run
// on pocketfft (lib/third_party/pocketfft), when executed, or when the graph
// runs that a stream captured them in. Unnormalized, as cuFFT's: a forward
// and an inverse transform multiply by the number of elements.
#pragma once

#ifndef _CUFFT_H_
#define _CUFFT_H_
#endif

#include <cstddef>

#include "cuComplex.h"
#include "cuda_runtime.h"
#include "library_types.h"

#define CUFFT_VER_MAJOR 11
#define CUFFT_VER_MINOR 0
#define CUFFT_VER_PATCH 0
#define CUFFT_VER_BUILD 21
#define CUFFT_VERSION 11000

#define CUFFT_FORWARD -1
#define CUFFT_INVERSE 1

extern "C" {

typedef enum cufftResult_t {
    CUFFT_SUCCESS                   = 0x0,
    CUFFT_INVALID_PLAN              = 0x1,
    CUFFT_ALLOC_FAILED              = 0x2,
    CUFFT_INVALID_TYPE              = 0x3,
    CUFFT_INVALID_VALUE             = 0x4,
    CUFFT_INTERNAL_ERROR            = 0x5,
    CUFFT_EXEC_FAILED               = 0x6,
    CUFFT_SETUP_FAILED              = 0x7,
    CUFFT_INVALID_SIZE              = 0x8,
    CUFFT_UNALIGNED_DATA            = 0x9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
    CUFFT_INVALID_DEVICE            = 0xB,
    CUFFT_PARSE_ERROR               = 0xC,
    CUFFT_NO_WORKSPACE              = 0xD,
    CUFFT_NOT_IMPLEMENTED           = 0xE,
    CUFFT_LICENSE_ERROR             = 0x0F,
    CUFFT_NOT_SUPPORTED             = 0x10,
    CUFFT_MISSING_DEPENDENCY        = 0x11,
    CUFFT_NVRTC_FAILURE             = 0x12,
    CUFFT_NVJITLINK_FAILURE         = 0x13,
    CUFFT_NVSHMEM_FAILURE           = 0x14
} cufftResult;

typedef float cufftReal;
typedef double cufftDoubleReal;
typedef cuComplex cufftComplex;
typedef cuDoubleComplex cufftDoubleComplex;

typedef enum cufftType_t {
    CUFFT_R2C = 0x2a,
    CUFFT_C2R = 0x2c,
    CUFFT_C2C = 0x29,
    CUFFT_D2Z = 0x6a,
    CUFFT_Z2D = 0x6c,
    CUFFT_Z2Z = 0x69
} cufftType;

typedef enum cufftCompatibility_t { CUFFT_COMPATIBILITY_FFTW_PADDING = 0x01 } cufftCompatibility;
#define CUFFT_COMPATIBILITY_DEFAULT CUFFT_COMPATIBILITY_FFTW_PADDING

typedef int cufftHandle;

cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch);
cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type);
cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type);
cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed,
                          int ostride, int odist, cufftType type, int batch);

cufftResult cufftCreate(cufftHandle *handle);
cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t *workSize);
cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t *workSize);
cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t *workSize);
cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed, int istride, int idist,
                              int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workSize);
cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed,
                                long long int istride, long long int idist, long long int *onembed,
                                long long int ostride, long long int odist, cufftType type, long long int batch,
                                size_t *workSize);

cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t *workSize);
cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t *workSize);
cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t *workSize);
cufftResult cufftEstimateMany(int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride,
                              int odist, cufftType type, int batch, size_t *workSize);
cufftResult cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t *workSize);
cufftResult cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t *workSize);
cufftResult cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t *workSize);
cufftResult cufftGetSizeMany(cufftHandle handle, int rank, int *n, int *inembed, int istride, int idist,
                             int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workArea);
cufftResult cufftGetSizeMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed,
                               long long int istride, long long int idist, long long int *onembed,
                               long long int ostride, long long int odist, cufftType type, long long int batch,
                               size_t *workSize);
cufftResult cufftGetSize(cufftHandle handle, size_t *workSize);
cufftResult cufftSetWorkArea(cufftHandle plan, void *workArea);
cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate);

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction);
cufftResult cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata);
cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftReal *odata);
cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata, int direction);
cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata, cufftDoubleComplex *odata);
cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleReal *odata);

cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream);
cufftResult cufftDestroy(cufftHandle plan);
cufftResult cufftGetVersion(int *version);
cufftResult cufftGetProperty(libraryPropertyType type, int *value);

} // extern "C"
