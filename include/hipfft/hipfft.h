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

// hipFFT's API on cuda4cpu's FFTs (cufft.h's): the same plans and
// transforms, with HIP's names and types (hipfftComplex is hipComplex), as on
// HIP's NVIDIA platform, where hipFFT is cuFFT. The functions are in the
// library, with C linkage.
#pragma once

#include <cstddef>

#include "../hip/hip_complex.h"
#include "../hip/hip_runtime_api.h"

#define hipfftVersionMajor 1
#define hipfftVersionMinor 0
#define hipfftVersionPatch 21
#define hipfftVersionTweak 0
#define HIPFFT_VERSION (hipfftVersionMajor * 10000 + hipfftVersionMinor * 100 + hipfftVersionPatch)

#define HIPFFT_FORWARD -1
#define HIPFFT_BACKWARD 1

extern "C" {

typedef enum hipfftResult_t {
    HIPFFT_SUCCESS                   = 0,
    HIPFFT_INVALID_PLAN              = 1,
    HIPFFT_ALLOC_FAILED              = 2,
    HIPFFT_INVALID_TYPE              = 3,
    HIPFFT_INVALID_VALUE             = 4,
    HIPFFT_INTERNAL_ERROR            = 5,
    HIPFFT_EXEC_FAILED               = 6,
    HIPFFT_SETUP_FAILED              = 7,
    HIPFFT_INVALID_SIZE              = 8,
    HIPFFT_UNALIGNED_DATA            = 9,
    HIPFFT_INCOMPLETE_PARAMETER_LIST = 10,
    HIPFFT_INVALID_DEVICE            = 11,
    HIPFFT_PARSE_ERROR               = 12,
    HIPFFT_NO_WORKSPACE              = 13,
    HIPFFT_NOT_IMPLEMENTED           = 14,
    HIPFFT_NOT_SUPPORTED             = 16
} hipfftResult;

typedef enum hipfftType_t {
    HIPFFT_R2C = 0x2a,
    HIPFFT_C2R = 0x2c,
    HIPFFT_C2C = 0x29,
    HIPFFT_D2Z = 0x6a,
    HIPFFT_Z2D = 0x6c,
    HIPFFT_Z2Z = 0x69
} hipfftType;

typedef enum hipfftLibraryPropertyType_t {
    HIPFFT_MAJOR_VERSION,
    HIPFFT_MINOR_VERSION,
    HIPFFT_PATCH_LEVEL
} hipfftLibraryPropertyType;

typedef float hipfftReal;
typedef double hipfftDoubleReal;
typedef hipComplex hipfftComplex;
typedef hipDoubleComplex hipfftDoubleComplex;

struct hipfftHandle_t;
typedef struct hipfftHandle_t *hipfftHandle;

hipfftResult hipfftPlan1d(hipfftHandle *plan, int nx, hipfftType type, int batch);
hipfftResult hipfftPlan2d(hipfftHandle *plan, int nx, int ny, hipfftType type);
hipfftResult hipfftPlan3d(hipfftHandle *plan, int nx, int ny, int nz, hipfftType type);
hipfftResult hipfftPlanMany(hipfftHandle *plan, int rank, int *n, int *inembed, int istride, int idist,
                            int *onembed, int ostride, int odist, hipfftType type, int batch);

hipfftResult hipfftCreate(hipfftHandle *plan);
hipfftResult hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t *workSize);
hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t *workSize);
hipfftResult hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t *workSize);
hipfftResult hipfftMakePlanMany(hipfftHandle plan, int rank, int *n, int *inembed, int istride, int idist,
                                int *onembed, int ostride, int odist, hipfftType type, int batch, size_t *workSize);
hipfftResult hipfftMakePlanMany64(hipfftHandle plan, int rank, long long int *n, long long int *inembed,
                                  long long int istride, long long int idist, long long int *onembed,
                                  long long int ostride, long long int odist, hipfftType type,
                                  long long int batch, size_t *workSize);

hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t *workSize);
hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t *workSize);
hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t *workSize);
hipfftResult hipfftEstimateMany(int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride,
                                int odist, hipfftType type, int batch, size_t *workSize);
hipfftResult hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t *workSize);
hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t *workSize);
hipfftResult hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t *workSize);
hipfftResult hipfftGetSizeMany(hipfftHandle plan, int rank, int *n, int *inembed, int istride, int idist,
                               int *onembed, int ostride, int odist, hipfftType type, int batch, size_t *workSize);
hipfftResult hipfftGetSizeMany64(hipfftHandle plan, int rank, long long int *n, long long int *inembed,
                                 long long int istride, long long int idist, long long int *onembed,
                                 long long int ostride, long long int odist, hipfftType type, long long int batch,
                                 size_t *workSize);
hipfftResult hipfftGetSize(hipfftHandle plan, size_t *workSize);
hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate);
hipfftResult hipfftSetWorkArea(hipfftHandle plan, void *workArea);

hipfftResult hipfftExecC2C(hipfftHandle plan, hipfftComplex *idata, hipfftComplex *odata, int direction);
hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal *idata, hipfftComplex *odata);
hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex *idata, hipfftReal *odata);
hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex *idata, hipfftDoubleComplex *odata,
                           int direction);
hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal *idata, hipfftDoubleComplex *odata);
hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex *idata, hipfftDoubleReal *odata);

hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream);
hipfftResult hipfftDestroy(hipfftHandle plan);
hipfftResult hipfftGetVersion(int *version);
hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int *value);

} // extern "C"
