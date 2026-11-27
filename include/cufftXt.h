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

// cuFFT's extensions (cufftXt.h): callbacks, functions that a transform
// calls for each element it loads or stores. On the GPU, they're device
// functions whose address a program copies from a __device__ variable
// (cudaMemcpyFromSymbol); here, those are the functions. Multiple GPUs
// (cufftXtSetGPUs, descriptors) aren't provided.
#pragma once

#include "cufft.h"

extern "C" {

typedef enum cufftXtCallbackType_t {
    CUFFT_CB_LD_COMPLEX        = 0x0,
    CUFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    CUFFT_CB_LD_REAL           = 0x2,
    CUFFT_CB_LD_REAL_DOUBLE    = 0x3,
    CUFFT_CB_ST_COMPLEX        = 0x4,
    CUFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    CUFFT_CB_ST_REAL           = 0x6,
    CUFFT_CB_ST_REAL_DOUBLE    = 0x7,
    CUFFT_CB_UNDEFINED         = 0x8
} cufftXtCallbackType;

typedef cufftComplex (*cufftCallbackLoadC)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftDoubleComplex (*cufftCallbackLoadZ)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftReal (*cufftCallbackLoadR)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftDoubleReal (*cufftCallbackLoadD)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef void (*cufftCallbackStoreC)(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                    void *sharedPointer);
typedef void (*cufftCallbackStoreZ)(void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo,
                                    void *sharedPointer);
typedef void (*cufftCallbackStoreR)(void *dataOut, size_t offset, cufftReal element, void *callerInfo,
                                    void *sharedPointer);
typedef void (*cufftCallbackStoreD)(void *dataOut, size_t offset, cufftDoubleReal element, void *callerInfo,
                                    void *sharedPointer);

cufftResult cufftXtSetCallback(cufftHandle plan, void **callback_routine, cufftXtCallbackType cbType,
                               void **caller_info);
cufftResult cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType);
cufftResult cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType, size_t sharedSize);

} // extern "C"
