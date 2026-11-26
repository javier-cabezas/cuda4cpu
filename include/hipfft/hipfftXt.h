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

// hipFFT's extensions: callbacks (see cufftXt.h). Multiple GPUs
// (hipfftXtSetGPUs, descriptors) aren't provided.
#pragma once

#include "hipfft.h"

extern "C" {

typedef enum hipfftXtCallbackType_t {
    HIPFFT_CB_LD_COMPLEX        = 0x0,
    HIPFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    HIPFFT_CB_LD_REAL           = 0x2,
    HIPFFT_CB_LD_REAL_DOUBLE    = 0x3,
    HIPFFT_CB_ST_COMPLEX        = 0x4,
    HIPFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    HIPFFT_CB_ST_REAL           = 0x6,
    HIPFFT_CB_ST_REAL_DOUBLE    = 0x7,
    HIPFFT_CB_UNDEFINED         = 0x8
} hipfftXtCallbackType;

typedef hipfftComplex (*hipfftCallbackLoadC)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef hipfftDoubleComplex (*hipfftCallbackLoadZ)(void *dataIn, size_t offset, void *callerInfo,
                                                   void *sharedPointer);
typedef hipfftReal (*hipfftCallbackLoadR)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef hipfftDoubleReal (*hipfftCallbackLoadD)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef void (*hipfftCallbackStoreC)(void *dataOut, size_t offset, hipfftComplex element, void *callerInfo,
                                     void *sharedPointer);
typedef void (*hipfftCallbackStoreZ)(void *dataOut, size_t offset, hipfftDoubleComplex element, void *callerInfo,
                                     void *sharedPointer);
typedef void (*hipfftCallbackStoreR)(void *dataOut, size_t offset, hipfftReal element, void *callerInfo,
                                     void *sharedPointer);
typedef void (*hipfftCallbackStoreD)(void *dataOut, size_t offset, hipfftDoubleReal element, void *callerInfo,
                                     void *sharedPointer);

hipfftResult hipfftXtSetCallback(hipfftHandle plan, void **callbacks, hipfftXtCallbackType cbtype,
                                 void **callbackData);
hipfftResult hipfftXtClearCallback(hipfftHandle plan, hipfftXtCallbackType cbtype);
hipfftResult hipfftXtSetCallbackSharedSize(hipfftHandle plan, hipfftXtCallbackType cbtype, size_t sharedSize);

} // extern "C"
