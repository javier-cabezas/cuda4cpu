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

// CUDA's library_types.h: the types that CUDA's libraries share, at global
// scope as in CUDA.
#pragma once

typedef enum libraryPropertyType_t {
    MAJOR_VERSION,
    MINOR_VERSION,
    PATCH_LEVEL,
} libraryPropertyType;

typedef enum cudaDataType_t {
    CUDA_R_32F  = 0,
    CUDA_R_64F  = 1,
    CUDA_R_16F  = 2,
    CUDA_R_8I   = 3,
    CUDA_C_32F  = 4,
    CUDA_C_64F  = 5,
    CUDA_C_16F  = 6,
    CUDA_C_8I   = 7,
    CUDA_R_8U   = 8,
    CUDA_C_8U   = 9,
    CUDA_R_32I  = 10,
    CUDA_C_32I  = 11,
    CUDA_R_32U  = 12,
    CUDA_C_32U  = 13,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_4I   = 16,
    CUDA_C_4I   = 17,
    CUDA_R_4U   = 18,
    CUDA_C_4U   = 19,
    CUDA_R_16I  = 20,
    CUDA_C_16I  = 21,
    CUDA_R_16U  = 22,
    CUDA_C_16U  = 23,
    CUDA_R_64I  = 24,
    CUDA_C_64I  = 25,
    CUDA_R_64U  = 26,
    CUDA_C_64U  = 27,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29,
} cudaDataType;
