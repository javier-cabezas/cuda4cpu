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

// hipRAND on HIP's NVIDIA platform: cuRAND's host API with HIP's names.
// Every name is a #define for cuRAND's (hiprandCreateGenerator is
// curandCreateGenerator), or a typedef of its type.
#pragma once

#include "../curand.h"
#include "../hip/hip_runtime_api.h"

#define HIPRAND_VERSION CURAND_VERSION

typedef curandStatus_t hiprandStatus_t;
typedef curandStatus_t hiprandStatus;
typedef curandRngType_t hiprandRngType_t;
typedef curandRngType_t hiprandRngType;
typedef curandOrdering_t hiprandOrdering_t;
typedef curandOrdering_t hiprandOrdering;
typedef curandDirectionVectorSet_t hiprandDirectionVectorSet_t;
typedef curandGenerator_t hiprandGenerator_t;
typedef curandDiscreteDistribution_t hiprandDiscreteDistribution_t;
typedef curandDirectionVectors32_t hiprandDirectionVectors32_t;
typedef curandDirectionVectors64_t hiprandDirectionVectors64_t;
#define HIPRAND_STATUS_SUCCESS CURAND_STATUS_SUCCESS
#define HIPRAND_STATUS_VERSION_MISMATCH CURAND_STATUS_VERSION_MISMATCH
#define HIPRAND_STATUS_NOT_INITIALIZED CURAND_STATUS_NOT_INITIALIZED
#define HIPRAND_STATUS_ALLOCATION_FAILED CURAND_STATUS_ALLOCATION_FAILED
#define HIPRAND_STATUS_TYPE_ERROR CURAND_STATUS_TYPE_ERROR
#define HIPRAND_STATUS_OUT_OF_RANGE CURAND_STATUS_OUT_OF_RANGE
#define HIPRAND_STATUS_LENGTH_NOT_MULTIPLE CURAND_STATUS_LENGTH_NOT_MULTIPLE
#define HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED CURAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define HIPRAND_STATUS_LAUNCH_FAILURE CURAND_STATUS_LAUNCH_FAILURE
#define HIPRAND_STATUS_PREEXISTING_FAILURE CURAND_STATUS_PREEXISTING_FAILURE
#define HIPRAND_STATUS_INITIALIZATION_FAILED CURAND_STATUS_INITIALIZATION_FAILED
#define HIPRAND_STATUS_ARCH_MISMATCH CURAND_STATUS_ARCH_MISMATCH
#define HIPRAND_STATUS_INTERNAL_ERROR CURAND_STATUS_INTERNAL_ERROR

#define HIPRAND_RNG_TEST CURAND_RNG_TEST
#define HIPRAND_RNG_PSEUDO_DEFAULT CURAND_RNG_PSEUDO_DEFAULT
#define HIPRAND_RNG_PSEUDO_XORWOW CURAND_RNG_PSEUDO_XORWOW
#define HIPRAND_RNG_PSEUDO_MRG32K3A CURAND_RNG_PSEUDO_MRG32K3A
#define HIPRAND_RNG_PSEUDO_MTGP32 CURAND_RNG_PSEUDO_MTGP32
#define HIPRAND_RNG_PSEUDO_MT19937 CURAND_RNG_PSEUDO_MT19937
#define HIPRAND_RNG_PSEUDO_PHILOX4_32_10 CURAND_RNG_PSEUDO_PHILOX4_32_10
#define HIPRAND_RNG_QUASI_DEFAULT CURAND_RNG_QUASI_DEFAULT
#define HIPRAND_RNG_QUASI_SOBOL32 CURAND_RNG_QUASI_SOBOL32
#define HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
#define HIPRAND_RNG_QUASI_SOBOL64 CURAND_RNG_QUASI_SOBOL64
#define HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64 CURAND_RNG_QUASI_SCRAMBLED_SOBOL64

#define HIPRAND_ORDERING_PSEUDO_BEST CURAND_ORDERING_PSEUDO_BEST
#define HIPRAND_ORDERING_PSEUDO_DEFAULT CURAND_ORDERING_PSEUDO_DEFAULT
#define HIPRAND_ORDERING_PSEUDO_SEEDED CURAND_ORDERING_PSEUDO_SEEDED
#define HIPRAND_ORDERING_PSEUDO_LEGACY CURAND_ORDERING_PSEUDO_LEGACY
#define HIPRAND_ORDERING_PSEUDO_DYNAMIC CURAND_ORDERING_PSEUDO_DYNAMIC
#define HIPRAND_ORDERING_QUASI_DEFAULT CURAND_ORDERING_QUASI_DEFAULT

#define HIPRAND_DIRECTION_VECTORS_32_JOEKUO6 CURAND_DIRECTION_VECTORS_32_JOEKUO6
#define HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
#define HIPRAND_DIRECTION_VECTORS_64_JOEKUO6 CURAND_DIRECTION_VECTORS_64_JOEKUO6
#define HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6

#define hiprandCreateGenerator curandCreateGenerator
#define hiprandCreateGeneratorHost curandCreateGeneratorHost
#define hiprandDestroyGenerator curandDestroyGenerator
#define hiprandGetVersion curandGetVersion
#define hiprandSetStream curandSetStream
#define hiprandSetPseudoRandomGeneratorSeed curandSetPseudoRandomGeneratorSeed
#define hiprandSetGeneratorOffset curandSetGeneratorOffset
#define hiprandSetGeneratorOrdering curandSetGeneratorOrdering
#define hiprandSetQuasiRandomGeneratorDimensions curandSetQuasiRandomGeneratorDimensions
#define hiprandGenerate curandGenerate
#define hiprandGenerateLongLong curandGenerateLongLong
#define hiprandGenerateUniform curandGenerateUniform
#define hiprandGenerateUniformDouble curandGenerateUniformDouble
#define hiprandGenerateNormal curandGenerateNormal
#define hiprandGenerateNormalDouble curandGenerateNormalDouble
#define hiprandGenerateLogNormal curandGenerateLogNormal
#define hiprandGenerateLogNormalDouble curandGenerateLogNormalDouble
#define hiprandGeneratePoisson curandGeneratePoisson
#define hiprandGenerateSeeds curandGenerateSeeds
#define hiprandGetDirectionVectors32 curandGetDirectionVectors32
#define hiprandGetDirectionVectors64 curandGetDirectionVectors64
#define hiprandCreatePoissonDistribution curandCreatePoissonDistribution
#define hiprandDestroyDistribution curandDestroyDistribution

// hipRAND's scramble constants are const
inline hiprandStatus_t hiprandGetScrambleConstants32(const unsigned int **constants)
{
    return curandGetScrambleConstants32(const_cast<unsigned int **>(constants));
}

inline hiprandStatus_t hiprandGetScrambleConstants32(unsigned int **constants)
{
    return curandGetScrambleConstants32(constants);
}

inline hiprandStatus_t hiprandGetScrambleConstants64(const unsigned long long **constants)
{
    return curandGetScrambleConstants64(const_cast<unsigned long long **>(constants));
}

inline hiprandStatus_t hiprandGetScrambleConstants64(unsigned long long **constants)
{
    return curandGetScrambleConstants64(constants);
}
