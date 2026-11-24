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

// cuRAND's host API: generators that fill arrays (in any memory, since the
// device is the CPU) with pseudorandom and quasirandom numbers, and the
// tables the device API (curand_kernel.h) takes. The functions are in the
// library, with C linkage, as cuRAND's are.
//
// The integers and uniform floats of XORWOW and Philox4_32_10 are CUDA's,
// bit for bit, with the default and legacy orderings; the other results are
// distributed as CUDA's but are not the same numbers (docs/features.md).
#pragma once

// cuRAND's include guard, which code (the CUDA samples' helper_cuda.h) tests
// before using its types
#ifndef CURAND_H_
#define CURAND_H_
#endif

#include <cstddef>

#include "cuda_runtime.h"
#include "library_types.h"

#define CURAND_VER_MAJOR 10
#define CURAND_VER_MINOR 3
#define CURAND_VER_PATCH 1
#define CURAND_VER_BUILD 0
#define CURAND_VERSION (CURAND_VER_MAJOR * 1000 + CURAND_VER_MINOR * 100 + CURAND_VER_PATCH)

extern "C" {

enum curandStatus {
    CURAND_STATUS_SUCCESS                   = 0,
    CURAND_STATUS_VERSION_MISMATCH          = 100,
    CURAND_STATUS_NOT_INITIALIZED           = 101,
    CURAND_STATUS_ALLOCATION_FAILED         = 102,
    CURAND_STATUS_TYPE_ERROR                = 103,
    CURAND_STATUS_OUT_OF_RANGE              = 104,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE       = 105,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    CURAND_STATUS_LAUNCH_FAILURE            = 201,
    CURAND_STATUS_PREEXISTING_FAILURE       = 202,
    CURAND_STATUS_INITIALIZATION_FAILED     = 203,
    CURAND_STATUS_ARCH_MISMATCH             = 204,
    CURAND_STATUS_INTERNAL_ERROR            = 999,
};
typedef enum curandStatus curandStatus_t;

enum curandRngType {
    CURAND_RNG_TEST                    = 0,
    CURAND_RNG_PSEUDO_DEFAULT          = 100,
    CURAND_RNG_PSEUDO_XORWOW           = 101,
    CURAND_RNG_PSEUDO_MRG32K3A         = 121,
    CURAND_RNG_PSEUDO_MTGP32           = 141,
    CURAND_RNG_PSEUDO_MT19937          = 142,
    CURAND_RNG_PSEUDO_PHILOX4_32_10    = 161,
    CURAND_RNG_QUASI_DEFAULT           = 200,
    CURAND_RNG_QUASI_SOBOL32           = 201,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
    CURAND_RNG_QUASI_SOBOL64           = 203,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204,
};
typedef enum curandRngType curandRngType_t;

enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST    = 100,
    CURAND_ORDERING_PSEUDO_DEFAULT = 101,
    CURAND_ORDERING_PSEUDO_SEEDED  = 102,
    CURAND_ORDERING_PSEUDO_LEGACY  = 103,
    CURAND_ORDERING_PSEUDO_DYNAMIC = 104,
    CURAND_ORDERING_QUASI_DEFAULT  = 201,
};
typedef enum curandOrdering curandOrdering_t;

enum curandDirectionVectorSet {
    CURAND_DIRECTION_VECTORS_32_JOEKUO6           = 101,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
    CURAND_DIRECTION_VECTORS_64_JOEKUO6           = 103,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104,
};
typedef enum curandDirectionVectorSet curandDirectionVectorSet_t;

enum curandMethod {
    CURAND_CHOOSE_BEST = 0,
    CURAND_ITR         = 1,
    CURAND_KNUTH       = 2,
    CURAND_HITR        = 3,
    CURAND_M1          = 4,
    CURAND_M2          = 5,
    CURAND_BINARY_SEARCH = 6,
    CURAND_DISCRETE_GAUSS = 7,
    CURAND_REJECTION   = 8,
    CURAND_DEVICE_API  = 9,
    CURAND_FAST_REJECTION = 10,
    CURAND_3RD         = 11,
    CURAND_DEFINITION  = 12,
    CURAND_POISSON     = 13,
};
typedef enum curandMethod curandMethod_t;

//! The 32 (or 64) direction vectors of one dimension of a Sobol' sequence
typedef unsigned int curandDirectionVectors32_t[32];
typedef unsigned long long curandDirectionVectors64_t[64];

struct curandGenerator_st;
typedef struct curandGenerator_st *curandGenerator_t;

//! A discrete distribution, sampled by inverting its cumulative distribution
//! function: probability is the CDF of the values from shift on, which
//! curand_discrete() searches
struct curandDiscreteDistribution_st {
    double *probability;
    unsigned int length;
    unsigned int shift;
    double lambda;
};
typedef struct curandDiscreteDistribution_st *curandDiscreteDistribution_t;

curandStatus_t curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type);
curandStatus_t curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type);
curandStatus_t curandDestroyGenerator(curandGenerator_t generator);
curandStatus_t curandGetVersion(int *version);
curandStatus_t curandGetProperty(libraryPropertyType type, int *value);
curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream);
curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed);
curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset);
curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order);
curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions);
curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int *output_ptr, size_t num);
curandStatus_t curandGenerateLongLong(curandGenerator_t generator, unsigned long long *output_ptr, size_t num);
curandStatus_t curandGenerateUniform(curandGenerator_t generator, float *output_ptr, size_t num);
curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double *output_ptr, size_t num);
curandStatus_t curandGenerateNormal(curandGenerator_t generator, float *output_ptr, size_t n, float mean,
                                    float stddev);
curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double *output_ptr, size_t n, double mean,
                                          double stddev);
curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float *output_ptr, size_t n, float mean,
                                       float stddev);
curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double *output_ptr, size_t n,
                                             double mean, double stddev);
curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int *output_ptr, size_t n,
                                     double lambda);
curandStatus_t curandGenerateSeeds(curandGenerator_t generator);
curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t *vectors[], curandDirectionVectorSet_t set);
curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t *vectors[], curandDirectionVectorSet_t set);
curandStatus_t curandGetScrambleConstants32(unsigned int **constants);
curandStatus_t curandGetScrambleConstants64(unsigned long long **constants);
curandStatus_t curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t *discrete_distribution);
curandStatus_t curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution);

}
