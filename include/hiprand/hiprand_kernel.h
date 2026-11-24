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

// hipRAND's device API on HIP's NVIDIA platform: cuRAND's (curand_kernel.h)
// with HIP's names. hiprand() is a function, since a macro of that name
// would rename every other use of it.
#pragma once

#include "../curand_kernel.h"
#include "hiprand.h"

typedef curandStateXORWOW_t hiprandState_t;
typedef curandStateXORWOW_t hiprandState;
typedef curandStateXORWOW_t hiprandStateXORWOW_t;
typedef curandStateXORWOW_t hiprandStateXORWOW;
typedef curandStatePhilox4_32_10_t hiprandStatePhilox4_32_10_t;
typedef curandStatePhilox4_32_10_t hiprandStatePhilox4_32_10;
typedef curandStateMRG32k3a_t hiprandStateMRG32k3a_t;
typedef curandStateMRG32k3a_t hiprandStateMRG32k3a;
typedef curandStateSobol32_t hiprandStateSobol32_t;
typedef curandStateSobol32_t hiprandStateSobol32;
typedef curandStateScrambledSobol32_t hiprandStateScrambledSobol32_t;
typedef curandStateScrambledSobol32_t hiprandStateScrambledSobol32;
typedef curandStateSobol64_t hiprandStateSobol64_t;
typedef curandStateSobol64_t hiprandStateSobol64;
typedef curandStateScrambledSobol64_t hiprandStateScrambledSobol64_t;
typedef curandStateScrambledSobol64_t hiprandStateScrambledSobol64;

template <typename State>
inline auto hiprand(State *state) -> decltype(curand(state))
{
    return curand(state);
}

inline uint4 hiprand4(curandStatePhilox4_32_10_t *state)
{
    return curand4(state);
}

#define hiprand_init curand_init
#define hiprand_uniform curand_uniform
#define hiprand_uniform_double curand_uniform_double
#define hiprand_uniform2_double curand_uniform2_double
#define hiprand_uniform4 curand_uniform4
#define hiprand_uniform4_double curand_uniform4_double
#define hiprand_normal curand_normal
#define hiprand_normal2 curand_normal2
#define hiprand_normal4 curand_normal4
#define hiprand_normal_double curand_normal_double
#define hiprand_normal2_double curand_normal2_double
#define hiprand_normal4_double curand_normal4_double
#define hiprand_log_normal curand_log_normal
#define hiprand_log_normal2 curand_log_normal2
#define hiprand_log_normal4 curand_log_normal4
#define hiprand_log_normal_double curand_log_normal_double
#define hiprand_log_normal2_double curand_log_normal2_double
#define hiprand_log_normal4_double curand_log_normal4_double
#define hiprand_poisson curand_poisson
#define hiprand_poisson4 curand_poisson4
#define hiprand_discrete curand_discrete
#define hiprand_discrete4 curand_discrete4
#define skipahead_subsequence skipahead_sequence
