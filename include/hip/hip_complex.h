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

// HIP's complex numbers: CUDA's (cuComplex.h) under HIP's names, as on HIP's
// NVIDIA platform. hipCsqabsf and hipCsqabs, the squared magnitude, have no
// CUDA counterpart.

#pragma once

#include "hip_runtime_api.h"
#include "../cuComplex.h"

#define hipFloatComplex cuFloatComplex
#define hipDoubleComplex cuDoubleComplex
#define hipComplex cuComplex
#define hipCrealf cuCrealf
#define hipCimagf cuCimagf
#define make_hipFloatComplex make_cuFloatComplex
#define make_hipComplex make_cuComplex
#define hipConjf cuConjf
#define hipCaddf cuCaddf
#define hipCsubf cuCsubf
#define hipCmulf cuCmulf
#define hipCdivf cuCdivf
#define hipCabsf cuCabsf
#define hipCfmaf cuCfmaf
#define hipCreal cuCreal
#define hipCimag cuCimag
#define make_hipDoubleComplex make_cuDoubleComplex
#define hipConj cuConj
#define hipCadd cuCadd
#define hipCsub cuCsub
#define hipCmul cuCmul
#define hipCdiv cuCdiv
#define hipCabs cuCabs
#define hipCfma cuCfma
#define hipComplexFloatToDouble cuComplexFloatToDouble
#define hipComplexDoubleToFloat cuComplexDoubleToFloat

namespace cuda4cpu {

inline namespace hip_api {

//! |z|^2, as HIP computes it: from the absolute value, squared
inline float hipCsqabsf(cuFloatComplex z)
{
    return cuCabsf(z) * cuCabsf(z);
}

inline double hipCsqabs(cuDoubleComplex z)
{
    return cuCabs(z) * cuCabs(z);
}

}

}
