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

// HIP's bfloat16: CUDA's __nv_bfloat16 and __nv_bfloat162, with the same
// intrinsics (cuda_bf16.h), as on HIP's NVIDIA platform

#pragma once

#include "hip_runtime_api.h"
#include "../cuda_bf16.h"

#define HIPRT_ONE_BF16        CUDART_ONE_BF16
#define HIPRT_ZERO_BF16       CUDART_ZERO_BF16
#define HIPRT_INF_BF16        CUDART_INF_BF16
#define HIPRT_MAX_NORMAL_BF16 CUDART_MAX_NORMAL_BF16
#define HIPRT_MIN_DENORM_BF16 CUDART_MIN_DENORM_BF16
#define HIPRT_NAN_BF16        CUDART_NAN_BF16
#define HIPRT_NEG_ZERO_BF16   CUDART_NEG_ZERO_BF16

typedef struct __nv_bfloat16 __hip_bfloat16;
typedef struct __nv_bfloat162 __hip_bfloat162;
typedef struct __nv_bfloat16_raw __hip_bfloat16_raw;
typedef struct __nv_bfloat162_raw __hip_bfloat162_raw;
