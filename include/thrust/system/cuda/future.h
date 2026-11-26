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

// cuda4cpu's overlay of NVIDIA's Thrust (CCCL 2.x): the CUDA system's
// futures need its CUDA backend, which cuda4cpu doesn't build Thrust with;
// with another device system, CCCL's own.
#pragma once

#include <thrust/detail/config.h>

// Without Thrust (THRUST_DEVICE_SYSTEM_OMP undefined), its configuration
// has said so already
#if defined(THRUST_DEVICE_SYSTEM_OMP) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP && !defined(ROCTHRUST_VERSION)
#error "cuda4cpu: Thrust's CUDA futures (thrust/system/cuda/future.h) need its CUDA backend; cuda4cpu builds Thrust on OpenMP"
#elif __has_include_next(<thrust/system/cuda/future.h>)
#include_next <thrust/system/cuda/future.h>
#endif
