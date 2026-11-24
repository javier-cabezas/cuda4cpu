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

// cuda4cpu's overlay of NVIDIA's Thrust (CCCL) on its OpenMP backend: every
// algorithm's dispatch includes this header, and with it, thrust::system_error,
// which the CUDA backend's headers bring along and programs written for it
// use without including it. rocThrust's passes through.
#pragma once

#include_next <thrust/system/omp/detail/execution_policy.h>

#if !defined(ROCTHRUST_VERSION)
#include <thrust/system_error.h>
#endif
