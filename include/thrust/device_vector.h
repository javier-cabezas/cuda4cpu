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

// cuda4cpu's overlay of NVIDIA's Thrust (CCCL): CCCL's device_vector.h, and on the
// OpenMP backend, which cuda4cpu builds CUDA programs' Thrust with, what the
// CUDA system brings along with it (thrust::cuda::par, thrust::system_error,
// ...: detail/cuda4cpu.h), for programs written for that system.
#pragma once

#if __has_include_next(<thrust/device_vector.h>)
#include_next <thrust/device_vector.h>
#include <thrust/system/cuda/detail/cuda4cpu.h>
#else
#error "cuda4cpu: Thrust is NVIDIA's CCCL: add its include directories (a CUDA toolkit's include, or a CCCL repository's thrust, libcudacxx/include and cub)"
#endif
