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

// How cuda4cpu builds NVIDIA's CCCL (Thrust, CUB, libcu++, from a CUDA
// toolkit or CCCL's repository), on the CPU: Thrust's device system is
// OpenMP's, as nvcc builds it with THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP;
// CCCL's NVTX ranges are off (cuda4cpu's NVTX does nothing); and so is its
// support of CUDA's half, bfloat16 and 8-bit types, which it declares at
// global scope, where they would collide with cuda4cpu's (in
// cuda4cpu::cuda_api). Algorithms of those types still work. Not on HIP's
// AMD platform, where rocThrust takes its HIP backend. cuda_runtime.h
// includes it, and so does Thrust's configuration (thrust/detail/config.h),
// for sources that include Thrust before cuda_runtime.h, or without it.
#pragma once

#if !(defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__))
#ifndef THRUST_DEVICE_SYSTEM
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
#endif
#ifndef CCCL_DISABLE_NVTX
#define CCCL_DISABLE_NVTX
#endif
#ifndef CCCL_DISABLE_FP16_SUPPORT
#define CCCL_DISABLE_FP16_SUPPORT
#endif
#endif
