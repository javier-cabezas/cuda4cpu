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

// HIP's platform macros. cuda4cpu compiles HIP code as HIP's NVIDIA platform
// does: on top of the CUDA runtime API, with 32-lane warps. Code that
// distinguishes the platforms takes its NVIDIA branch, which uses the CUDA
// features cuda4cpu implements, rather than the AMD one, which may use
// wavefronts of 64 lanes or AMD GPU builtins.

#pragma once

#ifndef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_NVIDIA__ 1
#endif

//! The deprecated name of __HIP_PLATFORM_NVIDIA__
#ifndef __HIP_PLATFORM_NVCC__
#define __HIP_PLATFORM_NVCC__ 1
#endif

// As with nvcc on that platform, __HIPCC__ goes with __CUDACC__, which
// cuda4cpu-rewrite defines in the files it rewrites
#if defined(__CUDACC__) && !defined(__HIPCC__)
#define __HIPCC__ 1
#endif
