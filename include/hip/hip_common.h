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

// HIP's platform macros. By default cuda4cpu compiles HIP code as HIP's
// NVIDIA platform does: on top of the CUDA runtime API, with 32-lane warps.
// Code that distinguishes the platforms takes its NVIDIA branch, which uses
// the CUDA features cuda4cpu implements, rather than the AMD one, which may
// use wavefronts of 64 lanes or AMD GPU builtins.
//
// With __HIP_PLATFORM_AMD__ defined (cuda4cpu-hipcc --cuda4cpu-platform=amd,
// which defines all of these), it emulates an RDNA3 GPU instead: gfx1100,
// unless one of the other __gfx11xx__ macros names the GPU, with 32-lane
// wavefronts and AMD's builtins (amdgcn.h). __HIP_DEVICE_COMPILE__ stays
// undefined, as __CUDA_ARCH__ does: host and device code compile together.

#pragma once

// A program that defines __HIP_PLATFORM_AMD__ itself, after the rewriter
// defined __HIP_PLATFORM_NVIDIA__, stays on the NVIDIA platform
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

//! The deprecated name of __HIP_PLATFORM_AMD__
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__ 1
#endif

// The GPU: its name, gfx version (major, minor) and family
#if defined(__gfx1101__)
#define CUDA4CPU_AMDGPU_TARGET "gfx1101"
#elif defined(__gfx1102__)
#define CUDA4CPU_AMDGPU_TARGET "gfx1102"
#elif defined(__gfx1103__)
#define CUDA4CPU_AMDGPU_TARGET "gfx1103"
#elif defined(__gfx1150__)
#define CUDA4CPU_AMDGPU_TARGET "gfx1150"
#elif defined(__gfx1151__)
#define CUDA4CPU_AMDGPU_TARGET "gfx1151"
#elif defined(__gfx1152__)
#define CUDA4CPU_AMDGPU_TARGET "gfx1152"
#elif defined(__gfx1153__)
#define CUDA4CPU_AMDGPU_TARGET "gfx1153"
#else
#ifndef __gfx1100__
#define __gfx1100__ 1
#endif
#define CUDA4CPU_AMDGPU_TARGET "gfx1100"
#endif

#ifndef __GFX11__
#define __GFX11__ 1
#endif
#ifndef __AMDGCN__
#define __AMDGCN__ 1
#endif
#ifndef __AMDGPU__
#define __AMDGPU__ 1
#endif
#ifndef __AMDGCN_WAVEFRONT_SIZE__
#define __AMDGCN_WAVEFRONT_SIZE__ 32
#endif
//! The deprecated name of __AMDGCN_WAVEFRONT_SIZE__
#ifndef __AMDGCN_WAVEFRONT_SIZE
#define __AMDGCN_WAVEFRONT_SIZE 32
#endif

#else

#ifndef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_NVIDIA__ 1
#endif

//! The deprecated name of __HIP_PLATFORM_NVIDIA__
#ifndef __HIP_PLATFORM_NVCC__
#define __HIP_PLATFORM_NVCC__ 1
#endif

#endif

// As with nvcc on that platform, __HIPCC__ goes with __CUDACC__, which
// cuda4cpu-rewrite defines in the files it rewrites
#if defined(__CUDACC__) && !defined(__HIPCC__)
#define __HIPCC__ 1
#endif
