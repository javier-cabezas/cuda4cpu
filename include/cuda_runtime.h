/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
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

//
// Drop-in replacement for the CUDA runtime header. It makes the CUDA API
// (cudaMalloc, dim3, atomicAdd, __shfl_sync, ...) visible in the global
// namespace, like the real header, without the rest of cuda4cpu: kernels are
// still launched with cuda4cpu::launch(kernel, grid, block).call(args...).
//
// nvcc includes this header implicitly in .cu files. To compile them
// unchanged, pass -include cuda_runtime.h to the compiler.
//

#pragma once

#include "cuda4cpu.hpp"

using namespace cuda4cpu::cuda_api;
