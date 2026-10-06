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


// CUDA's cuda.h, the driver API header. Programs often include it only for
// CUDA_VERSION or alongside the runtime API. cuda4cpu provides the runtime
// API and the version; the driver API (cuInit, cuModuleLoad, ...) isn't
// implemented, so programs that call it don't compile.

#pragma once

#include "cuda_runtime.h"

#ifndef CUDA_VERSION
#define CUDA_VERSION CUDART_VERSION
#endif

#ifndef __cuda_cuda_h__
#define __cuda_cuda_h__
#endif
