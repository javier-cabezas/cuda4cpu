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

// cuBLAS's API with handles (see cublas_api.h). cuBLAS's functions of this
// API are named with _v2 (cublasSgemm_v2), which this header defines
// cublasSgemm as: here they're the functions' own names, and cublas_api.h
// defines the _v2 names as macros for them, for code that calls those.
#pragma once

#ifndef CUBLAS_V2_H_
#define CUBLAS_V2_H_
#endif

#include "cublas_api.h"
