#
# CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
#
# Copyright (C) 2014 Javier Cabezas
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# A project written for CUDA, as projects are: CMake's CUDA language, the
# CUDA toolkit's targets, a static library of kernels compiled for separate
# compilation (which CMake device-links), a host .cpp file that includes
# cuda_runtime.h with C++17, and a program of one .cu file. The cmake_language
# test configures it with the toolkit's cuda4cpu.cmake toolchain file.

cmake_minimum_required(VERSION 3.25)
project(cuda4cpu_cuda_language LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(kernels STATIC kernels.cu)
set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

add_executable(host main.cpp)
target_compile_features(host PRIVATE cxx_std_17)
target_link_libraries(host PRIVATE kernels CUDA::cudart)
set_target_properties(host PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(single single.cu)
target_compile_features(single PRIVATE cuda_std_17)
