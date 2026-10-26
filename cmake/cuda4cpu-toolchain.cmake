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

# A CMake toolchain file for projects written for CUDA (LANGUAGES CUDA), or
# for HIP on its NVIDIA platform (LANGUAGES HIP), installed as cuda4cpu.cmake
# in the toolkit directory (cmake/cuda4cpu-toolkit.cmake):
#
#     cmake -DCMAKE_TOOLCHAIN_FILE=<prefix>/lib/cuda4cpu/toolkit/cuda4cpu.cmake ...
#
# CUDA and HIP sources compile with the toolkit's nvcc (cuda4cpu-nvcc), which
# CMake identifies as NVIDIA's. C++ sources compile with cuda4cpu-c++, which
# gives host code that includes CUDA's headers what they need: C++23, OpenMP
# and cuda4cpu's include directory. It also links programs with libcuda4cpu
# and an rpath to it.

set(CMAKE_CUDA_COMPILER "${CMAKE_CURRENT_LIST_DIR}/bin/nvcc" CACHE FILEPATH "The CUDA compiler: cuda4cpu-nvcc")
set(CMAKE_HIP_COMPILER "${CMAKE_CURRENT_LIST_DIR}/bin/nvcc" CACHE FILEPATH "The HIP compiler: cuda4cpu-nvcc")
set(CMAKE_HIP_PLATFORM "nvidia" CACHE STRING "HIP's platform: cuda4cpu's HIP is HIP's NVIDIA platform")
# Where CMake looks for HIP's installation (on the NVIDIA platform, it only
# checks there is one): the toolkit, which has HIP's headers
set(CMAKE_HIP_COMPILER_ROCM_ROOT "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "HIP's installation: cuda4cpu's toolkit")
set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_LIST_DIR}/bin/cuda4cpu-c++" CACHE FILEPATH "The C++ compiler: cuda4cpu-c++")
