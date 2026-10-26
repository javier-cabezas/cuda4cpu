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


# HIP's CMake package (find_package(hip)), for cuda4cpu's HIP, in the toolkit
# directory's lib/cmake/hip (cmake/cuda4cpu-toolkit.cmake). HIP is HIP's
# NVIDIA platform: hip::host and hip::device bring HIP's headers, and
# libcuda4cpu, the HIP runtime, through the toolkit's libamdhip64.

get_filename_component(_hip_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(hip_VERSION 7.2.0)
set(HIP_VERSION ${hip_VERSION})
set(HIP_PLATFORM "nvidia")
set(HIP_COMPILER "nvcc")
set(HIP_RUNTIME "cuda")
set(hip_INCLUDE_DIR "${_hip_root}/include")
set(hip_INCLUDE_DIRS "${hip_INCLUDE_DIR}")
set(hip_LIBRARIES hip::host)
set(hip_FOUND TRUE)

if(NOT TARGET hip::host)
    # By name, from the toolkit's lib, whose libamdhip64.so is a linker script
    # that links libcuda4cpu from the same directory
    add_library(hip::amdhip64 INTERFACE IMPORTED)
    set_target_properties(hip::amdhip64 PROPERTIES
        INTERFACE_LINK_DIRECTORIES "${_hip_root}/lib"
        INTERFACE_LINK_LIBRARIES amdhip64
        INTERFACE_INCLUDE_DIRECTORIES "${hip_INCLUDE_DIR}")
    add_library(hip::host INTERFACE IMPORTED)
    set_target_properties(hip::host PROPERTIES
        INTERFACE_LINK_LIBRARIES hip::amdhip64
        INTERFACE_COMPILE_DEFINITIONS __HIP_PLATFORM_NVIDIA__)
    add_library(hip::device INTERFACE IMPORTED)
    set_target_properties(hip::device PROPERTIES INTERFACE_LINK_LIBRARIES hip::host)
endif()

unset(_hip_root)
