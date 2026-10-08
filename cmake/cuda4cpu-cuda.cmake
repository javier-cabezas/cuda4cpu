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

#
# cuda4cpu_add_cuda_sources(<target> <source>...)
#
# Adds CUDA source files (.cu) to <target>. At build time, each one is
# rewritten by cuda4cpu-rewrite (kernel<<<...>>>(...) launches and extern
# __shared__ declarations, also in the local headers it includes) and compiled
# as C++ with cuda_runtime.h included implicitly, as nvcc does. .hip files
# among them are HIP sources, as with cuda4cpu_add_hip_sources. <target> is
# linked against cuda4cpu::cuda4cpu.
#
# cuda4cpu_add_hip_sources(<target> <source>...)
#
# Adds HIP source files to <target>, whatever their extension (.hip, .cpp,
# ...): HIP projects often compile .cpp files with hipcc. They are rewritten
# like CUDA sources, and also get cuda_runtime.h, which has the kernel
# language. They include hip/hip_runtime.h themselves, as with hipcc, after
# any macro that configures it (HIP_TEMPLATE_KERNEL_LAUNCH).
#

function(cuda4cpu_add_cuda_sources target)
    _cuda4cpu_add_sources(${target} "" ${ARGN})
endfunction()

function(cuda4cpu_add_hip_sources target)
    _cuda4cpu_add_sources(${target} hip ${ARGN})
endfunction()

# Adds the sources, in <language> (cuda or hip), or by extension if empty
function(_cuda4cpu_add_sources target language)
    get_property(rewrite GLOBAL PROPERTY CUDA4CPU_REWRITE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    foreach(source IN LISTS ARGN)
        cmake_path(ABSOLUTE_PATH source BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                   NORMALIZE OUTPUT_VARIABLE input)
        cmake_path(RELATIVE_PATH input BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                   OUTPUT_VARIABLE relative)
        string(REPLACE "../" "__/" relative "${relative}")
        set(output "${CMAKE_CURRENT_BINARY_DIR}/cuda4cpu/${target}/${relative}.cpp")

        set(source_language "${language}")
        if(source_language STREQUAL "")
            set(source_language cuda)
            if(input MATCHES "\\.hip$")
                set(source_language hip)
            endif()
        endif()
        add_custom_command(
            OUTPUT "${output}"
            COMMAND Python3::Interpreter "${rewrite}" "${input}" -o "${output}"
                    --language ${source_language}
                    --shadow-dir "${output}.headers" --depfile "${output}.d"
            DEPENDS "${input}" "${rewrite}"
            DEPFILE "${output}.d"
            COMMENT "Rewriting ${relative} for cuda4cpu"
            VERBATIM)

        target_sources(${target} PRIVATE "${output}")
        set_source_files_properties("${output}" PROPERTIES COMPILE_OPTIONS "-include;cuda_runtime.h")
    endforeach()

    target_link_libraries(${target} PRIVATE cuda4cpu::cuda4cpu)
endfunction()
