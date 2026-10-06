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
# as C++ with cuda_runtime.h included implicitly, as nvcc does. <target> is
# linked against cuda4cpu::cuda4cpu.
#

function(cuda4cpu_add_cuda_sources target)
    get_property(rewrite GLOBAL PROPERTY CUDA4CPU_REWRITE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    foreach(source IN LISTS ARGN)
        cmake_path(ABSOLUTE_PATH source BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                   NORMALIZE OUTPUT_VARIABLE input)
        cmake_path(RELATIVE_PATH input BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                   OUTPUT_VARIABLE relative)
        string(REPLACE "../" "__/" relative "${relative}")
        set(output "${CMAKE_CURRENT_BINARY_DIR}/cuda4cpu/${target}/${relative}.cpp")

        add_custom_command(
            OUTPUT "${output}"
            COMMAND Python3::Interpreter "${rewrite}" "${input}" -o "${output}"
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
