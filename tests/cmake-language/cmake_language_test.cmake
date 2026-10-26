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


# Configures a CMake project with the toolkit's toolchain file, builds it,
# and runs its programs, each of which must return 0.
# Usage: cmake -DSOURCE=<project> -DWORK=<build dir> -DTOOLCHAIN=<cuda4cpu.cmake>
#              -DGENERATOR=<generator> -DPROGRAMS=<a|b> -P cmake_language_test.cmake

file(REMOVE_RECURSE "${WORK}")

function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE output)
    if(status)
        message(FATAL_ERROR "failed (${status}): ${ARGN}\n${output}")
    endif()
endfunction()

run(${CMAKE_COMMAND} -S "${SOURCE}" -B "${WORK}" -G "${GENERATOR}" -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN}")
run(${CMAKE_COMMAND} --build "${WORK}")

string(REPLACE "|" ";" programs "${PROGRAMS}")
foreach(program IN LISTS programs)
    run("${WORK}/${program}")
endforeach()
