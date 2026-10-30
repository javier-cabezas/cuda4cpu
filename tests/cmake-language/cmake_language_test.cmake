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
# and runs its programs, each of which must return 0. A second build must do
# nothing, and with TOUCH (a file of the project), a build after touching it
# must compile again the objects that match REBUILDS. The project is copied
# into WORK, so that touching its files leaves the source tree alone.
# Usage: cmake -DSOURCE=<project> -DWORK=<build dir> -DTOOLCHAIN=<cuda4cpu.cmake>
#              -DGENERATOR=<generator> -DPROGRAMS=<a|b> [-DTOUCH=<file> -DREBUILDS=<regex>]
#              -P cmake_language_test.cmake

file(REMOVE_RECURSE "${WORK}")
file(COPY "${SOURCE}/" DESTINATION "${WORK}/source")

function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE output)
    if(status)
        message(FATAL_ERROR "failed (${status}): ${ARGN}\n${output}")
    endif()
    set(output "${output}" PARENT_SCOPE)
endfunction()

run(${CMAKE_COMMAND} -S "${WORK}/source" -B "${WORK}/build" -G "${GENERATOR}" -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN}")
run(${CMAKE_COMMAND} --build "${WORK}/build")

string(REPLACE "|" ";" programs "${PROGRAMS}")
foreach(program IN LISTS programs)
    run("${WORK}/build/${program}")
endforeach()

# The dependencies the drivers write name the original files, not the
# rewritten copies they delete, which would make every build compile again
run(${CMAKE_COMMAND} --build "${WORK}/build")
if(output MATCHES "Building|Linking")
    message(FATAL_ERROR "a second build compiled again:\n${output}")
endif()

if(DEFINED TOUCH)
    file(TOUCH "${WORK}/source/${TOUCH}")
    run(${CMAKE_COMMAND} --build "${WORK}/build")
    if(NOT output MATCHES "Building [A-Z]+ object [^\n]*(${REBUILDS})")
        message(FATAL_ERROR "touching ${TOUCH} didn't compile ${REBUILDS} again:\n${output}")
    endif()
endif()
