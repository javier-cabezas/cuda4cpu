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

# CUDA4CPU_INIT_MEMORY fills every kind of new allocation with its byte, and
# CUDA4CPU_LEAK_CHECK=warn reports, at exit, the allocations never freed, by
# the line that made them, graph allocations included. Memory that a static
# object's destructor or a destructor function frees after main returns isn't
# reported.
# Usage: cmake -DPROGRAM=... -P memory_checks_test.cmake

function(run expected_status)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env ${ARGN} OUTPUT_VARIABLE output ERROR_VARIABLE errors
                    RESULT_VARIABLE status)
    if(NOT status EQUAL expected_status)
        message(FATAL_ERROR "'${ARGN}' exited with ${status}, expected ${expected_status}:\n${output}\n${errors}")
    endif()
    set(output "${output}" PARENT_SCOPE)
    set(errors "${errors}" PARENT_SCOPE)
endfunction()

foreach(byte 0xa5 0 255)
    run(0 CUDA4CPU_INIT_MEMORY=${byte} "${PROGRAM}" fill)
    if(NOT output STREQUAL "filled\n")
        message(FATAL_ERROR "CUDA4CPU_INIT_MEMORY=${byte}:\n${output}\n${errors}")
    endif()
endforeach()

foreach(invalid 0x100 -1 invalid)
    run(1 CUDA4CPU_INIT_MEMORY=${invalid} "${PROGRAM}" fill)
    if(NOT errors MATCHES "cuda4cpu: ignoring CUDA4CPU_INIT_MEMORY=${invalid}: expected a byte")
        message(FATAL_ERROR "CUDA4CPU_INIT_MEMORY=${invalid} was not rejected:\n${errors}")
    endif()
endforeach()

run(0 CUDA4CPU_LEAK_CHECK=warn "${PROGRAM}" leaks)
foreach(leak device managed pinned graph)
    if(NOT output MATCHES "${leak} leaked at line ([0-9]+)")
        message(FATAL_ERROR "no ${leak} leak:\n${output}")
    endif()
    set(${leak}_line ${CMAKE_MATCH_1})
endforeach()
set(at "made at [^\n]*memory_checks\\.cu")
set(expected
    "cuda4cpu: warning: an allocation of graph memory \\(1000 bytes\\), ${at}:${graph_line}, was never freed\\.\n"
    "cuda4cpu: warning: 2 allocations of device memory \\(200 bytes in all\\), ${at}:${device_line}, were never freed\\.\n"
    "cuda4cpu: warning: an allocation of pinned host memory \\(50 bytes\\), ${at}:${pinned_line}, was never freed\\.\n"
    "cuda4cpu: warning: an allocation of managed memory \\(4 bytes\\), ${at}:${managed_line}, was never freed\\.\n")
string(JOIN "" expected ${expected})
if(NOT errors MATCHES "^${expected}$")
    message(FATAL_ERROR "expected exactly the four leaks, largest first, got:\n${errors}")
endif()

foreach(setting "" CUDA4CPU_LEAK_CHECK=ignore)
    run(0 ${setting} "${PROGRAM}" leaks)
    if(errors MATCHES "cuda4cpu")
        message(FATAL_ERROR "leaks reported with '${setting}':\n${errors}")
    endif()
endforeach()

run(0 CUDA4CPU_LEAK_CHECK=yes "${PROGRAM}" leaks)
if(NOT errors MATCHES "^cuda4cpu: ignoring CUDA4CPU_LEAK_CHECK=yes: expected warn or ignore\n$")
    message(FATAL_ERROR "an invalid CUDA4CPU_LEAK_CHECK was not rejected, or leaks were reported:\n${errors}")
endif()
