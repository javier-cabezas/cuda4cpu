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

# A kernel that crashes is reported with the CUDA thread, its block and the
# launch, and the program still dies (or runs its own handler). Under
# AddressSanitizer, its report follows, and the program exits with 1. UBSan
# may stop the program before the crash (a division by zero), with a report
# of its own: GCC's runtime for it doesn't tell cuda4cpu.
# Usage: cmake -DPROGRAM=... -P crash_test.cmake

set(dim "\\([0-9]+, [0-9]+, [0-9]+\\)")
set(cases
    "invalid_access|CUDA thread \\(17, 0, 0\\) of block \\(3, 0, 0\\) made an invalid memory access at address 0x[0-9a-f]+"
    "after_barrier|CUDA thread \\(5, 1, 0\\) of block \\(1, 2, 0\\) made an invalid memory access at address 0x[0-9a-f]+"
    "overflow|CUDA thread \\(5, 0, 0\\) of block \\(0, 0, 0\\) overflowed its stack of [0-9]+ bytes"
    "divide|CUDA thread \\(2, 0, 0\\) of block \\(0, 0, 0\\) divided an integer by zero"
    "assert|CUDA thread \\(9, 0, 0\\) of block \\(0, 0, 0\\) aborted \\(a failed assert\\(\\), or a call to abort\\(\\)\\)"
    "own_handler|CUDA thread \\(17, 0, 0\\) of block \\(3, 0, 0\\) made an invalid memory access at address 0x[0-9a-f]+")

foreach(case IN LISTS cases)
    string(FIND "${case}" "|" sep)
    string(SUBSTRING "${case}" 0 ${sep} arg)
    math(EXPR start "${sep} + 1")
    string(SUBSTRING "${case}" ${start} -1 pattern)

    execute_process(COMMAND "${PROGRAM}" ${arg} OUTPUT_VARIABLE output ERROR_VARIABLE errors
                    RESULT_VARIABLE status)
    if(status EQUAL 0 OR output MATCHES "no crash")
        message(FATAL_ERROR "'${arg}' did not crash (status ${status}):\n${output}\n${errors}")
    endif()
    if(NOT errors MATCHES "launch at line ([0-9]+)")
        message(FATAL_ERROR "'${arg}' did not launch:\n${errors}")
    endif()
    set(line ${CMAKE_MATCH_1})
    if(errors MATCHES "runtime error: ")
        set(pattern "(${pattern}|CUDA thread ${dim} of block ${dim} made the error that the sanitizer reports above)")
        if(NOT errors MATCHES "cuda4cpu: error")
            continue()
        endif()
    endif()
    string(REGEX MATCHALL "cuda4cpu: error: [^\n]*" reports "${errors}")
    list(LENGTH reports count)
    if(NOT count EQUAL 1 OR NOT errors MATCHES "cuda4cpu: error: ${pattern}, in the kernel launched at [^\n]*crash\\.cu:${line}\\.")
        message(FATAL_ERROR "'${arg}': expected one report matching '${pattern}' at crash.cu:${line}, "
                            "got (status ${status}):\n${errors}")
    endif()
    if(arg STREQUAL "own_handler" AND NOT (status EQUAL 3 AND errors MATCHES "the program's handler"))
        message(FATAL_ERROR "the program's own handler did not run (status ${status}):\n${errors}")
    endif()
endforeach()

execute_process(COMMAND "${PROGRAM}" host OUTPUT_VARIABLE output ERROR_VARIABLE errors RESULT_VARIABLE status)
if(status EQUAL 0 OR errors MATCHES "cuda4cpu: error")
    message(FATAL_ERROR "a crash outside kernels was reported, or not fatal (status ${status}):\n${errors}")
endif()
