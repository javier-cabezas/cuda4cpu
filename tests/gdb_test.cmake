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

# Stops in a kernel at a breakpoint conditioned on the CUDA thread, with the
# cuda4cpu gdb script, and checks the built-in variables and the backtrace.
# Usage: cmake -DGDB=... -DSCRIPT=... -DPROGRAM=... -DSOURCE=... -P gdb_test.cmake

file(STRINGS "${SOURCE}" lines)
set(number 0)
foreach(line IN LISTS lines)
    math(EXPR number "${number} + 1")
    if(line MATCHES "int v = s\\[\\(t \\+ N - 2\\) % N\\];")
        set(breakpoint ${number})
    endif()
endforeach()

execute_process(
    COMMAND "${GDB}" -q -batch -nx -ex "set debuginfod enabled off" -ex "source ${SCRIPT}"
            -ex "break barriers.cpp:${breakpoint} if \$threadIdx(\"x\") == 6 && \$blockIdx(\"x\") == 3"
            -ex run -ex "print \$threadIdx()" -ex "print \$blockIdx().x" -ex "print \$laneid()"
            -ex cuda4cpu -ex bt -ex kill
            "${PROGRAM}"
    OUTPUT_VARIABLE output ERROR_VARIABLE errors RESULT_VARIABLE status)

foreach(expected
        "\\$1 = {x = 6, y = 0, z = 0}"
        "\\$2 = 3"
        "\\$3 = 6"
        "thread \\(6, 0, 0\\) of block \\(3, 0, 0\\), lane 6; blocks of \\(64, 1, 1\\) threads, grid of \\(8, 1, 1\\) blocks"
        "#0 [^\n]*exit_between_barriers"
        "cuda4cpu::thread_block::fiber_main"
        "cuda4cpu_fiber_trampoline|__start_context")
    if(NOT output MATCHES "${expected}")
        message(FATAL_ERROR "gdb output does not match '${expected}':\n${output}\n${errors}")
    endif()
endforeach()

# The backtrace ends at the fiber trampoline (with the assembly context
# switch; the portable one starts fibers through makecontext)
if(output MATCHES "cuda4cpu_fiber_trampoline[^\n]*\n#")
    message(FATAL_ERROR "the backtrace continues past the fiber trampoline:\n${output}")
endif()
