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

# With CUDA4CPU_DIVERGENT_BARRIERS=error, divergent barriers abort the program
# with a single report. Usage: cmake -DPROGRAM=... -P divergent_barrier_error.cmake

execute_process(COMMAND ${CMAKE_COMMAND} -E env CUDA4CPU_DIVERGENT_BARRIERS=error "${PROGRAM}"
                OUTPUT_VARIABLE output ERROR_VARIABLE errors RESULT_VARIABLE status)

if(status EQUAL 0 OR output MATCHES "divergent: done")
    message(FATAL_ERROR "the program did not stop (status ${status}):\n${output}\n${errors}")
endif()
string(REGEX MATCHALL "error: threads of block" reports "${errors}")
list(LENGTH reports count)
if(NOT count EQUAL 1)
    message(FATAL_ERROR "expected one report, got ${count}:\n${errors}")
endif()
