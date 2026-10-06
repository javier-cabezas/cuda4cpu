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

# Builds a .cu file with the cuda4cpu-c++ driver, in one step and with
# separate compilation and linking, and runs the results.
# Usage: cmake -DDRIVER=... -DSOURCE=... -DWORK=... -P driver_test.cmake

file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

function(run)
    execute_process(COMMAND ${ARGN} WORKING_DIRECTORY "${WORK}" RESULT_VARIABLE status)
    if(status)
        message(FATAL_ERROR "failed (${status}): ${ARGN}")
    endif()
endfunction()

run("${DRIVER}" -O2 -o "${WORK}/one_step" "${SOURCE}")
run("${WORK}/one_step")

run("${DRIVER}" -O2 -c "${SOURCE}")
cmake_path(GET SOURCE STEM stem)
run("${DRIVER}" -o "${WORK}/two_steps" "${WORK}/${stem}.o")
run("${WORK}/two_steps")
