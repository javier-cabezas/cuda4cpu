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

# A directory laid out like a CUDA toolkit, for build systems written for
# nvcc, which find it through CUDA_PATH, CUDA_HOME or their own variable
# (Rodinia's CUDA_DIR): bin/nvcc is cuda4cpu-nvcc, include has cuda4cpu's
# headers, and lib64's libcudart, libcudart_static and libcuda are linker
# scripts that link libcuda4cpu (-lcuda4cpu, through a link to it in lib64,
# which -L<toolkit>/lib64 finds wherever cuda4cpu is installed) and its
# OpenMP runtime, for programs linked by the host compiler. Those find
# libcuda4cpu at run time as programs find CUDA's libcudart: through
# LD_LIBRARY_PATH (lib64 has a link to its soname too), unless it's
# installed where the dynamic loader looks. It's toolkit/ in the build tree,
# and <libdir>/cuda4cpu/toolkit in the installation.
#
# cuda4cpu.cmake there is a toolchain file for CMake projects with LANGUAGES
# CUDA (or HIP): CUDA (and HIP) compile with bin/nvcc, and C++ with
# bin/cuda4cpu-c++, for host code that includes CUDA's headers.
#
# It's laid out like a ROCm installation too, for HIP_PATH and ROCM_PATH:
# bin/hipcc and bin/hipconfig, and lib with libamdhip64 (a linker script like
# lib64's) and HIP's CMake package (lib/cmake/hip, for find_package(hip)).

# Writes the toolkit into directory: nvcc and include are what bin/nvcc and
# include link to, and libdir where lib64's links to libcuda4cpu point
function(cuda4cpu_toolkit directory nvcc include libdir)
    # nvvm/libdevice, which holds nvcc's device libraries, only for CMake to
    # find (it checks that the toolkit has one)
    file(MAKE_DIRECTORY ${directory}/bin ${directory}/lib64 ${directory}/lib/cmake/hip ${directory}/nvvm/libdevice)
    file(CREATE_LINK ${nvcc} ${directory}/bin/nvcc SYMBOLIC)
    foreach(tool IN ITEMS cuda4cpu-c++ hipcc hipconfig)
        set(target cuda4cpu-${tool})
        if(tool STREQUAL "cuda4cpu-c++")
            set(target ${tool})
        endif()
        cmake_path(REPLACE_FILENAME nvcc ${target} OUTPUT_VARIABLE path)
        file(CREATE_LINK ${path} ${directory}/bin/${tool} SYMBOLIC)
    endforeach()
    foreach(file IN ITEMS hip-config.cmake hip-config-version.cmake)
        configure_file(${PROJECT_SOURCE_DIR}/cmake/${file} ${directory}/lib/cmake/hip/${file} COPYONLY)
    endforeach()
    configure_file(${PROJECT_SOURCE_DIR}/cmake/cuda4cpu-toolchain.cmake ${directory}/cuda4cpu.cmake COPYONLY)
    file(CREATE_LINK ${include} ${directory}/include SYMBOLIC)
    if(BUILD_SHARED_LIBS)
        set(names ${CMAKE_SHARED_LIBRARY_PREFIX}cuda4cpu${CMAKE_SHARED_LIBRARY_SUFFIX})
        list(APPEND names ${names}.${PROJECT_VERSION_MAJOR})
    else()
        set(names ${CMAKE_STATIC_LIBRARY_PREFIX}cuda4cpu${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
    foreach(name IN LISTS names)
        foreach(dir IN ITEMS lib64 lib)
            file(CREATE_LINK ${libdir}/${name} ${directory}/${dir}/${name} SYMBOLIC)
        endforeach()
    endforeach()

    # A static libcuda4cpu needs its own dependencies too
    set(inputs -lcuda4cpu ${OpenMP_CXX_LIBRARIES})
    if(NOT BUILD_SHARED_LIBS AND NUMA_LIBRARY)
        list(APPEND inputs ${NUMA_LIBRARY})
    endif()
    list(JOIN inputs " " inputs)
    foreach(name IN ITEMS lib64/libcudart.so lib64/libcudart_static.a lib64/libcuda.so lib/libamdhip64.so)
        file(WRITE ${directory}/${name} "/* cuda4cpu: the CUDA and HIP runtime is libcuda4cpu */\nINPUT(${inputs})\n")
    endforeach()

    # The device runtime, which CMake links with CUDA programs: there's no
    # device code, so an empty archive
    file(WRITE ${directory}/lib64/libcudadevrt.a "!<arch>\n")
endfunction()

cuda4cpu_toolkit(${PROJECT_BINARY_DIR}/toolkit ${PROJECT_BINARY_DIR}/tools/cuda4cpu-nvcc
                 ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/lib)

# The installation's links are relative, so that it can be moved
set(toolkit_dir ${CMAKE_INSTALL_FULL_LIBDIR}/cuda4cpu/toolkit)
cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_BINDIR BASE_DIRECTORY ${toolkit_dir}/bin OUTPUT_VARIABLE nvcc_link)
cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_INCLUDEDIR BASE_DIRECTORY ${toolkit_dir} OUTPUT_VARIABLE include_link)
cuda4cpu_toolkit(${PROJECT_BINARY_DIR}/install/toolkit ${nvcc_link}/cuda4cpu-nvcc ${include_link}/cuda4cpu ../../..)
install(DIRECTORY ${PROJECT_BINARY_DIR}/install/toolkit DESTINATION ${CMAKE_INSTALL_LIBDIR}/cuda4cpu
        USE_SOURCE_PERMISSIONS)
unset(toolkit_dir)
unset(nvcc_link)
unset(include_link)
