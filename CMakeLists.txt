cmake_minimum_required(VERSION 3.25)

project(cuda4cpu
    VERSION 0.3.0
    DESCRIPTION "Library and headers to make CUDA codes run seamlessly on CPUs"
    HOMEPAGE_URL "https://github.com/javier-cabezas/cuda4cpu"
    LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_SHARED_LIBS "Build cuda4cpu as a shared library" ON)
option(CUDA4CPU_BUILD_TESTS "Build the cuda4cpu tests" ${PROJECT_IS_TOP_LEVEL})
option(CUDA4CPU_BUILD_BENCHMARKS "Build the cuda4cpu micro-benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(CUDA4CPU_ENABLE_NUMA "Use libnuma (if found) to discover the NUMA topology" ON)
option(CUDA4CPU_ENABLE_VALGRIND "Register fiber stacks with Valgrind" OFF)
option(CUDA4CPU_ENABLE_NATIVE "Compile in-tree targets with -march=native" OFF)
option(CUDA4CPU_PORTABLE_CONTEXT "Switch fibers with _setjmp/_longjmp instead of the x86-64 assembly" OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_subdirectory(lib)

# Compiling .cu files: cuda4cpu-rewrite and cuda4cpu_add_cuda_sources()
set_property(GLOBAL PROPERTY CUDA4CPU_REWRITE "${PROJECT_SOURCE_DIR}/tools/cuda4cpu-rewrite")
include(cmake/cuda4cpu-cuda.cmake)

if(CUDA4CPU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CUDA4CPU_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

#
# Installation and package config
#
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CUDA4CPU_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/cuda4cpu)

# The compiler drivers, configured for the build tree and, with paths relative
# to their own directory, for the installation: cuda4cpu-c++,
# cuda4cpu-hipcc, which compiles every C++ source as HIP, as hipcc does,
# cuda4cpu-nvcc, which takes nvcc's options, cuda4cpu-hipconfig, which
# answers as HIP's hipconfig, and cuda4cpu-mcp, an MCP server that compiles,
# runs and checks programs with them
set(driver_permissions OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
                       WORLD_READ WORLD_EXECUTE)

function(cuda4cpu_configure_drivers directory)
    foreach(driver IN ITEMS c++ hipcc nvcc)
        set(CUDA4CPU_DRIVER_MODE "")
        if(driver STREQUAL "hipcc")
            set(CUDA4CPU_DRIVER_MODE "hip")
        elseif(driver STREQUAL "nvcc")
            set(CUDA4CPU_DRIVER_MODE "nvcc")
        endif()
        configure_file(tools/cuda4cpu-c++.in ${directory}/cuda4cpu-${driver} @ONLY
                       FILE_PERMISSIONS ${driver_permissions})
    endforeach()
    configure_file(tools/cuda4cpu-hipconfig.in ${directory}/cuda4cpu-hipconfig @ONLY
                   FILE_PERMISSIONS ${driver_permissions})
    configure_file(tools/cuda4cpu-mcp.in ${directory}/cuda4cpu-mcp @ONLY FILE_PERMISSIONS ${driver_permissions})
endfunction()

set(CUDA4CPU_DRIVER_CXX "${CMAKE_CXX_COMPILER}")
set(CUDA4CPU_DRIVER_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(CUDA4CPU_DRIVER_LIBRARY_DIR "${PROJECT_BINARY_DIR}/lib")
set(CUDA4CPU_DRIVER_REWRITE "${PROJECT_SOURCE_DIR}/tools/cuda4cpu-rewrite")
set(CUDA4CPU_DRIVER_TOOLKIT_DIR "${PROJECT_BINARY_DIR}/toolkit")
set(CUDA4CPU_DRIVER_DOCS_DIR "${PROJECT_SOURCE_DIR}/docs")
cuda4cpu_configure_drivers(tools)

cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_INCLUDEDIR BASE_DIRECTORY "${CMAKE_INSTALL_FULL_BINDIR}"
           OUTPUT_VARIABLE CUDA4CPU_DRIVER_INCLUDE_DIR)
string(APPEND CUDA4CPU_DRIVER_INCLUDE_DIR "/cuda4cpu")
cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_LIBDIR BASE_DIRECTORY "${CMAKE_INSTALL_FULL_BINDIR}"
           OUTPUT_VARIABLE CUDA4CPU_DRIVER_LIBRARY_DIR)
set(CUDA4CPU_DRIVER_REWRITE "cuda4cpu-rewrite")
cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_LIBDIR BASE_DIRECTORY "${CMAKE_INSTALL_FULL_BINDIR}"
           OUTPUT_VARIABLE CUDA4CPU_DRIVER_TOOLKIT_DIR)
string(APPEND CUDA4CPU_DRIVER_TOOLKIT_DIR "/cuda4cpu/toolkit")
cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_DOCDIR BASE_DIRECTORY "${CMAKE_INSTALL_FULL_BINDIR}"
           OUTPUT_VARIABLE CUDA4CPU_DRIVER_DOCS_DIR)
cuda4cpu_configure_drivers(install)

include(cmake/cuda4cpu-toolkit.cmake)

install(PROGRAMS tools/cuda4cpu-rewrite "${PROJECT_BINARY_DIR}/install/cuda4cpu-c++"
                 "${PROJECT_BINARY_DIR}/install/cuda4cpu-hipcc" "${PROJECT_BINARY_DIR}/install/cuda4cpu-nvcc"
                 "${PROJECT_BINARY_DIR}/install/cuda4cpu-hipconfig" "${PROJECT_BINARY_DIR}/install/cuda4cpu-mcp"
        DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES tools/cuda4cpu-gdb.py DESTINATION ${CMAKE_INSTALL_DATADIR}/cuda4cpu)
# The reference, which cuda4cpu-mcp's capabilities tool reads
file(GLOB cuda4cpu_docs ${PROJECT_SOURCE_DIR}/docs/*.md)
install(FILES ${cuda4cpu_docs} DESTINATION ${CMAKE_INSTALL_DOCDIR})

install(TARGETS cuda4cpu
        EXPORT cuda4cpu-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuda4cpu)

install(EXPORT cuda4cpu-targets
        NAMESPACE cuda4cpu::
        DESTINATION ${CUDA4CPU_CMAKE_DIR})

configure_package_config_file(cmake/cuda4cpu-config.cmake.in
                              ${PROJECT_BINARY_DIR}/cuda4cpu-config.cmake
                              INSTALL_DESTINATION ${CUDA4CPU_CMAKE_DIR}
                              PATH_VARS CMAKE_INSTALL_BINDIR)

write_basic_package_version_file(${PROJECT_BINARY_DIR}/cuda4cpu-config-version.cmake
                                 COMPATIBILITY SameMinorVersion)

install(FILES ${PROJECT_BINARY_DIR}/cuda4cpu-config.cmake
              ${PROJECT_BINARY_DIR}/cuda4cpu-config-version.cmake
              cmake/cuda4cpu-cuda.cmake
        DESTINATION ${CUDA4CPU_CMAKE_DIR})
