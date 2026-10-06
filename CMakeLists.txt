cmake_minimum_required(VERSION 3.25)

project(cuda4cpu
    VERSION 0.1.0
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

# The cuda4cpu-c++ compiler driver, configured for the build tree and, with
# paths relative to its own directory, for the installation
set(driver_permissions OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
                       WORLD_READ WORLD_EXECUTE)

set(CUDA4CPU_DRIVER_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(CUDA4CPU_DRIVER_LIBRARY_DIR "${PROJECT_BINARY_DIR}/lib")
set(CUDA4CPU_DRIVER_REWRITE "${PROJECT_SOURCE_DIR}/tools/cuda4cpu-rewrite")
configure_file(tools/cuda4cpu-c++.in tools/cuda4cpu-c++ @ONLY FILE_PERMISSIONS ${driver_permissions})

cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_INCLUDEDIR BASE_DIRECTORY "${CMAKE_INSTALL_FULL_BINDIR}"
           OUTPUT_VARIABLE CUDA4CPU_DRIVER_INCLUDE_DIR)
string(APPEND CUDA4CPU_DRIVER_INCLUDE_DIR "/cuda4cpu")
cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_LIBDIR BASE_DIRECTORY "${CMAKE_INSTALL_FULL_BINDIR}"
           OUTPUT_VARIABLE CUDA4CPU_DRIVER_LIBRARY_DIR)
set(CUDA4CPU_DRIVER_REWRITE "cuda4cpu-rewrite")
configure_file(tools/cuda4cpu-c++.in install/cuda4cpu-c++ @ONLY FILE_PERMISSIONS ${driver_permissions})

install(PROGRAMS tools/cuda4cpu-rewrite "${PROJECT_BINARY_DIR}/install/cuda4cpu-c++"
        DESTINATION ${CMAKE_INSTALL_BINDIR})

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
