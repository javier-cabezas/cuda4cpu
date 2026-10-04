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
option(CUDA4CPU_ENABLE_NUMA "Use libnuma (if found) to discover the NUMA topology" ON)
option(CUDA4CPU_ENABLE_VALGRIND "Register fiber stacks with Valgrind" OFF)
option(CUDA4CPU_ENABLE_NATIVE "Compile in-tree targets with -march=native" OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_subdirectory(lib)

if(CUDA4CPU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#
# Installation and package config
#
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CUDA4CPU_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/cuda4cpu)

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
                              INSTALL_DESTINATION ${CUDA4CPU_CMAKE_DIR})

write_basic_package_version_file(${PROJECT_BINARY_DIR}/cuda4cpu-config-version.cmake
                                 COMPATIBILITY SameMinorVersion)

install(FILES ${PROJECT_BINARY_DIR}/cuda4cpu-config.cmake
              ${PROJECT_BINARY_DIR}/cuda4cpu-config-version.cmake
        DESTINATION ${CUDA4CPU_CMAKE_DIR})
