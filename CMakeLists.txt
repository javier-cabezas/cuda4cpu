project(cuda4cpu)
cmake_minimum_required(VERSION 2.8)

include(FindPackageHandleStandardArgs)

find_path(VALGRIND_INCLUDE_DIR memcheck.h
    /usr/include /usr/include/valgrind /usr/local/include /usr/local/include/valgrind
    ${VALGRIND_PREFIX}/include ${VALGRIND_PREFIX}/include/valgrind)

find_program(VALGRIND_PROGRAM NAMES valgrind PATH /usr/bin /usr/local/bin ${VALGRIND_PREFIX}/bin)

find_package_handle_standard_args(VALGRIND DEFAULT_MSG
                                  VALGRIND_INCLUDE_DIR
                                  VALGRIND_PROGRAM)

mark_as_advanced(VALGRIND_INCLUDE_DIR VALGRIND_PROGRAM)

set(DEBUG_FLAGS "")

if(${VALGRIND_FOUND})
    set(DEBUG_FLAGS "-DCUDA4CPU_HANDLE_VALGRIND ${DEBUG_FLAGS}")
endif(${VALGRIND_FOUND})

set(COMMON_CXX_FLAGS "-std=c++11 -g")

set(CMAKE_CXX_FLAGS_DEBUG   "${COMMON_CXX_FLAGS} ${DEBUG_FLAGS} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_CXX_FLAGS} -O3 -march=native")

set(CUDART_LIB "" CACHE INTERNAL "")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tests)
