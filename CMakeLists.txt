project(cuda4cpu)
cmake_minimum_required(VERSION 2.8)

set(COMMON_CXX_FLAGS "-std=c++11 -g")

set(CMAKE_CXX_FLAGS_DEBUG   "${COMMON_CXX_FLAGS} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_CXX_FLAGS} -O3 -march=native")

set(CUDART_LIB "" CACHE INTERNAL "")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tests)
