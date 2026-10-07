add_library(cuda4cpu cuda4cpu.cpp)
add_library(cuda4cpu::cuda4cpu ALIAS cuda4cpu)

set(CUDA4CPU_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

target_sources(cuda4cpu
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS ${CUDA4CPU_INCLUDE_DIR}
        FILES
            ${CUDA4CPU_INCLUDE_DIR}/cuda4cpu.hpp
            ${CUDA4CPU_INCLUDE_DIR}/cooperative_groups.h
            ${CUDA4CPU_INCLUDE_DIR}/cooperative_groups/reduce.h
            ${CUDA4CPU_INCLUDE_DIR}/cooperative_groups/scan.h
            ${CUDA4CPU_INCLUDE_DIR}/cuda.h
            ${CUDA4CPU_INCLUDE_DIR}/cuda_runtime.h
            ${CUDA4CPU_INCLUDE_DIR}/cuda_runtime_api.h
            ${CUDA4CPU_INCLUDE_DIR}/cuda_profiler_api.h
            ${CUDA4CPU_INCLUDE_DIR}/device_launch_parameters.h
            ${CUDA4CPU_INCLUDE_DIR}/vector_functions.h
            ${CUDA4CPU_INCLUDE_DIR}/vector_types.h
            ${CUDA4CPU_INCLUDE_DIR}/defines.hpp
            ${CUDA4CPU_INCLUDE_DIR}/launch.hpp
            ${CUDA4CPU_INCLUDE_DIR}/warp.hpp
            ${CUDA4CPU_INCLUDE_DIR}/cuda/atomics.hpp
            ${CUDA4CPU_INCLUDE_DIR}/cuda/device.hpp
            ${CUDA4CPU_INCLUDE_DIR}/cuda/error.hpp
            ${CUDA4CPU_INCLUDE_DIR}/cuda/events.hpp
            ${CUDA4CPU_INCLUDE_DIR}/cuda/math.hpp
            ${CUDA4CPU_INCLUDE_DIR}/cuda/memory.hpp
            ${CUDA4CPU_INCLUDE_DIR}/cuda/mycuda.hpp
            ${CUDA4CPU_INCLUDE_DIR}/cuda/streams.hpp
            ${CUDA4CPU_INCLUDE_DIR}/cuda/types.hpp)

target_compile_features(cuda4cpu PUBLIC cxx_std_23)

set_target_properties(cuda4cpu PROPERTIES
    CXX_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

# Kernels are launched from header code that uses OpenMP, so consumers need it too
target_link_libraries(cuda4cpu PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(cuda4cpu PRIVATE -Wall)

# Debug information in every build, so that debuggers can show the CUDA thread
# that each OS thread runs (see tools/cuda4cpu-gdb.py)
target_compile_options(cuda4cpu PRIVATE -g)

# Let the compiler inline the scheduler's helpers into each other; otherwise
# every call between them goes through the PLT of the shared library
target_compile_options(cuda4cpu PRIVATE -fno-semantic-interposition)

# Fibers switch stacks with _setjmp/_longjmp, which glibc's fortified longjmp
# rejects ("longjmp causes uninitialized stack frame"). Some toolchains (e.g.
# Ubuntu's GCC) enable _FORTIFY_SOURCE by default. All the fiber code lives in
# the library, so this does not affect consumers.
target_compile_options(cuda4cpu PRIVATE -U_FORTIFY_SOURCE)

if(CUDA4CPU_ENABLE_NUMA)
    find_path(NUMA_INCLUDE_DIR numa.h)
    find_library(NUMA_LIBRARY numa)
    mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARY)

    if(NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
        message(STATUS "cuda4cpu: using libnuma (${NUMA_LIBRARY})")
        target_compile_definitions(cuda4cpu PRIVATE CUDA4CPU_HAVE_NUMA)
        target_include_directories(cuda4cpu PRIVATE ${NUMA_INCLUDE_DIR})
        target_link_libraries(cuda4cpu PRIVATE ${NUMA_LIBRARY})
    else()
        message(STATUS "cuda4cpu: libnuma not found, assuming a single NUMA node")
    endif()
endif()

if(CUDA4CPU_ENABLE_VALGRIND)
    find_path(VALGRIND_INCLUDE_DIR valgrind/valgrind.h REQUIRED)
    mark_as_advanced(VALGRIND_INCLUDE_DIR)

    target_compile_definitions(cuda4cpu PRIVATE CUDA4CPU_HANDLE_VALGRIND)
    target_include_directories(cuda4cpu PRIVATE ${VALGRIND_INCLUDE_DIR})
endif()

if(CUDA4CPU_PORTABLE_CONTEXT)
    target_compile_definitions(cuda4cpu PRIVATE CUDA4CPU_PORTABLE_CONTEXT)
endif()

if(CUDA4CPU_ENABLE_NATIVE)
    target_compile_options(cuda4cpu PUBLIC $<BUILD_INTERFACE:-march=native>)
endif()
