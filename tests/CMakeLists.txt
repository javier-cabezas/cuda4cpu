foreach(test IN ITEMS atomics barriers cooperative_groups divergent_barrier events launch memcpy_direction runtime
                      schedule stacks stencil2d stencil3d warp)
    add_executable(${test} ${test}.cpp)
    set_target_properties(${test} PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(${test} PRIVATE cuda4cpu::cuda4cpu)
    if(NOT test MATCHES "^(divergent_barrier|memcpy_direction|schedule)$")
        add_test(NAME ${test} COMMAND ${test})
    endif()
endforeach()

# runtime checks the errors of copies whose pointers don't match their kind
set_tests_properties(runtime PROPERTIES ENVIRONMENT "CUDA4CPU_MEMCPY_DIRECTION=error")

# Those copies with each CUDA4CPU_MEMCPY_DIRECTION: warn (the default) reports
# each kind of mistake once
add_test(NAME memcpy_direction_warn COMMAND memcpy_direction warn)
set_tests_properties(memcpy_direction_warn PROPERTIES
    PASS_REGULAR_EXPRESSION "cuda4cpu: warning: a copy with cudaMemcpyHostToDevice has its destination in host memory that CUDA didn't allocate[^\n]*\ncuda4cpu: warning: a copy with cudaMemcpyHostToDevice has its source in device memory[^\n]*\ncuda4cpu: warning: [^\n]*cudaMemcpyDeviceToHost[^\n]*\n.*memcpy_direction: passed")
add_test(NAME memcpy_direction_error COMMAND memcpy_direction error)
set_tests_properties(memcpy_direction_error PROPERTIES
    ENVIRONMENT "CUDA4CPU_MEMCPY_DIRECTION=error" FAIL_REGULAR_EXPRESSION "warning")
add_test(NAME memcpy_direction_ignore COMMAND memcpy_direction ignore)
set_tests_properties(memcpy_direction_ignore PROPERTIES
    ENVIRONMENT "CUDA4CPU_MEMCPY_DIRECTION=ignore" FAIL_REGULAR_EXPRESSION "warning")

# A missing __syncthreads() that the default order hides and the others expose
add_test(NAME schedule_forward COMMAND schedule hidden)
add_test(NAME schedule_reverse COMMAND schedule exposed)
add_test(NAME schedule_warp_reverse COMMAND schedule warp_exposed)
add_test(NAME schedule_random COMMAND schedule exposed)
add_test(NAME schedule_warp_random COMMAND schedule warp_exposed)
add_test(NAME schedule_mixed COMMAND schedule exposed)
set_tests_properties(schedule_forward PROPERTIES ENVIRONMENT "CUDA4CPU_SCHEDULE=forward")
set_tests_properties(schedule_reverse PROPERTIES ENVIRONMENT "CUDA4CPU_SCHEDULE=reverse")
set_tests_properties(schedule_warp_reverse PROPERTIES ENVIRONMENT "CUDA4CPU_SCHEDULE=warp_reverse")
set_tests_properties(schedule_random PROPERTIES ENVIRONMENT "CUDA4CPU_SCHEDULE=random:12345")
set_tests_properties(schedule_warp_random PROPERTIES ENVIRONMENT "CUDA4CPU_SCHEDULE=warp_random:12345")
set_tests_properties(schedule_mixed PROPERTIES ENVIRONMENT "CUDA4CPU_SCHEDULE=mixed:12345")

# Threads that reach different __syncthreads() calls: warned about by default,
# fatal or ignored on request
add_test(NAME divergent_barrier_warn COMMAND divergent_barrier)
set_tests_properties(divergent_barrier_warn PROPERTIES
    PASS_REGULAR_EXPRESSION "warning: threads of block \\([0-9], 0, 0\\) reached different __syncthreads\\(\\) calls, at [^ ]*divergent_barrier.cpp:[0-9]+ and [^ ]*divergent_barrier.cpp:[0-9]+"
    ENVIRONMENT "CUDA4CPU_DIVERGENT_BARRIERS=warn")
# The report names the two calls of divergent_later(), not the barrier before
add_test(NAME divergent_barrier_later COMMAND divergent_barrier later)
set_tests_properties(divergent_barrier_later PROPERTIES
    PASS_REGULAR_EXPRESSION "warning: threads of block \\([0-9], 0, 0\\) reached different __syncthreads\\(\\) calls, at [^ ]*divergent_barrier.cpp:63 and [^ ]*divergent_barrier.cpp:66"
    ENVIRONMENT "CUDA4CPU_DIVERGENT_BARRIERS=warn")
add_test(NAME divergent_barrier_ignore COMMAND divergent_barrier)
set_tests_properties(divergent_barrier_ignore PROPERTIES
    FAIL_REGULAR_EXPRESSION "different __syncthreads"
    ENVIRONMENT "CUDA4CPU_DIVERGENT_BARRIERS=ignore")
add_test(NAME divergent_warp_warn COMMAND divergent_barrier warp)
set_tests_properties(divergent_warp_warn PROPERTIES
    PASS_REGULAR_EXPRESSION "warning: lanes [0-9]+ and [0-9]+ of warp [0-9]+ of block \\([0-9], 0, 0\\) reached mismatched warp operations \\(__shfl_\\*_sync with mask 0xffffffff and warp vote with mask 0xffffffff\\)"
    ENVIRONMENT "CUDA4CPU_DIVERGENT_BARRIERS=warn")
add_test(NAME divergent_warp_mask COMMAND divergent_barrier warp_mask)
set_tests_properties(divergent_warp_mask PROPERTIES
    PASS_REGULAR_EXPRESSION "warning: lane 0 of warp [0-9]+ of block \\([0-9], 0, 0\\) called __shfl_\\*_sync with mask 0xfffffffe, which does not include its own lane"
    ENVIRONMENT "CUDA4CPU_DIVERGENT_BARRIERS=warn")
add_test(NAME divergent_warp_ignore COMMAND divergent_barrier warp)
set_tests_properties(divergent_warp_ignore PROPERTIES
    FAIL_REGULAR_EXPRESSION "mismatched warp operations"
    ENVIRONMENT "CUDA4CPU_DIVERGENT_BARRIERS=ignore")
add_test(NAME divergent_barrier_error
         COMMAND ${CMAKE_COMMAND} -DPROGRAM=$<TARGET_FILE:divergent_barrier>
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/divergent_barrier_error.cmake)

# The gdb script: a breakpoint conditioned on the CUDA thread, the built-in
# variables, and a backtrace that ends at the fiber trampoline
find_program(GDB gdb)
if(GDB)
    add_executable(barriers_debuginfo barriers.cpp)
    set_target_properties(barriers_debuginfo PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_options(barriers_debuginfo PRIVATE -g -O0)
    target_link_libraries(barriers_debuginfo PRIVATE cuda4cpu::cuda4cpu)
    add_test(NAME gdb
             COMMAND ${CMAKE_COMMAND} -DGDB=${GDB} -DSCRIPT=${PROJECT_SOURCE_DIR}/tools/cuda4cpu-gdb.py
                     -DPROGRAM=$<TARGET_FILE:barriers_debuginfo> -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/barriers.cpp
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/gdb_test.cmake)
endif()

# Ports of CUDA programs, checked against a host reference. They are .cu files
# in their original CUDA syntax, compiled through cuda4cpu_add_cuda_sources.
foreach(sample IN ITEMS async_api device_query histogram managed_add matmul nbody reduction scan
                        softmax vector_add)
    add_executable(sample_${sample})
    set_target_properties(sample_${sample} PROPERTIES CXX_EXTENSIONS OFF)
    cuda4cpu_add_cuda_sources(sample_${sample} samples/${sample}.cu)
    add_test(NAME sample_${sample} COMMAND sample_${sample})
endforeach()

# The same tests with the loop over the CUDA threads vectorized: Clang with
# CUDA4CPU_VECTORIZE_THREADS, and AVX2 for the masked loads and stores of
# bounds checks. Only where the machine running the tests has AVX2.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_FLAGS -march=x86-64-v3)
    check_cxx_source_runs("int main() { return __builtin_cpu_supports(\"avx2\") ? 0 : 1; }"
                          CUDA4CPU_HOST_HAS_AVX2)
    unset(CMAKE_REQUIRED_FLAGS)
endif()
add_executable(vectorize)
set_target_properties(vectorize PROPERTIES CXX_EXTENSIONS OFF)
cuda4cpu_add_cuda_sources(vectorize vectorize.cu)
add_test(NAME vectorize COMMAND vectorize)
if(CUDA4CPU_HOST_HAS_AVX2)
    add_executable(vectorize_vectorized)
    set_target_properties(vectorize_vectorized PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_definitions(vectorize_vectorized PRIVATE CUDA4CPU_VECTORIZE_THREADS)
    target_compile_options(vectorize_vectorized PRIVATE -march=x86-64-v3)
    cuda4cpu_add_cuda_sources(vectorize_vectorized vectorize.cu)
    add_test(NAME vectorize_vectorized COMMAND vectorize_vectorized)
    foreach(test IN ITEMS atomics barriers launch stencil2d stencil3d warp)
        add_executable(${test}_vectorized ${test}.cpp)
        set_target_properties(${test}_vectorized PROPERTIES CXX_EXTENSIONS OFF)
        target_compile_definitions(${test}_vectorized PRIVATE CUDA4CPU_VECTORIZE_THREADS)
        target_compile_options(${test}_vectorized PRIVATE -march=x86-64-v3)
        target_link_libraries(${test}_vectorized PRIVATE cuda4cpu::cuda4cpu)
        add_test(NAME ${test}_vectorized COMMAND ${test}_vectorized)
    endforeach()
    foreach(sample IN ITEMS histogram matmul softmax vector_add)
        add_executable(sample_${sample}_vectorized)
        set_target_properties(sample_${sample}_vectorized PROPERTIES CXX_EXTENSIONS OFF)
        target_compile_definitions(sample_${sample}_vectorized PRIVATE CUDA4CPU_VECTORIZE_THREADS)
        target_compile_options(sample_${sample}_vectorized PRIVATE -march=x86-64-v3)
        cuda4cpu_add_cuda_sources(sample_${sample}_vectorized samples/${sample}.cu)
        add_test(NAME sample_${sample}_vectorized COMMAND sample_${sample}_vectorized)
    endforeach()
endif()

# Half precision against an exact reference, and in kernels
add_executable(fp16)
set_target_properties(fp16 PROPERTIES CXX_EXTENSIONS OFF)
cuda4cpu_add_cuda_sources(fp16 fp16.cu)
add_test(NAME fp16 COMMAND fp16)

# Loops with interleaved threads (loop ticks inserted by the rewriter), with
# the interleaving chosen per kernel, off, and fixed
add_executable(interleave)
set_target_properties(interleave PROPERTIES CXX_EXTENSIONS OFF)
cuda4cpu_add_cuda_sources(interleave interleave.cu)
foreach(mode IN ITEMS auto off 1 3)
    add_test(NAME interleave_${mode} COMMAND interleave)
    set_tests_properties(interleave_${mode} PROPERTIES ENVIRONMENT "CUDA4CPU_INTERLEAVE=${mode}")
endforeach()

# CUDA graphs: capture, explicit graphs, conditionals and graph memory. A .cu
# file, so that (void *)kernel and params.gridDim go through the rewriter.
add_executable(graphs)
set_target_properties(graphs PROPERTIES CXX_EXTENSIONS OFF)
cuda4cpu_add_cuda_sources(graphs graphs.cu)
add_test(NAME graphs COMMAND graphs)

# Every launch form and extern __shared__ placement, checked at run time
add_executable(rewrite_features)
set_target_properties(rewrite_features PROPERTIES CXX_EXTENSIONS OFF)
cuda4cpu_add_cuda_sources(rewrite_features rewrite/features.cu)
add_test(NAME rewrite_features COMMAND rewrite_features)

# HIP programs, through HIP's headers and names only: the runtime API, the
# host APIs for memory and contexts, the kernel language, and
# hipLaunchKernelGGL as a function template in a .cpp file, which
# cuda4cpu_add_hip_sources compiles as HIP whatever its extension
foreach(source IN ITEMS hip/runtime.hip hip/memory.hip hip/kernels.hip hip/template_launch.cpp)
    cmake_path(GET source STEM stem)
    add_executable(hip_${stem})
    set_target_properties(hip_${stem} PROPERTIES CXX_EXTENSIONS OFF)
    cuda4cpu_add_hip_sources(hip_${stem} ${source})
    add_test(NAME hip_${stem} COMMAND hip_${stem})
endforeach()
set_tests_properties(hip_memory PROPERTIES ENVIRONMENT "CUDA4CPU_MEMCPY_DIRECTION=error")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME rewrite_unit
         COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/rewrite/test_rewrite.py
                 ${PROJECT_SOURCE_DIR}/tools/cuda4cpu-rewrite)

# The cuda4cpu-c++ driver, with its default compiler, the one that builds
# the project. CXX names the driver itself, as in CXX=cuda4cpu-c++ make,
# which it must skip instead of running itself until the timeout.
add_test(NAME driver
         COMMAND ${CMAKE_COMMAND} -DDRIVER=${PROJECT_BINARY_DIR}/tools/cuda4cpu-c++
                 -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/samples/vector_add.cu
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/driver
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/rewrite/driver_test.cmake)
set_tests_properties(driver PROPERTIES
                     ENVIRONMENT_MODIFICATION "CUDA4CPU_CXX=unset:;CXX=set:${PROJECT_BINARY_DIR}/tools/cuda4cpu-c++"
                     TIMEOUT 60)

# cuda4cpu-hipcc compiles a .cpp file as HIP, and drops options for AMD GPUs;
# cuda4cpu-c++ does with -x hip
add_test(NAME driver_hipcc
         COMMAND ${CMAKE_COMMAND} -DDRIVER=${PROJECT_BINARY_DIR}/tools/cuda4cpu-hipcc
                 -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/hip/template_launch.cpp
                 -DFLAGS=--offload-arch=gfx90a|-fgpu-rdc
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/driver_hipcc
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/rewrite/driver_test.cmake)
add_test(NAME driver_x_hip
         COMMAND ${CMAKE_COMMAND} -DDRIVER=${PROJECT_BINARY_DIR}/tools/cuda4cpu-c++
                 -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/hip/template_launch.cpp
                 -DFLAGS=-x|hip
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/driver_x_hip
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/rewrite/driver_test.cmake)
set_tests_properties(driver_hipcc driver_x_hip PROPERTIES ENVIRONMENT "CUDA4CPU_CXX=${CMAKE_CXX_COMPILER}")

# Passing the address of a symbol to cudaMemcpyToSymbol, which cuda4cpu
# accepted before it took symbols by name as CUDA does, must not compile
add_test(NAME symbol_address_rejected
         COMMAND ${CMAKE_CXX_COMPILER} -std=c++23 -fopenmp -fsyntax-only
                 -I${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/symbol_address.cpp)
set_tests_properties(symbol_address_rejected PROPERTIES
                     PASS_REGULAR_EXPRESSION "takes the symbol itself")
