foreach(test IN ITEMS atomics barriers events launch runtime stacks stencil2d stencil3d warp)
    add_executable(${test} ${test}.cpp)
    set_target_properties(${test} PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(${test} PRIVATE cuda4cpu::cuda4cpu)
    add_test(NAME ${test} COMMAND ${test})
endforeach()

# Ports of CUDA programs, checked against a host reference. They are .cu files
# in their original CUDA syntax, compiled through cuda4cpu_add_cuda_sources.
foreach(sample IN ITEMS async_api device_query histogram managed_add matmul nbody reduction scan
                        softmax vector_add)
    add_executable(sample_${sample})
    set_target_properties(sample_${sample} PROPERTIES CXX_EXTENSIONS OFF)
    cuda4cpu_add_cuda_sources(sample_${sample} samples/${sample}.cu)
    add_test(NAME sample_${sample} COMMAND sample_${sample})
endforeach()

# Every launch form and extern __shared__ placement, checked at run time
add_executable(rewrite_features)
set_target_properties(rewrite_features PROPERTIES CXX_EXTENSIONS OFF)
cuda4cpu_add_cuda_sources(rewrite_features rewrite/features.cu)
add_test(NAME rewrite_features COMMAND rewrite_features)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME rewrite_unit
         COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/rewrite/test_rewrite.py
                 ${PROJECT_SOURCE_DIR}/tools/cuda4cpu-rewrite)

# The cuda4cpu-c++ driver, with the compiler that builds the project
add_test(NAME driver
         COMMAND ${CMAKE_COMMAND} -DDRIVER=${PROJECT_BINARY_DIR}/tools/cuda4cpu-c++
                 -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/samples/vector_add.cu
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/driver
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/rewrite/driver_test.cmake)
set_tests_properties(driver PROPERTIES ENVIRONMENT "CUDA4CPU_CXX=${CMAKE_CXX_COMPILER}")

# Passing the address of a symbol to cudaMemcpyToSymbol, which cuda4cpu
# accepted before it took symbols by name as CUDA does, must not compile
add_test(NAME symbol_address_rejected
         COMMAND ${CMAKE_CXX_COMPILER} -std=c++23 -fopenmp -fsyntax-only
                 -I${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/symbol_address.cpp)
set_tests_properties(symbol_address_rejected PROPERTIES
                     PASS_REGULAR_EXPRESSION "takes the symbol itself")
