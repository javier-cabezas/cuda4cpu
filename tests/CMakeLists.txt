foreach(test IN ITEMS atomics barriers events launch runtime stacks stencil2d stencil3d warp)
    add_executable(${test} ${test}.cpp)
    set_target_properties(${test} PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(${test} PRIVATE cuda4cpu::cuda4cpu)
    add_test(NAME ${test} COMMAND ${test})
endforeach()

# Ports of CUDA samples, checked against a host reference. They also print
# the time of the cuda4cpu version and of the host reference.
foreach(sample IN ITEMS async_api device_query histogram managed_add matmul nbody reduction scan
                        softmax vector_add)
    add_executable(sample_${sample} samples/${sample}.cpp)
    set_target_properties(sample_${sample} PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(sample_${sample} PRIVATE cuda4cpu::cuda4cpu)
    add_test(NAME sample_${sample} COMMAND sample_${sample})
endforeach()

# Like the .cu file it comes from, managed_add includes no CUDA header: nvcc
# includes cuda_runtime.h implicitly, which -include reproduces
target_compile_options(sample_managed_add PRIVATE -include cuda_runtime.h)

# Passing the address of a symbol to cudaMemcpyToSymbol, which cuda4cpu
# accepted before it took symbols by name as CUDA does, must not compile
add_test(NAME symbol_address_rejected
         COMMAND ${CMAKE_CXX_COMPILER} -std=c++23 -fopenmp -fsyntax-only
                 -I${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/symbol_address.cpp)
set_tests_properties(symbol_address_rejected PROPERTIES
                     PASS_REGULAR_EXPRESSION "takes the symbol itself")
