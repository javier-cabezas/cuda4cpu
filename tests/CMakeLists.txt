foreach(test IN ITEMS atomics barriers events launch stacks stencil2d stencil3d warp)
    add_executable(${test} ${test}.cpp)
    set_target_properties(${test} PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(${test} PRIVATE cuda4cpu::cuda4cpu)
    add_test(NAME ${test} COMMAND ${test})
endforeach()

# Ports of CUDA samples, checked against a host reference. They also print
# the time of the cuda4cpu version and of the host reference.
foreach(sample IN ITEMS histogram matmul nbody reduction scan)
    add_executable(sample_${sample} samples/${sample}.cpp)
    set_target_properties(sample_${sample} PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(sample_${sample} PRIVATE cuda4cpu::cuda4cpu)
    add_test(NAME sample_${sample} COMMAND sample_${sample})
endforeach()
