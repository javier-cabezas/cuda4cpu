add_executable(microbench microbench.cpp)
set_target_properties(microbench PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(microbench PRIVATE cuda4cpu::cuda4cpu)

# The same, with the loop over the CUDA threads vectorized (Clang, with AVX2
# or later in CMAKE_CXX_FLAGS)
add_executable(microbench_vectorized microbench.cpp)
set_target_properties(microbench_vectorized PROPERTIES CXX_EXTENSIONS OFF)
target_compile_definitions(microbench_vectorized PRIVATE CUDA4CPU_VECTORIZE_THREADS)
target_link_libraries(microbench_vectorized PRIVATE cuda4cpu::cuda4cpu)
