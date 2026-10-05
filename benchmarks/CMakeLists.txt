add_executable(microbench microbench.cpp)
set_target_properties(microbench PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(microbench PRIVATE cuda4cpu::cuda4cpu)
