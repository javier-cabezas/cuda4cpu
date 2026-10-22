# A project that builds its CUDA code with cuda4cpu, fetched from GitHub.
# -DFETCHCONTENT_SOURCE_DIR_CUDA4CPU=<path> uses a local checkout instead.
cmake_minimum_required(VERSION 3.25)
project(cuda4cpu_example CXX)

include(FetchContent)
FetchContent_Declare(cuda4cpu
    GIT_REPOSITORY https://github.com/javier-cabezas/cuda4cpu.git
    GIT_TAG v0.1.0)
FetchContent_MakeAvailable(cuda4cpu)

add_executable(vecadd)
cuda4cpu_add_cuda_sources(vecadd vecadd.cu)

enable_testing()
add_test(NAME vecadd COMMAND vecadd)
