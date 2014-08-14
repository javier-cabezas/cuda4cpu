aux_source_directory(lib LIB_SRC)
aux_source_directory(include LIB_INCLUDE)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")

add_executable(stencil2d stencil2d.cpp)
add_executable(stencil3d stencil3d.cpp)
target_link_libraries(stencil2d ${CMAKE_IMPORT_LIBRARY_PREFIX}cuda4cpu${CMAKE_IMPORT_LIBRARY_SUFFIX})
target_link_libraries(stencil3d ${CMAKE_IMPORT_LIBRARY_PREFIX}cuda4cpu${CMAKE_IMPORT_LIBRARY_SUFFIX})
