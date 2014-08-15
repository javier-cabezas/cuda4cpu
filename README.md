cuda4cpu
========

Library and headers to make CUDA codes run seamlessly on CPUs.

Using cuda4cpu
--------------

1. Include `cuda4hpc.hpp` file in those source files containing CUDA code.
```cpp
#include <cuda4hpc.hpp>
```
2. Use the `cuda4hpc` namespace to override CUDA keywords and types.
```cpp
using namespace cuda4hpc;
```
3. Use the `launch` function instead of the CUDA `<<<...>>>` notation to launch the cuda kernel. It returns a temporary object that you must use to pass the kernel arguments:
```cpp
launch(my_cuda_kernel, grid, block)(arguments...);
```
4. Link your program against `libcuda4cpu`
```
g++ -o my_app object_files... -lcuda4cpu
```
