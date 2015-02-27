![two-level clustering](https://raw.githubusercontent.com/wiki/javier-cabezas/cuda4cpu/images/cuda4cpu_logo_300x75.png)

Library and headers to make CUDA codes run seamlessly on CPUs.

Using cuda4cpu
--------------

Include `cuda4hpc.hpp` file in those source files containing CUDA code:
```cpp
#include <cuda4hpc.hpp>
```
Use the `cuda4hpc` namespace to override CUDA keywords and types:
```cpp
using namespace cuda4hpc;
```
Use the `launch` function instead of the CUDA `<<<...>>>` notation to launch the cuda kernel. It returns a temporary object that you must use to pass the kernel arguments:
```cpp
launch(my_cuda_kernel, grid, block)(arguments...);
```
Compiler your code using C++11:
```
g++ -o object_file -c source_file -std=c++11
```
Link your program against `libcuda4cpu`:
```
g++ -o my_app object_files... -lcuda4cpu
```

Example
-------

This is the code of a vector addition using cuda4cpu:
```CUDA
#include <cuda4hpc.hpp>

using namespace cuda4hpc;

__global__
void vecadd(float *C, const float *A, const float *B, size_t elems)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < elems)
        C[i] = A[i] + B[i];
}

int main()
{
    float *A = new float[4096];
    float *B = new float[4096];
    float *C = new float[4096];

    // Initialize input data
    ...

    // Launch the kernel
    launch(vecadd, 4096 / 512, 512)(C, A, B);

    // Use output data
    ...

    delete []A;
    delete []B;
    delete []C;
}
```
