![cuda4cpu](https://raw.githubusercontent.com/wiki/javier-cabezas/cuda4cpu/images/cuda4cpu_logo_300x75.png)

[![CI](https://github.com/javier-cabezas/cuda4cpu/actions/workflows/ci.yml/badge.svg)](https://github.com/javier-cabezas/cuda4cpu/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**cuda4cpu** is a C++ library and set of headers that lets you compile and run
CUDA kernels on CPUs with a regular C++ compiler. There's no GPU, no `nvcc` and
no source-to-source translation. You include one header and replace the
`<<<...>>>` launch syntax with a function call.

It's useful for debugging kernels with ordinary CPU tools (gdb, Valgrind,
sanitizers), for running CUDA code on machines without a GPU, and for
prototyping.

## Contents

- [Quick example](#quick-example)
- [Building](#building)
- [Using cuda4cpu in your project](#using-cuda4cpu-in-your-project)
- [Porting a CUDA program](#porting-a-cuda-program)
- [Supported CUDA features](#supported-cuda-features)
- [How it works](#how-it-works)
- [Limitations](#limitations)
- [License](#license)

## Quick example

```cpp
#include <cuda4cpu.hpp>

using namespace cuda4cpu;

__global__
void vecadd(float *C, const float *A, const float *B, size_t elems)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < elems)
        C[i] = A[i] + B[i];
}

int main()
{
    constexpr size_t N = 4096;

    float *A, *B, *C;
    cudaMalloc((void **)&A, N * sizeof(float));
    cudaMalloc((void **)&B, N * sizeof(float));
    cudaMalloc((void **)&C, N * sizeof(float));

    // Initialize input data
    // ...

    // Equivalent to: vecadd<<<N / 512, 512>>>(C, A, B, N);
    launch(vecadd, N / 512, 512).call(C, A, B, N);
    cudaDeviceSynchronize();

    // Use output data
    // ...

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
```

## Building

### Requirements

| Requirement | Notes |
|---|---|
| C++23 compiler | GCC 14+ or Clang 18+ |
| CMake 3.25+ | Ninja is recommended (the presets use it) |
| OpenMP | Bundled with GCC; Clang needs `libomp` |
| Linux / glibc | Fibers use `ucontext` and `_setjmp`/`_longjmp` |
| libnuma (optional) | Used for NUMA topology discovery |
| Valgrind (optional) | Headers needed for `CUDA4CPU_ENABLE_VALGRIND` |

On Debian/Ubuntu:

```sh
sudo apt-get install cmake ninja-build g++ libnuma-dev
```

### Configure, build and test

The repository ships CMake presets for `debug` and `release`:

```sh
cmake --workflow --preset release      # configure + build + test
```

Or step by step:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Build trees go to `build/<preset>/`. To install:

```sh
cmake --install build/release --prefix /usr/local
```

### Build options

| Option | Default | Description |
|---|---|---|
| `BUILD_SHARED_LIBS` | `ON` | Build `libcuda4cpu` as a shared library (`OFF` for static) |
| `CUDA4CPU_BUILD_TESTS` | `ON` when top-level | Build the tests and register them with CTest |
| `CUDA4CPU_ENABLE_NUMA` | `ON` | Use libnuma if found; otherwise assume a single NUMA node |
| `CUDA4CPU_ENABLE_VALGRIND` | `OFF` | Register fiber stacks with Valgrind so it doesn't report false errors |
| `CUDA4CPU_ENABLE_NATIVE` | `OFF` | Compile in-tree targets with `-march=native` |

Pass options at configure time, for example
`cmake --preset debug -DCUDA4CPU_ENABLE_VALGRIND=ON`.

## Using cuda4cpu in your project

### With CMake (recommended)

After installing cuda4cpu:

```cmake
find_package(cuda4cpu 0.1 REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE cuda4cpu::cuda4cpu)
```

Or vendor it as a subdirectory, with `add_subdirectory(cuda4cpu)` or
`FetchContent`, and link against the same `cuda4cpu::cuda4cpu` target.

The target brings in everything your code needs: the include path, C++23,
OpenMP and `-U_FORTIFY_SOURCE` (see [Limitations](#limitations)).

### Without CMake

```sh
g++ -std=c++23 -fopenmp -U_FORTIFY_SOURCE -I/usr/local/include/cuda4cpu -c main.cpp
g++ -fopenmp -o my_app main.o -lcuda4cpu
```

## Porting a CUDA program

1. Include `cuda4cpu.hpp` in every source file that contains CUDA code, and
   bring the CUDA runtime names into scope:

   ```cpp
   #include <cuda4cpu.hpp>
   using namespace cuda4cpu;
   ```

2. Replace each kernel launch with `launch(kernel, grid, block).call(args...)`:

   ```cpp
   // CUDA
   stencil<<<dim3(64, 512), dim3(32, 4)>>>(B, A, cols);

   // cuda4cpu
   launch(stencil, dim3(64, 512), dim3(32, 4)).call(B, A, cols);
   ```

   Template kernels work the same way: `launch(stencil<4>, grid, block).call(...)`.

3. Pass the address of `__constant__` variables to `cudaMemcpyToSymbol`:

   ```cpp
   cudaMemcpyToSymbol(&coeffs, host_coeffs, sizeof(coeffs));
   ```

4. Rename `.cu` files to `.cpp` (or tell your build system to compile them as
   C++) and build with your regular compiler.

The `tests/` directory has complete 2D and 3D stencil examples that use shared
memory, `__constant__` memory and `__syncthreads()`.

## Supported CUDA features

**Kernel language**

- `__global__`, `__device__`, `__host__`, `__shared__`, `__constant__`
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- `__syncthreads()`
- `dim3` and the built-in vector types (`float4`, `int2`, `uchar3`, ...)

**Runtime API**

| Area | Functions |
|---|---|
| Device | `cudaDeviceSynchronize` |
| Memory | `cudaMalloc`, `cudaFree`, `cudaMallocHost`, `cudaHostAlloc`, `cudaFreeHost`, `cudaMemcpy`, `cudaMemcpyToSymbol` |
| Streams | `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`, `cudaStreamDestroy`, `cudaStreamGetFlags`, `cudaStreamGetPriority`, `cudaStreamQuery`, `cudaStreamSynchronize`, `cudaStreamWaitEvent`, `cudaStreamAddCallback` |
| Events | `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventDestroy`, `cudaEventRecord`, `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventElapsedTime` |

## How it works

- **Grid → OpenMP threads.** `launch(...).call(...)` splits the grid's thread
  blocks into contiguous chunks and runs one chunk per CPU core inside an
  OpenMP parallel loop.
- **Block → fibers.** Within a block, each CUDA thread is a user-level fiber
  with its own stack. Fibers are created with `makecontext` and switched with
  `_setjmp`/`_longjmp`, which is much cheaper than switching OS threads.
- **`__syncthreads()` → cooperative switch.** Each CUDA thread runs until it
  reaches a barrier and then yields to the next fiber. Once every fiber has
  reached the barrier, execution continues past it.
- **`__shared__` → `static thread_local`.** A block always runs to completion
  on a single OS thread, so a thread-local variable behaves as per-block shared
  memory.
- **Device memory is host memory.** `cudaMalloc` is `malloc`, `cudaMemcpy` is
  `memcpy`, and streams and events run synchronously.

## Limitations

- **Linux/glibc on 64-bit only.** The fiber implementation relies on
  `ucontext` and passes pointers through `makecontext` as two 32-bit halves.
- **`_FORTIFY_SOURCE` must be off** in files that launch kernels. glibc's
  fortified `longjmp` aborts with *"longjmp causes uninitialized stack frame"*
  when fibers switch stacks. The CMake target passes `-U_FORTIFY_SOURCE` for
  you; otherwise add it yourself (Ubuntu's GCC enables fortify by default).
- **Small fiber stacks.** Each CUDA thread gets a `SIGSTKSZ`-sized stack. Deep
  recursion or large local arrays in kernels can overflow it.
- **Kernel launches are synchronous** and always run on the host. Streams and
  events exist for API compatibility only.
- **Not implemented yet:** atomics, warp-level intrinsics (`__shfl*`,
  `__ballot`, ...), dynamic shared memory, textures/surfaces, unified memory
  and error-reporting APIs. Every function returns `0` (success).

## License

cuda4cpu is licensed under the [Apache License, Version 2.0](LICENSE).
