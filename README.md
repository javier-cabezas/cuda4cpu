![cuda4cpu](https://raw.githubusercontent.com/wiki/javier-cabezas/cuda4cpu/images/cuda4cpu_logo_300x75.png)

[![CI](https://github.com/javier-cabezas/cuda4cpu/actions/workflows/ci.yml/badge.svg)](https://github.com/javier-cabezas/cuda4cpu/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**cuda4cpu** is a C++ library and set of headers that lets you compile and run
CUDA kernels on CPUs with a regular C++ compiler. There's no GPU, no `nvcc` and
no source-to-source translation. You include one header and replace the
`<<<...>>>` launch syntax with a function call.

It's useful for debugging kernels with ordinary CPU tools (gdb, Valgrind),
for running CUDA code on machines without a GPU, and for prototyping.

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
#include <cuda_runtime.h>

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
    cudaMalloc(&A, N * sizeof(float));
    cudaMalloc(&B, N * sizeof(float));
    cudaMalloc(&C, N * sizeof(float));

    // Initialize input data
    // ...

    // Equivalent to: vecadd<<<N / 512, 512>>>(C, A, B, N);
    cuda4cpu::launch(vecadd, N / 512, 512).call(C, A, B, N);
    if (cudaGetLastError() != cudaSuccess)
        return 1;
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
| Linux / glibc, 64-bit | Fibers use an assembly context switch on x86-64, and `ucontext` with `_setjmp`/`_longjmp` elsewhere |
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
| `CUDA4CPU_BUILD_BENCHMARKS` | `ON` when top-level | Build `benchmarks/microbench`, which measures launch, barrier, shuffle and scheduling costs |
| `CUDA4CPU_PORTABLE_CONTEXT` | `OFF` | Use the `_setjmp`/`_longjmp` fallback instead of the x86-64 assembly context switch |

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

The target brings in everything your code needs: the include path, C++23
and OpenMP.

To compile `.cu` files as they are, have CMake treat them as C++, and include
`cuda_runtime.h` implicitly as nvcc does:

```cmake
set_source_files_properties(kernels.cu PROPERTIES LANGUAGE CXX)
target_compile_options(my_app PRIVATE -include cuda_runtime.h)
```

### Without CMake

```sh
g++ -std=c++23 -fopenmp -I/usr/local/include/cuda4cpu -c main.cpp
g++ -fopenmp -o my_app main.o -lcuda4cpu
```

For a `.cu` file, add `-x c++ -include cuda_runtime.h`.

## Porting a CUDA program

1. Use cuda4cpu's `cuda_runtime.h`. It provides the CUDA API in the global
   namespace, like the real header. Files that include it keep compiling
   unchanged. For files that rely on nvcc including it implicitly, pass
   `-include cuda_runtime.h` (see [Using cuda4cpu in your
   project](#using-cuda4cpu-in-your-project)).

2. Replace each kernel launch with `cuda4cpu::launch(kernel, grid, block).call(args...)`:

   ```cpp
   // CUDA
   stencil<<<dim3(64, 512), dim3(32, 4)>>>(B, A, cols);

   // cuda4cpu
   cuda4cpu::launch(stencil, dim3(64, 512), dim3(32, 4)).call(B, A, cols);
   ```

   Template kernels work the same way: `launch(stencil<4>, grid, block).call(...)`.
   The optional third and fourth launch parameters (dynamic shared memory size
   and stream) go after the block: `launch(kernel, grid, block, shared_bytes)`.
   Code that includes `cuda4cpu.hpp` and uses `using namespace cuda4cpu` can
   write `launch` without the namespace.

3. Replace `extern __shared__` arrays (dynamic shared memory) with a pointer.
   C++ has no way to express an array whose size is set at launch time:

   ```cpp
   // CUDA
   extern __shared__ float sdata[];

   // cuda4cpu
   float *sdata = dynamic_shared<float>();
   ```

4. Compile `.cu` files as C++ with your regular compiler. GCC warns about
   `#pragma unroll` with `-Wall`; add `-Wno-unknown-pragmas` if your kernels
   use it.

`tests/samples/` has ports of ten CUDA programs, each checked against a host
reference. Five are CUDA samples using kernel-language features: reduction,
matrix multiplication, histogram, prefix scan and n-body. The other five use
the runtime API as real applications do: vectorAdd, asyncAPI, deviceQuery, the
unified-memory program from NVIDIA's introduction to CUDA, and a softmax kernel.
Apart from dynamic shared memory, the only change from the CUDA originals is
the kernel launch.

## Supported CUDA features

**Kernel language**

- `__global__`, `__device__`, `__host__`, `__shared__`, `__constant__`,
  `__forceinline__`, `__launch_bounds__`
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, `warpSize`
- `__syncthreads()`, and dynamic shared memory through `dynamic_shared<T>()`
- Warp functions: `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`,
  `__shfl_xor_sync` (any type of up to 8 bytes, with `width`),
  `__ballot_sync`, `__any_sync`, `__all_sync`, `__activemask`, `__syncwarp`
- Atomics: `atomicAdd`, `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`,
  `atomicInc`, `atomicDec`, `atomicCAS`, `atomicAnd`, `atomicOr`, `atomicXor`
  for the same types as CUDA, and `__threadfence`, `__threadfence_block`,
  `__threadfence_system`
- `dim3`, the built-in vector types (`float4`, `int2`, `uchar3`, ...) and their
  `make_<type>` functions
- Math: the C++ standard library, plus `rsqrtf`, `rsqrt` and the fast
  intrinsics `__expf`, `__exp10f`, `__logf`, `__log2f`, `__log10f`, `__sinf`,
  `__cosf`, `__tanf`, `__sincosf`, `__powf`, `__fdividef` and `__saturatef`.
  They are computed at full precision.
- Integer intrinsics: `__popc`, `__popcll`, `__clz`, `__clzll`, `__ffs`,
  `__ffsll`, `__brev`, `__brevll`, and `__ldg`

**Runtime API**

| Area | Functions |
|---|---|
| Errors | `cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorName`, `cudaGetErrorString`, and `cudaError_t` with CUDA's codes |
| Device | `cudaGetDeviceCount`, `cudaSetDevice`, `cudaGetDevice`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, `cudaDriverGetVersion`, `cudaRuntimeGetVersion`, `cudaDeviceSynchronize`, `cudaDeviceReset`, `cudaDeviceSetLimit`, `cudaDeviceGetLimit` (`cudaLimitStackSize` only) |
| Memory | `cudaMalloc`, `cudaMallocHost`, `cudaHostAlloc`, `cudaMallocManaged` (with typed overloads, as in CUDA), `cudaFree`, `cudaFreeHost`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`, `cudaMemsetAsync`, `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`, `cudaGetSymbolAddress`, `cudaGetSymbolSize` |
| Streams | `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`, `cudaStreamDestroy`, `cudaStreamGetFlags`, `cudaStreamGetPriority`, `cudaStreamQuery`, `cudaStreamSynchronize`, `cudaStreamWaitEvent`, `cudaStreamAddCallback` |
| Events | `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventDestroy`, `cudaEventRecord`, `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventElapsedTime` |

## How it works

- **Grid → OpenMP threads.** `launch(...).call(...)` splits the grid's thread
  blocks into contiguous chunks, about 8 per CPU core, and hands them out to
  the cores dynamically, so blocks of uneven cost balance out.
- **Block → fibers.** Within a block, each CUDA thread is a user-level fiber
  with its own stack. On x86-64, switching fibers takes a dozen instructions
  that save and restore the callee-saved registers. That's about 12–20 ns per
  switch on a single core, including scheduling. Each OS thread creates
  fibers only once per block size, and reuses them for every block and every
  later launch.
- **No barriers, no switching.** A block starts by running its CUDA threads
  back to back, each one a direct call to the kernel. Kernels that never call
  `__syncthreads()` never switch fibers. `threadIdx`, `blockIdx`, `blockDim`
  and `gridDim` are thread-local values that the scheduler keeps up to date,
  so reading one is a single memory load.
- **`__syncthreads()` → cooperative switch.** When a CUDA thread first reaches
  a barrier, the block switches to fiber mode. From then on, each CUDA thread
  runs until it reaches a barrier and then yields to the next one. Once every
  live thread has reached the barrier, execution continues past it. As on
  GPUs, threads that return early don't hold back the barrier.
- **Warp functions → warp barriers.** A shuffle or vote is a barrier among the
  lanes of one warp. Each lane publishes its value and yields. When the last
  participating lane arrives, the results are computed for all of them and
  they continue.
- **Atomics → `std::atomic_ref`.** Thread blocks run in parallel on different
  cores, so atomics on global memory are real atomic operations. They are
  relaxed, as in CUDA.
- **`__shared__` → `static thread_local`.** A block always runs to completion
  on a single OS thread, so a thread-local variable behaves as per-block shared
  memory.
- **Device memory is host memory.** `cudaMalloc` is `malloc`, `cudaMemcpy` is
  `memcpy`, and streams and events run synchronously.

## Limitations

- **Linux/glibc on 64-bit only.** The context switch is written for x86-64.
  Other architectures use the `ucontext` and `_setjmp`/`_longjmp` fallback,
  which passes pointers through `makecontext` as two 32-bit halves.
- **`threadIdx` and the other built-in variables are values**, not
  variables you can take the address of. Reading their fields and copying
  them work as in CUDA.
- **Fixed-size stacks.** Each CUDA thread gets a 64 KiB stack with a guard
  page below it, so an overflow crashes with `SIGSEGV` instead of corrupting
  memory. Kernels with deep recursion or large local arrays can request more
  with `cudaDeviceSetLimit(cudaLimitStackSize, bytes)` before launching.
  Smaller requests are rounded up to 64 KiB.
- **No AddressSanitizer or ThreadSanitizer for kernels with barriers.** The
  sanitizers don't know about the fiber stack switches and report false
  errors. Valgrind works with `CUDA4CPU_ENABLE_VALGRIND=ON`.
- **Kernel launches are synchronous** and always run on the host. Streams and
  events exist for API compatibility only, and asynchronous copies complete
  before they return, so `cudaErrorNotReady` never occurs.
- **One device: the CPU.** `cudaGetDeviceProperties` reports the CPU's name,
  memory, L2 cache and clock, one multiprocessor per OS thread, CUDA's limits
  for blocks and grids, and compute capability 6.0. That's the newest whose
  features cuda4cpu implements (no tensor cores), so code that checks it picks
  supported paths.
- **Errors follow CUDA where the CPU can detect them.** Invalid arguments,
  failed allocations, and launches outside CUDA's limits (more than 1024
  threads per block, zero sizes, grids too large) return or record the CUDA
  error code, and rejected launches don't run. There are no
  device-side errors: an out-of-bounds access in a kernel is an ordinary
  segmentation fault.
- **Warp functions approximate convergence.** cuda4cpu can't see which lanes
  are in the same branch. A warp function waits for every lane in its mask
  that hasn't returned and isn't waiting at `__syncthreads()`. So
  `__activemask()` returns those lanes, and lanes in different branches that
  run warp functions at the same time must use disjoint masks, as CUDA
  requires.
- **Not implemented yet:** textures and surfaces, cooperative groups,
  `__match_*_sync` and `__reduce_*_sync`, 16-bit and scoped atomics
  (`atomicAdd_block`, ...), 2D/3D copies (`cudaMemcpy2D`, `cudaMallocPitch`),
  mapped host memory (`cudaHostGetDevicePointer`), `cudaFuncSetAttribute`, the
  double-precision and rounding-mode intrinsics, and the driver API
  (`cuda.h`).

## License

cuda4cpu is licensed under the [Apache License, Version 2.0](LICENSE).
