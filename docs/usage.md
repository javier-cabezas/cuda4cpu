# Building and using cuda4cpu

## Requirements

| Requirement | Notes |
|---|---|
| C++23 compiler | GCC 14+ or Clang 18+ |
| CMake 3.25+ | Ninja is recommended (the presets use it) |
| OpenMP | Bundled with GCC; Clang needs `libomp` |
| Linux / glibc, 64-bit | Fibers use an assembly context switch on x86-64, and `ucontext` with `_setjmp`/`_longjmp` elsewhere |
| Python 3 | Runs `cuda4cpu-rewrite` when compiling `.cu` and HIP files (standard library only) |
| libnuma (optional) | Used for NUMA topology discovery |
| Valgrind (optional) | Headers needed for `CUDA4CPU_ENABLE_VALGRIND` |

On Debian/Ubuntu:

```sh
sudo apt-get install cmake ninja-build g++ libnuma-dev
```

## Configure, build and test

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

## Build options

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

After installing cuda4cpu, add `.cu` files with `cuda4cpu_add_cuda_sources`:

```cmake
find_package(cuda4cpu 0.1 REQUIRED)

add_executable(my_app main.cpp)
cuda4cpu_add_cuda_sources(my_app kernels.cu solver.cu)
```

At build time, each `.cu` file is rewritten by `cuda4cpu-rewrite`, along with
the local headers it includes with `#include "..."`. The result is compiled as
C++, with `cuda_runtime.h` included implicitly as nvcc does. Changing a `.cu`
file or one of its headers triggers a new rewrite. The function also links
the target against `cuda4cpu::cuda4cpu`, which brings in the include path,
C++23 and OpenMP. C++ files that use cuda4cpu directly only need
`target_link_libraries(my_app PRIVATE cuda4cpu::cuda4cpu)`.

HIP sources go through `cuda4cpu_add_hip_sources`, whatever their extension,
since HIP projects often compile `.cpp` files with hipcc.
`cuda4cpu_add_cuda_sources` also takes `.hip` files.

```cmake
cuda4cpu_add_hip_sources(my_app kernels.hip main.cpp)
```

You can also vendor cuda4cpu as a subdirectory, with `add_subdirectory(cuda4cpu)`
or `FetchContent`. The functions and the target are then available.
[examples/cmake-project](../examples/cmake-project) is a complete project that
fetches cuda4cpu from GitHub.

### Projects written for CMake's CUDA or HIP language

A project that already builds with nvcc through CMake's own CUDA support
(`project(... LANGUAGES CUDA)`, `enable_language(CUDA)`,
`find_package(CUDAToolkit)` and `CUDA::cudart`) builds unchanged with a
toolchain file, with CMake 4.1 or later (older CMake only accepts NVIDIA's
own nvcc):

```sh
cmake -B build -DCMAKE_TOOLCHAIN_FILE=<prefix>/lib/cuda4cpu/toolkit/cuda4cpu.cmake
```

In the build tree, it's `build/<preset>/toolkit/cuda4cpu.cmake`. It makes
`cuda4cpu-nvcc` the CUDA compiler, which CMake identifies as NVIDIA's nvcc
12.0, with the toolkit directory (below) as the CUDA toolkit. It makes
`cuda4cpu-c++` the C++ compiler, so that `.cpp` files that include
`cuda_runtime.h` get what cuda4cpu's headers need (C++23, OpenMP and the
include path), and programs get an rpath to libcuda4cpu. CMake's CUDA
architectures, separable compilation (`CUDA_SEPARABLE_COMPILATION`, device
linking), `CUDA_STANDARD` and the toolkit's runtime target
(`CUDA::cudart`) work; libraries that cuda4cpu doesn't
implement (cuBLAS, ...) aren't found. With it, all 43 of NVIDIA's samples that
pass in the compatibility suite build with their own `CMakeLists.txt` and
pass.

The same file sets up HIP on its NVIDIA platform (`project(... LANGUAGES
HIP)`, with `CMAKE_HIP_PLATFORM=nvidia`), whose compiler is nvcc.

### Without CMake

`cuda4cpu-c++` is a compiler driver: use it like your C++ compiler, with `.cu`
files among the inputs.

```sh
cuda4cpu-c++ -O2 -o my_app main.cpp kernels.cu
cuda4cpu-c++ -O2 -c kernels.cu        # kernels.o, as with any compiler
```

It rewrites the `.cu` files, adds the C++ standard, OpenMP and the cuda4cpu
include directory, and links `libcuda4cpu` when it links. It runs
`$CUDA4CPU_CXX`, `$CXX` (unless that's a cuda4cpu driver, as in
`CXX=cuda4cpu-c++ make`), or else the compiler that built `libcuda4cpu`. Use
the same kind of compiler as the library: GCC and Clang have their own
OpenMP runtimes, and a program with both has two pools of threads. It takes
the compiler's options; `cuda4cpu-nvcc` (below) takes nvcc's. `.hip` files
are HIP sources, and `-x cu` or `-x hip` set the language of the files after
them, as with clang.

`cuda4cpu-hipcc` is the same driver for HIP projects. Like hipcc, it compiles
every C++ source file (`.cpp`, `.cc`, `.cu`, `.hip`, ...) as HIP. It drops
hipcc's options for AMD GPUs, such as `--offload-arch=gfx90a`, `-fgpu-rdc`
and `-D__HIP_PLATFORM_AMD__`, so build scripts written for hipcc work unchanged:

```sh
cuda4cpu-hipcc -O2 --offload-arch=gfx90a -o my_app main.cpp kernels.cpp
```

As hipcc, it answers `--version` (HIP 7.2.0), builds a static library from
sources with `--emit-static-lib`, adds `HIPCC_COMPILE_FLAGS_APPEND` (and
`HIPCC_LINK_FLAGS_APPEND` when it links), and shows its commands with
`HIPCC_VERBOSE=1`. `cuda4cpu-hipconfig` answers as HIP's `hipconfig`:
`--platform` is `nvidia`, `--compiler` `nvcc`, `--path` and `--rocmpath` the
toolkit directory (below, or `HIP_PATH`/`ROCM_PATH`), and `--cpp_config`,
`--version` and `--full` what HIP's print.

To run the rewriter on its own: `cuda4cpu-rewrite kernels.cu -o kernels.cpp`.

### Build scripts written for nvcc

`cuda4cpu-nvcc` takes nvcc's command line, so Makefiles and scripts that call
nvcc build unchanged:

```sh
cuda4cpu-nvcc -O3 -arch=sm_80 -Xcompiler -Wall,-fPIC -std=c++17 \
    --expt-relaxed-constexpr -lineinfo -o my_app main.cu -lcudart
cuda4cpu-nvcc -dc -o kernels.o kernels.cu          # separate compilation:
cuda4cpu-nvcc -dlink -o dlink.o kernels.o          #   an empty object
cuda4cpu-nvcc -lib -o libkernels.a kernels.o dlink.o
```

- Options for the GPU and for nvcc's own steps are accepted and ignored:
  `-arch`, `-gencode`, `-code`, `-Xptxas`, `-rdc`, `-maxrregcount`,
  `--expt-relaxed-constexpr`, `--extended-lambda`, `-lineinfo`,
  `-use_fast_math`, `--default-stream`, `-Wno-deprecated-gpu-targets`, ...
- nvcc's names for the host compiler's options are translated:
  `-Xcompiler`/`--compiler-options` (comma-separated), `-Xlinker`, `-ccbin`
  (`-ccbin gcc-14` compiles with `g++-14`), `-dc`, `-G`, `--include-path`,
  `--define-macro`, `--generate-dependencies-with-compile`, ... Other options
  go to the host compiler as they are.
- `-lcudart`, `-lcudart_static`, `-lcuda`, `-lcudadevrt` and `-lnvToolsExt`
  are dropped: libcuda4cpu provides them.
- `-cuda` preprocesses each `.cu` file into `<file>.cpp.ii`, `-dlink` writes
  an empty object (there's no device code to link), and `-lib` archives
  objects. `-ptx`, `-cubin` and `-fatbin` fail: they make GPU code.
- `--version` prints a banner with nvcc's version line (12.0, the version of
  CUDA that cuda4cpu implements), `-v` shows each command before it runs, and
  `--dryrun` only shows them. `NVCC_PREPEND_FLAGS` and `NVCC_APPEND_FLAGS`
  add options, as with nvcc.
- It defines `__NVCC__`, `__CUDACC_VER_MAJOR__` (12), `__CUDACC_VER_MINOR__`
  and `__CUDACC_VER_BUILD__`, as nvcc does. `.c` files are compiled as C, as
  nvcc does.
- cuda4cpu needs C++23, so an older `-std` (`c++17`, `gnu++20`, ...) is
  raised to it, in the same dialect: nvcc's default is C++17, and host code
  that includes `cuda_runtime.h` needs C++23 too. `cuda4cpu-c++` and
  `cuda4cpu-hipcc` do the same.

Build systems that look for nvcc in a CUDA toolkit (`$(CUDA_PATH)/bin/nvcc`,
`-I$(CUDA_PATH)/include`, `-L$(CUDA_PATH)/lib64 -lcudart`) find one in
`<prefix>/lib/cuda4cpu/toolkit` (`build/<preset>/toolkit` in the build
tree): `bin/nvcc` is `cuda4cpu-nvcc`, `include` has cuda4cpu's headers, and
`lib64`'s `libcudart.so`, `libcudart_static.a` and `libcuda.so` link
libcuda4cpu and its OpenMP runtime.

```sh
make CUDA_PATH=/usr/local/lib/cuda4cpu/toolkit
```

Programs that `cuda4cpu-nvcc` links find libcuda4cpu at run time by
themselves. Those that the host compiler links with `-lcudart` find it as
programs find CUDA's: through `LD_LIBRARY_PATH=<toolkit>/lib64`, unless
cuda4cpu is installed where the dynamic loader looks.

The toolkit is laid out like a ROCm installation too, for build systems
written for hipcc (`$(ROCM_PATH)/bin/hipcc`, `HIP_PATH`, `hipconfig`,
`-L$(ROCM_PATH)/lib -lamdhip64`, `find_package(hip)`): `bin/hipcc` is
`cuda4cpu-hipcc`, `bin/hipconfig` `cuda4cpu-hipconfig`, and `lib` has
`libamdhip64.so` (which links libcuda4cpu) and HIP's CMake package
(`hip::host`, `hip::device`). With it, AMD's rocm-examples build with their
own Makefiles (`make ROCM_INSTALL_DIR=<toolkit>`, or `GPU_RUNTIME=CUDA` with
the toolkit's nvcc) and their own CMake files (the toolchain file, with
`-DROCM_ROOT=<toolkit>` and `GPU_RUNTIME` `HIP` or `CUDA`): of the 70 that
pass in the compatibility suite, 56 to 62 build and run that way, depending
on the build. Most of the others use AMD's own tools (assembly, LLVM IR,
hipify) or AMD-only build steps. A few compile with the plain C++ compiler,
which `CXX=<toolkit>/bin/cuda4cpu-c++` fixes.

## Porting a CUDA program

CUDA code compiles as it is: there are no source changes. The rewriting turns
the two pieces of syntax C++ lacks into cuda4cpu calls:

```cpp
// CUDA                                    // what the compiler sees
kernel<<<grid, block, smem, stream>>>(x);  cuda4cpu::launch([&](const auto &...a) { kernel(a...); },
                                                           grid, block, smem, stream).call(x);
extern __shared__ float sdata[];           float *sdata = cuda4cpu::dynamic_shared<float>();
```

The kernel, the launch configuration and the arguments are copied as they are.
So template arguments are deduced from the arguments and default arguments
apply, as in CUDA. Because the compiler knows which kernel each launch runs, it
can inline it. Line numbers are preserved, and diagnostics and `__FILE__` refer
to the original file.

The rewriter also handles a few things that only nvcc understands:

- **`(void *)kernel`** becomes `cuda4cpu::kernel_address(kernel)` for any
  `__global__` function of the file or its local headers. It records the
  kernel's parameter types, which graph kernel nodes and `cudaLaunchKernel`
  need to unpack an argument array. The kernel passed to `cudaLaunchKernel`
  just loses the cast.
- **`params.gridDim` and `params.blockDim`** become `params.cuda4cpu_gridDim`
  and `params.cuda4cpu_blockDim`. `gridDim` and `blockDim` are macros for the
  built-in variables, so the fields of `cudaKernelNodeParams` have other names.
- **`clock()` in `__global__` and `__device__` functions** becomes the device
  clock, which counts at the reported `clockRate`, like `clock64()`. On the
  host, `clock()` stays the C library's.
- **`__CUDACC__`** is defined, as nvcc does, so code that checks for a CUDA
  compiler takes the CUDA branch.

C++ code can also use the API directly, without the rewriter:
`cuda4cpu::launch(kernel, grid, block[, smem]).call(args...)` and
`cuda4cpu::dynamic_shared<T>()`. Code that includes `cuda4cpu.hpp` and uses
`using namespace cuda4cpu` can write `launch` without the namespace.

GCC warns about `#pragma unroll` with `-Wall`; add `-Wno-unknown-pragmas` if
your kernels use it.

`tests/samples/` has ten CUDA programs, unmodified `.cu` files built with
`cuda4cpu_add_cuda_sources`, each checked against a host reference:

- **Kernel-language features:** five CUDA samples (reduction, matrix
  multiplication, histogram, prefix scan and n-body).
- **The runtime API, as real applications use it:** vectorAdd, asyncAPI,
  deviceQuery, the unified-memory program from NVIDIA's introduction to CUDA,
  and a softmax kernel.
