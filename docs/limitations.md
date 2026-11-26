# Limitations

- **Linux/glibc on 64-bit only.** The context switch is written for x86-64.
  Other architectures use the `ucontext` and `_setjmp`/`_longjmp` fallback,
  which passes pointers through `makecontext` as two 32-bit halves.
- **`threadIdx` and the other built-in variables are values**, not
  variables you can take the address of. Reading their fields and copying
  them work as in CUDA.
- **Fixed-size stacks.** Each CUDA thread gets a 64 KiB stack with a guard
  page below it, so an overflow is reported and crashes with `SIGSEGV`
  instead of corrupting memory. Kernels with deep recursion or large local arrays can request more
  with `cudaDeviceSetLimit(cudaLimitStackSize, bytes)` before launching.
  Smaller requests are rounded up to 64 KiB. Launches whose arguments take
  4 KiB or more get room for copies of them on top.
- **No ThreadSanitizer yet.** It reports false races between kernels and
  host code, because OpenMP synchronizes its threads in ways it can't see.
  AddressSanitizer, UBSan and Valgrind work (see [Debugging](debugging.md)).
- **Kernel launches are synchronous** and always run on the host. Streams and
  events exist for API compatibility only, and asynchronous copies complete
  before they return, so `cudaErrorNotReady` never occurs.
- **`cuda4cpu_add_cuda_sources` rewrites local headers only**, and so does
  `cuda4cpu_add_hip_sources`. Through CMake's functions, headers included with `#include "..."` relative to the including
  file are rewritten; headers found through include directories (`-I`) or
  `#include <...>` are not, so a launch or `extern __shared__` array in one of
  those doesn't compile. The drivers (`cuda4cpu-c++`, `cuda4cpu-nvcc`,
  `cuda4cpu-hipcc`) rewrite those too.
- **`extern __shared__` at namespace scope** becomes an object that converts
  to a pointer and supports subscripts, but not `reinterpret_cast` or
  `sizeof`. Inside functions it is a plain pointer. Alignment specifiers are
  dropped: dynamic shared memory is always 64-byte aligned.
- **One device: the CPU.** `cudaGetDeviceProperties` reports the CPU's name,
  memory, L2 cache and clock, one multiprocessor per OS thread, CUDA's limits
  for blocks and grids, and compute capability 6.0. That's the newest whose
  features cuda4cpu has all of, so code that checks it picks supported paths.
  Code that checks it before using tensor cores (`mma.h`) needs
  `CUDA4CPU_COMPUTE_CAPABILITY=7.0` or later (8.0 for bfloat16, tf32 and
  double), which may also lead it to features that cuda4cpu doesn't have.
- **`__CUDA_ARCH__` isn't defined**, because host and device code compile
  together: code inside `#if __CUDA_ARCH__ >= 800` doesn't compile in.
  Defining it yourself (`-D__CUDA_ARCH__=800`) compiles that code in, but
  also drops the host code under `#ifndef __CUDA_ARCH__`.
- **Errors follow CUDA where the CPU can detect them.** Invalid arguments,
  failed allocations, and launches outside CUDA's limits (more than 1024
  threads per block, zero sizes, grids too large) return or record the CUDA
  error code, and rejected launches don't run. A failed `assert()` in a
  kernel is CUDA's: each thread that fails it prints CUDA's message and
  ends, and `cudaErrorAssert` is sticky until `cudaDeviceReset()`. The
  synchronizing calls, copies, memsets, launches (graphs too) and
  `cudaGetLastError` return it, and launches and blocks that haven't started
  don't run; other calls still work, where CUDA's would fail. cuda4cpu
  catches the assert through glibc's `__assert_fail`, which programs that
  link cuda4cpu find first: code in a library that a program without it
  loads with `dlopen` aborts instead, and so does a fully static program
  (`-static`) that also uses glibc's `__assert_perror_fail`, which doesn't
  link. Other
  device-side errors end the program: an invalid access is reported with
  the CUDA thread, its block and the kernel's launch, instead of making
  later calls return `cudaErrorIllegalAddress`.
- **Warp functions approximate convergence.** cuda4cpu can't see which lanes
  are in the same branch. A warp function waits for every lane in its mask
  that hasn't returned and isn't waiting at `__syncthreads()`. Lanes in
  different branches that run warp functions at the same time must use
  disjoint masks, as CUDA requires. `__activemask()` and `coalesced_threads()`
  wait until every lane of the warp is blocked somewhere, and return the lanes
  that called them from the same place. Lanes in another branch don't call
  them from there, so they are left out. Lanes that reach that place in
  different loop iterations may still be grouped together, unlike on a GPU.
- **The divergent barrier check can't tell cooperative groups' `sync()`
  calls apart**: it knows `__syncthreads()` by its line, but `sync()` has
  CUDA's signature, without one. Threads at two different `block.sync()`
  calls aren't reported, and a `sync()` that diverges from a
  `__syncthreads()` is reported by its name.
- **Cooperative launches are limited to one block per OS thread**
  (`multiProcessorCount` blocks), because all their blocks must run at the
  same time. Larger grids fail with `cudaErrorCooperativeLaunchTooLarge`, as
  on a GPU, and `grid.sync()` in a kernel launched otherwise aborts with a
  message.
- **Graph kernel nodes need the kernel's type.** The `void *` in
  `cudaKernelNodeParams::func` must come from `cuda4cpu::kernel_address`.
  The rewriter inserts it for `(void *)kernel`. A kernel known only by an
  address taken elsewhere fails with `cudaErrorInvalidDeviceFunction`. Code
  that doesn't go through the rewriter names the dimension fields
  `cuda4cpu_gridDim` and `cuda4cpu_blockDim`. Graphs can't be launched into a
  stream being captured.
- **HIP runs as on HIP's NVIDIA platform.** Warps have 32 lanes, and
  `__ballot` returns 64-bit masks whose high half is zero. Code that only
  supports AMD GPUs compiles on HIP's AMD platform
  (`--cuda4cpu-platform=amd`), which emulates an RDNA3 or RDNA4 GPU with
  32-lane wavefronts and has part of AMD's builtins (`__builtin_amdgcn_*`):
  not 64-lane wavefronts (CDNA) and their builtins, and the matrix
  operations only with Clang. See
  [HIP](hip.md#hips-amd-platform).
- **cuRAND gives CUDA's numbers only for XORWOW and Philox4_32_10**, as
  integers and uniform floats. Its other generators and distributions are
  distributed as CUDA's, but their values differ. MTGP32's device API isn't
  provided, and its host generator is `std::mt19937`. See
  [features](features.md#libraries).
- **Thrust runs on its OpenMP backend**, with CCCL's support of the half,
  bfloat16 and 8-bit types off. CUB's device-wide algorithms don't build for
  CUDA; hipCUB's do on HIP's AMD platform.
- **Not implemented yet:** CUDA arrays and their textures and surfaces (texture
  objects of linear memory work), dynamic parallelism, the sub-byte types of
  `wmma::experimental`, inline PTX, the driver API, and CUDA's libraries
  other than cuRAND, cuBLAS and Thrust (cuFFT, cuSPARSE, CUB's device
  algorithms, ...).
- **cuBLAS:** cublas_v2.h's API.
  - Not provided: cuBLASLt, cuBLASXt, the legacy API of `cublas.h`, and the
    64-bit integer forms (`cublasSgemm_64`).
  - Speed: without a CBLAS library, cuda4cpu's own BLAS is much slower than
    an optimized one for large products (see [usage](usage.md#blas)).
  - Rounding: half-precision products compute in float. [Compatibility with real CUDA and HIP
  code](compatibility.md) shows which programs each
  one blocks.
