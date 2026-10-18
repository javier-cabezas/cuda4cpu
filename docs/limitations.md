# Limitations

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
- **No ThreadSanitizer yet.** It reports false races between kernels and
  host code, because OpenMP synchronizes its threads in ways it can't see.
  AddressSanitizer, UBSan and Valgrind work (see [Debugging](debugging.md)).
- **Kernel launches are synchronous** and always run on the host. Streams and
  events exist for API compatibility only, and asynchronous copies complete
  before they return, so `cudaErrorNotReady` never occurs.
- **The rewriter finds local headers through relative paths only.** Headers
  included with `#include "..."` relative to the including file are rewritten.
  Headers found through include directories (`-I`) or `#include <...>` are not.
  A launch or `extern __shared__` array in one of those doesn't compile.
- **`extern __shared__` at namespace scope** becomes an object that converts
  to a pointer and supports subscripts, but not `reinterpret_cast` or
  `sizeof`. Inside functions it is a plain pointer. Alignment specifiers are
  dropped: dynamic shared memory is always 64-byte aligned.
- **`cuda4cpu-c++` with `-MD`** writes dependencies on the rewritten copies,
  which are temporary. Use `cuda4cpu_add_cuda_sources` for builds that need
  dependency tracking.
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
  that hasn't returned and isn't waiting at `__syncthreads()`. Lanes in
  different branches that run warp functions at the same time must use
  disjoint masks, as CUDA requires. `__activemask()` and `coalesced_threads()`
  wait until every lane of the warp is blocked somewhere, and return the lanes
  that called them from the same place. Lanes in another branch don't call
  them from there, so they are left out. Lanes that reach that place in
  different loop iterations may still be grouped together, unlike on a GPU.
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
  supports AMD GPUs doesn't compile: AMD builtins (`__builtin_amdgcn_*`),
  `__HIP_PLATFORM_AMD__` branches, and code that assumes 64-lane wavefronts.
- **Not implemented yet:** textures and surfaces, dynamic parallelism,
  bfloat16 and 8-bit floats, `__reduce_*_sync`, `memcpy_async`, 16-bit
  integer atomics, inline PTX, and the driver API. [Compatibility with real CUDA and HIP
  code](compatibility.md) shows which programs each
  one blocks.
