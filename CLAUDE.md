# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## What this is

cuda4cpu runs CUDA kernels on the CPU with a plain C++23 compiler. Kernels are
compiled unchanged; CUDA keywords and built-ins are redefined as macros, and the
`<<<grid, block>>>` launch syntax becomes `launch(kernel, grid, block).call(args...)`.
See README.md for the user-facing description.

## Commands

```sh
cmake --workflow --preset debug     # configure + build + test (Debug)
cmake --workflow --preset release   # same, Release
ctest --preset debug -R stencil2d   # run a single test
cmake --preset debug -DCUDA4CPU_ENABLE_VALGRIND=ON   # opt-in options, see CMakeLists.txt
build/release/benchmarks/microbench   # launch, barrier, shuffle and scheduling costs
```

Build trees live in `build/<preset>/` (git-ignored). The presets export
`compile_commands.json` there.

## Layout

- `include/cuda4cpu.hpp`: umbrella header that users include
- `include/defines.hpp`: macro overrides for `__global__`, `__shared__`,
  `threadIdx`, `__syncthreads`, and similar. It's included **last** on purpose,
  because these macros would otherwise break the library's own headers.
- `include/launch.hpp`: the public side of the execution engine
  (`launch_conf`, `system`, `thread_block`, `grid_launcher`, `launch()`,
  `dynamic_shared<T>()`). It holds only the launcher template and the inline
  built-in variable getters.
- `include/warp.hpp`: `warpSize`, shuffles, votes and `__syncwarp`, built on
  `thread_block::warp_shuffle`/`warp_ballot`/`syncwarp`.
- `include/cuda/*.hpp`: CUDA runtime API shims (memory, streams, events,
  device, types), plus device-side `atomics.hpp` (on `std::atomic_ref`) and
  `math.hpp`. All are header-only inline functions in namespace `cuda4cpu`.
- `lib/cuda4cpu.cpp`: the only compiled source. It holds all the fiber code
  (guarded stacks, the x86-64 assembly context switch and its
  `ucontext`/`_setjmp` fallback, the barrier scheduler, the per-OS-thread
  fiber cache) and `system` (topology via OpenMP plus optional libnuma).
- `benchmarks/microbench.cpp`: micro-benchmarks. Use them to back any
  performance claim, and compare medians: run-to-run noise is large.
- `tests/`: each test returns non-zero on failure.
  - `barriers`: early exits and divergent-block barrier semantics
  - `launch`: thread and block numbering, and fiber cache reuse and eviction
  - `stacks`: deep stacks, `cudaLimitStackSize`, and the guard page (forks a
    child that must die with `SIGSEGV`)
  - `warp`: every shuffle and vote variant (widths, partial warps), exited
    lanes, and lanes that branch to `__syncthreads()`
  - `atomics`: every atomic function, racing across blocks
  - `events`: event timing
  - `stencil{2,3}d`: shared memory, `__constant__` and `__syncthreads`, checked
    against a host reference
  - `samples/`: ports of CUDA samples (reduction, matmul, histogram, scan,
    nbody), checked against a host reference. They also print timings.

## How execution works (read before touching `launch.hpp`)

- `grid_launcher::call` stores the kernel pointer and its arguments (a
  `std::tuple`) in a `detail::kernel_call<Args...>`. That type also provides
  the two thread loops instantiated for the kernel's signature: `run_direct`
  and `run_one`. The blocks run in an `omp for schedule(dynamic, chunk)`
  loop, with about 8 contiguous chunks per OS thread.
- An OS thread that gets blocks calls `thread_block::acquire`, which returns a
  `thread_block` from a small `thread_local` cache keyed by block shape and
  stack size. Creating one maps its stacks.
- Contexts (`fibers::context`). On x86-64 a context is just a stack pointer:
  `cuda4cpu_switch_context` pushes rbp, rbx and r12–r15, swaps the stack
  pointer, and pops. `start_fiber` writes a fresh frame at the top of the
  stack: six zeroed registers plus `cuda4cpu_fiber_trampoline` as the return
  address. The trampoline aligns the stack, marks the call stack's end
  (`.cfi_undefined rip`) and calls `fiber_access::start()` →
  `fiber_main()`. With `CUDA4CPU_PORTABLE_CONTEXT`, or on other
  architectures, a context is a `jmp_buf`, and each fiber's entry point is
  captured once at creation through `makecontext`.
- `execute` runs one block. It always starts in **direct mode**: `fiber_main`
  runs on fiber 0's stack and calls `run_direct`, which loops over the
  threads calling the kernel directly. `next_direct_thread()` moves to the
  next thread. With no barriers there are no switches at all.
- The first `__syncthreads()` or warp function calls `promote()`: threads
  before `cur_` are finished, and the current thread stays on fiber 0's stack
  (its own stack is unused for that block).
- In fiber mode, the scheduling state is one 32-bit word per warp for each of
  `live`, `started`, `block_wait` and `warp_wait`. A thread can run when it is
  live and not waiting. A waiting thread saves itself in `ctx[i]` and jumps to
  the next runnable thread in index order (`next_runnable`). A thread that
  hasn't started begins from `start[j]`.
- `__syncthreads()` is released when `block_arrived == live_count`. A warp
  function records its kind, mask, value and source lane, and is released by
  `try_release_warp` once every participating lane is in `warp_wait`. The
  releasing lane computes every lane's result. Participants are the live lanes
  in the mask, minus lanes in `block_wait`, which are in another branch.
- Releases are rechecked whenever the set of lanes being waited for shrinks:
  when a thread returns, or starts waiting at `__syncthreads()`. That is why
  early returns don't block barriers, and why no barrier can deadlock.
  `deadlock()` is an internal consistency check.
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim` and the lane are stored in
  `thread_block::Vars_`, which `set_current_thread()` and `execute()` keep up
  to date. The getters return `dim3` by value.
- `__shared__` is `static thread_local`. That's correct only because a block
  always runs to completion on one OS thread.

## Gotchas

- Keep every `setjmp`/`longjmp`/`ucontext` use in `lib/cuda4cpu.cpp`. The
  fortified `longjmp` aborts with "longjmp causes uninitialized stack frame",
  and the library is built with `-U_FORTIFY_SOURCE` PRIVATE so that consumers
  don't need it. The same goes for `CUDA4CPU_HANDLE_VALGRIND`, which must not
  affect the header layout.
- `Current_` and `Vars_` are `__thread`, not `thread_local`, and `Vars_` is a
  trivial type (`raw_dim3`). GCC makes every access to an `extern
  thread_local` from another file call a TLS initialization check, which
  more than doubled the cost of reading `threadIdx`. Check the kernel's
  assembly after touching them.
- Change `cuda4cpu_switch_context` and `start_fiber` together, because they
  share the frame layout. Test both context implementations (CI has a
  `portable context` job).
- Never resume a fiber that is `finished`. Its kernel frame has returned, and
  its context was saved into `discarded`. With the fallback, resuming it
  would return from `fiber_entry`, which ends the process
  (`uc_link == nullptr`).
- Stacks come from one `mmap` with a guard page below each stack, installed
  with `MADV_GUARD_INSTALL` (Linux 6.13+), or `mprotect` on older kernels.
  `mprotect` takes the address-space lock for writing and splits the mapping:
  OS threads creating fibers in parallel serialize on it (a 1024-thread block
  launch was 6× slower), and each fiber costs two mappings against
  `vm.max_map_count`.
  Each stack top is shifted by `(i % 64) * 64` bytes, because aligned stacks
  thrash the L1 cache sets on every switch: about 15% on stencil3d.
- `-fno-semantic-interposition` lets the scheduler helpers inline into each
  other. Without it they are PLT calls, which made each switch about a third more
  expensive.
- The stack size is read in `grid_launcher::call` (header side) from
  `detail::stack_size` and passed to `acquire`. Don't read it from the library.
- The `makecontext` trampoline passes a pointer as two 32-bit ints, which
  assumes a 64-bit platform.
- `extern __shared__ T name[];` can't work: `__shared__` expands to
  `static thread_local`, and no definition could be sized at launch time.
  Users write `T *name = dynamic_shared<T>();` instead.
- glibc 2.41+ declares the C23 `rsqrt`/`rsqrtf` globally, so `math.hpp` only
  defines its own when the C library doesn't. Otherwise the call is ambiguous
  under `using namespace cuda4cpu`.
- Header functions must stay `inline` (or templates). A non-inline definition
  causes duplicate symbols when `libcuda4cpu` is linked statically.
- Never include `<numa.h>` from public headers. libnuma is an optional,
  PRIVATE dependency, guarded by `CUDA4CPU_HAVE_NUMA`.

## Conventions

- C++23 with `CXX_EXTENSIONS OFF`; the target requires it through
  `target_compile_features(... cxx_std_23)`.
- Four-space indentation, snake_case for library types and functions, and CUDA
  names exactly as in the CUDA Runtime API.
- Every source file starts with the Apache-2.0 header and
  `SPDX-License-Identifier: Apache-2.0`.
- New public headers must be added to the `FILE_SET HEADERS` list in
  `lib/CMakeLists.txt`, or they won't be installed.
- Add a CTest test (in `tests/CMakeLists.txt`) for each new CUDA feature, and
  check results against a host reference.
- Changes to `main` go through pull requests. CI (`.github/workflows/ci.yml`)
  builds and tests with GCC and Clang in Debug and Release.
