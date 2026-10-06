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
python3 tests/rewrite/test_rewrite.py tools/cuda4cpu-rewrite   # rewriter unit tests
```

Build trees live in `build/<preset>/` (git-ignored). The presets export
`compile_commands.json` there.

## Layout

- `include/cuda4cpu.hpp`: umbrella header
- `include/cuda_runtime.h`: drop-in for CUDA's header. It includes
  `cuda4cpu.hpp` and does `using namespace cuda4cpu::cuda_api`.
- `include/defines.hpp`: macro overrides for `__global__`, `__shared__`,
  `threadIdx`, `__syncthreads`, and similar. It's included **last** on purpose,
  because these macros would otherwise break the library's own headers.
- `include/launch.hpp`: the public side of the execution engine
  (`launch_conf`, `system`, `thread_block`, `grid_launcher`, `launch()`,
  `dynamic_shared<T>()`). It holds only the launcher template and the inline
  built-in variable getters.
- `include/warp.hpp`: `warpSize`, shuffles, votes and `__syncwarp`, built on
  `thread_block::warp_shuffle`/`warp_ballot`/`syncwarp`.
- `include/cuda/*.hpp`: the CUDA runtime API (types, errors, device, memory,
  streams, events), plus device-side `atomics.hpp` (on `std::atomic_ref`) and
  `math.hpp`. All CUDA names live in the **inline namespace
  `cuda4cpu::cuda_api`**. `cuda_runtime.h` exports exactly that namespace to
  global scope, so helpers go in `cuda4cpu::detail`, never in `cuda_api`.
  Most functions are header-only. Device properties and aligned allocation
  are in the library.
- `lib/cuda4cpu.cpp`: the only compiled source. It holds all the fiber code
  (guarded stacks, the x86-64 assembly context switch and its
  `ucontext`/`_setjmp` fallback, the barrier scheduler, the per-OS-thread
  fiber cache) and `system` (topology via OpenMP plus optional libnuma).
- `tools/cuda4cpu-gdb.py`: gdb convenience functions (`$threadIdx()`, ...)
  and the `cuda4cpu` command, which read `thread_block::Vars_`. The library is
  always built with `-g` so that gdb knows that variable's type.
- `tools/cuda4cpu-rewrite`: rewrites `kernel<<<...>>>(...)` and
  `extern __shared__ T x[];` in a `.cu` file and in the local headers it
  includes. It uses Python 3, standard library only.
  `tools/cuda4cpu-c++.in`: the compiler driver, configured into
  `build/<preset>/tools/cuda4cpu-c++` (absolute build-tree paths) and, for
  installation, with paths relative to its own directory.
- `cmake/cuda4cpu-cuda.cmake`: `cuda4cpu_add_cuda_sources()`, used in-tree and
  installed with the package config. It finds the rewriter through the
  `CUDA4CPU_REWRITE` global property.
- `compat/`: the compatibility suite. `suites.json` lists programs from
  cuda-samples and Rodinia, pinned by revision, with their sources, include
  directories, arguments and checks: `expect` and `reject` are output
  regexes, `verified: false` marks programs that don't check their results,
  and `skip` gives a reason not to run one. `run.py` fetches the suites,
  builds each program with `cuda4cpu-c++` (`.c` files with the C compiler),
  runs it, and reports. `baseline.json` holds the expected status of each
  program: CI fails if a passing program regresses. When a change makes
  programs pass, update the baseline in the same PR with `--update-baseline`.
  Run it locally with
  `compat/run.py --driver build/release/tools/cuda4cpu-c++ --work /tmp/compat --filter 'rodinia/*'`.
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
  - `runtime`: error codes and the last error, launch limits, allocation,
    copies, memset, symbols, events, device queries and intrinsics, through
    `cuda_runtime.h` only
  - `symbol_address_rejected`: a compile-only test. Passing `&symbol` to
    `cudaMemcpyToSymbol` must fail with the static_assert message.
  - `events`: event timing
  - `stencil{2,3}d`: shared memory, `__constant__` and `__syncthreads`, checked
    against a host reference
  - `samples/*.cu`: ten unmodified CUDA programs, built with
    `cuda4cpu_add_cuda_sources` and checked against a host reference.
    Five use kernel features (reduction, matmul, histogram, scan, nbody, which
    also print timings). Five use the runtime API like real code (vector_add,
    async_api, device_query, managed_add, softmax).
  - `rewrite/features.cu` (+ `features.cuh`): every launch form and
    `extern __shared__` placement, checked at run time, including that
    `__LINE__` matches the original file. Update its expected line if you
    edit the file above that check.
  - `schedule_{forward,reverse,random}`: a kernel with a missing
    `__syncthreads()` that the default order hides and the others expose
  - `divergent_barrier_{warn,ignore,error}`: the divergent barrier check in
    each mode (error mode must abort with a single report)
  - `gdb` (`gdb_test.cmake`, if gdb is installed): a breakpoint conditioned on
    `$threadIdx("x")`, the built-in variables, and a backtrace that ends where
    the fiber started
  - `rewrite_unit` (`rewrite/test_rewrite.py`): exact rewriter output, the
    syntax that must stay untouched, errors, and shadow headers with depfiles.
  - `driver`: builds `samples/vector_add.cu` with the build-tree
    `cuda4cpu-c++`, in one step and with separate compile and link.

## How execution works (read before touching `launch.hpp`)

- `launch(kernel, ...)` returns a `grid_launcher<Args...>`, and
  `launch(callable, ...)` returns a `callable_launcher<F>` (that's what the
  rewriter emits, with a lambda that calls the kernel). Their `call(args...)`
  stores the callable and the arguments (a `std::tuple`) in a
  `detail::kernel_call<F, Stored...>`. That type provides the two thread loops
  instantiated for it, `run_direct` and `run_one`, and `detail::run_grid`
  runs the blocks in an `omp for schedule(dynamic, chunk)` loop, with about 8
  contiguous chunks per OS thread. With a lambda, the kernel is known at
  compile time and gets inlined into the loop: rewritten vecadd matches a
  plain OpenMP loop, while launching through a function pointer costs about
  1.3x.
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

- `cuda_runtime.h` defines the include guards of CUDA's own headers
  (`__CUDA_RUNTIME_H__`, `__DRIVER_TYPES_H__`), because code such as the CUDA
  samples' `helper_cuda.h` tests them before using the runtime API.
- CUDA's `min`/`max` overloads are declared as `(min)`/`(max)`, so that
  programs that define `min`/`max` macros don't break them.
- glibc's `<math.h>` declares `__expf`, `__sinf`, `__powf`, ... without
  exporting them, so they don't link. `math.hpp` defines them as
  `extern "C" inline` at global scope, which completes those declarations.
  Defining them in a namespace instead would make unqualified calls ambiguous.
  The same applies to `rsqrtf` with glibc 2.41+.
- Allocations go through `detail::allocate_aligned` in the library. When
  inline, GCC may elide an allocation whose result is never dereferenced,
  and treat it as successful.
- Runtime calls that fail must return the error through
  `detail::record_error(err)`, which also makes it the thread's last error.
  `grid_launcher::call` rejects configurations outside CUDA's limits with
  `cudaErrorInvalidConfiguration`.
- Don't define `__noinline__` as a macro: libstdc++ uses
  `__attribute__((__noinline__))`.

- Debugging settings come from the environment, read once in `settings`:
  `CUDA4CPU_SCHEDULE` (forward, reverse, random[:seed]) and
  `CUDA4CPU_DIVERGENT_BARRIERS` (warn, error, ignore).
- Running order: with `CUDA4CPU_SCHEDULE` other than forward, `order_` points
  to a permutation (`fibers::order`, inverse in `position`). It's reversed once
  per thread_block, or shuffled per block from the seed and the block index.
  Direct mode walks it with `pos_`. `promote()` marks `order_[0..pos_)` as
  finished, and `next_runnable_after()` follows the order. Anything that
  reasons about which threads ran in direct mode must use the order, not
  `t < cur_`: `warp_live_mask` once got this wrong, and `__activemask()` before
  the first barrier gave wrong lanes in random order. Run the tests with
  `CUDA4CPU_SCHEDULE=random:<several seeds>` after scheduler changes. Direct mode runs on
  the stack of `order_[0]` (`direct_stack`), never stack 0, because a thread
  that hasn't started yet must not start on the stack the promoted thread is
  still using.
- `__syncthreads()` is a function-like macro passing `__FILE__` and `__LINE__`.
  The divergent barrier check compares them, not return addresses: the
  compiler inlines a lambda-launched kernel into both `run_direct` and
  `run_one`, so one call site has several addresses.
- AddressSanitizer: `__sanitizer_start/finish_switch_fiber` are weak
  references, called only when the runtime is present. Every switch goes
  through `fibers::switch_to`/`start_fiber`, which call `sanitizer_leave` as
  the last thing before the raw switch. A context saved into `discarded`
  leaves its fake stack in `stack_fake_stacks[running_stack]`, and the next
  fiber started on that stack takes it in `sanitizer_started`. The fake stack
  may hold the caller's addressable locals, so nothing may touch them after
  `sanitizer_leave`; and destroying fake stacks instead of reusing them costs
  two mmap calls per CUDA thread. Every context records the stack it runs on
  (`stack_index`, set in `start_fiber` and `promote`).
- ThreadSanitizer isn't supported: GCC's libgomp synchronizes through raw
  futexes that it can't see, and modeling barriers for it requires keeping it
  out of the scheduler state and `Vars_`, which all CUDA threads share.
- The rewriter must never add or remove newlines, so that line numbers stay
  the same; it also prepends `#line 1 "<original path>"`. It copies the
  kernel expression, configuration and arguments verbatim, and never splits
  them at commas: that is ambiguous without a C++ parser (`a < b, c > d`).
  The lexer has to skip comments, string, character and raw-string literals,
  and digit separators (`1'000`).
- Shadow headers are written under `<output>.headers/<absolute original
  path>`, and the rewritten `#include` points to them with an absolute path.
  Only quote includes that resolve relative to the including file are
  rewritten.

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
