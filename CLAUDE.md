# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## What this is

cuda4cpu runs CUDA kernels on the CPU with a plain C++23 compiler. Kernels are
compiled unchanged; CUDA keywords and built-ins are redefined as macros, and the
`<<<grid, block>>>` launch syntax becomes `launch(kernel, grid, block).call(args...)`.
HIP code compiles too, as on HIP's NVIDIA platform: a thin layer of HIP names
over the CUDA implementation. See README.md for the user-facing description.

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
- `include/warp.hpp`: `warpSize`, shuffles, votes, matches, `__activemask`
  and `__syncwarp`, built on `thread_block::warp_shuffle`/`warp_ballot`/
  `warp_match`/`warp_active_mask`/`syncwarp`.
- `include/cooperative_groups.h` (+ `cooperative_groups/reduce.h`, `scan.h`):
  CUDA's cooperative groups, in the global namespace `cooperative_groups` as
  in CUDA, with helpers in `cuda4cpu::detail::cg`. `thread_group` is the
  type-erased base that every group converts to. Tiles of up to 32 threads
  use the warp functions with the tile's lanes as the mask; larger tiles use
  `thread_block::sync_warps`. Block and grid `sync()` take a defaulted
  `std::source_location`, so the divergent barrier check reports the
  caller's line.
- `include/cuda/*.hpp`: the CUDA runtime API (types, errors, device, memory,
  streams, events), plus device-side `atomics.hpp` (on `std::atomic_ref`) and
  `math.hpp`. All CUDA names live in the **inline namespace
  `cuda4cpu::cuda_api`**. `cuda_runtime.h` exports exactly that namespace to
  global scope, so helpers go in `cuda4cpu::detail`, never in `cuda_api`.
  Most functions are header-only. Device properties and aligned allocation
  are in the library.
- `include/cuda/graphs.hpp` + `lib/graphs.cpp`: CUDA graphs and stream
  capture. The API is declared in the header and defined in the library,
  except what needs a kernel's type: `detail::bound_kernel` (a kernel with
  copies of its arguments, which a kernel node runs), the capture of launches
  and `kernel_address()`, all in `launch.hpp`. Async functions in `memory.hpp`,
  `streams.hpp` and `events.hpp` check `detail::capturing(stream)` and call
  the `detail::capture_*` functions of the library.
- `include/nvtx3/nvToolsExt.h` (and `nvToolsExt.h`): NVTX stand-ins that do
  nothing.
- `include/cuda_fp16.h`: `__half`/`__half2` and their intrinsics, in
  `cuda_api`. A `__half` holds the binary16 bits. `detail::to_half_bits`
  rounds a double to half in any of the four modes, and every operation is
  "compute in double, round once", which is correctly rounded for half (see
  the comments on `__hfma`'s exactness). `hip/hip_fp16.h` includes it.
- `include/hip/`: HIP, on top of the CUDA API.
  - `hip_runtime_api.h`: HIP's runtime API. Every HIP name that is a CUDA
    name with another prefix is a `#define` for it (`hipMalloc` →
    `cudaMalloc`), listed by the CUDA header that defines it. Add the HIP
    macro when you add a CUDA function, type or constant that HIP has.
    `hipError_t` is `cudaError_t`: HIP's codes have CUDA's values.
  - What HIP does differently lives in the inline namespace
    `cuda4cpu::hip_api`, which the header exports like `cuda_api`:
    `hipDeviceProp_t` (derived from `cudaDeviceProp`),
    `hipDeviceAttribute_t`, `hipGetErrorName`, `hipHostMalloc`, and HIP's
    driver-API copies and memsets. Helpers go in `cuda4cpu::detail`: a
    `detail` namespace inside `hip_api` would hide it.
  - `hip_runtime.h` adds the kernel language: the mask-less warp functions,
    `hipLaunchKernelGGL` (a macro, or a function template with
    `HIP_TEMPLATE_KERNEL_LAUNCH`), `HIP_DYNAMIC_SHARED`, `hipThreadIdx_x`
    and the like, and the `__HIP_ARCH_HAS_*` macros.
  - `hip_common.h` defines `__HIP_PLATFORM_NVIDIA__`, and `__HIPCC__` with
    `__CUDACC__`.
- `lib/cuda4cpu.cpp`: the main compiled source. It holds all the fiber code
  (guarded stacks, the x86-64 assembly context switch and its
  `ucontext`/`_setjmp` fallback, the barrier scheduler, the per-OS-thread
  fiber cache) and `system` (topology via OpenMP plus optional libnuma).
- `tools/cuda4cpu-gdb.py`: gdb convenience functions (`$threadIdx()`, ...)
  and the `cuda4cpu` command, which read `thread_block::Vars_`. The library is
  always built with `-g` so that gdb knows that variable's type.
- `tools/cuda4cpu-rewrite`: rewrites `kernel<<<...>>>(...)` and
  `extern __shared__ T x[];` in a `.cu` file and in the local headers it
  includes. It also:
  - drops a `(void *)` cast on the kernel passed to
    `cudaLaunchKernel`/`cudaLaunchCooperativeKernel`
  - turns `(void *)kernel` elsewhere into `cuda4cpu::kernel_address(kernel)`,
    for the `__global__` functions it has seen (`find_kernels`)
  - turns `.gridDim`/`->gridDim` (and `blockDim`) into `cuda4cpu_gridDim`,
    the name of the `cudaKernelNodeParams` fields
  - turns `clock()` inside `__global__`/`__device__` bodies into
    `cuda4cpu::device_clock()`
  - inserts `cuda4cpu::loop_tick();` after the `{` of the outermost loops of
    `__global__`/`__device__` bodies (`find_loop_bodies`). Inner loops and
    bodies that aren't blocks are left alone, so that inner loops still
    vectorize.
  - rewrites `HIP_DYNAMIC_SHARED(T, name)` like `extern __shared__ T name[];`,
    and drops the cast in `hipLaunchKernel`/`hipLaunchCooperativeKernel`

  The rewritten `.cu` file starts by defining `__CUDACC__`, as nvcc does.
  `--language hip` (the default for `.hip` files) also defines `__HIPCC__` and
  `__HIP_PLATFORM_NVIDIA__`. It uses Python 3, standard library only.
  `tools/cuda4cpu-c++.in`: the compiler driver, configured into
  `build/<preset>/tools/cuda4cpu-c++` (absolute build-tree paths) and, for
  installation, with paths relative to its own directory. The same template
  also becomes `cuda4cpu-hipcc` (`CUDA4CPU_DRIVER_MODE` is `hip`), which
  compiles every C++ source as HIP and drops hipcc's options for AMD GPUs.
  Both handle `.hip` files and `-x cu`/`-x hip`.
- `cmake/cuda4cpu-cuda.cmake`: `cuda4cpu_add_cuda_sources()` and
  `cuda4cpu_add_hip_sources()` (any extension), used in-tree and installed
  with the package config. It finds the rewriter through the
  `CUDA4CPU_REWRITE` global property.
- `compat/`: the compatibility suite. `suites.json` lists programs from
  cuda-samples, Rodinia and rocm-examples (HIP), pinned by revision, with
  their sources, include directories, arguments and checks: `expect` and
  `reject` are output regexes, `verified: false` marks programs that don't
  check their results, `skip` gives a reason not to run one, and `timeout`
  gives a long program more time (such programs start first). `run.py`
  fetches the suites, builds each program with `cuda4cpu-c++`, or
  `cuda4cpu-hipcc` for suites with `"driver": "hipcc"` (`.c` files with the
  C compiler), runs it, and reports. `baseline.json` holds the expected status of each
  program: CI fails if a passing program regresses. When a change makes
  programs pass, update the baseline in the same PR with `--update-baseline`
  (with `--filter`, the other programs keep their status).
  Run it locally with
  `compat/run.py --driver build/release/tools/cuda4cpu-c++ --work /tmp/compat --filter 'rodinia/*'`.
- `benchmarks/microbench.cpp`: micro-benchmarks. Use them to back any
  performance claim, and compare medians: run-to-run noise is large.
- `tests/`: each test returns non-zero on failure.
  - `interleave_{auto,off,1,3}` (`interleave.cu`, through the rewriter):
    loops with barriers, shuffles and early exits give the same results with
    any interleaving, and a thread that spins on another thread of its block
    finishes when they take turns
  - `barriers`: early exits and divergent-block barrier semantics, and 3D
    blocks that reach a barrier or a shuffle in the middle of direct mode
  - `launch`: thread and block numbering, and fiber cache reuse and eviction
  - `stacks`: deep stacks, `cudaLimitStackSize`, and the guard page (forks a
    child that must die with `SIGSEGV`)
  - `warp`: every shuffle and vote variant (widths, partial warps), exited
    lanes, and lanes that branch to `__syncthreads()`
  - `atomics`: every atomic function, racing across blocks
  - `graphs` (`graphs.cu`, through the rewriter): stream capture with fork
    and join, explicit graphs with every node type, `cudaGraphExecUpdate`,
    conditional nodes, graph memory (reallocated at the same address after
    `cudaFree`), 2D/3D copies and the device clock
  - `cooperative_groups`: every group type and operation, including partial
    warps and tiles, tiles of several warps that sync independently,
    `coalesced_threads()` in a branch (warp-aggregated atomics), and
    `grid.sync()` across rounds of a cooperative launch
  - `runtime`: error codes and the last error, launch limits, allocation,
    copies, memset, symbols, events, device queries and intrinsics, through
    `cuda_runtime.h` only
  - `symbol_address_rejected`: a compile-only test. Passing `&symbol` to
    `cudaMemcpyToSymbol` must fail with the static_assert message.
  - `events`: event timing
  - `fp16` (`fp16.cu`): half precision against an exact reference, a table
    of every half decoded independently of the header. Every half's round
    trip, float conversions in all four modes around every rounding
    boundary, a million random add/sub/mul/fma (exact as `__int128`
    multiples of 2^-48) and div/sqrt (through 64-bit `long double`), the
    special values, and kernels with `half2`, shuffles and atomics.
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
    `driver_hipcc` and `driver_x_hip` build a HIP `.cpp` file with
    `cuda4cpu-hipcc` (and an AMD-only option) and with `cuda4cpu-c++ -x hip`.
  - `hip/` (through `cuda4cpu_add_hip_sources`, HIP names only):
    `hip_runtime` (errors and their names, device queries and attributes,
    host memory, driver-API copies and memsets, symbols, `hipStreamDefault`
    and `hipStreamPerThread`, pools, `hipLaunchKernel`, capture),
    `hip_kernels` (every `hipLaunchKernelGGL` form, `HIP_DYNAMIC_SHARED` in a
    kernel and at namespace scope, mask-less warp functions with widths,
    cooperative groups, bit, clock and atomic intrinsics, the platform
    macros) and `hip_template_launch` (`HIP_TEMPLATE_KERNEL_LAUNCH`, in a
    `.cpp` file).

## How execution works (read before touching `launch.hpp`)

- `launch(kernel, ...)` returns a `grid_launcher<Args...>`, and
  `launch(callable, ...)` returns a `callable_launcher<F>` (that's what the
  rewriter emits, with a lambda that calls the kernel). Their `call(args...)`
  stores the callable and the arguments (a `std::tuple`) in a
  `detail::kernel_call<F, Stored...>`. That type provides the two thread loops
  instantiated for it, `run_direct` and `run_one`, and `detail::run_grid`
  runs the blocks in an `omp for schedule(dynamic, chunk)` loop, with about 8
  contiguous chunks per OS thread. The parallel region has at most one OS
  thread per block, and a single-block grid runs on the calling thread
  without one: waking the OpenMP threads costs about 2 µs, 5x an empty
  256-thread block. With a lambda, the kernel is known at
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
  threads calling the kernel directly. With no barriers there are no switches
  at all. In the default order the loop iterates over z, y and x, and stores
  only `threadIdx`: `cur_`, `pos_` and the lane aren't kept up to date, and
  `promote()` and `lane_id()` compute them from `threadIdx`
  (`direct_thread()`). That made a CUDA thread that only checks bounds go from
  2.3 to 0.9 ns. With `CUDA4CPU_SCHEDULE`, `next_direct_thread()` moves to
  the next thread in the order and sets them all.
- The first `__syncthreads()` or warp function calls `promote()`: threads
  before `cur_` are finished, and the current thread stays on fiber 0's stack
  (its own stack is unused for that block).
- In fiber mode, the scheduling state is a `warp_state` per warp, 32 bytes so
  that a scheduling decision reads one cache line: 32-bit lane masks `live`,
  `started`, `block_wait`, `warp_wait`, `tile_wait` (at the barrier of a tile
  larger than a warp) and `converge_wait` (in `__activemask()`, also in
  `warp_wait`), plus the pending tile's range of warps. Keeping them together
  matters: as separate vectors, shuffles were 18% slower. A thread can run
  when it is live and not waiting. A waiting thread saves itself in `ctx[i]` and jumps to
  the next runnable thread in index order (`next_runnable`). A thread that
  hasn't started begins from `start[j]`.
- `__syncthreads()` is released when `block_arrived == live_count`. A warp
  function records its kind, mask, value and source lane, and is released by
  `try_release_warp` once every participating lane is in `warp_wait`. The
  releasing lane computes every lane's result. Participants are the live lanes
  in the mask, minus lanes in `block_wait` or `tile_wait`, which are in
  another branch. Lanes in `converge_wait` haven't arrived yet.
- `__activemask()` (and `coalesced_threads()`) is the `op_converge` warp
  operation, keyed by its call site. It is released once every live lane that
  isn't at a barrier is in `warp_wait`, whatever it waits for, and returns the
  lanes waiting at the same call site. This can't deadlock with other warp
  operations, which wait for converging lanes: those only need everyone to be
  blocked. The rare operations (matches, converge) are `[[gnu::noinline]]`
  methods of `fibers`, out of `try_release_warp`'s hot path.
- A tile barrier (`sync_warps`) is released when every live thread of its
  warps that isn't at `__syncthreads()` is in `tile_wait`.
- `grid_group::sync()` is a `__syncthreads()` that sets `grid_pending`;
  `release_block()` then waits at the launch's `std::barrier`
  (`kernel_closure::grid`) before releasing the block. Only
  `cudaLaunchCooperativeKernel` sets it: `run_grid_cooperative` runs each
  block on its own OS thread (OpenMP, or `std::jthread` when OpenMP can't
  promise one thread per block), and a block that returns drops out of the
  barrier.
- `loop_tick()` decrements `Vars_.tick`; at 0, `interleave()` promotes the
  block if needed and switches to the next runnable thread, which runs
  `slice_` iterations before the next switch. `execute()` sets `slice_` per
  block from the kernel's `interleave_tuner` (a static member of each
  `kernel_call` instantiation, so one per launch site): the first blocks try
  `interleave_slices` (off, 4, 16) in turn, three each, and the fastest block
  of each candidate decides. A grid-stride copy went from 199 to 35 ms, and
  page walks per thousand instructions from 297 to 1 in
  `cuda-samples/alignedTypes`. `CUDA4CPU_INTERLEAVE` fixes the slice.
- Releases are rechecked whenever the set of lanes being waited for shrinks:
  when a thread returns, or starts waiting at `__syncthreads()`, at a tile
  barrier, or (for `__activemask()`) at any warp operation. That is why
  early returns don't block barriers, and why no barrier can deadlock.
  `deadlock()` is an internal consistency check.
- `threadIdx`, `blockIdx`, `blockDim` and `gridDim` are stored in
  `thread_block::Vars_`, which `set_current_thread()`, `execute()` and the
  direct-mode loop keep up to date. The getters return `dim3` by value. The
  lane is computed from `threadIdx` and `blockDim`, and so is gdb's
  `$laneid()`.
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
- A barrier costs about 200 instructions per thread, and `perf annotate` is
  the way to find waste in it. Keep error paths out of the scheduler's frames
  (`report_divergent_barrier` is `[[gnu::noinline, gnu::cold]]`; inlined, it
  gave `check_barrier_site` a 200-byte frame with a stack protector), and
  divisions out of `next_runnable`. `switch_to_thread` prefetches the saved
  frames of the thread after `to`: a block's stacks rotate through more memory
  than L1 holds, and the resumed fiber's first loads missed.
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
- The intrinsics with a rounding mode (`__dadd_rd`, `__float2int_ru`, ...)
  set only the SSE rounding mode in MXCSR on x86-64 (`detail::rounded`):
  `fesetround` also sets the x87 control word and cost over 100 ns per
  call, 95% of `interval`'s run time. The fma variants still use
  `fesetround` unless `__FMA__` is defined, because glibc's software fma
  reads the x87 mode. `tests/runtime` checks every variant against
  `fesetround` on random operands.
- `gridDim` and `blockDim` are object-like macros, so any struct field with
  those names breaks. CUDA's `cudaKernelNodeParams` has them: ours are named
  `cuda4cpu_gridDim`/`cuda4cpu_blockDim`, and the rewriter renames member
  accesses. Name new fields like that if a CUDA struct needs them
  (`cudaLaunchConfig_t` does).
- Kernel nodes and `cudaLaunchKernel(const void *)` find a kernel's binder in
  a registry keyed by address, filled by `kernel_address()`. Launches of a
  typed function pointer don't need it.
- Graph allocations are `mmap(PROT_NONE)` ranges, committed with `mprotect`
  while allocated (`graph_allocation`). So `cudaFree`, `cudaFreeHost` and
  `cudaFreeAsync` go through `detail::free_memory` in the library, which
  checks the registry (skipped while there are no graph allocations) before
  `std::free`.
- An executable graph copies its nodes (kernels are shared `bound_kernel`s),
  so the graph can be destroyed right after instantiation, as CUDA allows.
  `exec_node::origin` keeps the graph node pointer only as a key for
  `cudaGraphExec*NodeSetParams` and updates; it may dangle.
- `clock()` can't be overloaded (it's the C function, and `std::clock` must
  keep working), so the rewriter replaces device calls. `clock64()` and
  `device_clock()` count at `clockRate`, from `steady_clock`.
- Copies and memsets go through `detail::copy_bytes`, `copy_rows` and
  `fill_rows` in the library, which split those of 4 MB or more across the
  OpenMP threads (at least 2 MB each, and none inside a parallel region).
  Copying into a fresh allocation is bound by page faults, which threads
  take in parallel: 256 MB goes from about 180 ms to 30 ms. Overlapping
  copies stay one `memmove`.
- Allocations go through `detail::allocate_aligned` in the library. When
  inline, GCC may elide an allocation whose result is never dereferenced,
  and treat it as successful.
- Runtime calls that fail must return the error through
  `detail::record_error(err)`, which also makes it the thread's last error.
  `grid_launcher::call` rejects configurations outside CUDA's limits with
  `cudaErrorInvalidConfiguration`.
- Don't define `__noinline__` as a macro: libstdc++ uses
  `__attribute__((__noinline__))`.
- `cudaStreamDefault` (and `cudaStreamNonBlocking`) are macros for `0x00`,
  as in CUDA: code passes `cudaStreamDefault`/`hipStreamDefault` where a
  stream goes, which needs a null pointer constant. An enumerator doesn't
  convert to a pointer. `cudaStreamPerThread` is a `thread_local` stream per
  OS thread, which `cudaStreamDestroy` refuses.
- HIP sources get `-include cuda_runtime.h`, like `.cu` files, not
  `hip/hip_runtime.h`: programs define `HIP_TEMPLATE_KERNEL_LAUNCH` before
  their own `#include <hip/hip_runtime.h>`, which an implicit include would
  preempt. Clang's HIP mode doesn't include it either.

- Debugging settings come from the environment, read once in `settings`:
  `CUDA4CPU_SCHEDULE` (forward, reverse, random[:seed]),
  `CUDA4CPU_DIVERGENT_BARRIERS` (warn, error, ignore) and
  `CUDA4CPU_INTERLEAVE` (auto, off, or a number of loop iterations).
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
