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
```

Build trees live in `build/<preset>/` (git-ignored). The presets export
`compile_commands.json` there.

## Layout

- `include/cuda4cpu.hpp`: umbrella header that users include
- `include/defines.hpp`: macro overrides for `__global__`, `__shared__`,
  `threadIdx`, `__syncthreads`, and similar. It's included **last** on purpose,
  because these macros would otherwise break the library's own headers.
- `include/launch.hpp`: the execution engine (`launch_conf`, `system`,
  `thread_block`, `grid_launcher`, `launch()`)
- `include/cuda/*.hpp`: CUDA runtime API shims (memory, streams, events,
  device, types). All are header-only `static inline` functions in namespace
  `cuda4cpu`.
- `lib/cuda4cpu.cpp`: the only compiled source. It defines `system::sys_`
  (topology via OpenMP plus optional libnuma) and `thread_block::Current_`.
- `tests/stencil{2,3}d.cpp`: kernels with shared memory, `__constant__` and
  `__syncthreads`, checked against a host reference. They return non-zero on
  mismatch.

## How execution works (read before touching `launch.hpp`)

- `grid_launcher::call` binds the arguments into a `std::function`, then splits
  the blocks across OpenMP threads (`schedule(static, 1)` over one chunk per
  CPU).
- Each OpenMP thread builds one `thread_block`, which holds one fiber per CUDA
  thread. Fibers are created once with `makecontext` and snapshotted as a
  `jmp_buf` (`callees_base_`). Each block execution copies the snapshots and
  switches with `_setjmp`/`_longjmp`.
- `__syncthreads()` is a round-robin yield to the next fiber. `threadIdx`
  resolves through `thread_id_barrier_`, the fiber that is running now.
- `thread_block::Current_` is `thread_local`; built-in variables read from it.
- `__shared__` is `static thread_local`. That's correct only because a block
  always runs to completion on one OS thread.

## Gotchas

- `_FORTIFY_SOURCE` breaks the fiber switch: the fortified `longjmp` aborts with
  "longjmp causes uninitialized stack frame". The CMake target passes
  `-U_FORTIFY_SOURCE` PUBLICly, and the tests also `#undef` it. Keep both.
- `CUDA4CPU_HANDLE_VALGRIND` changes the layout of `thread_block`, so it must be
  a PUBLIC definition, the same in every translation unit.
- Fiber stacks are `SIGSTKSZ` bytes, which is a runtime value (`sysconf`) on
  glibc 2.34+ and not a constant expression.
- The `makecontext` trampoline passes a pointer as two 32-bit ints, which
  assumes a 64-bit platform.
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
- Changes to `master` go through pull requests. CI (`.github/workflows/ci.yml`)
  builds and tests with GCC and Clang in Debug and Release.
