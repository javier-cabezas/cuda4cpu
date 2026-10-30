# Debugging

Kernels are ordinary CPU code, so CPU debugging tools work on them. cuda4cpu
also adds checks that only a CPU implementation can make.

**gdb.** Set breakpoints by line, step and inspect locals as in any program.
Each CUDA thread has its own stack, so a backtrace shows the kernel's frames
and ends where the thread started. cuda4cpu's gdb script adds the built-in
variables of the CUDA thread you're stopped in:

```
(gdb) source /usr/local/share/cuda4cpu/cuda4cpu-gdb.py
(gdb) break kernel.cu:42 if $threadIdx("x") == 6 && $blockIdx("x") == 2
(gdb) run
(gdb) print $threadIdx()
$1 = {x = 6, y = 0, z = 0}
(gdb) cuda4cpu
thread (6, 0, 0) of block (2, 0, 0), lane 6; blocks of (64, 1, 1) threads, grid of (8, 1, 1) blocks
(gdb) cuda4cpu threads
block (2, 0, 0), 64 threads (64 live):
  threads (0, 0, 0)..(5, 0, 0): at __syncthreads() (kernel.cu:45)
* thread (6, 0, 0), lane 6: running
  threads (7, 0, 0)..(63, 0, 0): runnable
(gdb) cuda4cpu thread 4
switched to thread (4, 0, 0) of block (2, 0, 0), lane 4 at kernel.cu:45
(gdb) cuda4cpu thread
thread (6, 0, 0) of block (2, 0, 0), lane 6; blocks of (64, 1, 1) threads, grid of (8, 1, 1) blocks
```

`$threadIdx()`, `$blockIdx()`, `$blockDim()` and `$gridDim()` return the whole
value. With a dimension (`$threadIdx("x")`), they return one component, which
is what breakpoint conditions need. `cuda4cpu threads` lists the CUDA threads
of the block and their states, and `cuda4cpu thread <x> [y] [z]` switches to
another started CUDA thread so `bt` and `print` inspect its stack; `cuda4cpu
thread` (or continuing execution) switches back to the running thread. The
program isn't modified: gdb shows that thread's frames as callers of the frame
the OS thread stopped in, which `bt` hides (frame numbers start at 1), so
continuing from an IDE is safe too. Set
`OMP_NUM_THREADS=1` to run all the blocks on one OS thread, one after another.

**Crashes in kernels.** A kernel that crashes is reported with the CUDA
thread and block it crashed in, and the line of its launch:

```
cuda4cpu: error: CUDA thread (17, 0, 0) of block (3, 0, 0) made an invalid memory access at
address 0x7f3a5c1e0100, in the kernel launched at main.cu:42. Build with -g -fsanitize=address
to find the access.
```

Invalid memory accesses, stack overflows (with the stack's size and how to
raise it), integer divisions by zero, illegal instructions and aborts, such
as a failed `assert()`, are reported. The program then dies of the signal as
it would without cuda4cpu (a core dump, or the program's own handler), and a
debugger stops at the fault before the report. In optimized code, a CUDA
thread's index may still be in a register when it crashes before a barrier,
so cuda4cpu runs its block once more, each thread on its own, to find the
thread. Under AddressSanitizer, the report of an error the sanitizer finds
follows the sanitizer's, which has the line.

**Missing `__syncthreads()`: `CUDA4CPU_SCHEDULE`.** Between barriers, the
threads of a block run one after another, in index order by default. A missing
barrier goes unnoticed when the thread that reads shared memory happens to run
after the one that writes it. Run the program again with another order; if the
results change, a barrier is missing:

- `reverse`, `random` (or `random:<seed>`): reverse or shuffle all the threads
  of each block, exposing missing `__syncthreads()` and `__syncwarp()`.
- `warp_reverse`, `warp_random` (or `warp_random:<seed>`): reverse or shuffle
  the warps of each block while keeping the lanes of each warp in forward
  order, for kernels that rely on lockstep execution within a warp.
- `mixed` (or `mixed:<seed>`): cycles the blocks of each grid through
  `reverse`, `warp_reverse`, `random` and `forward`, so a single run tests all
  of them.

`random`, `warp_random` and `mixed` print the seed they picked so the run can
be repeated.

**Divergent barriers and warp operations.** CUDA requires all the threads of a
block to reach the same `__syncthreads()`, and all the participating lanes of a
warp operation (`__shfl_*_sync`, votes, `__match_*_sync`, `__syncwarp`) to have
their own lane bit set in `mask` and to execute the same operation with the
same `mask`. cuda4cpu reports violations:

```
cuda4cpu: warning: threads of block (0, 0, 0) reached different __syncthreads() calls,
at kernel.cu:12 and kernel.cu:15. CUDA requires every thread of a block to reach the same one.
```

`CUDA4CPU_DIVERGENT_BARRIERS` chooses what happens: `warn` (the default), `error`
(abort, to stop in a debugger) or `ignore`.

**Threads taking turns in loops: `CUDA4CPU_INTERLEAVE`.** `auto` (the default)
chooses per kernel how often the threads of a block take turns in their loops.
`off` runs each thread to its next barrier, and a number makes each thread let
the next one run after that many iterations (`1`: every iteration). Like
`CUDA4CPU_SCHEDULE`, a different value can expose code that depends on the
order threads run in.

**Pointer checks in `cudaFree` and `cudaMemcpy`.** Because every allocation is
recorded with its kind, `cudaFree` rejects host stack/heap pointers, interior
pointers (`ptr + k`), symbol addresses and double frees with
`cudaErrorInvalidValue`. Copies (`cudaMemcpy`, `cudaMemcpy2D`, `cudaMemcpy3D`,
their async and HIP equivalents) check that their pointers are where their
`cudaMemcpyKind` says: a `cudaMalloc` pointer passed as the host side of
`cudaMemcpyHostToDevice`, or swapped `dst` and `src`, is reported:

```
cuda4cpu: warning: a copy with cudaMemcpyHostToDevice has its destination in host memory that CUDA
didn't allocate, which CUDA leaves undefined. The copy goes ahead; CUDA4CPU_MEMCPY_DIRECTION=error
makes it fail with cudaErrorInvalidMemcpyDirection.
```

CUDA leaves such copies undefined, and a GPU may make them anyway, so
`CUDA4CPU_MEMCPY_DIRECTION` chooses what happens: `warn` (the default, once
for each kind of mistake), `error` (the copy fails with
`cudaErrorInvalidMemcpyDirection`) or `ignore`. Pinned, registered and managed
memory can be either side. `cudaMemcpyToSymbol` and `cudaMemcpyFromSymbol`
always reject a kind they don't accept, as CUDA does.

**AddressSanitizer and UBSan.** Compile your program with `-fsanitize=address`
(and `undefined`). cuda4cpu tells AddressSanitizer about every switch between
fiber stacks, so out-of-bounds accesses in kernels are reported with the
kernel's source line. That covers local arrays and memory from `cudaMalloc`.
cuda4cpu rounds allocations up to 256 bytes and poisons the rest, so an
access just past the end of one is reported as `use-after-poison` (`is
located N bytes inside of` the rounded region, with N the size asked for or
more), and one past the rounding as `heap-buffer-overflow`. The library
doesn't need to be rebuilt. By default, AddressSanitizer aborts on
allocations too large to satisfy. Set `ASAN_OPTIONS=allocator_may_return_null=1`
to make `cudaMalloc` return `cudaErrorMemoryAllocation` instead.

**Valgrind.** Build cuda4cpu with `-DCUDA4CPU_ENABLE_VALGRIND=ON`, so that it
registers its fiber stacks with Valgrind. Otherwise Valgrind reports false
errors in the context switch.

## Checking kernels from an AI agent (MCP)

`cuda4cpu-mcp` is an MCP server (Model Context Protocol, over stdio) that
compiles, runs and checks CUDA and HIP programs with cuda4cpu, so that an
agent can validate kernels on a machine without a GPU. Register it with an
MCP client, for example Claude Code:

```sh
claude mcp add cuda4cpu -- /usr/local/bin/cuda4cpu-mcp     # or build/<preset>/tools/cuda4cpu-mcp
claude mcp add cuda4cpu -- docker run -i --rm -v "$PWD:$PWD" ghcr.io/javier-cabezas/cuda4cpu
```

In Docker, the server sees only the directories you mount: `paths` must be
in one of them (`-v "$PWD:$PWD"` mounts the current directory at the same
path), and `files` work without any. The server is in the
[MCP Registry](https://registry.modelcontextprotocol.io) as
`io.github.javier-cabezas/cuda4cpu`, with the Docker image as its package.

`compile`, `run` and `check` take the sources inline (`files`, contents by
file name) or by path (`paths`), nvcc's or hipcc's `options`, and for runs
the `args`, `stdin`, `environment`, `working_directory` and
`timeout_seconds` (60 by default) of the program. Results are JSON, with a
short summary first.

- **`compile`** builds the sources with `cuda4cpu-nvcc`, or `cuda4cpu-hipcc`
  for HIP (by default when a source is a `.hip` file). It returns the
  compiler's diagnostics at the sources' own lines, and names the features
  they use that cuda4cpu doesn't implement (textures, inline PTX, libcu++,
  the CUDA libraries, the driver API, ...).
- **`run`** builds and runs the program: exit code, output (up to 16 KB of
  each), and the problems cuda4cpu reports along the way.
- **`check`** looks for bugs, and returns a verdict and findings:
  1. A build with AddressSanitizer and UBSan, run in the default order:
     out-of-bounds and use-after-free accesses and undefined behavior, at the
     kernel's line, and the CUDA thread and block that made them. An access
     just past the end of an allocation, in the padding cuda4cpu poisons, is
     an `allocation-overflow`. If it finds any, the other orders aren't run: memory
     errors make their results unreliable, and a program that corrupts its
     heap may hang.
  2. The default order, twice: an exit code other than 0 is a finding, and
     output that differs between the two runs (timings) is compared by exit
     code only, unless `ignore_patterns` removes the lines that vary.
  3. The other orders (`reverse`, `warp_reverse`, and `random` and
     `warp_random` with each of `seeds`, 1 to 3 by default). A result that
     changes with the order is reported once, with the orders that changed
     it and the first line of output that differs: a thread reads what
     another writes without a `__syncthreads()` (or `__syncwarp()`) between
     them, or threads race.

  Every run reports divergent barriers and warp operations and copies in
  the wrong direction (in warn mode, so that the program's behavior doesn't
  change), and a time-out stops the check. A clean check means that nothing
  showed up in those orders and checks, not that the program has no race.
- **`test_kernel`** tests one kernel without a program. It takes the
  kernel's source and `name` (a template with its arguments works too), the
  `grid`, `block` and `shared_bytes`, and the kernel's `arguments` in order:
  a scalar is `{"type": "float", "value": 2.5}`, an array a pointer type with
  its `data`, given (`[1, 2, 3]`) or generated (`{"fill": 0, "count": n}`,
  `{"range": [start, step], "count": n}`, `{"random": [min, max], "count":
  n, "seed": s}`). An output array has `expected` values, or `"output":
  true` and a C++ `reference` function with the kernel's parameters, which
  gets host copies of the inputs and writes the expected outputs. Integers
  must match exactly, floating point within a `tolerance` (1e-5 absolute
  plus 1e-5 relative by default). The result lists each output's
  mismatches by index, with the values it got and expected, and with
  `check` (the default) the findings of the checks above: it passes only if
  every output matches and they find nothing.
- **`capabilities`** returns what cuda4cpu implements and doesn't (this
  documentation's `features.md` and `limitations.md`, installed with it).

Programs run with your permissions, each in its own process group (which a
time-out kills) in a temporary directory, without core dumps. Nothing else
isolates them.
