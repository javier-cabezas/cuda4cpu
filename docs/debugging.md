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
```

`$threadIdx()`, `$blockIdx()`, `$blockDim()` and `$gridDim()` return the whole
value. With a dimension (`$threadIdx("x")`), they return one component, which
is what breakpoint conditions need. gdb lists OS threads, not the CUDA threads
suspended at barriers. Set `OMP_NUM_THREADS=1` to run all the blocks on one
thread, one after another.

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

**AddressSanitizer and UBSan.** Compile your program with `-fsanitize=address`
(and `undefined`). cuda4cpu tells AddressSanitizer about every switch between
fiber stacks, so out-of-bounds accesses in kernels are reported with the
kernel's source line. That covers local arrays and memory from `cudaMalloc`.
The library doesn't need to be rebuilt. By default, AddressSanitizer aborts on
allocations too large to satisfy. Set `ASAN_OPTIONS=allocator_may_return_null=1`
to make `cudaMalloc` return `cudaErrorMemoryAllocation` instead.

**Valgrind.** Build cuda4cpu with `-DCUDA4CPU_ENABLE_VALGRIND=ON`, so that it
registers its fiber stacks with Valgrind. Otherwise Valgrind reports false
errors in the context switch.
