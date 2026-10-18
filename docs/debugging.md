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
after the one that writes it. Run the program again with
`CUDA4CPU_SCHEDULE=reverse` or `CUDA4CPU_SCHEDULE=random`; if the results
change, a barrier is missing. `random` prints its seed, and
`CUDA4CPU_SCHEDULE=random:<seed>` repeats that order.

**Divergent barriers.** CUDA requires all the threads of a block to reach the
same `__syncthreads()`. cuda4cpu reports blocks whose threads reach different
ones:

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
