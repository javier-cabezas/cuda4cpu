# How it works

- **Grid → OpenMP threads.** `launch(...).call(...)` splits the grid's thread
  blocks into contiguous chunks, about 8 per OS thread, and hands them out to
  the threads dynamically, so blocks of uneven cost balance out. There is one
  OS thread per CPU core, unless `OMP_NUM_THREADS` says otherwise.
- **Block → fibers.** Within a block, each CUDA thread is a user-level fiber
  with its own stack. On x86-64, switching fibers takes a dozen instructions
  that save and restore the callee-saved registers. Including scheduling, a
  CUDA thread spends about 6.5 ns at a barrier on a single core, and 10 ns at
  a shuffle. Each OS thread creates fibers only once per block size, and
  reuses them for every block and every later launch.
- **No barriers, no switching.** A block starts by running its CUDA threads
  back to back, each one a direct call to the kernel. Kernels that never call
  `__syncthreads()` never switch fibers. `threadIdx`, `blockIdx`, `blockDim`
  and `gridDim` are thread-local values that the scheduler keeps up to date,
  so reading one is a single memory load. With Clang, AVX2 or later
  (`-march=x86-64-v3`, or `native`) and `-DCUDA4CPU_VECTORIZE_THREADS`, the
  loop over the CUDA threads of a kernel without barriers vectorizes, as if
  the threads were the lanes of a SIMD loop: a 2D stencil ran 2.2 times
  faster on one core. It's off by default, because the threads out of bounds
  still issue the kernel's loads and stores, masked off: a launch with far
  more threads than elements ran 13 times slower or worse, when those
  addresses weren't mapped.
- **Loops → threads take turns.** GPU code expects the threads of a warp to
  touch neighboring addresses together. Run one after another, each thread of
  a grid-stride loop touches a new page per iteration, and page walks
  dominate. cuda4cpu-rewrite inserts a cheap tick at the top of the outermost
  loops of device functions, and every few iterations the next CUDA thread of
  the block runs. Each kernel launch site times its first blocks with no
  turns, with turns every 4 iterations and every 16, and keeps the fastest.
  A 50 MB grid-stride copy ran 5.7 times faster.
- **`__syncthreads()` → cooperative switch.** When a CUDA thread first reaches
  a barrier, the block switches to fiber mode. From then on, each CUDA thread
  runs until it reaches a barrier and then yields to the next one. Once every
  live thread has reached the barrier, execution continues past it. As on
  GPUs, threads that return early don't hold back the barrier.
- **Warp functions → warp barriers.** A shuffle or vote is a barrier among the
  lanes of one warp. Each lane publishes its value and yields. When the last
  participating lane arrives, the results are computed for all of them and
  they continue.
- **Cooperative groups → the same barriers.** A tile of up to 32 threads is
  a warp function with the tile's lanes as its mask. A larger tile waits at a
  barrier over its warps, released like `__syncthreads()`. A cooperative
  launch runs every block at the same time, each on its own OS thread, so
  `grid.sync()` is a `__syncthreads()` whose last thread waits for the other
  blocks.
- **Atomics → `std::atomic_ref`.** Thread blocks run in parallel on different
  cores, so atomics on global memory are real atomic operations. They are
  relaxed, as in CUDA.
- **`__shared__` → `static thread_local`.** A block always runs to completion
  on a single OS thread, so a thread-local variable behaves as per-block shared
  memory.
- **Device memory is host memory.** `cudaMalloc` is `malloc`, `cudaMemcpy` is
  `memcpy`, and streams and events run synchronously.
- **Graphs run their nodes in order.** An instantiated graph is a snapshot of
  its nodes in an order that respects their dependencies, and launching it
  runs them one after the other. A stream being captured adds each operation
  as a node that depends on the stream's previous work. Events carry those
  dependencies to other streams, which builds the same graph as CUDA. Graph
  allocations reserve their address range when the node is added, and are
  committed while allocated, so using them after they are freed faults.
