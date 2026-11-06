# Roadmap

What's planned next, roughly in order within each section. Contributions are
welcome: open an issue to discuss an item before starting on a large one.

## Features

The compatibility suite (see [docs/compatibility.md](docs/compatibility.md))
shows what blocks real programs. 133 of 159 pass today; most of the rest need
one of these:

- **Textures and surfaces** (texture objects, `tex1D`/`tex2D`/`tex3D`,
  `surf2Dwrite`, CUDA arrays): 10 programs.
- **Dynamic parallelism**: kernels that launch kernels. 2 programs.
- **Smaller API gaps**, one program each: device-side `assert` reported as
  a launch error, green contexts.
- **Kernel language:** `__reduce_*_sync`, 16-bit integer atomics, the
  sub-byte types of `wmma::experimental`, and pipelines that a group shares
  (`cuda::pipeline<cuda::thread_scope_block>`).
- **Platforms:** CI on AArch64, where the portable context switch
  (`_setjmp`/`_longjmp`) is used but hasn't been tested yet, and an assembly
  context switch for it. macOS and Windows would need their own fiber stacks
  and context switch.

Out of scope: inline PTX, the driver API, and graphics interop.

## Performance

Measured on the compatibility suite (124.5 s in total for the programs that
pass, Clang 22, `-march=native`, 16 OS threads). `cuda-samples/scan` takes 41%
of it: about half of its time goes to barriers, because it waits at
`__syncthreads()` after every two or three operations.

1. **Find where `scan`'s branch mispredictions come from.** Its barriers are
   bound by stalls, not instructions: inlining the barrier into the kernels
   (static library, link-time optimization) took 4.8% of its instructions
   away and none of its cycles, although a barrier went from 6.3 to 5.2 ns in
   the micro-benchmarks. It mispredicts about 4.6 * 10^10 branches, roughly a
   fifth of its cycles.
2. **Remove barriers at compile time.** Splitting a kernel at each barrier
   and running each piece as a loop over the block's threads (as POCL and
   CuPBoP do) avoids switching altogether, and is the only way to make
   barrier-dense kernels an order of magnitude faster. It needs a Clang-based
   transform, so it would be an option next to the plain C++ compiler path.
3. **Warp matrix functions in one synchronization.** `mma_sync` syncs the
   warp twice, once when the lanes have given their elements of A and B and
   once before the next one reuses the buffer, and those syncs take about
   half its time. A warp operation whose releasing lane computes D for every
   lane would need one. A 4096×4096×4096 GEMM (`cudaTensorCoreGemm`) takes
   about 14 s on 16 OS threads, and the 8192³ tensor-core samples, which
   the compatibility suite skips, from 45 s (bfloat16, tf32) to 5 minutes
   (double, whose 8×8×4 tiles need sixteen times as many `mma_sync` calls).
