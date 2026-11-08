# Roadmap

What's planned next, roughly in order within each section. Contributions are
welcome: open an issue to discuss an item before starting on a large one.

## Features

The compatibility suite (see [docs/compatibility.md](docs/compatibility.md))
shows what blocks real programs. 135 of 161 pass today; most of the rest need
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

## AMD GPU builtins

HIP code that only runs on AMD GPUs reaches `__builtin_amdgcn_*` through
`__HIP_PLATFORM_AMD__` and `__gfx1100__`-style branches, while cuda4cpu runs
HIP as HIP's NVIDIA platform does. Clang lists 870 of these builtins, but real
code uses few: rocPRIM 15, llama.cpp's HIP backend 23, rocWMMA 52 (most of
them matrix operations), composable_kernel 362. Both GCC and Clang accept
inline functions with those names. Without an AMD GPU, the semantics come from
AMD's ISA manuals and LLVM's lowering.

1. **An AMD platform emulating RDNA3 (`gfx1100`), with 32-lane wavefronts,
   and the core builtins.** An explicit switch (`--cuda4cpu-platform=amd`, or
   `CUDA4CPU_HIP_PLATFORM=amd`) defines `__HIP_PLATFORM_AMD__`, `__AMDGCN__`,
   `__gfx1100__`, `__GFX11__` and a wavefront size of 32, and
   `--offload-arch` picks the GPU. The builtins: dispatch IDs and sizes,
   lane operations (`readfirstlane`, `readlane`, `writelane`, `ballot`,
   `mbcnt`, `ds_bpermute`/`ds_permute`, `ds_swizzle`, `mov_dpp`/`update_dpp`,
   `mov_dpp8`, `permlane*`, `wave_reduce_*`), synchronization (`s_barrier`,
   `wave_barrier`, `fence`, `s_sleep`, clocks, and the scheduling hints, which
   do nothing), integer and bit operations (`perm`, `alignbit`, `ubfe`/`sbfe`,
   `lerp`, `sad`, the dot products), floating-point ones (`rcp`, `rsq`,
   `sin`/`cos` in turns, `exp2`, `log`, `ldexp`, `frexp`, `fract`, `class`,
   `fmed3`), and atomics. Then rocPRIM, built for `gfx1100`, in the
   compatibility suite. Done: rocPRIM 7.2.4's block and device sums pass,
   built with Clang.
2. **Conversions and buffer resources:** fp8/bf8 in AMD's FNUZ and the OCP
   formats, stochastic rounding, scaled FP4/FP6/FP8 (OCP MX), and buffer
   loads and stores with AMD's out-of-range rules.
3. **WMMA on 32-lane wavefronts**, RDNA3's and RDNA4's (`_gfx12`), and
   SWMMAC, with the hardware's lane layouts. They take `ext_vector_type`
   vectors, so they need Clang as the host compiler. Then RDNA4 (`gfx1200`,
   `gfx1201`) as a target, and rocWMMA's samples.
4. **64-lane wavefronts and CDNA's MFMA** (`gfx90a`, `gfx942`, `gfx950`),
   which need warps of 64 lanes in the scheduler: decided after step 3.

Out of scope: graphics (`image_*`, ray tracing, interpolation), GFX12.5's
cluster and tensor DMA builtins, and AMD's lockstep execution of a
wavefront's lanes, which code that skips `__syncwarp()` relies on.
`__has_builtin(__builtin_amdgcn_*)` stays false, since they aren't the
compiler's builtins.

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
