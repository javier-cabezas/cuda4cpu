# Roadmap

What's planned next, roughly in order within each section. Contributions are
welcome: open an issue to discuss an item before starting on a large one.

## Features

The compatibility suite (see [docs/compatibility.md](docs/compatibility.md))
shows what blocks real programs. 129 of 154 pass today; most of the rest need
one of these:

- **Textures and surfaces** (texture objects, `tex1D`/`tex2D`/`tex3D`,
  `surf2Dwrite`, CUDA arrays): 10 programs.
- **Dynamic parallelism**: kernels that launch kernels. 2 programs.
- **Smaller API gaps**, one program each: libcu++ headers (`<cuda/...>`),
  device-side `assert` reported as a launch error, `hip/hip_complex.h`, 8-bit
  floats (`hip/hip_fp8.h`), green contexts.
- **Kernel language:** bfloat16, `__reduce_*_sync`, `memcpy_async`, 16-bit
  integer atomics.
- **Platforms:** CI on AArch64, where the portable context switch
  (`_setjmp`/`_longjmp`) is used but hasn't been tested yet, and an assembly
  context switch for it. macOS and Windows would need their own fiber stacks
  and context switch.

Out of scope: inline PTX, the driver API, and graphics interop.

## Performance

Measured on the compatibility suite (131.8 s in total for the programs that
pass, Clang 22, `-march=native`, 16 OS threads). `cuda-samples/scan` takes 42%
of it: about half of its time goes to barriers, because it waits at
`__syncthreads()` after every two or three operations.

1. **Find out where idle OpenMP time comes from.** OpenMP threads wait for
   9.5% of `scan`'s time and 16% of `streamcluster`'s. If it's uneven work at
   the end of each launch, smaller chunks of blocks would help.
2. **Inline the barrier fast path into kernels.** A barrier costs about 6.5 ns
   per CUDA thread on one core, and a bare fiber switch 3.5 ns. Moving the
   fast path into the header would save the call and some of the rest, at
   most about a quarter of `scan`'s time, at the cost of exposing the
   scheduler's state in a public header.
3. **Remove barriers at compile time.** Splitting a kernel at each barrier
   and running each piece as a loop over the block's threads (as POCL and
   CuPBoP do) avoids switching altogether, and is the only way to make
   barrier-dense kernels an order of magnitude faster. It needs a Clang-based
   transform, so it would be an option next to the plain C++ compiler path.
