# Bugs in the compatibility suite's programs

cuda4cpu-mcp's `check` found these by running the programs that pass in
`baseline.json` with AddressSanitizer and UBSan, and with the threads of each
block in several orders. Each one was confirmed in the source. Each patch
applies to the revision `suites.json` pins (`git apply <patch>` in the
suite's checkout), and with it `check` finds nothing in 9 orders and under
the sanitizers. None has been reported upstream.

The suite runs the programs unpatched: they pass in the default order, as
they do on a GPU.

| Patch | Program | Bug |
|---|---|---|
| `cuda-samples-dxtc.patch` | cuda-samples dxtc | A data race in `colorSums` (`CudaMath.h`): each step of the in-place butterfly sum reads `sums[idx ^ mask]` while that thread writes it, which works only if the tile's 16 threads run in lockstep, and Volta's independent thread scheduling doesn't promise that. In reverse order the RMS error is 560.6 and the test fails. |
| `rodinia-srad_v2.patch` | Rodinia srad_v2 | Out-of-bounds reads at the grid's edges (`srad_kernel.cu`): both kernels load `index_n`, `index_s`, `index_w` and `index_e` before replacing them at the edges, so they read up to a row before the array and past its end. With cuda4cpu at 256×256, 14 of 100 runs crashed with SIGSEGV, and none with the patch, whose results are identical. |
| `cuda-samples-sobolqrng.patch` | cuda-samples SobolQRNG | Arrays allocated with `new[]` freed with `delete` (`sobol.cpp`). |
| `cuda-samples-graphmemorynodes.patch` | cuda-samples graphMemoryNodes | `foundValidationFailure`, from `cudaMallocManaged`, is read before anything sets it, and CUDA doesn't clear managed memory (`graphMemoryNodes.cu`). |

Other reports were undefined behavior that does no harm: signed overflow in
warp lanes whose sums are discarded (reduction), in a host timing loop
(shfl_scan) and in a linear congruential generator (Rodinia particlefilter),
and `1 << tid` for `tid >= 32` (systemWideAtomics).
