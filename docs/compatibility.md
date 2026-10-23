# Compatibility with real CUDA and HIP code

`compat/` builds and runs real CUDA programs, unmodified, with `cuda4cpu-c++`,
and HIP programs with `cuda4cpu-hipcc`:

- **cuda-samples:** a curated set of NVIDIA's CUDA samples, those that use
  only the runtime API and the kernel language. Samples that need graphics
  interop, the driver API, NVRTC, the CUDA libraries, Thrust/CUB, MPI or
  several GPUs are excluded. These samples check their own results.
- **Rodinia:** the Rodinia benchmarks that generate their own inputs. Except
  `lud`, they don't check their results, so for them "passing" means they run
  to completion.
- **rocm-examples:** AMD's HIP examples: HIP-Basic, the examples of the HIP
  documentation, and the reduction tutorial, excluding those that need
  graphics interop, the module API, HIPRTC, ROCm libraries or several GPUs.
  Most check their results, or print results that the suite checks. Many
  of the documentation's examples only check for API errors.

The suites are fetched at the revisions pinned in `compat/suites.json`. CI runs
them and fails if a program that passes in `compat/baseline.json` stops
passing. Today:

| Suite | Pass | Doesn't pass yet |
|---|---|---|
| cuda-samples | 43 of 61 (1 more waives itself, and 2 are skipped as too slow) | 15 |
| Rodinia | 10 of 11 | 1 (needs OpenGL) |
| rocm-examples | 76 of 82 | 6 |

Most of what's missing is a few features:

| Programs | Missing |
|---|---|
| 10 | Textures and surfaces |
| 2 | Dynamic parallelism (kernels that launch kernels) |
| 2 | Inline PTX |
| 1 each | libcu++ (`<cuda/...>`), green contexts, device-side `assert` reported as an error, OpenGL, 8-bit floats (`hip/hip_fp8.h`), complex numbers (`hip/hip_complex.h`) |

Two more, versions 5 and 6 of the reduction tutorial, have a race:
their last warp reduces through shared memory without `__syncwarp()` between
steps. That relies on the lanes of a warp running in lockstep, as AMD's do.
CUDA doesn't promise it, and on cuda4cpu each lane runs to its next barrier,
so the result is wrong.

To run the programs yourself (they need git and network access):

```sh
compat/run.py --driver build/release/tools/cuda4cpu-c++ --work /tmp/compat --report report.md
```

HIP programs are built with the `cuda4cpu-hipcc` next to `--driver`
(`--hipcc` chooses another).
