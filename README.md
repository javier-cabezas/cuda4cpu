![cuda4cpu](https://raw.githubusercontent.com/wiki/javier-cabezas/cuda4cpu/images/cuda4cpu_logo_300x75.png)

[![CI](https://github.com/javier-cabezas/cuda4cpu/actions/workflows/ci.yml/badge.svg)](https://github.com/javier-cabezas/cuda4cpu/actions/workflows/ci.yml)
[![Real programs](https://img.shields.io/badge/real%20CUDA%2FHIP%20programs-130%20of%20154%20pass-brightgreen)](docs/compatibility.md)
[![MCP server](https://img.shields.io/badge/MCP-kernel%20checks%20for%20AI%20agents-8A2BE2)](#use-it-from-an-ai-agent-mcp)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](docs/usage.md#requirements)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Run CUDA and HIP programs on your CPU, unmodified, with your regular C++
compiler.** No GPU, no `nvcc`, no CUDA toolkit: GCC or Clang compile your
`.cu` files, and the kernels run as native code on every core.

```console
$ cuda4cpu-c++ -O2 vecadd.cu -o vecadd
$ ./vecadd
c[42] = 126, computed on cuda4cpu: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
```

- **Test CUDA code where there's no GPU:** in CI on GitHub's hosted runners,
  on a laptop, or in a classroom.
- **Let AI agents test the kernels they write:** `cuda4cpu-mcp`, an MCP
  server for Claude Code, Cursor or any MCP client, compiles, runs and checks
  CUDA and HIP code, and reports wrong results, missing barriers and memory
  errors at their source lines.
- **Debug kernels with the tools you already use:** gdb stops in a single CUDA
  thread, and AddressSanitizer, UBSan and Valgrind point at the kernel's
  source line.
- **Catch the bugs a GPU hides:** missing `__syncthreads()`, and threads of a
  block that reach different barriers.
- **Real code works:** 130 of 154 programs from NVIDIA's cuda-samples,
  Rodinia and AMD's rocm-examples build unmodified and pass, and CI checks
  that they keep passing.
- **The CUDA you write today:** shared memory, warp shuffles and votes,
  cooperative groups (including `grid.sync()`), atomics, half precision,
  bfloat16 and 8-bit floats, streams, events, and CUDA graphs with stream
  capture. HIP code compiles too.

## Contents

- [Quick start](#quick-start)
- [See it catch bugs](#see-it-catch-bugs)
- [Use it from an AI agent (MCP)](#use-it-from-an-ai-agent-mcp)
- [Use it in your project](#use-it-in-your-project)
- [What's supported](#whats-supported)
- [How it works](#how-it-works)
- [How it compares](#how-it-compares)
- [FAQ](#faq)
- [Roadmap and contributing](#roadmap-and-contributing)

## Quick start

You need Linux (or WSL 2) on x86-64, with GCC 14 or later and Python 3, as
on Ubuntu 24.04 (`sudo apt-get install g++-14 python3`). Install the latest
release in `~/.local`:

```sh
mkdir -p ~/.local
curl -fsSL https://github.com/javier-cabezas/cuda4cpu/releases/latest/download/cuda4cpu-linux-x86_64.tar.gz |
    tar -xz -C ~/.local --strip-components=1
```

Then compile any `.cu` file with `cuda4cpu-c++`, which takes the same
options as your C++ compiler, or with `cuda4cpu-nvcc`, which takes nvcc's:

```sh
cuda4cpu-c++ -O2 vecadd.cu -o vecadd
./vecadd
```

**With Docker**, there's nothing else to install: the image has cuda4cpu,
GCC and CMake.

```sh
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/src" -w /src ghcr.io/javier-cabezas/cuda4cpu \
    cuda4cpu-nvcc -O2 vecadd.cu -o vecadd
```

**From source**, which needs CMake 3.25+ and Ninja too, and builds and runs
the tests:

```sh
git clone https://github.com/javier-cabezas/cuda4cpu.git
cd cuda4cpu
cmake --workflow --preset release
cmake --install build/release --prefix /usr/local
```

See [docs/usage.md](docs/usage.md) for the other ways to install it and the
build options.

## See it catch bugs

**A missing `__syncthreads()`.** On a GPU, this kernel can pass its tests for
a long time: only the first thread of each warp reads what another warp
wrote. cuda4cpu runs the threads of a block in index order between barriers,
which hides the bug too. Run them in reverse or in random order, and it
shows ([examples/bugs/missing_barrier.cu](examples/bugs/missing_barrier.cu)):

```cuda
__global__ void pair_sum(const int *in, int *out)
{
    __shared__ int tile[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    tile[t] = in[i];
    // Bug: a __syncthreads() is missing here
    out[i] = tile[t] + (t > 0 ? tile[t - 1] : 0);
}
```

```console
$ ./missing_barrier
PASSED: 0 wrong results
$ CUDA4CPU_SCHEDULE=reverse ./missing_barrier
FAILED: 65279 wrong results
$ CUDA4CPU_SCHEDULE=random ./missing_barrier
cuda4cpu: running threads in random order, seed 1277581447 (CUDA4CPU_SCHEDULE=random:1277581447 reproduces it)
FAILED: 32593 wrong results
```

**An out-of-bounds access**, reported by AddressSanitizer with the kernel's
line ([examples/bugs/out_of_bounds.cu](examples/bugs/out_of_bounds.cu)):

```console
$ cuda4cpu-c++ -g -fsanitize=address out_of_bounds.cu -o out_of_bounds
$ ./out_of_bounds
==8457==ERROR: AddressSanitizer: use-after-poison on address 0x7c81561e2630 ...
READ of size 4 at 0x7c81561e2630 thread T4
    #0 0x65457dfe1ad5 in scale(float*, int) .../out_of_bounds.cu:27
    ...
0x7c81561e2630 is located 4400 bytes inside of 4608-byte region [0x7c81561e1500,0x7c81561e2700)
```

The array has 1100 floats, 4400 bytes: cuda4cpu rounds allocations up to
256 bytes and poisons the rest, so the first element past the end is
reported, as `use-after-poison`.

**Divergent barriers.** CUDA requires every thread of a block to reach the
same `__syncthreads()`, and cuda4cpu reports blocks whose threads don't:

```
cuda4cpu: warning: threads of block (0, 0, 0) reached different __syncthreads() calls,
at kernel.cu:12 and kernel.cu:15. CUDA requires every thread of a block to reach the same one.
```

**gdb in a CUDA thread.** Each CUDA thread is a fiber with its own stack, so
breakpoints, stepping and backtraces work as in any program. cuda4cpu's gdb
script adds the built-in variables:

```
(gdb) break kernel.cu:42 if $threadIdx("x") == 6 && $blockIdx("x") == 2
(gdb) run
(gdb) cuda4cpu
thread (6, 0, 0) of block (2, 0, 0), lane 6; blocks of (64, 1, 1) threads, grid of (8, 1, 1) blocks
```

[docs/debugging.md](docs/debugging.md) has the details.

## Use it from an AI agent (MCP)

A coding agent on a machine without a GPU can write CUDA, but it can't run
it, so it can't tell whether a kernel works. `cuda4cpu-mcp` gives it a way
to find out: an MCP server that compiles, runs and tests CUDA and HIP code
with cuda4cpu, and runs the checks above for it. Register it with your
agent, for example Claude Code:

```sh
claude mcp add cuda4cpu -- cuda4cpu-mcp    # installed from a release, or build/release/tools/cuda4cpu-mcp
claude mcp add cuda4cpu -- docker run -i --rm -v "$PWD:$PWD" ghcr.io/javier-cabezas/cuda4cpu
```

Other clients take the same command, for example in Cursor's `mcp.json` or
Claude Desktop's configuration: `{"mcpServers": {"cuda4cpu": {"command":
"cuda4cpu-mcp"}}}`. The server speaks MCP over stdio, and is listed in the
[MCP Registry](https://registry.modelcontextprotocol.io) as
`io.github.javier-cabezas/cuda4cpu`. In Docker, it sees only the
directories you mount (`-v "$PWD:$PWD"` above): sources given inline need
none.

| Tool | What the agent gets |
| --- | --- |
| `compile` | nvcc's or hipcc's diagnostics at the source's lines, and the features it uses that cuda4cpu doesn't implement |
| `run` | the exit code, the output, and cuda4cpu's warnings |
| `check` | a verdict, and findings with their source lines: memory errors and undefined behavior (AddressSanitizer and UBSan), results that change with the order of a block's threads (a missing `__syncthreads()` or `__syncwarp()`, a race), divergent barriers, copies in the wrong direction, hangs |
| `test_kernel` | a test of one kernel, without writing a program: the agent gives the launch configuration and the arguments (given or generated), with expected values or a C++ reference function, and gets the elements that differ and the findings of `check` |
| `capabilities` | what cuda4cpu implements and doesn't, so that the agent can tell a bug in its code from a missing feature |

Results are JSON, with a summary first. For the missing `__syncthreads()`
above, whose program passes in the default order, `check` answers:

```
problems found
  [depends on the thread order] running the threads of each block in reverse order (reverse)
  ends with exit code 1 instead of exit code 0; 8 of 8 other orders change the result
    a thread reads data that another thread of its block writes without a __syncthreads()
    between them (or, within a warp, a __syncwarp()), or threads race on the same data.
```

And for a SAXPY that computes its index with `blockDim.y` instead of
`blockDim.x`, `test_kernel` with a reference that loops over the elements
answers:

```
failed
  argument 3: 998 of 1000 elements differ (largest error 1998)
    [1] 5, expected 3
    [2] 13, expected 5
    ...
```

A clean `check` means that nothing showed up in the orders and checks it
ran, not that a program has no race. Programs run with your permissions,
each in a temporary directory and its own process group, killed at a
time-out: the server doesn't sandbox them.
[docs/debugging.md](docs/debugging.md#checking-kernels-from-an-ai-agent-mcp)
describes every tool and its arguments.

## Use it in your project

**CMake.** Fetch cuda4cpu and add your `.cu` files to a target
([examples/cmake-project](examples/cmake-project)):

```cmake
include(FetchContent)
FetchContent_Declare(cuda4cpu
    GIT_REPOSITORY https://github.com/javier-cabezas/cuda4cpu.git
    GIT_TAG v0.4.0)
FetchContent_MakeAvailable(cuda4cpu)

add_executable(my_app main.cpp)
cuda4cpu_add_cuda_sources(my_app kernels.cu solver.cu)
```

With cuda4cpu installed, `find_package(cuda4cpu REQUIRED)` does the same.
HIP sources go through `cuda4cpu_add_hip_sources`.

**Any build system.** `cuda4cpu-c++` is a compiler driver: use it like your
C++ compiler, with `.cu` files among the inputs. `cuda4cpu-hipcc` does the
same for HIP projects, and drops hipcc's options for AMD GPUs, so build
scripts written for hipcc work unchanged:

```sh
cuda4cpu-c++ -O2 -o my_app main.cpp kernels.cu
cuda4cpu-hipcc -O2 --offload-arch=gfx90a -o my_app main.cpp kernels.cpp
```

**GitHub Actions.** Hosted runners have no GPU, but they can run your CUDA
tests with cuda4cpu. The action installs a release, with the drivers on the
`PATH` and the CMake package where `find_package(cuda4cpu)` finds it:

```yaml
jobs:
  test:
    runs-on: ubuntu-24.04
    env:
      CXX: g++-14
    steps:
      - uses: actions/checkout@v4
      - uses: javier-cabezas/cuda4cpu@v0.4.0
      - run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
      - run: cmake --build build
      - run: ctest --test-dir build --output-on-failure
      # Again with the threads of each block in another order, which exposes
      # missing __syncthreads()
      - run: CUDA4CPU_SCHEDULE=reverse ctest --test-dir build --output-on-failure
```

Its outputs give the `toolkit` directory, for Makefiles written for nvcc
(`make CUDA_PATH=...`), and the `toolchain-file` for CMake's CUDA and HIP
languages. A project that fetches cuda4cpu with `FetchContent` needs no
action.

## What's supported

- **Kernel language:** `__global__`, `__device__`, `__shared__` (static and
  dynamic), `__constant__`, `__syncthreads()`, every warp shuffle, vote and
  match, `__activemask()` and `__syncwarp()`, atomics, `cuda_fp16.h`,
  `cuda_bf16.h`, `cuda_fp8.h`, `cuComplex.h`, math and integer intrinsics,
  rounding-mode intrinsics, `clock64()`.
- **Cooperative groups:** thread blocks, tiles of up to 1024 threads,
  coalesced groups, partitions, `reduce` and `scan`, and `grid.sync()` in
  cooperative launches.
- **Runtime API:** devices, memory (including pitched, 3D, managed, pinned
  and stream-ordered allocation), symbols, streams, events, occupancy, and
  CUDA graphs: explicit graphs with every node type but external
  semaphores, conditional nodes, stream capture and `cudaGraphExecUpdate`.
- **HIP**, as on HIP's NVIDIA platform: the runtime API under HIP's names,
  `hipLaunchKernelGGL`, `HIP_DYNAMIC_SHARED`, the mask-less warp functions
  and HIP's own additions, such as pointer attributes, virtual memory
  management and contexts.

The full lists: [CUDA features](docs/features.md), [HIP](docs/hip.md),
[limitations](docs/limitations.md). The [compatibility
suite](docs/compatibility.md) shows what real programs still miss: mostly
textures and dynamic parallelism.

## How it works

1. **A small rewriter** turns the two pieces of CUDA syntax that C++ can't
   express, `kernel<<<grid, block>>>(args)` and `extern __shared__` arrays,
   into library calls. Everything else is C++ with macros: `__global__`,
   `threadIdx` and the rest are defined by the headers. Line numbers stay the
   same, so diagnostics point at your file.
2. **Thread blocks run in parallel** on the CPU's cores, through OpenMP.
3. **The CUDA threads of a block run one after another.** A kernel without
   barriers is a plain loop over the threads, with the kernel inlined into
   it: a vector add runs as fast as the same loop written with OpenMP.
4. **At a barrier or a warp operation, each CUDA thread switches** to the
   next one: CUDA threads are fibers, with their own stacks and a context
   switch of a dozen instructions. A thread spends about 6.5 ns at a
   `__syncthreads()` on one core, including scheduling.
5. **In loops, the threads of a block take turns** every few iterations, so
   that they touch neighboring addresses together as on a GPU. Each launch
   site measures how often to switch. A grid-stride copy ran 5.7 times
   faster.

[docs/how-it-works.md](docs/how-it-works.md) explains each piece, and
[docs/usage.md](docs/usage.md#porting-a-cuda-program) shows what the rewriter
does.

## How it compares

Other projects run CUDA or HIP code on CPUs:

- [HIP-CPU](https://github.com/ROCm-Developer-Tools/HIP-CPU), AMD's
  header-only library, runs HIP code on CPUs.
- [chipStar](https://github.com/CHIP-SPV/chipStar) compiles CUDA and HIP to
  SPIR-V with its LLVM-based toolchain, and runs it on OpenCL or Level Zero
  devices, including CPUs through an OpenCL CPU runtime such as PoCL.
- [CuPBoP](https://arxiv.org/abs/2206.07896), a research framework, compiles
  CUDA to LLVM IR with Clang and translates it for CPUs.

cuda4cpu needs no special toolchain: the kernels are ordinary C++ functions,
compiled by your GCC or Clang. So every CPU tool works on them as it does on
the rest of your program. The price is that barriers cost a fiber switch per
CUDA thread, which compilers that restructure kernels around barriers avoid:
kernels that synchronize every few instructions are where cuda4cpu is
slowest.

## FAQ

**Is it an emulator?** No. Kernels compile to native code for your CPU, with
your compiler's optimizations, and run on all cores. Nothing interprets PTX.

**Can I keep my build system?** Usually. CMake projects written for CUDA
(`LANGUAGES CUDA`, `find_package(CUDAToolkit)`) build with
`-DCMAKE_TOOLCHAIN_FILE=<prefix>/lib/cuda4cpu/toolkit/cuda4cpu.cmake` (CMake
4.1 or later), and so
do NVIDIA's samples with their own `CMakeLists.txt`. `cuda4cpu-nvcc` takes
nvcc's options, and `<prefix>/lib/cuda4cpu/toolkit` is laid out like a CUDA
toolkit, so `make CUDA_PATH=<prefix>/lib/cuda4cpu/toolkit` builds Makefiles
that call `$(CUDA_PATH)/bin/nvcc`. See [Using cuda4cpu in your
project](docs/usage.md#using-cuda4cpu-in-your-project).

**Does it need CUDA installed?** No. cuda4cpu provides its own
`cuda_runtime.h`, `cooperative_groups.h`, `cuda_fp16.h` and the rest.

**Do I need to change my code?** Usually not. Programs that use textures,
dynamic parallelism, inline PTX, the driver API or CUDA's libraries (cuBLAS,
Thrust, CUB, ...) don't build yet. See [limitations](docs/limitations.md).

**Will my kernels be fast?** Faster than you might expect, but it's a CPU:
expect the performance of a well-parallelized CPU program, not of a GPU.
Kernels without barriers fare best.

**macOS or Windows?** On Windows, it works in WSL 2. Otherwise, cuda4cpu
needs Linux with glibc on a 64-bit CPU, and is tested on x86-64.

## Roadmap and contributing

[ROADMAP.md](ROADMAP.md) lists what's planned: textures, dynamic parallelism,
the remaining API gaps, and faster barriers. Bug reports with a program that
fails are the most useful contribution. See [CONTRIBUTING.md](CONTRIBUTING.md)
for how to build, test and send changes.

If cuda4cpu is useful to you, a star helps others find it.

cuda4cpu is licensed under the [Apache License, Version 2.0](LICENSE).
