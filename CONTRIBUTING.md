# Contributing to cuda4cpu

Thanks for your interest! Bug reports, programs that don't compile or run,
and pull requests are all welcome.

## Reporting a problem

The most useful report is a CUDA or HIP program that fails: a compile error,
a crash, or wrong results. Please include:

- the smallest program you can make that shows it, or a link to the project
  and the file
- the command you ran (`cuda4cpu-c++ ...` or your CMake setup)
- your compiler and its version (`g++ --version`), and your Linux
  distribution

If the program uses a CUDA feature that cuda4cpu doesn't implement yet,
[ROADMAP.md](ROADMAP.md) may already list it: a report still helps us decide
what comes first.

## Building and testing

```sh
cmake --workflow --preset debug     # configure, build and run the tests
cmake --workflow --preset release
ctest --preset debug -R stencil2d   # one test
```

You need GCC 14+ or Clang 18+, CMake 3.25+, Ninja and Python 3. See
[docs/usage.md](docs/usage.md) for the build options.

Changes to the scheduler (`lib/cuda4cpu.cpp`) should also pass with the
threads of each block in other orders, which exposes code that depends on the
default one:

```sh
CUDA4CPU_SCHEDULE=reverse ctest --preset release -E '^schedule_'
CUDA4CPU_SCHEDULE=random:1 ctest --preset release -E '^schedule_'
```

### Real programs

`compat/run.py` builds and runs a curated set of NVIDIA's cuda-samples,
Rodinia and AMD's rocm-examples, unmodified (it needs git and network
access):

```sh
compat/run.py --driver build/release/tools/cuda4cpu-c++ --work /tmp/compat --filter 'rodinia/*'
```

CI fails if a program that passes in `compat/baseline.json` stops passing.
If your change makes more programs pass, update the baseline in the same pull
request with `--update-baseline`.

### Performance

Back performance claims with `build/release/benchmarks/microbench`, or with
timings of the compatibility programs, and compare medians of several runs:
run-to-run noise is large.

## Pull requests

- Add a test in `tests/` for each new CUDA feature (registered in
  `tests/CMakeLists.txt`), and check results against a host reference.
- C++23, four-space indentation, snake_case for the library's own names, and
  CUDA's names exactly as in the CUDA Runtime API.
- Every source file starts with the Apache-2.0 header and
  `SPDX-License-Identifier: Apache-2.0`.
- [CLAUDE.md](CLAUDE.md) describes the code's layout and how execution works
  in detail, with the pitfalls we've hit. It's written for AI assistants, and
  it's the best map of the code for people too.

By contributing, you agree that your contributions are licensed under the
[Apache License, Version 2.0](LICENSE).
