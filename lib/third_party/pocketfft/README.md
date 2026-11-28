# pocketfft

`pocketfft_hdronly.h` and `LICENSE.md` from Martin Reinecke's pocketfft,
the C++ header-only version (https://github.com/mreineck/pocketfft, branch
`cpp`, commit 4d865cba24359b05b37da4e75aee24498c3b78e7), under the BSD
3-clause license in `LICENSE.md`, with one change: `threading::thread_map`
runs its shares on the OpenMP threads when `POCKETFFT_OPENMP` is defined
(the block marked `cuda4cpu:`), which `lib/fft.cpp` does, instead of its
own thread pool, whose threads competed with the OpenMP threads that kernels
run on. `lib/fft.cpp` includes it: cuFFT, hipFFT and rocFFT's transforms run
on it. To update it, copy the two files from a later commit, apply the change
again, and change the commit here and in `THIRD_PARTY_NOTICES.md`.
