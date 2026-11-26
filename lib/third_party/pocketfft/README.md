# pocketfft

`pocketfft_hdronly.h` and `LICENSE.md` from Martin Reinecke's pocketfft,
the C++ header-only version (https://github.com/mreineck/pocketfft, branch
`cpp`, commit 4d865cba24359b05b37da4e75aee24498c3b78e7), unchanged, under
the BSD 3-clause license in `LICENSE.md`. `lib/fft.cpp` includes it: cuFFT,
hipFFT and rocFFT's transforms run on it. To update it, copy the two files
from a later commit and change the commit here and in
`THIRD_PARTY_NOTICES.md`.
