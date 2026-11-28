# Third-party notices

cuda4cpu is licensed under the Apache License 2.0 (`LICENSE`). It includes,
or derives from, the following works.

## Joe and Kuo's direction numbers for Sobol' sequences

`lib/sobol_directions.inc`, which `lib/make-sobol-directions.py` packs from
`new-joe-kuo-6.21201` (https://web.maths.unsw.edu.au/~fkuo/sobol/), holds
the direction numbers of cuRAND's Sobol' generators. libcuda4cpu contains
them.

```
Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the names of the copyright holders nor the names of the
      University of New South Wales and the University of Waikato
      and its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## pocketfft

`lib/third_party/pocketfft/pocketfft_hdronly.h`, Martin Reinecke's pocketfft
(https://github.com/mreineck/pocketfft, the C++ header-only version, commit
4d865cba24359b05b37da4e75aee24498c3b78e7), with one change (its threads
are OpenMP's, `lib/third_party/pocketfft/README.md`), computes cuFFT's,
hipFFT's and rocFFT's transforms. libcuda4cpu contains it. Its notice, as
the header gives it:

```
This file is part of pocketfft.

Copyright (C) 2010-2024 Max-Planck-Society
Copyright (C) 2019-2020 Peter Bell

For the odd-sized DCT-IV transforms:
  Copyright (C) 2003, 2007-14 Matteo Frigo
  Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology

For the prev_good_size search:
  Copyright (C) 2024 Tan Ping Liang, Peter Bell

For the safeguards against integer overflow in good_size search:
  Copyright (C) 2024 Cris Luengo

Authors: Martin Reinecke, Peter Bell

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Algorithms

cuRAND's generators are written for cuda4cpu from published descriptions:

- **XORWOW** (Marsaglia), with cuRAND's seeding constants and subsequence
  spacing as public reimplementations give them (mojolearn, Apache-2.0).
- **Philox4x32-10**, from Salmon, Moraes, Dror and Shaw, "Parallel random
  numbers: as easy as 1, 2, 3" (SC'11), checked against Random123's known
  answers.
- **MRG32k3a** (L'Ecuyer), seeded as rocRAND (MIT) seeds it.
- **AS 241** (Wichura) for the inverse normal distribution, Knuth's method
  and Hormann's PTRS for Poisson values.

They copy no code from those projects.
