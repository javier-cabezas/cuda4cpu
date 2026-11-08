#!/usr/bin/env python3
#
# CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
#
# Copyright (C) 2014 Javier Cabezas
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The toolkit as a HIP installation: hipconfig's answers, hipcc's
--version, HIPCC_COMPILE_FLAGS_APPEND, HIPCC_LINK_FLAGS_APPEND and
HIPCC_VERBOSE, a HIP program that the host compiler links from the
toolkit's lib (-lamdhip64), and HIP's AMD platform, which
--cuda4cpu-platform=amd or CUDA4CPU_HIP_PLATFORM=amd choose.

Usage: test_hipcc.py <toolkit> <a .hip program that returns 0> <C++ compiler>
"""

import os
import subprocess
import sys
import tempfile
import unittest

TOOLKIT = None
PROGRAM = None
CXX = None


def tool(name, *args, env=None, cwd=None):
    environment = {k: v for k, v in os.environ.items()
                   if k not in ('CUDA4CPU_CXX', 'CXX', 'HIP_PATH', 'ROCM_PATH') and not k.startswith('HIPCC_')}
    environment.update(env or {})
    return subprocess.run([os.path.join(TOOLKIT, 'bin', name)] + list(args), capture_output=True, text=True,
                          env=environment, cwd=cwd)


class Hipconfig(unittest.TestCase):

    def test_values(self):
        # Without a newline, unless -n is given, as hipconfig prints them
        self.assertEqual(tool('hipconfig', '--platform').stdout, 'nvidia')
        self.assertEqual(tool('hipconfig', '-P', '-n').stdout, 'nvidia\n')
        self.assertEqual(tool('hipconfig', '--compiler').stdout, 'nvcc')
        self.assertEqual(tool('hipconfig', '--runtime').stdout, 'cuda')
        self.assertEqual(tool('hipconfig', '--path').stdout, TOOLKIT)
        self.assertEqual(tool('hipconfig', '--rocmpath').stdout, TOOLKIT)
        self.assertEqual(tool('hipconfig', '--path', env={'HIP_PATH': '/opt/hip'}).stdout, '/opt/hip')
        self.assertRegex(tool('hipconfig', '--version').stdout, r'^7\.2\.0-')
        self.assertIn(f'-I{os.path.join(TOOLKIT, "include")}', tool('hipconfig', '--cpp_config').stdout)
        self.assertIn('HIP_PLATFORM       : nvidia', tool('hipconfig').stdout)
        self.assertNotEqual(tool('hipconfig', '--bogus').returncode, 0)

    def test_installation(self):
        self.assertTrue(os.path.isfile(os.path.join(TOOLKIT, 'include', 'hip', 'hip_runtime.h')))
        self.assertTrue(os.path.isfile(os.path.join(TOOLKIT, 'lib', 'cmake', 'hip', 'hip-config.cmake')))


class Hipcc(unittest.TestCase):

    def test_version(self):
        result = tool('hipcc', '--version')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith('HIP version: 7.2.0-'), result.stdout)

    def test_environment_flags_and_verbose(self):
        with tempfile.TemporaryDirectory() as work:
            result = tool('hipcc', '-o', 'program', PROGRAM, cwd=work,
                          env={'HIPCC_VERBOSE': '1', 'HIPCC_COMPILE_FLAGS_APPEND': '-DAPPENDED=1 -O1',
                               'HIPCC_LINK_FLAGS_APPEND': '-Wl,--as-needed'})
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [line for line in result.stderr.splitlines() if line.startswith('hipcc-cmd: ')]
            self.assertTrue(commands, result.stderr)
            self.assertIn('-DAPPENDED=1', commands[-1])
            self.assertIn('-Wl,--as-needed', commands[-1])
            # Link flags only when linking
            result = tool('hipcc', '-c', '-o', 'program.o', PROGRAM, cwd=work,
                          env={'HIPCC_VERBOSE': '1', 'HIPCC_LINK_FLAGS_APPEND': '-Wl,--as-needed'})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('--as-needed', result.stderr)

    def test_emit_static_lib(self):
        """--emit-static-lib compiles the sources into a static library"""
        with tempfile.TemporaryDirectory() as work:
            result = tool('hipcc', '--emit-static-lib', '-fPIC', '--offload-arch=gfx90a', '-o', 'libprogram.a',
                          PROGRAM, cwd=work)
            self.assertEqual(result.returncode, 0, result.stderr)
            members = subprocess.run(['ar', 't', os.path.join(work, 'libprogram.a')], capture_output=True, text=True)
            self.assertEqual(members.returncode, 0, members.stderr)
            self.assertEqual(len(members.stdout.split()), 1, members.stdout)
            result = tool('hipcc', '-o', 'program', 'libprogram.a', cwd=work)
            self.assertEqual(result.returncode, 0, result.stderr)
            run = subprocess.run([os.path.join(work, 'program')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_link_from_rocm_lib(self):
        """A program the host compiler links as ROCm's: -L$ROCM_PATH/lib -lamdhip64"""
        with tempfile.TemporaryDirectory() as work:
            result = tool('hipcc', '-c', '--offload-arch=gfx90a', '-o', 'program.o', PROGRAM, cwd=work)
            self.assertEqual(result.returncode, 0, result.stderr)
            link = subprocess.run([CXX, '-o', 'program', 'program.o', '-L' + os.path.join(TOOLKIT, 'lib'),
                                   '-lamdhip64'], capture_output=True, text=True, cwd=work)
            self.assertEqual(link.returncode, 0, link.stderr)
            run = subprocess.run([os.path.join(work, 'program')], capture_output=True, text=True,
                                 env=dict(os.environ, LD_LIBRARY_PATH=os.path.join(TOOLKIT, 'lib')))
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


# A program that prints its platform, its GPU and a builtin's lane IDs
AMD_PROGRAM = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void lanes(unsigned *out)
{
#if defined(__HIP_PLATFORM_AMD__)
    out[threadIdx.x] = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
#else
    out[threadIdx.x] = __lane_id();
#endif
}
int main()
{
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    unsigned out[64];
    lanes<<<1, 64>>>(out);
    unsigned sum = 0;
    for (unsigned v : out)
        sum += v;
#if defined(__HIP_PLATFORM_AMD__) && defined(__AMDGCN__) && __AMDGCN_WAVEFRONT_SIZE__ == 32 && defined(__GFX12__)
    std::printf("amd %s %d.%d %u rdna4\n", prop.gcnArchName, prop.major, prop.minor, sum);
#elif defined(__HIP_PLATFORM_AMD__) && defined(__AMDGCN__) && __AMDGCN_WAVEFRONT_SIZE__ == 32 && defined(__GFX11__)
    std::printf("amd %s %d.%d %u\n", prop.gcnArchName, prop.major, prop.minor, sum);
#elif defined(__HIP_PLATFORM_NVIDIA__)
    std::printf("nvidia %u\n", sum);
#endif
}
"""


class AmdPlatform(unittest.TestCase):

    def build_and_run(self, *options, env=None):
        with tempfile.TemporaryDirectory() as work:
            with open(os.path.join(work, 'amd.hip'), 'w') as f:
                f.write(AMD_PROGRAM)
            result = tool('hipcc', *options, '-o', 'amd', 'amd.hip', cwd=work, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            run = subprocess.run([os.path.join(work, 'amd')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            return run.stdout.strip()

    def test_switch_and_target(self):
        # Two warps of lanes 0 to 31: 2 * 496
        self.assertEqual(self.build_and_run('--cuda4cpu-platform=amd', '--offload-arch=gfx1102:xnack-'),
                         'amd gfx1102 11.0 992')
        self.assertEqual(self.build_and_run(env={'CUDA4CPU_HIP_PLATFORM': 'amd'}), 'amd gfx1100 11.0 992')
        self.assertEqual(self.build_and_run('--offload-arch', 'gfx1151', env={'CUDA4CPU_HIP_PLATFORM': 'amd'}),
                         'amd gfx1151 11.5 992')
        # RDNA4
        self.assertEqual(self.build_and_run('--cuda4cpu-platform=amd', '--offload-arch=gfx1201'),
                         'amd gfx1201 12.0 992 rdna4')
        self.assertEqual(self.build_and_run('--cuda4cpu-platform=amd', '--offload-arch=gfx1200'),
                         'amd gfx1200 12.0 992 rdna4')

    def test_cuda_sources_keep_cuda(self):
        """The AMD platform's macros are for HIP sources: a CUDA file that
        cuda4cpu-c++ compiles with them doesn't get them"""
        with tempfile.TemporaryDirectory() as work:
            with open(os.path.join(work, 'plain.cu'), 'w') as f:
                f.write('#ifdef __HIP_PLATFORM_AMD__\n#error AMD macros in a CUDA file\n#endif\nint main() {}\n')
            result = tool('cuda4cpu-c++', '-c', 'plain.cu', cwd=work, env={'CUDA4CPU_HIP_PLATFORM': 'amd'})
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_program_defining_amd_stays_nvidia(self):
        """A program that defines __HIP_PLATFORM_AMD__ itself compiles on the
        NVIDIA platform, as before"""
        with tempfile.TemporaryDirectory() as work:
            with open(os.path.join(work, 'self.hip'), 'w') as f:
                f.write('#define __HIP_PLATFORM_AMD__ 1\n#include <hip/hip_runtime.h>\n'
                        '#ifndef __HIP_PLATFORM_NVIDIA__\n#error not NVIDIA\n#endif\nint main() {}\n')
            result = tool('hipcc', '-c', 'self.hip', cwd=work)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_nvidia_by_default(self):
        # --offload-arch alone doesn't choose AMD's platform
        self.assertEqual(self.build_and_run('--offload-arch=gfx1100'), 'nvidia 992')
        self.assertEqual(self.build_and_run('--cuda4cpu-platform=nvidia', env={'CUDA4CPU_HIP_PLATFORM': 'amd'}),
                         'nvidia 992')

    def test_launch_in_a_library_header(self):
        """A launch in a header that -I finds, as rocPRIM's, is rewritten"""
        with tempfile.TemporaryDirectory() as work:
            library = os.path.join(work, 'include', 'lib')
            os.makedirs(library)
            with open(os.path.join(library, 'fill.hpp'), 'w') as f:
                f.write('#pragma once\n__global__ void fill(int *out) { out[threadIdx.x] = 7; }\n'
                        '#define FILL(out) fill<<<1, 4>>>(out)\n')
            with open(os.path.join(work, 'main.hip'), 'w') as f:
                f.write('#include <hip/hip_runtime.h>\n#include <lib/fill.hpp>\n#include <cstdio>\n'
                        'int main() { int out[4] = {}; FILL(out); std::printf("%d\\n", out[0] + out[3]); }\n')
            result = tool('hipcc', '-I', os.path.join(work, 'include'), '-o', 'main', 'main.hip', cwd=work)
            self.assertEqual(result.returncode, 0, result.stderr)
            run = subprocess.run([os.path.join(work, 'main')], capture_output=True, text=True)
            self.assertEqual(run.stdout.strip(), '14', run.stderr)

    def test_errors(self):
        cases = [(['--cuda4cpu-platform=amd', '--offload-arch=gfx90a'], 'emulates RDNA3 and RDNA4 GPUs'),
                 (['--cuda4cpu-platform=amd', '--offload-arch=gfx1100', '--offload-arch=gfx1101'], 'one AMD GPU'),
                 (['--cuda4cpu-platform=amd', '-mwavefrontsize64'], '32-lane wavefronts'),
                 (['--cuda4cpu-platform=amd', '--offload-arch'], 'needs a GPU'),
                 (['--cuda4cpu-platform=intel'], 'unknown HIP platform')]
        for options, message in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as work:
                # The options last, so that a trailing --offload-arch has no value
                result = tool('hipcc', '-c', PROGRAM, *options, cwd=work)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == '__main__':
    TOOLKIT, PROGRAM, CXX = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3]
    unittest.main(argv=sys.argv[:1], verbosity=1)
