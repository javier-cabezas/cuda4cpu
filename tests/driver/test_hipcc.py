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
HIPCC_VERBOSE, and a HIP program that the host compiler links from the
toolkit's lib (-lamdhip64).

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


if __name__ == '__main__':
    TOOLKIT, PROGRAM, CXX = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3]
    unittest.main(argv=sys.argv[:1], verbosity=1)
