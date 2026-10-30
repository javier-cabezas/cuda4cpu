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

"""cuda4cpu-nvcc's command line: the host compiler's command that nvcc's
options become (from --dryrun), --version, a build with nvcc's steps (-dc,
-dlink, -lib, then a link) that runs, and the toolkit directory: its
bin/nvcc, and a program linked by the host compiler with -lcudart.

Usage: test_nvcc.py <cuda4cpu-nvcc> <a .cu program that returns 0> <toolkit> <C++ compiler>
"""

import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest

NVCC = None
PROGRAM = None
TOOLKIT = None
CXX = None


def nvcc(*args, env=None, cwd=None):
    environment = dict(os.environ)
    for name in ('CUDA4CPU_CXX', 'CXX', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS'):
        environment.pop(name, None)
    environment.update(env or {})
    return subprocess.run([NVCC] + list(args), capture_output=True, text=True, env=environment, cwd=cwd)


def commands(*args, env=None):
    """The commands --dryrun shows, each as a list of arguments (without the
    lines that show nvcc's configuration, NAME=value)"""
    result = nvcc('--dryrun', *args, env=env)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return [shlex.split(line[3:]) for line in result.stderr.splitlines()
            if line.startswith('#$ ') and not re.match(r'#\$ [^= ]+=', line)]


def compile_command(*args, env=None):
    """The host compiler's command, the last one"""
    return commands(*args, env=env)[-1]


class Translation(unittest.TestCase):

    def test_gpu_options_are_dropped(self):
        command = compile_command('-arch=sm_80', '-arch', 'sm_70', '-gencode', 'arch=compute_80,code=sm_80',
                                  '--generate-code=arch=compute_90,code=[compute_90,sm_90]', '-code=sm_80',
                                  '--expt-relaxed-constexpr', '--extended-lambda', '-lineinfo', '-use_fast_math',
                                  '-Xptxas', '-v', '--ptxas-options=-O3', '-maxrregcount', '32', '-rdc=true',
                                  '--default-stream', 'per-thread', '-Wno-deprecated-gpu-targets', '--keep',
                                  '--forward-unknown-to-host-compiler', '-c', PROGRAM)
        for dropped in ('sm_80', 'sm_70', 'arch=compute_80,code=sm_80', '-v', '-O3', '32', 'per-thread', 'true'):
            self.assertNotIn(dropped, command)
        self.assertFalse([a for a in command if 'expt' in a or 'arch' in a or 'gencode' in a or 'lineinfo' in a])

    def test_host_options(self):
        command = compile_command('-Xcompiler', '-Wall,-Wextra', '--compiler-options', '-O2 -fno-strict-aliasing',
                                  '-Xcompiler=-fPIC', '-Xlinker', '--as-needed,-z,now', '--linker-options=-s',
                                  '--include-path', '/inc', '--define-macro=FOO=1', '--undefine-macro', 'BAR',
                                  '--pre-include', 'pre.h', '--system-include', '/sys', '--optimize', '2',
                                  '-dc', '-G', '--generate-dependencies-with-compile', '--dependency-output', 'x.d',
                                  '--output-file', 'out.o', PROGRAM)
        for expected in ('-Wall', '-Wextra', '-O2', '-fno-strict-aliasing', '-fPIC', '-Wl,--as-needed',
                         '-Wl,-z', '-Wl,now', '-Wl,-s', '-O2', '-c', '-g', '-MD'):
            self.assertIn(expected, command)
        text = ' '.join(command)
        for expected in ('-I /inc', '-D FOO=1', '-U BAR', '-include pre.h', '-isystem /sys', '-MF x.d', '-o out.o'):
            self.assertIn(expected, text)

    def test_nvcc_macros(self):
        command = compile_command('-c', PROGRAM)
        for expected in ('-D__NVCC__', '-D__CUDACC_VER_MAJOR__=12', '-D__CUDACC_VER_MINOR__=0',
                         '-D__CUDACC_VER_BUILD__=140', '-std=c++23', '-include'):
            self.assertIn(expected, command)

    def test_older_standards_are_raised(self):
        for given, expected in (('-std=c++17', '-std=c++23'), ('--std=c++14', '-std=c++23'),
                                ('-std=gnu++20', '-std=gnu++23'), ('-std=c++26', '-std=c++26')):
            command = compile_command(*given.split('='), '-c', PROGRAM) if given.startswith('--') else \
                compile_command(given, '-c', PROGRAM)
            self.assertIn(expected, command, given)
            self.assertEqual(len([a for a in command if a.startswith('-std=')]), 1, given)
        self.assertIn('-std=c++23', compile_command('-std', 'c++11', '-c', PROGRAM))

    def test_provided_libraries_are_dropped(self):
        command = compile_command('-lcudart', '-l', 'cudart_static', '--library', 'cuda', '-lcudadevrt',
                                  '-lnvToolsExt', '-lm', '-l', 'pthread', '-o', 'app', PROGRAM)
        self.assertNotIn('-lcudart', command)
        self.assertNotIn('cudart_static', command)
        self.assertNotIn('cuda', command)
        self.assertIn('-lm', command)
        self.assertIn('pthread', command)
        # Links go through cuda4cpu-c++, which adds libcuda4cpu, with the
        # toolkit's libcudart_static, as nvcc links its static runtime
        self.assertEqual(os.path.basename(command[0]), 'cuda4cpu-c++')
        self.assertIn('-L' + os.path.join(TOOLKIT, 'lib64'), command)
        self.assertIn('-lcudart_static', command)

    def test_verbose_shows_configuration(self):
        """What CMake reads in nvcc -v's output: the toolkit's root, include
        and library directories, the default architecture, and a link line
        with the library directory, whose first word links CUDA programs"""
        output = nvcc('--dryrun', '-o', 'app', PROGRAM).stderr
        lib64 = os.path.join(TOOLKIT, 'lib64')
        self.assertIn(f'#$ TOP={TOOLKIT}\n', output)
        self.assertIn(f'#$ NVVMIR_LIBRARY_DIR={os.path.join(TOOLKIT, "nvvm", "libdevice")}\n', output)
        self.assertIn(f'#$ INCLUDES="-I{os.path.join(TOOLKIT, "include")}"\n', output)
        self.assertIn(f'#$ LIBRARIES=  "-L{lib64}"\n', output)
        self.assertRegex(output, r'-arch compute_52')
        link = output.splitlines()[-1]
        self.assertIn(f'"-L{lib64}"', link)
        self.assertTrue(link.startswith('#$ ' + os.path.join(os.path.dirname(NVCC), 'cuda4cpu-c++') + ' '), link)
        self.assertTrue(os.path.isdir(os.path.join(TOOLKIT, 'nvvm', 'libdevice')))

    def test_options_file(self):
        with tempfile.TemporaryDirectory() as work:
            options = os.path.join(work, 'options.txt')
            with open(options, 'w') as f:
                f.write('-DFROM_FILE "-I/a dir" -arch=sm_80\n')
            command = compile_command('--options-file', options, '-c', PROGRAM)
            self.assertIn('-DFROM_FILE', command)
            self.assertIn('-I/a dir', command)
            self.assertFalse([a for a in command if 'sm_80' in a])

    def test_host_compiler(self):
        self.assertEqual(compile_command('-ccbin', 'gcc-14', '-c', PROGRAM)[0], 'g++-14')
        self.assertEqual(compile_command('--compiler-bindir=/opt/bin/clang', '-c', PROGRAM)[0], '/opt/bin/clang++')
        self.assertEqual(compile_command('-ccbin=g++', '-c', PROGRAM)[0], 'g++')
        # CUDA4CPU_CXX still wins
        self.assertEqual(compile_command('-ccbin', 'gcc', '-c', PROGRAM, env={'CUDA4CPU_CXX': 'my-c++'})[0], 'my-c++')

    def test_environment_flags(self):
        command = compile_command('-c', PROGRAM, env={'NVCC_PREPEND_FLAGS': '-DFIRST -arch=sm_80',
                                                      'NVCC_APPEND_FLAGS': "-Xcompiler -Wshadow -DLAST"})
        self.assertIn('-DFIRST', command)
        self.assertIn('-DLAST', command)
        self.assertIn('-Wshadow', command)
        self.assertFalse([a for a in command if 'sm_80' in a])

    def test_werror_kinds(self):
        self.assertNotIn('-Werror', compile_command('-Werror', 'cross-execution-space-call,reorder', '-c', PROGRAM))
        self.assertIn('-Werror', compile_command('-Werror=all-warnings', '-c', PROGRAM))
        self.assertIn('-Werror', compile_command('-Werror', '-c', PROGRAM))

    def test_c_sources_stay_c(self):
        command = compile_command('-c', 'helper.c')
        self.assertEqual(command[command.index('helper.c') - 2:command.index('helper.c') + 3],
                         ['-x', 'c', 'helper.c', '-x', 'none'])

    def test_steps(self):
        link = commands('-dlink', '-arch=sm_80', '-Xcompiler', '-fPIC', 'a.o', 'b.o', '-o', 'dlink.o')
        self.assertEqual(len(link), 1)
        self.assertIn('-c', link[0])
        self.assertIn(os.devnull, link[0])
        self.assertIn('-fPIC', link[0])
        self.assertEqual(link[0][link[0].index('-o') + 1], 'dlink.o')
        archive = commands('-lib', 'a.o', 'b.o', '-o', 'libx.a')
        self.assertEqual(archive, [[os.environ.get('AR', 'ar'), 'rcs', 'libx.a', 'a.o', 'b.o']])
        self.assertNotEqual(nvcc('-lib', PROGRAM, '-o', 'libx.a').returncode, 0)
        # -cuda preprocesses each .cu file into <file>.cpp.ii
        preprocess = commands('-cuda', PROGRAM)[-1]
        self.assertIn('-E', preprocess)
        self.assertEqual(preprocess[preprocess.index('-o') + 1], os.path.basename(PROGRAM) + '.cpp.ii')
        # GPU code can't be made
        result = nvcc('-ptx', PROGRAM)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('-ptx makes GPU code', result.stderr)

    def test_version(self):
        result = nvcc('--version')
        self.assertEqual(result.returncode, 0)
        # What CMake looks for to identify nvcc and its version
        self.assertRegex(result.stdout, r'nvcc: [^\n]+ Cuda compiler driver')
        self.assertRegex(result.stdout, r'release 12\.0, V12\.0\.140')
        self.assertEqual(nvcc('-V').stdout, result.stdout)


class Build(unittest.TestCase):

    def test_separate_compilation(self):
        with tempfile.TemporaryDirectory() as work:
            steps = [['-dc', '-arch=sm_80', '-Xcompiler', '-fPIC', '-std=c++17', '-o', 'program.o', PROGRAM],
                     ['-dlink', '-arch=sm_80', '-Xcompiler', '-fPIC', 'program.o', '-o', 'dlink.o'],
                     ['-lib', 'program.o', 'dlink.o', '-o', 'libprogram.a'],
                     ['-arch=sm_80', '-o', 'program', 'libprogram.a', '-lcudart']]
            for step in steps:
                result = nvcc(*step, cwd=work)
                self.assertEqual(result.returncode, 0, f'{step}: {result.stderr}')
            run = subprocess.run([os.path.join(work, 'program')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_toolkit(self):
        """bin/nvcc, and a program that the host compiler links with
        -lcudart, which finds libcuda4cpu at run time through lib64"""
        with tempfile.TemporaryDirectory() as work:
            toolkit_nvcc = os.path.join(TOOLKIT, 'bin', 'nvcc')
            environment = {name: value for name, value in os.environ.items() if name not in ('CUDA4CPU_CXX', 'CXX')}
            for command in ([toolkit_nvcc, '-c', '-arch=sm_80', '-o', 'program.o', PROGRAM],
                            [CXX, '-o', 'program', 'program.o', '-L' + os.path.join(TOOLKIT, 'lib64'), '-lcudart']):
                result = subprocess.run(command, capture_output=True, text=True, cwd=work, env=environment)
                self.assertEqual(result.returncode, 0, f'{command}: {result.stderr}')
            run = subprocess.run([os.path.join(work, 'program')], capture_output=True, text=True,
                                 env=dict(os.environ, LD_LIBRARY_PATH=os.path.join(TOOLKIT, 'lib64')))
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertTrue(os.path.isfile(os.path.join(TOOLKIT, 'include', 'cuda_runtime.h')))

    def test_dependencies(self):
        """Every way of asking for dependencies names the original source and
        its local header, not the rewritten copies, which are deleted"""
        def dependencies(text):
            text = text.replace('\\\n', ' ')
            words = re.findall(r'(?:\\[ #]|\$\$|[^\s\\])+', text)
            return [re.sub(r'\\([ #])|\$(\$)', lambda m: m.group(1) or m.group(2), w) for w in words]

        with tempfile.TemporaryDirectory() as top:
            work = os.path.join(top, 'with space')
            os.mkdir(work)
            source, header = os.path.join(work, 'program.cu'), os.path.join(work, 'kernel.cuh')
            with open(header, 'w') as f:
                f.write('__global__ void kernel(int *x) { x[threadIdx.x] = 1; }\n')
            with open(source, 'w') as f:
                f.write('#include "kernel.cuh"\nint main() { int *x; cudaMalloc(&x, 4); kernel<<<1, 1>>>(x); }\n')
            cases = [(['-MD', '-MF', 'deps.d', '-c', '-o', 'program.o'], 'deps.d'),
                     (['-MD', '-c', '-o', 'program.o'], 'program.d'),
                     (['-MMD', '-MP', '-c'], 'program.d'),
                     (['-M', '-o', 'deps.txt'], 'deps.txt'),
                     (['-M'], None)]
            for options, depfile in cases:
                with self.subTest(options=options):
                    result = nvcc(*options, source, cwd=work)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if depfile is None:
                        text = result.stdout
                    else:
                        with open(os.path.join(work, depfile)) as f:
                            text = f.read()
                    words = dependencies(text)
                    self.assertIn(source, words)
                    self.assertIn(header + (':' if '-MP' in options else ''), words)
                    for word in words:
                        self.assertNotIn('cuda4cpu-', word)
                        if not word.endswith(':'):
                            self.assertTrue(os.path.exists(word), word)

    def test_verbose_shows_commands(self):
        with tempfile.TemporaryDirectory() as work:
            result = nvcc('-v', '-c', PROGRAM, '-o', 'program.o', cwd=work)
            self.assertEqual(result.returncode, 0, result.stderr)
            shown = [line for line in result.stderr.splitlines() if line.startswith('#$ ')]
            self.assertTrue(any('cuda4cpu-rewrite' in line for line in shown))
            self.assertTrue(re.search(r'-c .*program.o|-o program.o', shown[-1]))
            self.assertTrue(os.path.exists(os.path.join(work, 'program.o')))


if __name__ == '__main__':
    NVCC, PROGRAM, TOOLKIT, CXX = (os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]),
                                   os.path.abspath(sys.argv[3]), sys.argv[4])
    unittest.main(argv=sys.argv[:1], verbosity=1)
