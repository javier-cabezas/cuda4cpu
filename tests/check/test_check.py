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

"""cuda4cpu-check, check's command line: on sources and on a program built
already, the findings at their lines, the exit codes, the JSON and GitHub
formats, and its errors.

Usage: test_check.py <cuda4cpu-check> <examples/bugs> <tests/samples>
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

CHECK = BUGS = SAMPLES = None


def check(*arguments, cwd=None, env=None):
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('CUDA4CPU_', 'GITHUB_'))}
    if 'CUDA4CPU_CXX' in os.environ:
        environment['CUDA4CPU_CXX'] = os.environ['CUDA4CPU_CXX']
    environment.update(env or {})
    return subprocess.run([CHECK, *arguments], capture_output=True, text=True, cwd=cwd, env=environment,
                          timeout=600)


class Sources(unittest.TestCase):

    def test_missing_barrier(self):
        result = check(os.path.join(BUGS, 'missing_barrier.cu'))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('cuda4cpu-check: problems found (11 runs', result.stdout)
        self.assertIn('[depends on the thread order]', result.stdout)

    def test_out_of_bounds_at_its_line(self):
        result = check(os.path.join(BUGS, 'out_of_bounds.cu'))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('[AddressSanitizer: allocation-overflow]', result.stdout)
        self.assertIn('out_of_bounds.cu:27', result.stdout)
        self.assertIn('the other orders were not run', result.stdout)

    def test_correct_program(self):
        result = check(os.path.join(SAMPLES, 'vector_add.cu'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('cuda4cpu-check: no problems found', result.stdout)

    def test_arguments_after_dashes(self):
        with tempfile.TemporaryDirectory() as work:
            source = os.path.join(work, 'arguments.cu')
            with open(source, 'w') as f:
                f.write('#include <cstdio>\n#include <cstring>\n'
                        'int main(int argc, char **argv) { return argc == 3 && !strcmp(argv[2], "-v") ? 0 : 7; }\n')
            result = check('--no-sanitizers', '--seeds', '5', source, '--', 'x', '-v')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('(6 runs', result.stdout)
            result = check('--no-sanitizers', '--seeds', '5', source)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn('[fails in the default order]', result.stdout)

    def test_failed_assert(self):
        with tempfile.TemporaryDirectory() as work:
            with open(os.path.join(work, 'asserts.cu'), 'w') as f:
                f.write('#include <cassert>\n'
                        '__global__ void k(int n) { assert(int(threadIdx.x) < n); }\n'
                        'int main() { k<<<1, 8>>>(6); return 0; }\n')
            result = check('--no-sanitizers', 'asserts.cu', cwd=work)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('[failed assert in a kernel] assert(int(threadIdx.x) < n) failed in void k(int) at '
                          'asserts.cu:2', result.stdout)

    def test_compiler_options_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as work:
            with open(os.path.join(work, 'size.cu'), 'w') as f:
                f.write('static_assert(N == 64);\nint main() { return 0; }\n')
            result = check('size.cu', cwd=work)
            self.assertEqual(result.returncode, 2)
            self.assertIn('does not compile', result.stdout)
            self.assertIn('size.cu:1:', result.stdout)          # relative to where it runs
            self.assertTrue(result.stderr)                      # the compiler's output
            result = check('-X', '-DN=64 -O1', '--no-sanitizers', 'size.cu', cwd=work)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class Prebuilt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp()
        nvcc = os.path.join(os.path.dirname(CHECK), 'cuda4cpu-nvcc')
        cls.binary = os.path.join(cls.work, 'missing_barrier')
        subprocess.run([nvcc, '-O2', '-g', os.path.join(BUGS, 'missing_barrier.cu'), '-o', cls.binary],
                       check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work)

    def test_program_built_already(self):
        result = check('-v', self.binary)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('[depends on the thread order]', result.stdout)
        self.assertIn('(10 runs', result.stdout)                # no sanitizer build
        self.assertIn('run reverse: exit code 1', result.stdout)
        self.assertIn('built without AddressSanitizer and UBSan', result.stdout)
        self.assertNotIn('warning', result.stderr)

    def test_json(self):
        result = check('--format', 'json', self.binary)
        self.assertEqual(result.returncode, 1)
        content = json.loads(result.stdout)
        self.assertEqual(content['verdict'], 'problems found')
        self.assertEqual(content['findings'][0]['kind'], 'depends on the thread order')

    def test_github(self):
        with tempfile.TemporaryDirectory() as work:
            summary = os.path.join(work, 'summary.md')
            result = check('--format', 'github', self.binary, env={'GITHUB_WORKSPACE': work,
                                                                   'GITHUB_STEP_SUMMARY': summary})
            self.assertEqual(result.returncode, 1)
            self.assertIn('::error title=cuda4cpu-check%3A depends on the thread order::', result.stdout)
            with open(summary) as f:
                self.assertIn('### cuda4cpu-check: problems found', f.read())

    def test_built_with_the_sanitizers(self):
        """A program built with them is the sanitizer run: their errors stop
        check, at the line, and their environment keeps leaks quiet"""
        nvcc = os.path.join(os.path.dirname(CHECK), 'cuda4cpu-nvcc')
        sanitize = ['-g', '-fsanitize=address,undefined']
        bad = os.path.join(self.work, 'out_of_bounds')
        subprocess.run([nvcc, *sanitize, os.path.join(BUGS, 'out_of_bounds.cu'), '-o', bad], check=True)
        result = check(bad)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('[AddressSanitizer: allocation-overflow]', result.stdout)
        self.assertIn('out_of_bounds.cu:27', result.stdout)
        self.assertIn('(1 run,', result.stdout)
        self.assertNotIn('fails in the default order', result.stdout)
        good = os.path.join(self.work, 'vector_add')
        subprocess.run([nvcc, *sanitize, os.path.join(SAMPLES, 'vector_add.cu'), '-o', good], check=True)
        result = check(good)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('(11 runs', result.stdout)
        self.assertNotIn('built without AddressSanitizer', result.stdout)

    def test_not_cuda4cpu(self):
        result = check(shutil.which('true'))
        self.assertEqual(result.returncode, 0)
        self.assertIn('does not seem to be built with cuda4cpu', result.stderr)


class Annotations(unittest.TestCase):

    def test_at_the_line_in_the_workspace(self):
        with tempfile.TemporaryDirectory() as work:
            shutil.copy(os.path.join(BUGS, 'out_of_bounds.cu'), work)
            result = check('--format', 'github', 'out_of_bounds.cu', cwd=work, env={'GITHUB_WORKSPACE': work})
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('::error file=out_of_bounds.cu,line=27,title=cuda4cpu-check%3A AddressSanitizer%3A '
                          'allocation-overflow::', result.stdout)


    def test_with_another_working_directory(self):
        """Locations stay relative to where check runs, not to -C"""
        with tempfile.TemporaryDirectory() as work:
            os.makedirs(os.path.join(work, 'src'))
            os.makedirs(os.path.join(work, 'run'))
            shutil.copy(os.path.join(BUGS, 'out_of_bounds.cu'), os.path.join(work, 'src'))
            result = check('--format', 'github', '-C', 'run', 'src/out_of_bounds.cu', cwd=work,
                           env={'GITHUB_WORKSPACE': work})
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('::error file=src/out_of_bounds.cu,line=27,', result.stdout)

    def test_compile_errors_in_the_summary(self):
        with tempfile.TemporaryDirectory() as work:
            with open(os.path.join(work, 'bad.cu'), 'w') as f:
                f.write('int main() { return undeclared | 1; }\n')
            summary = os.path.join(work, 'summary.md')
            result = check('--format', 'github', 'bad.cu', cwd=work,
                           env={'GITHUB_WORKSPACE': work, 'GITHUB_STEP_SUMMARY': summary})
            self.assertEqual(result.returncode, 2)
            self.assertIn('::error file=bad.cu,line=1,', result.stdout)
            with open(summary) as f:
                text = f.read()
            self.assertIn('| compiler error | bad.cu:1 |', text)


class Errors(unittest.TestCase):

    def test_usage(self):
        for arguments in ([], ['--seeds', 'x', 'a.cu'], ['-e', 'NOVALUE', 'a.cu'], ['-X', '-O2', '/bin/true'],
                          ['-X', '', '/bin/true'], ['-X', '-I "unclosed', 'a.cu'], ['a.cu', 'notes.txt'],
                          ['--timeout', 'nan', '/bin/true']):
            result = check(*arguments)
            self.assertEqual(result.returncode, 2, arguments)
            self.assertIn('error:', result.stderr, arguments)

    def test_missing(self):
        for name in ('missing.cu', './missing'):
            result = check(name)
            self.assertEqual(result.returncode, 2, name)
            self.assertIn('cuda4cpu-check: error:', result.stderr)

    def test_version(self):
        result = check('--version')
        self.assertEqual(result.returncode, 0)
        self.assertRegex(result.stdout, r'^cuda4cpu-check \d+\.\d+\.\d+')


if __name__ == '__main__':
    CHECK, BUGS, SAMPLES = (os.path.abspath(a) for a in sys.argv[1:4])
    del sys.argv[1:4]
    unittest.main()
