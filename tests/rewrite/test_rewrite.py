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

"""Unit tests of cuda4cpu-rewrite. Usage: test_rewrite.py <path to cuda4cpu-rewrite>"""

import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest

# Importing the tool would otherwise leave bytecode next to it, in tools/
sys.dont_write_bytecode = True

TOOL = sys.argv.pop(1) if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), '..', '..', 'tools', 'cuda4cpu-rewrite')
loader = importlib.machinery.SourceFileLoader('cuda4cpu_rewrite', TOOL)
spec = importlib.util.spec_from_loader('cuda4cpu_rewrite', loader)
rw = importlib.util.module_from_spec(spec)
loader.exec_module(rw)

LAMBDA = 'cuda4cpu::launch([&](const auto &...cuda4cpu_args) { '


def rewrite(text):
    out = rw.rewrite_text(text, 'x.cu')
    header = '#line 1 "x.cu"\n'
    assert out.startswith(header)
    return out[len(header):]


def launch(kernel, config, args):
    return f'{LAMBDA}{kernel}(cuda4cpu_args...); }}, {config}).call({args})'


class Launches(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(rewrite('k<<<g, b>>>(x, y);'), launch('k', 'g, b', 'x, y') + ';')

    def test_no_arguments_and_full_configuration(self):
        self.assertEqual(rewrite('k<<<g, b, s, st>>>();'), launch('k', 'g, b, s, st', '') + ';')

    def test_template_kernel_with_nested_templates(self):
        self.assertEqual(rewrite('k<std::pair<int, std::map<int, int>>, 3><<<1, 2>>>(p);'),
                         launch('k<std::pair<int, std::map<int, int>>, 3>', '1, 2', 'p') + ';')

    def test_qualified_and_member_kernels(self):
        self.assertEqual(rewrite('a::b::k<<<1, 1>>>();'), launch('a::b::k', '1, 1', '') + ';')
        self.assertEqual(rewrite('::k<<<1, 1>>>();'), launch('::k', '1, 1', '') + ';')
        self.assertEqual(rewrite('obj.template f<int><<<1, 1>>>();'),
                         launch('obj.template f<int>', '1, 1', '') + ';')
        self.assertEqual(rewrite('table[i]<<<1, 1>>>();'), launch('table[i]', '1, 1', '') + ';')
        self.assertEqual(rewrite('get()<<<1, 1>>>();'), launch('get()', '1, 1', '') + ';')

    def test_statement_context(self):
        self.assertEqual(rewrite('if (c) a<<<1, 1>>>(); else b<<<1, 1>>>();'),
                         f'if (c) {launch("a", "1, 1", "")}; else {launch("b", "1, 1", "")};')

    def test_void_cast_of_launched_kernel(self):
        self.assertEqual(rewrite('cudaLaunchCooperativeKernel((void *)k, g, b, args);'),
                         'cudaLaunchCooperativeKernel(k, g, b, args);')
        self.assertEqual(rewrite('cudaLaunchKernel((const void*)k<int>, g, b, args, 0, s);'),
                         'cudaLaunchKernel(k<int>, g, b, args, 0, s);')
        self.assertEqual(rewrite('cudaLaunchKernel(reinterpret_cast<void *>(&k), g, b, a);'),
                         'cudaLaunchKernel((&k), g, b, a);')
        self.assertEqual(rewrite('cudaLaunchKernel((void\n*)k, g, b, a);'),
                         'cudaLaunchKernel(\nk, g, b, a);')
        # Other calls, and casts elsewhere, are untouched
        for text in ('f((void *)k);', 'cudaLaunchKernel(k, g, b, (void *)a);',
                     'obj.cudaLaunchKernel((void *)k);'):
            self.assertEqual(rewrite(text), text)

    def test_spaces_between_brackets(self):
        self.assertEqual(rewrite('k << < g, b >> > (x);'), launch('k', ' g, b ', 'x') + ';')

    def test_shift_in_configuration(self):
        self.assertEqual(rewrite('k<<<n >> 1, dim3(m >> 2)>>>(x);'),
                         launch('k', 'n >> 1, dim3(m >> 2)', 'x') + ';')

    def test_multiline_keeps_line_count(self):
        text = 'k<<<g,\n  b>>>(x,\n  y);\nint after;'
        out = rewrite(text)
        self.assertEqual(out.count('\n'), text.count('\n'))
        self.assertTrue(out.endswith('\nint after;'))

    def test_launch_in_launch_arguments(self):
        out = rewrite('outer<<<1, 1>>>([&] { inner<<<2, 2>>>(); });')
        self.assertEqual(out, launch('outer', '1, 1', '[&] { ' + launch('inner', '2, 2', '') + '; }') + ';')

    def test_macro(self):
        self.assertEqual(rewrite('#define L(k) k<<<1, 32>>>(0)'), '#define L(k) ' + launch('k', '1, 32', '0'))


class Untouched(unittest.TestCase):
    def check(self, text):
        self.assertEqual(rewrite(text), text)

    def test_comments_and_strings(self):
        self.check('// k<<<1, 1>>>();\n/* k<<<1, 1>>>(); */\nconst char *s = "k<<<1, 1>>>()";')

    def test_raw_strings_and_characters(self):
        self.check('auto r = R"d(k<<<1, 1>>>(")")d"; auto u = u8R"(<<<)"; char c = \'<\';')

    def test_digit_separators(self):
        self.check("int n = 1'000'000; double d = 1e+5;")

    def test_operator_template(self):
        self.check('template <class T> std::ostream &operator<<<T>(std::ostream &, T);')


class ExternShared(unittest.TestCase):
    def test_function_scope(self):
        self.assertEqual(rewrite('void k() { extern __shared__ float s[]; }'),
                         'void k() { float *s = cuda4cpu::dynamic_shared<float>(); }')

    def test_qualifiers_and_alignment(self):
        self.assertEqual(rewrite('void k() { extern __shared__ __align__(16) volatile T s[]; }'),
                         'void k() { volatile T *s = cuda4cpu::dynamic_shared<volatile T>(); }')
        self.assertEqual(rewrite('void k() { extern __shared__ alignas(8) char s[]; }'),
                         'void k() { char *s = cuda4cpu::dynamic_shared<char>(); }')

    def test_namespace_scope(self):
        self.assertEqual(rewrite('extern __shared__ int s[];'),
                         'static cuda4cpu::dynamic_shared_array<int> s;')
        self.assertEqual(rewrite('namespace a { extern "C" { extern __shared__ int s[]; } }'),
                         'namespace a { extern "C" { static cuda4cpu::dynamic_shared_array<int> s; } }')

    def test_other_externs_untouched(self):
        text = 'extern int x; extern "C" void f(); extern __device__ int y[];'
        self.assertEqual(rewrite(text), text)


class Errors(unittest.TestCase):
    def test_launch_without_arguments(self):
        with self.assertRaises(rw.RewriteError):
            rewrite('k<<<1, 1>>>;')

    def test_unterminated_configuration(self):
        with self.assertRaises(rw.RewriteError):
            rewrite('k<<<1, 1(x);')

    def test_extern_shared_that_is_not_an_array(self):
        with self.assertRaises(rw.RewriteError):
            rewrite('void k() { extern __shared__ float s; }')


class Headers(unittest.TestCase):
    def test_local_headers_and_depfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'src')
            os.makedirs(os.path.join(src, 'sub'))
            with open(os.path.join(src, 'main.cu'), 'w') as f:
                f.write('#include "sub/a.cuh"\n#include <vector>\n#include "missing.h"\nint main() {}\n')
            with open(os.path.join(src, 'sub', 'a.cuh'), 'w') as f:
                f.write('#include "b.cuh"\nvoid a() { k<<<1, 1>>>(); }\n')
            with open(os.path.join(src, 'sub', 'b.cuh'), 'w') as f:
                f.write('__global__ void k() { extern __shared__ int s[]; }\n')

            out = os.path.join(tmp, 'out', 'main.cpp')
            dep = out + '.d'
            self.assertEqual(rw.main([os.path.join(src, 'main.cu'), '-o', out, '--depfile', dep]), 0)

            with open(out) as f:
                main = f.read()
            self.assertTrue(main.startswith(rw.PROLOGUE + f'#line 1 "{os.path.join(src, "main.cu")}"'))
            shadow_a = os.path.join(out + '.headers', src.lstrip(os.sep), 'sub', 'a.cuh')
            shadow_b = os.path.join(out + '.headers', src.lstrip(os.sep), 'sub', 'b.cuh')
            self.assertIn(f'#include "{shadow_a}"', main)
            self.assertIn('#include <vector>', main)
            self.assertIn('#include "missing.h"', main)       # not found locally: left alone
            with open(shadow_a) as f:
                a = f.read()
            self.assertIn(f'#include "{shadow_b}"', a)
            self.assertIn(LAMBDA, a)
            self.assertTrue(a.startswith(f'#line 1 "{os.path.join(src, "sub", "a.cuh")}"'))
            with open(shadow_b) as f:
                self.assertIn('cuda4cpu::dynamic_shared<int>()', f.read())
            with open(dep) as f:
                deps = f.read()
            for name in ('main.cu', 'a.cuh', 'b.cuh'):
                self.assertIn(name, deps)

    def test_error_reports_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'bad.cu')
            with open(path, 'w') as f:
                f.write('int x;\n\nk<<<1, 1>>>;\n')
            with self.assertRaises(rw.RewriteError) as ctx:
                rw.Rewriter(os.path.join(tmp, 'shadow')).rewrite_file(path)
            self.assertIn(f'{path}:3: error:', ctx.exception.args[1])


if __name__ == '__main__':
    unittest.main()
