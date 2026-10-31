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

"""cuda4cpu-mcp, driven as an MCP client drives it, over stdio: the protocol,
and each tool on programs whose problems are known. check must find each bug
(a missing __syncthreads(), an out-of-bounds access, a divergent barrier, a
copy in the wrong direction, undefined behavior, a failure, a hang) where it
is, and nothing in a correct program. test_kernel must pass correct kernels
and name the elements that wrong ones get wrong.

Usage: test_mcp.py <cuda4cpu-mcp> <examples/bugs>
"""

import json
import os
import subprocess
import sys
import unittest

SERVER = None
BUGS = None


class Client:
    """An MCP client of one server process"""

    def __init__(self):
        environment = {k: v for k, v in os.environ.items() if not k.startswith('CUDA4CPU_')}
        if 'CUDA4CPU_CXX' in os.environ:
            environment['CUDA4CPU_CXX'] = os.environ['CUDA4CPU_CXX']
        self.process = subprocess.Popen([SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                        env=environment)
        self.next_id = 0

    def send(self, message):
        self.process.stdin.write(json.dumps(message) + '\n')
        self.process.stdin.flush()

    def request(self, method, params=None):
        self.next_id += 1
        self.send({'jsonrpc': '2.0', 'id': self.next_id, 'method': method, 'params': params or {}})
        reply = json.loads(self.process.stdout.readline())
        assert reply['id'] == self.next_id, reply
        return reply

    def call(self, tool, **arguments):
        reply = self.request('tools/call', {'name': tool, 'arguments': arguments})
        result = reply['result']
        return result, result.get('structuredContent')

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)
        self.process.stdout.close()


CLIENT = None


def setUpModule():
    global CLIENT
    CLIENT = Client()
    reply = CLIENT.request('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                                          'clientInfo': {'name': 'test_mcp', 'version': '1'}})
    assert reply['result']['serverInfo']['name'] == 'cuda4cpu', reply
    CLIENT.send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})


def tearDownModule():
    CLIENT.close()


def kernel_program(body, kernel, launch='kernel<<<2, 64>>>(d);', check='return 0;'):
    """A program with one kernel over an array of 128 ints: kernel is its
    body, check runs on the host copy h after the launch"""
    return f'''#include <cstdio>
#include <climits>
{kernel}
int main()
{{
    const int n = 128;
    int h[n];
    for (int i = 0; i < n; ++i)
        h[i] = i;
    int *d;
    cudaMalloc(&d, n * sizeof(int));
    cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice);
    {launch}
    cudaMemcpy(h, d, n * sizeof(int), cudaMemcpyDeviceToHost);
    {body}
    cudaFree(d);
    {check}
}}
'''


CORRECT = kernel_program('', '''
__global__ void kernel(int *d)
{
    __shared__ int s[64];
    const int i = blockIdx.x * 64 + threadIdx.x;
    s[threadIdx.x] = d[i];
    __syncthreads();
    d[i] = s[63 - threadIdx.x];
}
''', check='''for (int i = 0; i < n; ++i)
        if (h[i] != (i / 64) * 64 + 63 - i % 64)
            return 1;
    std::printf("reversed\\n");
    return 0;''')


class Protocol(unittest.TestCase):

    def test_initialize(self):
        client = Client()
        try:
            reply = client.request('initialize', {'protocolVersion': '2024-11-05', 'capabilities': {}})
            self.assertEqual(reply['result']['protocolVersion'], '2024-11-05')
            self.assertIn('tools', reply['result']['capabilities'])
            reply = client.request('initialize', {'protocolVersion': '1999-01-01', 'capabilities': {}})
            self.assertEqual(reply['result']['protocolVersion'], '2025-06-18')
            self.assertIn('instructions', reply['result'])
        finally:
            client.close()

    def test_tools(self):
        tools = CLIENT.request('tools/list')['result']['tools']
        self.assertEqual([t['name'] for t in tools], ['compile', 'run', 'check', 'test_kernel', 'capabilities'])
        for tool in tools:
            self.assertEqual(tool['inputSchema']['type'], 'object')
            self.assertTrue(tool['description'])

    def test_errors(self):
        self.assertEqual(CLIENT.request('ping')['result'], {})
        self.assertEqual(CLIENT.request('resources/list')['error']['code'], -32601)
        self.assertEqual(CLIENT.request('tools/call', {'name': 'nonexistent'})['error']['code'], -32602)
        # A request the tool can't serve is the tool's error, not the protocol's
        result, _ = CLIENT.call('compile', paths=['/nonexistent.cu'])
        self.assertTrue(result['isError'])
        result, _ = CLIENT.call('compile', files={'../escape.cu': ''})
        self.assertTrue(result['isError'])
        CLIENT.process.stdin.write('not json\n')
        CLIENT.process.stdin.flush()
        self.assertEqual(json.loads(CLIENT.process.stdout.readline())['error']['code'], -32700)


class Compile(unittest.TestCase):

    def test_compiles(self):
        result, content = CLIENT.call('compile', files={'k.cu': CORRECT})
        self.assertFalse(result['isError'])
        self.assertTrue(content['compiled'])

    def test_diagnostics_at_the_sources_lines(self):
        source = '__global__ void k(int *x)\n{\n    x[threadIdx.x] = undeclared;\n}\nint main() { return 0; }\n'
        _, content = CLIENT.call('compile', files={'dir/k.cu': source})
        self.assertFalse(content['compiled'])
        errors = [d for d in content['diagnostics'] if d['severity'] == 'error']
        self.assertTrue(errors)
        self.assertEqual((errors[0]['file'], errors[0]['line']), ('dir/k.cu', 3))
        self.assertIn('undeclared', errors[0]['message'])

    def test_unsupported_features(self):
        source = ('__global__ void k(cudaTextureObject_t t, float *out)\n'
                  '{\n    out[threadIdx.x] = tex1D<float>(t, threadIdx.x);\n}\nint main() { return 0; }\n')
        result, content = CLIENT.call('compile', files={'tex.cu': source})
        self.assertFalse(content['compiled'])
        self.assertTrue(any('textures' in note for note in content['unsupported']), content)
        self.assertIn('unsupported: textures', result['content'][0]['text'])

    def test_hip(self):
        source = ('#include <hip/hip_runtime.h>\n__global__ void k(int *x) { x[hipThreadIdx_x] = 1; }\n'
                  'int main() { int *x; hipMalloc(&x, 64 * sizeof(int)); hipLaunchKernelGGL(k, 1, 64, 0, 0, x);\n'
                  'return hipDeviceSynchronize() == hipSuccess ? 0 : 1; }\n')
        _, content = CLIENT.call('compile', files={'k.hip': source})
        self.assertTrue(content['compiled'], content)
        self.assertEqual(content['language'], 'hip')


class Run(unittest.TestCase):

    def test_output_and_exit_code(self):
        source = ('#include <cstdio>\n#include <cstdlib>\n__global__ void k() { if (threadIdx.x == 3) '
                  'printf("lane %d\\n", int(threadIdx.x)); }\nint main(int argc, char **argv) {\n'
                  'k<<<1, 32>>>(); cudaDeviceSynchronize();\nchar line[64]; if (!std::fgets(line, sizeof line, stdin)) '
                  'return 9;\nstd::printf("%s %s %s", argv[1], std::getenv("GREETING"), line);\nreturn 7; }\n')
        _, content = CLIENT.call('run', files={'p.cu': source}, args=['first'], stdin='from stdin\n',
                                 environment={'GREETING': 'hello'})
        self.assertEqual(content['exit_code'], 7, content)
        self.assertIn('lane 3', content['stdout'])
        self.assertIn('first hello from stdin', content['stdout'])

    def test_timeout(self):
        source = '__global__ void k(volatile int *x) { while (*x == 0) {} }\n' \
                 'int main() { int *x; cudaMallocManaged(&x, 4); *x = 0; k<<<1, 1>>>(x); return 0; }\n'
        _, content = CLIENT.call('run', files={'hang.cu': source}, timeout_seconds=2)
        self.assertTrue(content['timed_out'])
        self.assertTrue(any(f['kind'] == 'timeout' for f in content['findings']))


class Check(unittest.TestCase):

    def check(self, **arguments):
        result, content = CLIENT.call('check', seeds=[1, 2], **arguments)
        self.assertFalse(result['isError'], result)
        return content

    def kinds(self, content):
        return [f['kind'] for f in content['findings']]

    def test_correct_program(self):
        content = self.check(files={'k.cu': CORRECT})
        self.assertEqual(content['verdict'], 'no problems found', content['findings'])
        self.assertIn('caveat', content)
        self.assertTrue(content['deterministic_output'])
        self.assertEqual(content['forward']['stdout'], 'reversed\n')
        self.assertEqual({r['order'] for r in content['runs']},
                         {'sanitizers', 'forward', 'init_memory:0xff', 'reverse', 'warp_reverse', 'random:1',
                          'warp_random:1', 'random:2', 'warp_random:2'})

    def test_missing_barrier(self):
        content = self.check(paths=[os.path.join(BUGS, 'missing_barrier.cu')])
        self.assertEqual(content['verdict'], 'problems found')
        finding = content['findings'][0]
        self.assertEqual(finding['kind'], 'depends on the thread order')
        self.assertIn('reverse', finding['orders'])
        self.assertIn('__syncthreads()', finding['meaning'])

    def test_out_of_bounds(self):
        content = self.check(paths=[os.path.join(BUGS, 'out_of_bounds.cu')])
        finding = content['findings'][0]
        self.assertEqual(finding['kind'], 'AddressSanitizer: allocation-overflow')
        location = finding['locations'][0]
        self.assertTrue(location['file'].endswith('out_of_bounds.cu'))
        self.assertEqual(location['line'], 27)
        self.assertIn('scale', location['function'])
        # x[1100], the first float past the end, in block 4 of 256 threads
        self.assertIn('by CUDA thread (76, 0, 0) of block (4, 0, 0)', finding['message'])
        # The other orders aren't run after memory errors
        self.assertEqual([r['order'] for r in content['runs']], ['sanitizers'])

    def test_crash_in_a_kernel(self):
        kernel = '''
__global__ void kernel(int *d)
{
    if (blockIdx.x == 1 && threadIdx.x == 37)
        d[1 << 30] = 1;
}
'''
        files = {'wild.cu': kernel_program('', kernel)}
        # The sanitizer's report has the line, and cuda4cpu's the thread
        finding = self.check(files=files)['findings'][0]
        self.assertEqual(finding['kind'], 'AddressSanitizer: SEGV')
        self.assertEqual(finding['locations'][0]['line'], 7)
        self.assertIn('by CUDA thread (37, 0, 0) of block (1, 0, 0)', finding['message'])
        # Without it, the launch's
        content = self.check(files=files, sanitizers=False)
        self.assertEqual(self.kinds(content), ['crash in a kernel'])
        finding = content['findings'][0]
        self.assertTrue(finding['message'].startswith('CUDA thread (37, 0, 0) of block (1, 0, 0) made an invalid '
                                                      'memory access'), finding['message'])
        self.assertEqual(finding['locations'], [{'file': 'wild.cu', 'line': 19}])

    def test_divergent_barrier(self):
        kernel = '''
__global__ void kernel(int *d)
{
    if (threadIdx.x < 16)
        __syncthreads();
    else
        __syncthreads();
    d[blockIdx.x * 64 + threadIdx.x] += 1;
}
'''
        content = self.check(files={'divergent.cu': kernel_program('', kernel)}, sanitizers=False)
        finding = [f for f in content['findings'] if f['kind'] == 'divergent barrier'][0]
        self.assertEqual(sorted(l['line'] for l in finding['locations']), [7, 9])

    def test_copy_in_the_wrong_direction(self):
        kernel = '__global__ void kernel(int *d) { d[threadIdx.x] *= 2; }\n'
        program = kernel_program('int back[n];\n    cudaMemcpy(back, d, n * sizeof(int), cudaMemcpyHostToDevice);',
                                 kernel)
        content = self.check(files={'direction.cu': program}, sanitizers=False)
        self.assertIn('copy in the wrong direction', self.kinds(content))

    def test_memory_leak(self):
        program = CORRECT.replace('cudaFree(d);', '')
        line = program.splitlines().index('    cudaMalloc(&d, n * sizeof(int));') + 1
        content = self.check(files={'leak.cu': program}, sanitizers=False)
        self.assertEqual(self.kinds(content), ['memory leak'])
        finding = content['findings'][0]
        self.assertIn('an allocation of device memory (512 bytes)', finding['message'])
        self.assertEqual(finding['locations'], [{'file': 'leak.cu', 'line': line}])

    def test_uninitialized_memory(self):
        source = """#include <cstdio>
__global__ void count(int *total) { atomicAdd(total, 1); }
int main()
{
    int *total;
    cudaMallocManaged(&total, sizeof(int));
    count<<<4, 64>>>(total);
    cudaDeviceSynchronize();
    std::printf("%d threads\\n", *total);
    cudaFree(total);
    return 0;
}
"""
        content = self.check(files={'count.cu': source}, sanitizers=False)
        self.assertEqual(self.kinds(content), ['reads uninitialized memory'])
        finding = content['findings'][0]
        self.assertIn('CUDA4CPU_INIT_MEMORY=0xff', finding['message'])
        # The default run's memory is usually zeros (256 threads), but malloc may fill it
        self.assertEqual(finding['first_difference']['filled'], '255 threads')

    def test_undefined_behavior(self):
        kernel = '''
__global__ void kernel(int *d)
{
    int x = INT_MAX - 10 + d[threadIdx.x];
    d[threadIdx.x] = x + 1;
}
'''
        content = self.check(files={'overflow.cu': kernel_program('', kernel, launch='kernel<<<1, 64>>>(d);')})
        finding = [f for f in content['findings'] if f['kind'] == 'undefined behavior'][0]
        self.assertIn('overflow', finding['message'])
        self.assertEqual(finding['locations'][0], {'file': 'overflow.cu', 'line': 7})

    def test_failure_in_every_order(self):
        content = self.check(files={'fails.cu': CORRECT.replace('63 - i % 64', '62 - i % 64')}, sanitizers=False)
        self.assertEqual(self.kinds(content), ['fails in the default order'])

    def test_hang(self):
        source = '__global__ void k(volatile int *x) { while (*x == 0) {} }\n' \
                 'int main() { int *x; cudaMallocManaged(&x, 4); *x = 0; k<<<1, 1>>>(x); return 0; }\n'
        content = self.check(files={'hang.cu': source}, sanitizers=False, timeout_seconds=2)
        self.assertEqual(self.kinds(content), ['timeout'])
        self.assertEqual([r['order'] for r in content['runs']], ['forward'])

    def test_output_that_varies(self):
        timing = CORRECT.replace('std::printf("reversed\\n");',
                                 'std::printf("reversed\\nclock %ld\\n", (long)clock());')
        content = self.check(files={'timing.cu': timing}, sanitizers=False)
        self.assertFalse(content['deterministic_output'])
        self.assertIn('note', content)
        self.assertEqual(content['verdict'], 'no problems found')
        content = self.check(files={'timing.cu': timing}, sanitizers=False, ignore_patterns=['^clock'])
        self.assertTrue(content['deterministic_output'])


SAXPY = {
    'kernel': '__global__ void saxpy(float a, const float *x, float *y, int n)\n'
              '{\n'
              '    int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
              '    if (i < n)\n'
              '        y[i] = a * x[i] + y[i];\n'
              '}\n',
    'name': 'saxpy', 'grid': 4, 'block': 64,
    'arguments': [{'name': 'a', 'type': 'float', 'value': 2.5},
                  {'name': 'x', 'type': 'const float*', 'data': {'random': [-1, 1], 'count': 250, 'seed': 7}},
                  {'name': 'y', 'type': 'float*', 'data': {'range': [0, 0.5], 'count': 250}, 'output': True},
                  {'name': 'n', 'type': 'int', 'value': 250}],
    'reference': 'void reference(float a, const float *x, float *y, int n)\n'
                 '{\n    for (int i = 0; i < n; ++i)\n        y[i] = a * x[i] + y[i];\n}\n',
    'seeds': [1],
}


class TestKernel(unittest.TestCase):

    def call_tool(self, **arguments):
        result, content = CLIENT.call('test_kernel', **arguments)
        self.assertFalse(result['isError'], result)
        return result, content

    def test_correct_kernel(self):
        result, content = self.call_tool(**SAXPY)
        self.assertTrue(content['passed'], content)
        self.assertEqual(content['outputs'], [{'output': 'y', 'elements': 250, 'first_mismatches': [],
                                               'mismatches': 0, 'max_abs_error': content['outputs'][0]['max_abs_error']}])
        self.assertEqual(content['verdict'], 'no problems found')
        self.assertTrue(result['content'][0]['text'].startswith('passed'))

    def test_wrong_element(self):
        wrong = dict(SAXPY, kernel=SAXPY['kernel'].replace('i < n', 'i <= n'),
                     arguments=SAXPY['arguments'][:3] + [{'name': 'n', 'type': 'int', 'value': 249}])
        _, content = self.call_tool(**wrong)
        self.assertFalse(content['passed'])
        output = content['outputs'][0]
        self.assertEqual(output['mismatches'], 1)
        self.assertEqual(output['first_mismatches'][0]['index'], 249)

    def test_expected_values_and_template(self):
        kernel = ('template <typename T>\n__global__ void square(T *d)\n{\n'
                  '    d[threadIdx.x] = d[threadIdx.x] * d[threadIdx.x];\n}\n')
        _, content = self.call_tool(kernel=kernel, name='square<long long>', grid=1, block=[4, 1, 1],
                               arguments=[{'type': 'long long *', 'data': [1, -2, 3, 3000000000],
                                           'expected': [1, 4, 9, 9000000000000000000]}], check=False)
        self.assertTrue(content['passed'], content)
        _, content = self.call_tool(kernel=kernel, name='square<int>', grid=1, block=3,
                               arguments=[{'type': 'int*', 'data': [1, 2, 3], 'expected': [1, 4, 10]}], check=False)
        self.assertFalse(content['passed'])
        self.assertEqual(content['outputs'][0]['first_mismatches'], [{'index': 2, 'got': 9, 'expected': 10}])

    def test_order_dependent_kernel(self):
        """Each thread reads its predecessor's element of shared memory
        without a barrier: right in the default order only"""
        kernel = ('__global__ void shift(int *d)\n{\n    __shared__ int s[64];\n'
                  '    s[threadIdx.x] = d[threadIdx.x];\n'
                  '    d[threadIdx.x] = threadIdx.x > 0 ? s[threadIdx.x - 1] : 0;\n}\n')
        _, content = self.call_tool(kernel=kernel, name='shift', grid=1, block=64, seeds=[1],
                               arguments=[{'type': 'int*', 'data': {'range': [1, 1], 'count': 64},
                                           'expected': {'range': [0, 1], 'count': 64}}])
        self.assertEqual(content['outputs'][0]['mismatches'], 0)
        self.assertFalse(content['passed'])
        self.assertEqual([f['kind'] for f in content['findings']], ['depends on the thread order'])

    def test_overflow_within_an_allocation_s_rounding(self):
        """128 threads write 100 ints: the overflow fits in the rounding of
        the allocation to 256 bytes, which cuda4cpu poisons"""
        kernel = '__global__ void inc(int *y)\n{\n    y[blockIdx.x * blockDim.x + threadIdx.x] += 1;\n}\n'
        _, content = self.call_tool(kernel=kernel, name='inc', grid=2, block=64, seeds=[1],
                               arguments=[{'type': 'int*', 'data': {'fill': 0, 'count': 100},
                                           'expected': {'fill': 1, 'count': 100}}])
        self.assertFalse(content['passed'])
        finding = content['findings'][0]
        self.assertEqual(finding['kind'], 'AddressSanitizer: allocation-overflow')
        self.assertEqual((finding['locations'][0]['file'], finding['locations'][0]['line']), ('kernel.cu', 3))

    def test_hip_kernel(self):
        kernel = ('#include <hip/hip_runtime.h>\n__global__ void halve(double *d) '
                  '{ d[hipThreadIdx_x] /= 2; }\n')
        _, content = self.call_tool(language='hip', kernel=kernel, name='halve', grid=1, block=8, check=False,
                               tolerance={'absolute': 0, 'relative': 1e-12},
                               arguments=[{'type': 'double*', 'data': {'fill': 3, 'count': 8},
                                           'expected': {'fill': 1.5, 'count': 8}}])
        self.assertTrue(content['passed'], content)
        self.assertEqual(content['language'], 'hip')

    def test_compile_error_in_the_kernel(self):
        kernel = '__global__ void k(int *d)\n{\n    d[threadIdx.x] = undeclared;\n}\n'
        _, content = self.call_tool(kernel=kernel, name='k', grid=1, block=1,
                               arguments=[{'type': 'int*', 'data': [0], 'expected': [0]}])
        self.assertFalse(content['compiled'])
        error = [d for d in content['diagnostics'] if d['severity'] == 'error'][0]
        self.assertEqual((error['file'], error['line']), ('kernel.cu', 3))

    def test_invalid_requests(self):
        for bad in ({'arguments': [{'type': 'struct params', 'value': 1}]},
                    {'arguments': [{'type': 'int*', 'data': [0]}]},              # no output
                    {'arguments': [{'type': 'int*', 'data': [0, 1], 'expected': [0]}]},
                    {'arguments': [{'type': 'int*', 'data': [0], 'output': True}]},   # no expected, no reference
                    {'arguments': [{'type': 'int*', 'data': [0], 'expected': [0]}], 'block': 2048}):
            request = dict({'kernel': '__global__ void k(int *) {}', 'name': 'k', 'grid': 1, 'block': 1}, **bad)
            result, _ = CLIENT.call('test_kernel', **request)
            self.assertTrue(result['isError'], bad)


class Capabilities(unittest.TestCase):

    def test_capabilities(self):
        result, content = CLIENT.call('capabilities')
        self.assertIn('cuda4cpu', content['text'])
        self.assertIn('Not implemented yet', content['text'])
        self.assertEqual(result['content'][0]['text'], content['text'])


if __name__ == '__main__':
    SERVER, BUGS = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    unittest.main(argv=sys.argv[:1], verbosity=2)
