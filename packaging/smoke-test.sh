#!/usr/bin/env bash
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

# Checks an installed cuda4cpu, as the release archive and the Docker image
# ship it: the drivers build programs that run, the CMake package and the
# toolkit's toolchain file work (when CMake is installed), and the MCP server
# finds a missing __syncthreads() and passes a correct kernel.
#
#     packaging/smoke-test.sh <prefix> [<source directory>]
#
# The source directory, the repository by default, has the example programs.

set -euo pipefail

prefix=$(realpath "$1")
src=$(realpath "${2:-$(dirname "$0")/..}")
toolkit=$(find "$prefix" -maxdepth 3 -type d -path '*/cuda4cpu/toolkit' | head -n 1)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PATH="$prefix/bin:$PATH"

step() { echo "== $*"; }

step cuda4cpu-c++
cuda4cpu-c++ -O2 "$src/examples/cmake-project/vecadd.cu" -o vecadd
./vecadd | grep 'computed on cuda4cpu'

step "cuda4cpu-nvcc, and the toolkit's bin/nvcc"
cuda4cpu-nvcc -O2 -arch=sm_80 "$src/examples/cmake-project/vecadd.cu" -o vecadd_nvcc
./vecadd_nvcc | grep 'computed on cuda4cpu'
"$toolkit/bin/nvcc" --version | grep 'cuda4cpu'

step cuda4cpu-hipcc
cat > square.cpp <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void square(int *v)
{
    v[threadIdx.x] *= v[threadIdx.x];
}

int main()
{
    int host[64];
    for (int i = 0; i < 64; ++i)
        host[i] = i;
    int *v;
    hipMalloc(&v, sizeof(host));
    hipMemcpy(v, host, sizeof(host), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(square, dim3(1), dim3(64), 0, 0, v);
    hipMemcpy(host, v, sizeof(host), hipMemcpyDeviceToHost);
    hipFree(v);
    std::printf("%d\n", host[7]);
    return host[7] == 49 ? 0 : 1;
}
EOF
cuda4cpu-hipcc -O2 --offload-arch=gfx90a square.cpp -o square
./square

if command -v cmake > /dev/null; then
    step "find_package(cuda4cpu)"
    mkdir package
    cp "$src/examples/cmake-project/vecadd.cu" package/
    cat > package/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.25)
project(smoke CXX)
find_package(cuda4cpu REQUIRED)
add_executable(vecadd)
cuda4cpu_add_cuda_sources(vecadd vecadd.cu)
EOF
    cmake -S package -B package/build -DCMAKE_PREFIX_PATH="$prefix" > /dev/null
    cmake --build package/build > /dev/null
    package/build/vecadd | grep 'computed on cuda4cpu'

    # CMake's CUDA language needs CMake 4.1 or later with cuda4cpu-nvcc
    version=$(cmake --version | sed -n 's/^cmake version \([0-9]*\.[0-9]*\).*/\1/p')
    if printf '4.1\n%s\n' "$version" | sort -V -C; then
        step "LANGUAGES CUDA with the toolkit's toolchain file"
        mkdir language
        cp "$src/examples/cmake-project/vecadd.cu" language/
        cat > language/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.25)
project(smoke LANGUAGES CUDA)
add_executable(vecadd vecadd.cu)
EOF
        cmake -S language -B language/build -DCMAKE_TOOLCHAIN_FILE="$toolkit/cuda4cpu.cmake" > /dev/null
        cmake --build language/build > /dev/null
        language/build/vecadd | grep 'computed on cuda4cpu'
    fi
fi

step cuda4cpu-mcp
python3 - "$src/examples/bugs/missing_barrier.cu" <<'EOF'
import json
import subprocess
import sys

server = subprocess.Popen(['cuda4cpu-mcp'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
calls = 0


def call(method, params):
    global calls
    calls += 1
    server.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': calls, 'method': method, 'params': params}) + '\n')
    server.stdin.flush()
    reply = json.loads(server.stdout.readline())
    if 'error' in reply:
        sys.exit(f'{method}: {reply["error"]}')
    return reply['result']


def tool(name, arguments):
    result = call('tools/call', {'name': name, 'arguments': arguments})
    return result['structuredContent']


call('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                    'clientInfo': {'name': 'smoke-test', 'version': '1'}})
tools = {t['name'] for t in call('tools/list', {})['tools']}
assert tools == {'compile', 'run', 'check', 'test_kernel', 'capabilities'}, tools

found = tool('check', {'paths': [sys.argv[1]]})
assert any(f['kind'] == 'depends on the thread order' for f in found['findings']), found
print('check: the missing __syncthreads() is found')

saxpy = tool('test_kernel', {
    'kernel': '__global__ void saxpy(int n, float a, const float *x, float *y)\n'
              '{\n'
              '    int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
              '    if (i < n)\n'
              '        y[i] = a * x[i] + y[i];\n'
              '}\n',
    'name': 'saxpy', 'grid': [4], 'block': [256],
    'arguments': [
        {'type': 'int', 'value': 1000},
        {'type': 'float', 'value': 2},
        {'type': 'const float *', 'data': {'range': [0, 1], 'count': 1000}},
        {'type': 'float *', 'data': {'fill': 1, 'count': 1000}, 'output': True}],
    'reference': 'void reference(int n, float a, const float *x, float *y)\n'
                 '{\n'
                 '    for (int i = 0; i < n; ++i)\n'
                 '        y[i] = a * x[i] + y[i];\n'
                 '}\n'})
assert saxpy['passed'], saxpy
print('test_kernel: a correct kernel passes')
server.stdin.close()
server.wait()
EOF

step "all checks passed"
