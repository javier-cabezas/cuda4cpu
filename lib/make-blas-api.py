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

"""Generates the BLAS routines' declarations of cuBLAS, hipBLAS and rocBLAS,
and the library's functions that implement them, from one table.

The three APIs take the same arguments in the same order: only the names of
their types, their enumerations' values and their statuses differ. Each
function converts its arguments (blas::arg, blas::ptr, blas::scalar_in) and
calls the routine of lib/blas.cpp, blas::<core>, with the handle's context.

Usage: lib/make-blas-api.py   (rewrites the headers' generated blocks and
lib/blas_api.inc in place)
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LICENSE = '''/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
'''

# Parameter kinds: T is the routine's type, R its real type
#   int, stride            integers (a stride is 64 bits)
#   op, fill, diag, side   enumerations
#   sT, sR                 a scalar: a pointer the handle's pointer mode says
#                          is the host's (read when called) or the device's
#   cT, cR                 an input array (const)
#   T, R, I                an output (or in-out) array or result
#   cTb, Tb                batched arrays: an array of pointers
P = {
    'n': ('int', 'n'), 'm': ('int', 'm'), 'k': ('int', 'k'), 'kl': ('int', 'kl'), 'ku': ('int', 'ku'),
    'lda': ('int', 'lda'), 'ldb': ('int', 'ldb'), 'ldc': ('int', 'ldc'),
    'incx': ('int', 'incx'), 'incy': ('int', 'incy'), 'batch': ('int', 'batchCount'),
    'trans': ('op', 'trans'), 'transa': ('op', 'transa'), 'transb': ('op', 'transb'),
    'uplo': ('fill', 'uplo'), 'diag': ('diag', 'diag'), 'side': ('side', 'side'), 'mode': ('side', 'mode'),
    'alpha': ('sT', 'alpha'), 'beta': ('sT', 'beta'), 'ralpha': ('sR', 'alpha'), 'rbeta': ('sR', 'beta'),
    'A': ('cT', 'A'), 'B': ('cT', 'B'), 'AP': ('cT', 'AP'), 'x': ('cT', 'x'), 'y': ('cT', 'y'),
    'oA': ('T', 'A'), 'oB': ('T', 'B'), 'oC': ('T', 'C'), 'oAP': ('T', 'AP'), 'ox': ('T', 'x'), 'oy': ('T', 'y'),
    'sA': ('stride', 'strideA'), 'sB': ('stride', 'strideB'), 'sC': ('stride', 'strideC'),
    'sx': ('stride', 'stridex'), 'sy': ('stride', 'stridey'),
    'Ab': ('cTb', 'Aarray'), 'Bb': ('cTb', 'Barray'), 'xb': ('cTb', 'xarray'),
    'oBb': ('Tb', 'Barray'), 'oCb': ('Tb', 'Carray'), 'oyb': ('Tb', 'yarray'),
    'result': ('T', 'result'), 'rresult': ('R', 'result'), 'iresult': ('I', 'result'),
}

SDCZ = 'SDCZ'

# (cuBLAS's name for each type, rocBLAS's name (lowercase, with
# underscores) for each type, core routine, parameters (keys of P, or
# (kind, name) pairs), whether cuBLAS's has a _v2 name, which APIs)
ROUTINES = []


def routine(names, core, params, v2=True, apis='chr', roc=None):
    """names: {type: cuBLAS's name without 'cublas'}; roc: {type: rocBLAS's
    name without 'rocblas_'}, by default the cuBLAS name in lowercase"""
    params = [P[p] if isinstance(p, str) else p for p in params]
    if roc is None:
        roc = {t: re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name).lower() for t, name in names.items()}
    ROUTINES.append((names, roc, core, params, v2, apis))


def each(types, pattern):
    """{type: pattern with {t} as the type's letter}"""
    return {t: pattern.format(t=t) for t in types}


# Level 1
routine({'S': 'Isamax', 'D': 'Idamax', 'C': 'Icamax', 'Z': 'Izamax'}, 'iamax', ['n', 'x', 'incx', 'iresult'])
routine({'S': 'Isamin', 'D': 'Idamin', 'C': 'Icamin', 'Z': 'Izamin'}, 'iamin', ['n', 'x', 'incx', 'iresult'])
routine({'S': 'Sasum', 'D': 'Dasum', 'C': 'Scasum', 'Z': 'Dzasum'}, 'asum', ['n', 'x', 'incx', 'rresult'])
routine(each(SDCZ, '{t}axpy'), 'axpy', ['n', 'alpha', 'x', 'incx', 'oy', 'incy'])
routine(each(SDCZ, '{t}copy'), 'copy', ['n', 'x', 'incx', 'oy', 'incy'])
routine({'S': 'Sdot', 'D': 'Ddot'}, 'dot', ['n', 'x', 'incx', 'y', 'incy', 'result'])
routine({'C': 'Cdotu', 'Z': 'Zdotu'}, 'dot', ['n', 'x', 'incx', 'y', 'incy', 'result'])
routine({'C': 'Cdotc', 'Z': 'Zdotc'}, 'dotc', ['n', 'x', 'incx', 'y', 'incy', 'result'])
routine({'S': 'Snrm2', 'D': 'Dnrm2', 'C': 'Scnrm2', 'Z': 'Dznrm2'}, 'nrm2', ['n', 'x', 'incx', 'rresult'])
routine(each(SDCZ, '{t}rot'), 'rot', ['n', 'ox', 'incx', 'oy', 'incy', ('sR', 'c'), ('sT', 's')])
routine({'C': 'Csrot', 'Z': 'Zdrot'}, 'rot', ['n', 'ox', 'incx', 'oy', 'incy', ('sR', 'c'), ('sR', 's')])
routine(each(SDCZ, '{t}rotg'), 'rotg', [('T', 'a'), ('T', 'b'), ('R', 'c'), ('T', 's')])
routine(each('SD', '{t}rotm'), 'rotm', ['n', 'ox', 'incx', 'oy', 'incy', ('cT', 'param')])
routine(each('SD', '{t}rotmg'), 'rotmg', [('T', 'd1'), ('T', 'd2'), ('T', 'x1'), ('cT', 'y1'), ('T', 'param')])
routine(each(SDCZ, '{t}scal'), 'scal', ['n', 'alpha', 'ox', 'incx'])
routine({'C': 'Csscal', 'Z': 'Zdscal'}, 'scal', ['n', 'ralpha', 'ox', 'incx'])
routine(each(SDCZ, '{t}swap'), 'swap', ['n', 'ox', 'incx', 'oy', 'incy'])

# Level 2
routine(each(SDCZ, '{t}gbmv'), 'gbmv',
        ['trans', 'm', 'n', 'kl', 'ku', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'oy', 'incy'])
routine(each(SDCZ, '{t}gemv'), 'gemv', ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'oy', 'incy'])
routine(each(SDCZ, '{t}gemvBatched'), 'gemv_batched',
        ['trans', 'm', 'n', 'alpha', 'Ab', 'lda', 'xb', 'incx', 'beta', 'oyb', 'incy', 'batch'], v2=False)
routine(each(SDCZ, '{t}gemvStridedBatched'), 'gemv_strided_batched',
        ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'sA', 'x', 'incx', 'sx', 'beta', 'oy', 'incy', 'sy', 'batch'],
        v2=False)
routine({'S': 'Sger', 'D': 'Dger', 'C': 'Cgeru', 'Z': 'Zgeru'}, 'ger',
        ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'oA', 'lda'])
routine({'C': 'Cgerc', 'Z': 'Zgerc'}, 'gerc', ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'oA', 'lda'])
routine(each('SD', '{t}sbmv'), 'sbmv', ['uplo', 'n', 'k', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'oy', 'incy'])
routine(each('SD', '{t}spmv'), 'spmv', ['uplo', 'n', 'alpha', 'AP', 'x', 'incx', 'beta', 'oy', 'incy'])
routine(each('SD', '{t}spr'), 'spr', ['uplo', 'n', 'ralpha', 'x', 'incx', 'oAP'])
routine(each('SD', '{t}spr2'), 'spr2', ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'oAP'])
routine(each(SDCZ, '{t}symv'), 'symv', ['uplo', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'oy', 'incy'])
routine(each(SDCZ, '{t}syr'), 'syr', ['uplo', 'n', 'alpha', 'x', 'incx', 'oA', 'lda'])
routine(each(SDCZ, '{t}syr2'), 'syr2', ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'oA', 'lda'])
routine(each(SDCZ, '{t}tbmv'), 'tbmv', ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'ox', 'incx'])
routine(each(SDCZ, '{t}tbsv'), 'tbsv', ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'ox', 'incx'])
routine(each(SDCZ, '{t}tpmv'), 'tpmv', ['uplo', 'trans', 'diag', 'n', 'AP', 'ox', 'incx'])
routine(each(SDCZ, '{t}tpsv'), 'tpsv', ['uplo', 'trans', 'diag', 'n', 'AP', 'ox', 'incx'])
routine(each(SDCZ, '{t}trmv'), 'trmv', ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'ox', 'incx'])
routine(each(SDCZ, '{t}trsv'), 'trsv', ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'ox', 'incx'])
routine(each('CZ', '{t}hemv'), 'hemv', ['uplo', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'oy', 'incy'])
routine(each('CZ', '{t}hbmv'), 'hbmv', ['uplo', 'n', 'k', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'oy', 'incy'])
routine(each('CZ', '{t}hpmv'), 'hpmv', ['uplo', 'n', 'alpha', 'AP', 'x', 'incx', 'beta', 'oy', 'incy'])
routine(each('CZ', '{t}her'), 'her', ['uplo', 'n', 'ralpha', 'x', 'incx', 'oA', 'lda'])
routine(each('CZ', '{t}her2'), 'her2', ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'oA', 'lda'])
routine(each('CZ', '{t}hpr'), 'hpr', ['uplo', 'n', 'ralpha', 'x', 'incx', 'oAP'])
routine(each('CZ', '{t}hpr2'), 'hpr2', ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'oAP'])

# Level 3
GEMM = ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'oC', 'ldc']
routine(each(SDCZ, '{t}gemm'), 'gemm', GEMM)
routine(each('CZ', '{t}gemm3m'), 'gemm', GEMM, v2=False, apis='c')
routine(each(SDCZ, '{t}gemmBatched'), 'gemm_batched',
        ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Ab', 'lda', 'Bb', 'ldb', 'beta', 'oCb', 'ldc', 'batch'],
        v2=False)
routine(each(SDCZ, '{t}gemmStridedBatched'), 'gemm_strided_batched',
        ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'sA', 'B', 'ldb', 'sB', 'beta', 'oC', 'ldc', 'sC',
         'batch'], v2=False)
routine(each(SDCZ, '{t}symm'), 'symm', ['side', 'uplo', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'oC', 'ldc'])
routine(each('CZ', '{t}hemm'), 'hemm', ['side', 'uplo', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'oC', 'ldc'])
routine(each(SDCZ, '{t}syrk'), 'syrk', ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'oC', 'ldc'])
routine(each('CZ', '{t}herk'), 'herk', ['uplo', 'trans', 'n', 'k', 'ralpha', 'A', 'lda', 'rbeta', 'oC', 'ldc'])
routine(each(SDCZ, '{t}syr2k'), 'syr2k',
        ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'oC', 'ldc'])
routine(each('CZ', '{t}her2k'), 'her2k',
        ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'rbeta', 'oC', 'ldc'])
routine(each(SDCZ, '{t}syrkx'), 'syrkx',
        ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'oC', 'ldc'], v2=False)
routine(each('CZ', '{t}herkx'), 'herkx',
        ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'rbeta', 'oC', 'ldc'], v2=False)
routine(each(SDCZ, '{t}trmm'), 'trmm',
        ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'oC', 'ldc'])
routine(each(SDCZ, '{t}trsm'), 'trsm', ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'oB', 'ldb'])
routine(each(SDCZ, '{t}trsmBatched'), 'trsm_batched',
        ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'Ab', 'lda', 'oBb', 'ldb', 'batch'], v2=False)
routine(each(SDCZ, '{t}geam'), 'geam',
        ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'oC', 'ldc'], v2=False)
routine(each(SDCZ, '{t}dgmm'), 'dgmm', ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'oC', 'ldc'], v2=False)


# Each API's names of the types, by kind and the routine's type
REAL = {'S': 'float', 'D': 'double', 'C': 'float', 'Z': 'double'}
APIS = {
    'c': {'prefix': 'cublas', 'handle': 'cublasHandle_t', 'status': 'cublasStatus_t', 'int': 'int',
          'stride': 'long long int', 'op': 'cublasOperation_t', 'fill': 'cublasFillMode_t',
          'diag': 'cublasDiagType_t', 'side': 'cublasSideMode_t',
          'T': {'S': 'float', 'D': 'double', 'C': 'cuComplex', 'Z': 'cuDoubleComplex'}},
    'h': {'prefix': 'hipblas', 'handle': 'hipblasHandle_t', 'status': 'hipblasStatus_t', 'int': 'int',
          'stride': 'hipblasStride', 'op': 'hipblasOperation_t', 'fill': 'hipblasFillMode_t',
          'diag': 'hipblasDiagType_t', 'side': 'hipblasSideMode_t',
          'T': {'S': 'float', 'D': 'double', 'C': 'hipComplex', 'Z': 'hipDoubleComplex'}},
    'r': {'prefix': 'rocblas_', 'handle': 'rocblas_handle', 'status': 'rocblas_status', 'int': 'rocblas_int',
          'stride': 'rocblas_stride', 'op': 'rocblas_operation', 'fill': 'rocblas_fill',
          'diag': 'rocblas_diagonal', 'side': 'rocblas_side',
          'T': {'S': 'float', 'D': 'double', 'C': 'rocblas_float_complex', 'Z': 'rocblas_double_complex'}},
}


def c_type(api, kind, t):
    a = APIS[api]
    T, R = a['T'][t], REAL[t]
    return {'int': a['int'], 'stride': a['stride'], 'op': a['op'], 'fill': a['fill'], 'diag': a['diag'],
            'side': a['side'], 'sT': f'const {T} *', 'sR': f'const {R} *', 'cT': f'const {T} *',
            'cR': f'const {R} *', 'T': f'{T} *', 'R': f'{R} *', 'I': f'{a["int"]} *',
            'cTb': f'const {T} *const ', 'Tb': f'{T} *const '}[kind]


def parameter(api, kind, name, t):
    ctype = c_type(api, kind, t)
    if kind in ('cTb', 'Tb'):
        return f'{ctype}{name}[]'
    return f'{ctype}{name}' if ctype.endswith('*') else f'{ctype} {name}'


def function_name(api, names, roc, t):
    return 'rocblas_' + roc[t] if api == 'r' else APIS[api]['prefix'] + names[t]


def signature(api, names, roc, params, t):
    a = APIS[api]
    args = [f'{a["handle"]} handle'] + [parameter(api, kind, name, t) for kind, name in params]
    return f'{a["status"]} {function_name(api, names, roc, t)}({", ".join(args)})'


def call_argument(kind, name):
    if kind in ('op', 'fill', 'diag', 'side'):
        return f'blas::arg({name})'
    if kind in ('sT', 'sR'):
        return f'blas::scalar_in(*context, {name})'
    if kind in ('cT', 'cR', 'T', 'R', 'I', 'cTb', 'Tb'):
        return f'blas::ptr({name})'
    return name


def wrap(text, indent, width=118):
    """Breaks a declaration after commas, with continuation lines aligned
    to the opening parenthesis"""
    if len(text) + indent <= width:
        return ' ' * indent + text
    open_at = text.index('(') + 1
    lines, line = [], ' ' * indent + text[:open_at]
    for piece in re.split(r'(?<=,) ', text[open_at:]):
        if len(line) + len(piece) + 1 > width and line.strip():
            lines.append(line.rstrip())
            line = ' ' * (indent + open_at) + piece
        else:
            line += ('' if line.endswith('(') else ' ') + piece
    lines.append(line)
    return '\n'.join(lines)


def declarations(api):
    out = []
    for names, roc, core, params, v2, apis in ROUTINES:
        if api in apis:
            for t in names:
                out.append(wrap(signature(api, names, roc, params, t) + ';', 0))
    return '\n'.join(out) + '\n'


def v2_macros():
    out = []
    for names, roc, core, params, v2, apis in ROUTINES:
        if v2:
            out += [f'#define cublas{name}_v2 cublas{name}' for name in names.values()]
    return '\n'.join(out) + '\n'


def definitions():
    out = [LICENSE + '\n// Generated by lib/make-blas-api.py: do not edit.\n']
    for api in 'chr':
        a = APIS[api]
        out.append(f'// {a["prefix"].rstrip("_")}\n')
        for names, roc, core, params, v2, apis in ROUTINES:
            if api not in apis:
                continue
            for t in names:
                args = ', '.join(['*context'] + [call_argument(kind, name) for kind, name in params])
                out.append(wrap(f'extern "C" {signature(api, names, roc, params, t)}', 0) + '\n{\n'
                           f'    CUDA4CPU_BLAS_CONTEXT({a["status"]}, handle);\n'
                           f'    return blas::api_status<{a["status"]}>(blas::{core}({args}));\n}}\n')
    return '\n'.join(out)


def replace_block(path, begin, end, text):
    with open(path) as f:
        content = f.read()
    start = content.index(begin) + len(begin)
    stop = content.index(end)
    content = content[:start] + '\n' + text + content[stop:]
    with open(path, 'w') as f:
        f.write(content)


def main():
    include = os.path.join(ROOT, 'include')
    blocks = [(os.path.join(include, 'cublas_api.h'), "cuBLAS's routines", declarations('c')),
              (os.path.join(include, 'cublas_v2.h'), "cuBLAS's _v2 names", v2_macros()),
              (os.path.join(include, 'hipblas', 'hipblas.h'), "hipBLAS's routines", declarations('h')),
              (os.path.join(include, 'rocblas', 'rocblas.h'), "rocBLAS's routines", declarations('r'))]
    for path, what, text in blocks:
        replace_block(path, f'// cuda4cpu-blas: begin {what} (generated by lib/make-blas-api.py)',
                      f'// cuda4cpu-blas: end {what}', text)
    with open(os.path.join(ROOT, 'lib', 'blas_api.inc'), 'w') as f:
        f.write(definitions())


if __name__ == '__main__':
    sys.exit(main())
