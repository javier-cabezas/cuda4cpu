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

"""Sets the release's version everywhere it appears: CMake's project
version, the MCP Registry entry and its image, Glama's image, and the pins
of the latest release in the README, the docs, the action and the example
project.

Usage: packaging/set-version.py X.Y.Z   writes X.Y.Z in every place
       packaging/set-version.py --check fails unless every place has
                                        CMake's version (CI runs it)
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = r'\d+\.\d+\.\d+'

# Each file, and the patterns in it whose group is the version. Every
# pattern must match at least once.
PLACES = {
    'CMakeLists.txt': [r'^project\(cuda4cpu\n +VERSION (%s)$'],
    'server.json': [r'^  "version": "(%s)",$', r'"ghcr\.io/javier-cabezas/cuda4cpu:(%s)"'],
    'glama/Dockerfile': [r'^FROM ghcr\.io/javier-cabezas/cuda4cpu:(%s)$'],
    'action.yml': [r'such as v(%s),', r'javier-cabezas/cuda4cpu@v(%s)\)'],
    'examples/cmake-project/CMakeLists.txt': [r'GIT_TAG v(%s)\)'],
    'README.md': [r'GIT_TAG v(%s)\)', r'uses: javier-cabezas/cuda4cpu@v(%s)$'],
    'docs/usage.md': [r'uses: javier-cabezas/cuda4cpu@v(%s)$'],
    'docs/debugging.md': [r'uses: javier-cabezas/cuda4cpu@v(%s)$'],
}


def occurrences():
    """Each place's file, pattern and the versions it holds."""
    for path, patterns in PLACES.items():
        with open(os.path.join(ROOT, path)) as f:
            content = f.read()
        for pattern in patterns:
            found = re.findall(pattern % VERSION, content, re.MULTILINE)
            yield path, pattern, found


def check():
    cmake = next(found for path, _, found in occurrences() if path == 'CMakeLists.txt')
    if len(cmake) != 1:
        print('set-version.py: CMakeLists.txt: no single project VERSION', file=sys.stderr)
        return 1
    expected = cmake[0]
    wrong = []
    for path, pattern, found in occurrences():
        if not found:
            wrong.append(f'{path}: no version where expected ({pattern % "X.Y.Z"})')
        wrong += [f"{path}: {v}, not CMake's {expected}" for v in found if v != expected]
    # The MCP Registry reads server.json's fields, not its text
    try:
        with open(os.path.join(ROOT, 'server.json')) as f:
            server = json.load(f)
        if server['version'] != expected:
            wrong.append(f"server.json: version {server['version']}, not CMake's {expected}")
        image = f'ghcr.io/javier-cabezas/cuda4cpu:{expected}'
        if server['packages'][0]['identifier'] != image:
            wrong.append(f"server.json: the first package is {server['packages'][0]['identifier']}, not {image}")
    except (ValueError, KeyError, IndexError, TypeError) as e:
        wrong.append(f'server.json: not an MCP server entry ({e!r})')
    for line in wrong:
        print(f'set-version.py: {line}', file=sys.stderr)
    if wrong:
        print('set-version.py: run packaging/set-version.py X.Y.Z', file=sys.stderr)
    return 1 if wrong else 0


def set_version(version):
    # Every file's new contents first, so that a place not found leaves them
    # all as they were
    files = {}
    for path, patterns in PLACES.items():
        with open(os.path.join(ROOT, path)) as f:
            content = f.read()
        for pattern in patterns:
            content, count = re.subn(pattern % VERSION,
                                     lambda m: m.group(0)[:m.start(1) - m.start(0)] + version +
                                     m.group(0)[m.end(1) - m.start(0):],
                                     content, flags=re.MULTILINE)
            if not count:
                sys.exit(f'set-version.py: {path}: no version where expected ({pattern % "X.Y.Z"}); '
                         'nothing was changed')
        files[path] = content
    for path, content in files.items():
        with open(os.path.join(ROOT, path), 'w') as f:
            f.write(content)
    return 0


def main():
    args = sys.argv[1:]
    if args == ['--check']:
        return check()
    if len(args) != 1 or not re.fullmatch(VERSION, args[0]):
        sys.exit(__doc__.strip().split('\n\n')[-1])
    return set_version(args[0])


if __name__ == '__main__':
    sys.exit(main())
