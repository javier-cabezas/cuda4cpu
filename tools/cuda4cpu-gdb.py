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

"""gdb support for cuda4cpu. Load it with: source <prefix>/share/cuda4cpu/cuda4cpu-gdb.py

Convenience functions for the CUDA thread that the selected OS thread runs:

    $threadIdx()  $blockIdx()  $blockDim()  $gridDim()  $laneid()

With a dimension, they return that component, which also works in breakpoint
conditions (where gdb can't access the members of a function's result):

    (gdb) break kernel.cu:42 if $threadIdx("x") == 6 && $blockIdx("x") == 7
    (gdb) print $threadIdx()

and the command `cuda4cpu`, which shows the current CUDA thread and block.
"""

import gdb

BUILTINS = 'cuda4cpu::thread_block::Vars_'


def builtins():
    try:
        return gdb.parse_and_eval(BUILTINS)
    except gdb.error as e:
        raise gdb.GdbError(f'cuda4cpu: cannot read the CUDA built-in variables ({e}); '
                           'is the program stopped in a thread that runs a kernel?')


class Builtin(gdb.Function):
    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def invoke(self, component=None):
        value = builtins()[self.field]
        if component is None:
            return value
        name = component.string()
        if name not in ('x', 'y', 'z'):
            raise gdb.GdbError(f'cuda4cpu: no component "{name}"')
        return value[name]


for name, field in (('threadIdx', 'thread_idx'), ('blockIdx', 'block_idx'),
                    ('blockDim', 'block_dim'), ('gridDim', 'grid_dim')):
    Builtin(name, field)


def lane(v):
    """The lane of the CUDA thread, from its index in the block, as lane_id() computes it."""
    t, b = v['thread_idx'], v['block_dim']
    return ((int(t['z']) * int(b['y']) + int(t['y'])) * int(b['x']) + int(t['x'])) % 32


class Laneid(gdb.Function):
    def __init__(self):
        super().__init__('laneid')

    def invoke(self):
        return lane(builtins())


Laneid()


def dim(value):
    return f"({int(value['x'])}, {int(value['y'])}, {int(value['z'])})"


class Cuda4cpuCommand(gdb.Command):
    """Shows the CUDA thread that the selected OS thread runs."""

    def __init__(self):
        super().__init__('cuda4cpu', gdb.COMMAND_DATA)

    def invoke(self, argument, from_tty):
        v = builtins()
        print(f"thread {dim(v['thread_idx'])} of block {dim(v['block_idx'])}, "
              f"lane {lane(v)}; blocks of {dim(v['block_dim'])} threads, "
              f"grid of {dim(v['grid_dim'])} blocks")


Cuda4cpuCommand()
