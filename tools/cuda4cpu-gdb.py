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

Commands:

    cuda4cpu                     show the current CUDA thread and block
    cuda4cpu threads             list the CUDA threads of the current block and their states
    cuda4cpu thread <x> [y] [z]  switch to another started CUDA thread of the block
    cuda4cpu thread              switch back to the running CUDA thread
"""

import os
import gdb

BUILTINS = 'cuda4cpu::thread_block::Vars_'
CURRENT_BLOCK = 'cuda4cpu::thread_block::Current_'

WARP_OPS = {
    0: '__syncwarp',
    1: '__shfl_*_sync',
    2: 'warp vote',
    3: '__match_any_sync',
    4: '__match_all_sync',
    5: '__activemask',
}

CALLEE_SAVED = ('r15', 'r14', 'r13', 'r12', 'rbx', 'rbp', 'rip')

# Saved register state of OS threads currently inspecting another CUDA fiber,
# keyed by (inferior.num, ptid)
_saved_threads = {}


def builtins():
    try:
        return gdb.parse_and_eval(BUILTINS)
    except gdb.error as e:
        raise gdb.GdbError(f'cuda4cpu: cannot read the CUDA built-in variables ({e}); '
                           'is the program stopped in a thread that runs a kernel?')


def current_block():
    try:
        ptr = gdb.parse_and_eval(CURRENT_BLOCK)
    except gdb.error as e:
        raise gdb.GdbError(f'cuda4cpu: cannot read the current thread block ({e}); '
                           'is the program stopped in a thread that runs a kernel?')
    if int(ptr) == 0:
        raise gdb.GdbError('cuda4cpu: the selected OS thread is not running a kernel')
    return ptr.dereference()


def _first_ptr(container):
    """Data pointer at offset 0 of a std::unique_ptr<T> or std::vector<T>."""
    elem_type = container.type.strip_typedefs().template_argument(0)
    return container.address.cast(elem_type.pointer().pointer()).dereference()


def _fibers(block):
    return _first_ptr(block['fibers_']).dereference()


def _coords(linear, bdim):
    bx, by = int(bdim['x']), int(bdim['y'])
    return (linear % bx, (linear // bx) % by, linear // (bx * by))


def _format_coords(c):
    return f'({c[0]}, {c[1]}, {c[2]})'


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


def _thread_key():
    t = gdb.selected_thread()
    if t is None:
        raise gdb.GdbError('cuda4cpu: no thread selected')
    return (gdb.selected_inferior().num, t.ptid)


def _running_tid(block, v):
    saved = _saved_threads.get(_thread_key())
    if saved is not None:
        return saved['running_tid']
    if bool(block['direct_']) and int(block['order_']) == 0:
        t, b = v['thread_idx'], v['block_dim']
        return (int(t['z']) * int(b['y']) + int(t['y'])) * int(b['x']) + int(t['x'])
    return int(block['cur_'])


def _thread_states(block, f, nthreads, running_tid):
    """Returns a list of human-readable state strings for threads 0..nthreads-1."""
    if bool(block['direct_']):
        order_ptr = int(block['order_'])
        if order_ptr == 0:
            return ['returned' if t < running_tid else ('running' if t == running_tid else 'not started')
                    for t in range(nthreads)]
        order = block['order_']
        pos = int(block['pos_'])
        states = ['not started'] * nthreads
        for p in range(pos):
            states[int(order[p])] = 'returned'
        states[running_tid] = 'running'
        return states

    warps = _first_ptr(f['warps'])
    warp_kind = _first_ptr(f['warp_kind'])
    warp_mask = _first_ptr(f['warp_mask'])
    bfile_ptr = f['barrier_file']
    bline = int(f['barrier_line'])
    bloc = ''
    if int(bfile_ptr) != 0 and bline > 0:
        bloc = f" ({os.path.basename(bfile_ptr.string())}:{bline})"

    states = []
    for t in range(nthreads):
        if t == running_tid:
            states.append('running')
            continue
        w = warps[t // 32]
        bit = 1 << (t % 32)
        if (int(w['live']) & bit) == 0:
            states.append('returned')
        elif (int(w['block_wait']) & bit) != 0:
            states.append(f'at __syncthreads(){bloc}')
        elif (int(w['tile_wait']) & bit) != 0:
            states.append('at tile.sync()')
        elif (int(w['converge_wait']) & bit) != 0:
            states.append('at __activemask()')
        elif (int(w['warp_wait']) & bit) != 0:
            op = WARP_OPS.get(int(warp_kind[t]), 'warp op')
            mask = int(warp_mask[t]) & 0xffffffff
            states.append(f'at {op} (mask 0x{mask:08x})')
        elif (int(w['started']) & bit) != 0:
            states.append('runnable')
        else:
            states.append('not started')
    return states


def _ror64(val, bits):
    val &= 0xffffffffffffffff
    return ((val >> bits) | (val << (64 - bits))) & 0xffffffffffffffff


def _load_fiber_registers(ctx_entry):
    fields = {field.name for field in ctx_entry.type.fields()}
    if 'sp' in fields:
        u64_ptr = gdb.lookup_type('unsigned long long').pointer()
        sp = ctx_entry['sp'].cast(u64_ptr)
        if int(sp) == 0:
            raise gdb.GdbError('cuda4cpu: fiber has no saved stack pointer')
        regs = {reg: int(sp[i]) for i, reg in enumerate(CALLEE_SAVED)}
        regs['rsp'] = int(sp) + 8 * len(CALLEE_SAVED)
        return regs

    # Portable context (jmp_buf on x86-64 glibc: RBX=0, RBP=1, R12..R15=2..5, RSP=6, PC=7)
    jb = ctx_entry['buf'][0]['__jmpbuf']
    u64_ptr = gdb.lookup_type('unsigned long long').pointer()
    fs_base = int(gdb.parse_and_eval('$fs_base'))
    guard = int(gdb.Value(fs_base + 0x30).cast(u64_ptr).dereference())
    return {
        'rbx': int(jb[0]) & 0xffffffffffffffff,
        'rbp': _ror64(int(jb[1]), 17) ^ guard,
        'r12': int(jb[2]) & 0xffffffffffffffff,
        'r13': int(jb[3]) & 0xffffffffffffffff,
        'r14': int(jb[4]) & 0xffffffffffffffff,
        'r15': int(jb[5]) & 0xffffffffffffffff,
        'rsp': _ror64(int(jb[6]), 17) ^ guard,
        'rip': _ror64(int(jb[7]), 17) ^ guard,
    }


def _apply_registers(regs, coords):
    gdb.newest_frame().select()
    for reg, val in regs.items():
        gdb.execute(f'set ${reg} = 0x{val:x}', to_string=True)
    gdb.execute(f'set {BUILTINS}.thread_idx.x = {coords[0]}', to_string=True)
    gdb.execute(f'set {BUILTINS}.thread_idx.y = {coords[1]}', to_string=True)
    gdb.execute(f'set {BUILTINS}.thread_idx.z = {coords[2]}', to_string=True)
    gdb.newest_frame().select()


def _select_kernel_frame():
    frame = gdb.newest_frame()
    while frame is not None:
        name = frame.name() or ''
        if not name.startswith('cuda4cpu::thread_block::') and name != 'cuda4cpu_switch_context':
            frame.select()
            return frame
        frame = frame.older()
    return gdb.newest_frame()


def _restore_thread(key):
    saved = _saved_threads.pop(key, None)
    if saved is None:
        return False
    _apply_registers(saved['regs'], saved['coords'])
    return True


def _restore_all_before_continue(event=None):
    if not _saved_threads:
        return
    orig_thread = gdb.selected_thread()
    for (inf_num, ptid), saved in list(_saved_threads.items()):
        for t in gdb.selected_inferior().threads():
            if t.ptid == ptid and t.is_stopped():
                t.switch()
                _apply_registers(saved['regs'], saved['coords'])
                break
        _saved_threads.pop((inf_num, ptid), None)
    if orig_thread is not None and orig_thread.is_valid():
        orig_thread.switch()


for _cmd in ('continue', 'step', 'next', 'stepi', 'nexti', 'finish', 'until'):
    gdb.execute(f'define hook-{_cmd}\npython _restore_all_before_continue()\nend', to_string=True)


class Cuda4cpuCommand(gdb.Command):
    """Inspect the running CUDA block and switch between its CUDA threads.

    cuda4cpu                     show the current CUDA thread and block
    cuda4cpu threads             list the CUDA threads of the current block and their states
    cuda4cpu thread <x> [y] [z]  switch to another started CUDA thread of the block
    cuda4cpu thread              switch back to the running CUDA thread
    """

    def __init__(self):
        super().__init__('cuda4cpu', gdb.COMMAND_DATA)

    def invoke(self, argument, from_tty):
        args = gdb.string_to_argv(argument)
        if not args:
            self._status()
        elif args[0] == 'threads':
            self._threads()
        elif args[0] == 'thread':
            self._switch_thread(args[1:])
        else:
            raise gdb.GdbError(f'cuda4cpu: unknown subcommand "{args[0]}" '
                               '(expected "threads" or "thread")')

    def _status(self):
        v = builtins()
        print(f"thread {dim(v['thread_idx'])} of block {dim(v['block_idx'])}, "
              f"lane {lane(v)}; blocks of {dim(v['block_dim'])} threads, "
              f"grid of {dim(v['grid_dim'])} blocks")

    def _threads(self):
        v = builtins()
        block = current_block()
        f = _fibers(block)
        bdim = v['block_dim']
        nthreads = int(block['nthreads_'])
        running_tid = _running_tid(block, v)
        selected_t = v['thread_idx']
        selected_tid = ((int(selected_t['z']) * int(bdim['y']) + int(selected_t['y'])) *
                        int(bdim['x']) + int(selected_t['x']))

        states = _thread_states(block, f, nthreads, running_tid)
        live_count = sum(1 for s in states if s != 'returned')
        print(f"block {dim(v['block_idx'])}, {nthreads} threads ({live_count} live):")

        t = 0
        while t < nthreads:
            state = states[t]
            # Keep the running and selected threads on their own lines
            if t == running_tid or t == selected_tid:
                end = t
            else:
                end = t
                while (end + 1 < nthreads and states[end + 1] == state and
                       end + 1 != running_tid and end + 1 != selected_tid):
                    end += 1
            mark = '*' if t == selected_tid else ' '
            if t == end:
                c = _format_coords(_coords(t, bdim))
                print(f"{mark} thread {c}, lane {t % 32}: {state}")
            else:
                c0 = _format_coords(_coords(t, bdim))
                c1 = _format_coords(_coords(end, bdim))
                print(f"  threads {c0}..{c1}: {state}")
            t = end + 1

    def _switch_thread(self, args):
        v = builtins()
        block = current_block()
        bdim = v['block_dim']
        bx, by, bz = int(bdim['x']), int(bdim['y']), int(bdim['z'])
        key = _thread_key()
        running_tid = _running_tid(block, v)

        if not args:
            _restore_thread(key)
            self._status()
            return

        if len(args) > 3:
            raise gdb.GdbError('usage: cuda4cpu thread [<x> [<y> [<z>]]]')
        try:
            x = int(args[0], 0)
            y = int(args[1], 0) if len(args) > 1 else 0
            z = int(args[2], 0) if len(args) > 2 else 0
        except ValueError:
            raise gdb.GdbError('usage: cuda4cpu thread [<x> [<y> [<z>]]]')

        if not (0 <= x < bx and 0 <= y < by and 0 <= z < bz):
            raise gdb.GdbError(f'cuda4cpu: thread ({x}, {y}, {z}) is outside blockDim {dim(bdim)}')

        target_tid = (z * by + y) * bx + x
        if target_tid == running_tid:
            _restore_thread(key)
            self._status()
            return

        if bool(block['direct_']):
            raise gdb.GdbError('cuda4cpu: block is in direct mode; only the running thread has a stack')

        f = _fibers(block)
        warps = _first_ptr(f['warps'])
        w = warps[target_tid // 32]
        bit = 1 << (target_tid % 32)
        if (int(w['live']) & bit) == 0:
            raise gdb.GdbError(f'cuda4cpu: thread ({x}, {y}, {z}) has already returned')
        if (int(w['started']) & bit) == 0:
            raise gdb.GdbError(f'cuda4cpu: thread ({x}, {y}, {z}) has not started yet')

        if key not in _saved_threads:
            gdb.newest_frame().select()
            orig_regs = {reg: int(gdb.parse_and_eval(f'${reg}')) & 0xffffffffffffffff
                         for reg in CALLEE_SAVED + ('rsp',)}
            _saved_threads[key] = {
                'running_tid': running_tid,
                'coords': _coords(running_tid, bdim),
                'regs': orig_regs,
            }

        ctx = _first_ptr(f['ctx'])
        regs = _load_fiber_registers(ctx[target_tid])
        _apply_registers(regs, (x, y, z))
        frame = _select_kernel_frame()
        sal = frame.find_sal()
        where = f' at {os.path.basename(sal.symtab.filename)}:{sal.line}' if sal.symtab else ''
        print(f"switched to thread ({x}, {y}, {z}) of block {dim(v['block_idx'])}, "
              f"lane {target_tid % 32}{where}")


Cuda4cpuCommand()
