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
import struct

import gdb
from gdb.unwinder import Unwinder, register_unwinder

BUILTINS = 'cuda4cpu::thread_block::Vars_'
CURRENT_BLOCK = 'cuda4cpu::thread_block::Current_'

# Names of thread_block::fibers::op's enumerators, as reports call them
WARP_OP_NAMES = {
    'op_sync': '__syncwarp',
    'op_shuffle': '__shfl_*_sync',
    'op_ballot': 'warp vote',
    'op_match_any': '__match_any_sync',
    'op_match_all': '__match_all_sync',
    'op_converge': '__activemask',
}

# The frame that cuda4cpu_switch_context pushes, from its stack pointer up
SWITCH_FRAME = ('r15', 'r14', 'r13', 'r12', 'rbx', 'rbp', 'rip')

# The CUDA thread that `cuda4cpu thread` selected in each OS thread, keyed by
# (inferior number, ptid). Nothing in the program changes: an unwinder makes
# the frames of the selected CUDA thread's stack the callers of the frame
# the OS thread stopped in, which the backtrace hides. Resuming forgets them.
_selected = {}


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
        selected = _selection()
        if self.field == 'thread_idx' and selected is not None:
            value = _dim3_value(selected['coords'], value.type)
        if component is None:
            return value
        name = component.string()
        if name not in ('x', 'y', 'z'):
            raise gdb.GdbError(f'cuda4cpu: no component "{name}"')
        return value[name]


for name, field in (('threadIdx', 'thread_idx'), ('blockIdx', 'block_idx'),
                    ('blockDim', 'block_dim'), ('gridDim', 'grid_dim')):
    Builtin(name, field)


def thread_coords(v):
    """The index of the selected CUDA thread in its block."""
    selected = _selection()
    if selected is not None:
        return selected['coords']
    t = v['thread_idx']
    return (int(t['x']), int(t['y']), int(t['z']))


def _linear(coords, bdim):
    return (coords[2] * int(bdim['y']) + coords[1]) * int(bdim['x']) + coords[0]


def lane(v):
    """The lane of the CUDA thread, from its index in the block, as lane_id() computes it."""
    return _linear(thread_coords(v), v['block_dim']) % 32


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
        return None
    return (gdb.selected_inferior().num, t.ptid)


def _selection():
    """What `cuda4cpu thread` selected in the selected OS thread, or None."""
    if not _selected:
        return None
    return _selected.get(_thread_key())


def _forget(key=None):
    """Goes back to the running CUDA thread, in one OS thread or in all of them."""
    if key is None:
        _selected.clear()
    else:
        _selected.pop(key, None)
    gdb.invalidate_cached_frames()


def _dim3_value(coords, value_type):
    return gdb.Value(struct.pack('=3I', *coords), value_type)


_op_names = None


def _warp_op_name(kind):
    global _op_names
    if _op_names is None:
        _op_names = {}
        try:
            for field in gdb.lookup_type('cuda4cpu::thread_block::fibers::op').fields():
                name = field.name.split('::')[-1]
                _op_names[field.enumval] = WARP_OP_NAMES.get(name, name)
        except gdb.error:
            pass
    return _op_names.get(kind, 'warp operation')


def _running_tid(block, v):
    """The CUDA thread that the OS thread runs (direct mode in the default order
    doesn't keep cur_ up to date)."""
    if bool(block['direct_']) and int(block['order_']) == 0:
        t = v['thread_idx']
        return _linear((int(t['x']), int(t['y']), int(t['z'])), v['block_dim'])
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
            mask = int(warp_mask[t]) & 0xffffffff
            states.append(f'at {_warp_op_name(int(warp_kind[t]))} (mask 0x{mask:08x})')
        elif (int(w['started']) & bit) != 0:
            states.append('runnable')
        else:
            states.append('not started')
    return states


def _ror64(val, bits):
    val &= 0xffffffffffffffff
    return ((val >> bits) | (val << (64 - bits))) & 0xffffffffffffffff


def _fiber_registers(ctx_entry):
    """The registers of a suspended fiber, as they are once its switch returns."""
    u64_ptr = gdb.lookup_type('unsigned long long').pointer()
    fields = {field.name for field in ctx_entry.type.fields()}
    if 'sp' in fields:
        sp = ctx_entry['sp'].cast(u64_ptr)
        if int(sp) == 0:
            raise gdb.GdbError('cuda4cpu: the fiber has no saved stack pointer')
        regs = {reg: int(sp[i]) for i, reg in enumerate(SWITCH_FRAME)}
        regs['rsp'] = int(sp) + 8 * len(SWITCH_FRAME)
        return regs

    # Portable context: a jmp_buf (x86-64 glibc: RBX=0, RBP=1, R12..R15=2..5,
    # RSP=6, PC=7), with RBP, RSP and PC mangled with the pointer guard
    jb = ctx_entry['buf'][0]['__jmpbuf']
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


class _FrameId:
    def __init__(self, sp, pc):
        self.sp = sp
        self.pc = pc


class FiberUnwinder(Unwinder):
    """Makes the selected CUDA thread's suspended frames the callers of the
    frame its OS thread stopped in. The program's registers stay as they are,
    so resuming it in any way (including from an IDE, through MI) is safe."""

    def __init__(self):
        super().__init__('cuda4cpu fibers')

    def __call__(self, pending_frame):
        selected = _selection()
        if selected is None:
            return None
        sp = pending_frame.read_register('rsp')
        pc = pending_frame.read_register('rip')
        if int(sp) != selected['stop_sp'] or int(pc) != selected['stop_pc']:
            return None
        # gdb stops a backtrace whose caller frame is below its callee, and
        # fiber stacks can be anywhere: the stop frame's ID takes the fiber's
        # stack pointer, so that its first frame is above it
        info = pending_frame.create_unwind_info(_FrameId(selected['regs']['rsp'], pc))
        for reg, val in selected['regs'].items():
            info.add_saved_register(reg, gdb.Value(val).cast(pending_frame.read_register(reg).type))
        return info


class HideStopFrames:
    """Hides the frames the OS thread stopped in from the backtrace of a
    selected CUDA thread."""

    def __init__(self):
        self.name = 'cuda4cpu fibers'
        self.priority = 100
        self.enabled = True
        gdb.frame_filters[self.name] = self

    def filter(self, frames):
        selected = _selection()
        if selected is None:
            return frames
        first = selected['first_level']
        return (f for f in frames if f.inferior_frame().level() >= first)


register_unwinder(None, FiberUnwinder(), replace=True)
HideStopFrames()
gdb.events.cont.connect(lambda event: _forget())
gdb.events.exited.connect(lambda event: _forget())


def _select_kernel_frame(frame):
    """The first frame from frame on that isn't the scheduler's."""
    first = frame
    while frame is not None:
        name = frame.name() or ''
        if not name.startswith('cuda4cpu::thread_block::') and name != 'cuda4cpu_switch_context':
            frame.select()
            return frame
        frame = frame.older()
    first.select()
    return first


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
        print(f"thread {_format_coords(thread_coords(v))} of block {dim(v['block_idx'])}, "
              f"lane {lane(v)}; blocks of {dim(v['block_dim'])} threads, "
              f"grid of {dim(v['grid_dim'])} blocks")

    def _threads(self):
        v = builtins()
        block = current_block()
        f = _fibers(block)
        bdim = v['block_dim']
        nthreads = int(block['nthreads_'])
        running_tid = _running_tid(block, v)
        selected_tid = _linear(thread_coords(v), bdim)

        states = _thread_states(block, f, nthreads, running_tid)
        live_count = sum(1 for s in states if s != 'returned')
        print(f"block {dim(v['block_idx'])}, {nthreads} threads ({live_count} live):")

        t = 0
        while t < nthreads:
            state = states[t]
            # Keep the running and selected threads on their own lines
            end = t
            if t != running_tid and t != selected_tid:
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

        if not args:
            if key in _selected:
                _forget(key)
                gdb.newest_frame().select()
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
        if target_tid == _running_tid(block, v):
            if key in _selected:
                _forget(key)
                gdb.newest_frame().select()
            self._status()
            return

        if bool(block['direct_']):
            raise gdb.GdbError('cuda4cpu: block is in direct mode; only the running thread has a stack')

        f = _fibers(block)
        w = _first_ptr(f['warps'])[target_tid // 32]
        bit = 1 << (target_tid % 32)
        if (int(w['live']) & bit) == 0:
            raise gdb.GdbError(f'cuda4cpu: thread ({x}, {y}, {z}) has already returned')
        if (int(w['started']) & bit) == 0:
            raise gdb.GdbError(f'cuda4cpu: thread ({x}, {y}, {z}) has not started yet')

        _forget(key)
        stop = gdb.newest_frame()
        if 'x86-64' not in stop.architecture().name():
            raise gdb.GdbError('cuda4cpu: switching CUDA threads is only supported on x86-64')
        stop_sp, stop_pc = int(stop.read_register('rsp')), int(stop.pc())
        _selected[key] = {
            'coords': (x, y, z),
            'regs': _fiber_registers(_first_ptr(f['ctx'])[target_tid]),
            'stop_sp': stop_sp,
            'stop_pc': stop_pc,
            'first_level': 0,
        }
        gdb.invalidate_cached_frames()

        # The selected thread's frames start after the frames at the stop
        # (several, if functions are inlined there)
        frame = gdb.newest_frame()
        while (frame is not None and int(frame.read_register('rsp')) == stop_sp and
               int(frame.pc()) == stop_pc):
            frame = frame.older()
        if frame is None:
            _forget(key)
            raise gdb.GdbError(f'cuda4cpu: cannot unwind the stack of thread ({x}, {y}, {z})')
        _selected[key]['first_level'] = frame.level()

        frame = _select_kernel_frame(frame)
        sal = frame.find_sal()
        where = f' at {os.path.basename(sal.symtab.filename)}:{sal.line}' if sal.symtab else ''
        print(f"switched to thread ({x}, {y}, {z}) of block {dim(v['block_idx'])}, "
              f"lane {target_tid % 32}{where}")


Cuda4cpuCommand()
