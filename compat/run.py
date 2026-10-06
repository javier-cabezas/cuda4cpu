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

"""Builds and runs real CUDA programs, unmodified, with cuda4cpu.

The programs come from public suites (NVIDIA's cuda-samples, Rodinia), fetched
at the revisions pinned in suites.json. Each program is compiled with
cuda4cpu-c++ (C files with the C compiler), run, and classified:

    passed        exit code 0, the expected output if the manifest gives one
                  ("expect"), and none of the output it rejects ("reject")
    waived        the program skipped itself (cuda-samples' exit code 2)
    failed        nonzero exit code, or the expected output is missing
    crashed       killed by a signal
    timeout       did not finish in time
    build_failed  did not compile or link
    skipped       not run, for the reason the manifest gives ("skip")

The report lists every program with its status and the first error. With
--baseline, programs that passed in the baseline and don't pass now are
regressions, and make the script exit with status 1.

    compat/run.py --driver build/release/tools/cuda4cpu-c++ --work /tmp/compat \\
                  --report report.md --json results.json --baseline compat/baseline.json
"""

import argparse
import concurrent.futures
import fnmatch
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATUSES = ('passed', 'waived', 'failed', 'crashed', 'timeout', 'build_failed', 'skipped')


def fetch(suite, url, revision, directory):
    """Clones url at revision (a tag, branch or commit) into directory."""
    if os.path.isdir(os.path.join(directory, '.git')):
        return
    os.makedirs(directory, exist_ok=True)
    run_git = lambda *args: subprocess.run(['git', '-C', directory, *args], check=True,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'fetching {suite} {revision} from {url}', flush=True)
    run_git('init', '-q')
    run_git('remote', 'add', 'origin', url)
    run_git('fetch', '-q', '--depth', '1', 'origin', revision)
    run_git('checkout', '-q', 'FETCH_HEAD')


def first_error(text):
    """The most useful line of compiler or program output."""
    for pattern in (r'error:.*', r'fatal error:.*', r'undefined reference to .*', r'.*Error.*', r'.*FAIL.*'):
        m = re.search(pattern, text)
        if m:
            line = m.group(0).strip()
            return line[:200]
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1][:200] if lines else ''


def build(program, source_dir, work_dir, driver, cc):
    """Compiles and links program; returns (ok, log)."""
    directory = os.path.join(source_dir, program['dir'])
    out = os.path.join(work_dir, 'build', program['suite'], program['name'])
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    sources = []
    for pattern in program['sources']:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        sources += [m for m in matches if not any(fnmatch.fnmatch(os.path.basename(m), x)
                                                  for x in program.get('exclude', []))]
    if not sources:
        return False, f'no sources match {program["sources"]} in {program["dir"]}'

    includes = [f'-I{os.path.join(source_dir, i)}' for i in program.get('include', [])]
    includes.append(f'-I{directory}')
    flags = ['-O2', '-w'] + includes + program.get('flags', [])

    objects = []
    log = []
    for i, source in enumerate(sources):
        obj = os.path.join(out, f'{i}_{os.path.basename(source)}.o')
        if source.endswith('.c'):
            command = [cc, '-c', '-o', obj, source] + flags + program.get('cflags', [])
        else:
            command = [driver, '-c', '-o', obj, source] + flags
        p = subprocess.run(command, cwd=directory, capture_output=True, text=True)
        log.append(p.stdout + p.stderr)
        if p.returncode != 0:
            return False, '\n'.join(log)
        objects.append(obj)

    exe = os.path.join(out, program['name'])
    p = subprocess.run([driver, '-o', exe] + objects + program.get('libs', []) + ['-lm'],
                       capture_output=True, text=True)
    log.append(p.stdout + p.stderr)
    return p.returncode == 0, '\n'.join(log)


def run_program(program, source_dir, work_dir, timeout):
    exe = os.path.join(work_dir, 'build', program['suite'], program['name'], program['name'])
    run_dir = os.path.join(source_dir, program.get('run_dir', program['dir']))
    start = time.monotonic()
    try:
        p = subprocess.run([exe] + program.get('args', []), cwd=run_dir, capture_output=True,
                           text=True, errors='replace', timeout=program.get('timeout', timeout))
    except subprocess.TimeoutExpired as e:
        return 'timeout', time.monotonic() - start, (e.stdout or '') if isinstance(e.stdout, str) else ''
    elapsed = time.monotonic() - start
    output = p.stdout + p.stderr
    if p.returncode < 0:
        return 'crashed', elapsed, output + f'\nkilled by signal {-p.returncode}'
    if p.returncode == program.get('waived_exit', -1):
        return 'waived', elapsed, output
    if p.returncode != 0:
        return 'failed', elapsed, output + f'\nexit code {p.returncode}'
    expected = program.get('expect')
    if expected and not re.search(expected, output):
        return 'failed', elapsed, output + f'\nexpected output not found: {expected}'
    rejected = program.get('reject')
    if rejected and re.search(rejected, output):
        return 'failed', elapsed, re.search(rejected + '.*', output).group(0)
    return 'passed', elapsed, output


def evaluate(program, source_dir, work_dir, driver, cc, timeout):
    if program.get('skip'):
        return {'status': 'skipped', 'detail': program['skip']}
    start = time.monotonic()
    ok, log = build(program, source_dir, work_dir, driver, cc)
    build_time = time.monotonic() - start
    if not ok:
        return {'status': 'build_failed', 'build_time': build_time, 'detail': first_error(log)}
    status, run_time, output = run_program(program, source_dir, work_dir, timeout)
    detail = '' if status == 'passed' else first_error(output)
    return {'status': status, 'build_time': build_time, 'run_time': run_time, 'detail': detail}


def load_manifest(path):
    with open(path) as f:
        manifest = json.load(f)
    programs = []
    for suite, spec in manifest['suites'].items():
        defaults = spec.get('defaults', {})
        for p in spec['programs']:
            program = dict(defaults)
            program.update(p)
            program['suite'] = suite
            program.setdefault('name', os.path.basename(program['dir']))
            programs.append(program)
    return manifest, programs


def markdown(results, programs, regressions, new_passes):
    counts = {s: sum(1 for r in results.values() if r['status'] == s) for s in STATUSES}
    lines = ['## cuda4cpu compatibility', '',
             f'{counts["passed"]} of {len(results)} programs pass. ' +
             ', '.join(f'{counts[s]} {s.replace("_", " ")}' for s in STATUSES[1:] if counts[s]) + '.', '']
    if regressions:
        lines += ['**Regressions** (passed in the baseline): ' + ', '.join(regressions), '']
    if new_passes:
        lines += ['**Now passing** (update the baseline): ' + ', '.join(new_passes), '']
    lines += ['| Suite | Program | Status | Run time | Note |', '|---|---|---|---|---|']
    for p in programs:
        key = f'{p["suite"]}/{p["name"]}'
        r = results[key]
        run_time = f'{r["run_time"]:.2f} s' if 'run_time' in r else ''
        note = r.get('detail', '').replace('|', '\\|')
        if r['status'] == 'passed' and p.get('verified') is False:
            note = 'runs to completion; the program does not check its results'
        lines.append(f'| {p["suite"]} | {p["name"]} | {r["status"].replace("_", " ")} | {run_time} | {note} |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--driver', required=True, help='path to cuda4cpu-c++')
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'), help='C compiler for .c files')
    parser.add_argument('--work', required=True, help='directory for the suites and the builds')
    parser.add_argument('--manifest', default=os.path.join(HERE, 'suites.json'))
    parser.add_argument('--baseline', help='expected results; regressions make the script fail')
    parser.add_argument('--update-baseline', action='store_true', help='write the results to --baseline')
    parser.add_argument('--report', help='write a Markdown report')
    parser.add_argument('--json', help='write the results as JSON')
    parser.add_argument('--filter', default='*', help='only programs whose suite/name match this pattern')
    parser.add_argument('--jobs', type=int, default=2, help='programs built and run at the same time')
    parser.add_argument('--timeout', type=float, default=300, help='seconds each program may run')
    args = parser.parse_args(argv)

    driver = os.path.abspath(args.driver)
    manifest, programs = load_manifest(args.manifest)
    programs = [p for p in programs if fnmatch.fnmatch(f'{p["suite"]}/{p["name"]}', args.filter)]

    sources = {}
    for suite, spec in manifest['suites'].items():
        if any(p['suite'] == suite for p in programs):
            sources[suite] = os.path.join(args.work, 'src', suite)
            fetch(suite, spec['url'], spec['revision'], sources[suite])

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(evaluate, p, sources[p['suite']], args.work, driver, args.cc, args.timeout):
                   f'{p["suite"]}/{p["name"]}' for p in programs}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            print(f'{key:55} {results[key]["status"]:13} {results[key].get("detail", "")[:80]}', flush=True)

    regressions, new_passes = [], []
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline) as f:
            baseline = json.load(f)
        for key, r in results.items():
            before = baseline.get(key)
            if before == 'passed' and r['status'] not in ('passed', 'waived'):
                regressions.append(key)
            elif before != 'passed' and r['status'] == 'passed':
                new_passes.append(key)

    report = markdown(results, programs, regressions, new_passes)
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
    if args.update_baseline and args.baseline:
        with open(args.baseline, 'w') as f:
            json.dump({k: results[k]['status'] for k in sorted(results)}, f, indent=2)
            f.write('\n')

    passed = sum(1 for r in results.values() if r['status'] == 'passed')
    print(f'\n{passed} of {len(results)} programs pass')
    if regressions:
        print('regressions: ' + ', '.join(regressions))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
