#!/usr/bin/env python3
"""Measure selected, identity-bound guest threads on different CPU sets.

The owning debug-run supervisor applies every policy, including newly created
threads. The renderer and all unselected workers retain their original policy.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import uuid

from debug_cpu_policy import (atomic_json, parse_cpus, renderer_for_request, start_ticks,
                              thread_overrides_for_request, validate_request)

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('abba_switch', Path(__file__).with_name('abba-switch.py'))
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def check_process(receipt_path, identity):
    receipt = json.loads(receipt_path.read_text())
    ab.require_stationary_receipt(receipt)
    pid = identity['pid']
    if (receipt['pid'] != pid or start_ticks(pid) != identity['start_ticks'] or
            hashlib.file_digest((Path('/proc') / str(pid) / 'exe').open('rb'), 'sha256').hexdigest()
            != identity['binary_sha256']):
        raise RuntimeError('Retained process identity changed')


def await_policy(evidence, request, allowed, default_renderer_cpu):
    identity = request['identity']
    renderer_cpu = renderer_for_request(request, identity, allowed, default_renderer_cpu)
    workers = validate_request(request, identity, allowed, renderer_cpu)
    overrides = thread_overrides_for_request(request, allowed, renderer_cpu)
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        ack = json.loads((evidence / 'cpu-policy-applied.json').read_text())
        if ack.get('identity') == identity and ack.get('token') == request['token']:
            if ack.get('error'):
                raise RuntimeError(ack['error'])
            # Older supervisors silently ignore unknown request fields.
            if overrides and ack.get('thread_overrides') != request['thread_overrides']:
                raise RuntimeError('Supervisor does not acknowledge per-thread policies; restart with current debug-run')
            names = ack['thread_names']
            renderer = request.get('renderer')
            renderers = ([renderer['tid']] if renderer else
                         [int(tid) for tid, name in names.items() if name == 'Kyty.Gpu'])
            if len(renderers) != 1 or not ack['affinities']:
                raise RuntimeError('Renderer identity or applied masks are absent')
            for tid_text, actual in ack['affinities'].items():
                tid = int(tid_text)
                expected = ({renderer_cpu} if tid == renderers[0] else
                            overrides[tid]['cpus'] if tid in overrides else workers)
                if set(actual) != expected or os.sched_getaffinity(tid) != expected:
                    raise RuntimeError(f'Actual affinity changed for thread {tid}')
            for tid, value in overrides.items():
                if str(tid) not in names or start_ticks(tid) != value['start_ticks']:
                    raise RuntimeError('Selected guest thread ended or changed identity')
            return ack
        time.sleep(0.1)
    raise RuntimeError('Supervisor did not acknowledge CPU policy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--threads', required=True, help='Observed guest TIDs, comma-separated')
    parser.add_argument('--cpus', required=True, help='Candidate CPU set; state 0 keeps original policy')
    parser.add_argument('--sequence', default='0,1,1,0,1,0,0,1')
    parser.add_argument('--seconds', type=float, default=25)
    parser.add_argument('--settle', type=float, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sequence = [int(x) for x in args.sequence.split(',')]
    tids = [int(x) for x in args.threads.split(',')]
    if (not sequence or any(x not in (0, 1) for x in sequence) or
            not tids or len(tids) != len(set(tids)) or args.settle < 0 or not 2 <= args.seconds <= 60):
        parser.error('Use unique guest TIDs, states 0/1 and measurement windows from 2 to 60 seconds')
    receipt = json.loads(args.receipt.read_text())
    identity = {'pid': receipt['pid'], 'start_ticks': receipt['pid_start_ticks'],
                'binary_sha256': receipt['binary_sha256']}
    check_process(args.receipt, identity)
    evidence = Path(receipt['evidence'])
    policy_path = evidence / 'cpu-policy.json'
    original_bytes = policy_path.read_bytes()
    original = json.loads(original_bytes)
    config = json.loads(Path(receipt['config']).read_text())
    allowed = set(config['cpu_affinity'])
    renderer_cpu = int(config['command'][config['command'].index('--render-cpu') + 1])
    if original.get('identity') != identity or original.get('thread_overrides'):
        raise RuntimeError('Expected the original process policy without prior thread overrides')
    overrides = []
    for tid in tids:
        if not (Path('/proc') / str(identity['pid']) / 'task' / str(tid)).is_dir():
            raise RuntimeError(f'Thread {tid} is not in the retained process')
        overrides.append({'tid': tid, 'start_ticks': start_ticks(tid), 'cpus': sorted(parse_cpus(args.cpus))})
    candidate = copy.deepcopy(original)
    candidate['thread_overrides'] = overrides
    thread_overrides_for_request(candidate, allowed, renderer_cpu)
    await_policy(evidence, original, allowed, renderer_cpu)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'original-policy.json').write_bytes(original_bytes)
    report = {'receipt': str(args.receipt.resolve()), 'identity': identity, 'overrides': overrides,
              'sequence': sequence, 'seconds': args.seconds, 'settle': args.settle, 'windows': []}
    def save():
        atomic_json(args.out / 'report.json', report)
    try:
        for index, state in enumerate(sequence):
            check_process(args.receipt, identity)
            request = copy.deepcopy(candidate if state else original)
            request['token'] = str(uuid.uuid4())
            atomic_json(policy_path, request)
            await_policy(evidence, request, allowed, renderer_cpu)
            time.sleep(args.settle)
            before = await_policy(evidence, request, allowed, renderer_cpu)
            label = f'{index}-thread-affinity-{state}'
            phase = ab.measure(identity['pid'], args.seconds, label, args.out)
            after = await_policy(evidence, request, allowed, renderer_cpu)
            report['windows'].append({'index': index, 'value': state, 'frames': phase['frames'],
                                      'elapsed_ns': phase['end_ns'] - phase['start_ns'],
                                      'fps': phase['measured_fps'], 'before': before, 'after': after})
            report['combined'] = ab.combine(report['windows'])
            save()
            print(f"{label}: {phase['measured_fps']:.4f} FPS", flush=True)
    finally:
        # Restore exact bytes (including the original request token), then verify.
        temporary = policy_path.with_name(policy_path.name + '.restore.tmp')
        temporary.write_bytes(original_bytes)
        temporary.replace(policy_path)
        try:
            report['restoration'] = await_policy(evidence, original, allowed, renderer_cpu)
        except Exception as exc:
            report['restoration_error'] = str(exc)
            raise
        finally:
            save()
    print(json.dumps(report['combined'], indent=2))


if __name__ == '__main__':
    main()
