#!/usr/bin/env python3
"""Run a bounded native-resource verification and alternating FPS sequence."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PHASES = ((0, 1, None, 'predicate'), (0, 0, None, 'off1'), (1, 1, None, 'combined1'),
          (0, 0, None, 'off2'), (1, 1, None, 'combined2'), (0, 0, None, 'off3'))
PREPARATION_PHASES = ((1, 1, 0, 'off1'), (1, 1, 1, 'on1'), (1, 1, 1, 'on2'),
                      (1, 1, 0, 'off2'), (0, 0, 1, 'scratch-only'), (0, 0, 0, 'original'))
LOOKUP_PHASES = ((1, 1, 1, 'off1'), (1, 1, 1, 'on1'),
                 (1, 1, 1, 'on2'), (1, 1, 1, 'off2'))


def main():
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, choices=range(10, 31), default=20)
    parser.add_argument('--label', default='native-v3')
    parser.add_argument('--suite', choices=('resources', 'preparation', 'lookup', 'trim', 'residency', 'finish', 'upload'), default='resources')
    args = parser.parse_args()
    if not args.label.replace('-', '').replace('_', '').isalnum():
        parser.error('label must be letters, digits, - or _')
    receipt = json.loads(args.receipt.read_text())
    if receipt.get('status') not in ('measured', 'diagnostics-only'):
        raise ValueError('The owned run must have finished its initial timing windows')
    phases = {'resources': PHASES, 'preparation': PREPARATION_PHASES,
              'lookup': LOOKUP_PHASES, 'trim': LOOKUP_PHASES, 'residency': LOOKUP_PHASES,
              'finish': LOOKUP_PHASES, 'upload': LOOKUP_PHASES}[args.suite]
    required = 60 + len(phases) * (args.seconds + 10)
    if receipt.get('diagnostics_deadline_monotonic', 0) - time.monotonic() < required:
        raise ValueError('Insufficient diagnostic retention for the complete sequence')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(receipt=str(args.receipt.resolve()), pid=receipt['pid'],
                   pid_start_ticks=receipt['pid_start_ticks'],
                   binary_sha256=receipt['binary_sha256'], suite=args.suite,
                   phases=[], status='running')

    def save():
        (args.output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')

    def control(name, resource, predicate, diagnostics=0, reset=False, scratch=None, lookup=None, trim=None, residency=None, finish=None, upload=None):
        command = [sys.executable, str(ROOT / 'tools/local/native-resource-control.py'),
                   str(args.receipt.resolve()), '--output', str((args.output / name).resolve()),
                   '--mode', str(resource), '--predicate-mode', str(predicate),
                   '--diagnostics', str(diagnostics)]
        if reset:
            command.append('--reset-stats')
        if scratch is not None:
            command.extend(['--scratch-mode', str(scratch)])
        if lookup is not None:
            command.extend(['--lookup-mode', str(lookup)])
        if trim is not None:
            command.extend(['--trim-mode', str(trim)])
        if residency is not None:
            command.extend(['--residency-mode', str(residency)])
        if finish is not None:
            command.extend(['--finish-mode', str(finish)])
        if upload is not None:
            command.extend(['--upload-mode', str(upload)])
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=40)
        (args.output / (name + '.log')).write_text(result.stdout + result.stderr)
        result.check_returncode()
        return json.loads(result.stdout)

    save()
    try:
        summary['verification_start'] = control('verify-start', 2, 2, 1, True,
                                               1 if args.suite != 'resources' else None,
                                               2 if args.suite in ('lookup', 'trim', 'residency', 'finish', 'upload') else None,
                                               1 if args.suite in ('trim', 'residency', 'finish', 'upload') else None,
                                               2 if args.suite in ('residency', 'finish', 'upload') else None,
                                               1 if args.suite == 'finish' else None,
                                               1 if args.suite == 'upload' else None)
        save()
        time.sleep(12)
        summary['verification'] = control('verify-finish', 0, 0,
                                         scratch=0 if args.suite != 'resources' else None,
                                         lookup=0 if args.suite in ('lookup', 'trim', 'residency', 'finish', 'upload') else None,
                                         trim=0 if args.suite in ('trim', 'residency', 'finish', 'upload') else None,
                                         residency=0 if args.suite in ('residency', 'finish', 'upload') else None,
                                         finish=0 if args.suite == 'finish' else None,
                                         upload=0 if args.suite == 'upload' else None)
        stats = summary['verification']['stats']
        pred = summary['verification']['predicate']['stats']
        if not (stats['verified_calls'] > 0 and stats['verified_calls'] == stats['native_calls']
                and stats['failed_transactions'] == 0 and pred[1] > 0 and pred[1] == pred[2]):
            raise ValueError('The live verification did not execute successfully')
        if args.suite in ('lookup', 'trim', 'residency', 'finish', 'upload') and not all(summary['verification']['lookup']['stats']):
            raise ValueError('Lookup verification did not cover both image paths and backing transfers')
        if args.suite in ('residency', 'finish', 'upload'):
            residency_stats = summary['verification']['residency']['stats']
            if not (residency_stats[1] > 0 and residency_stats[1] == residency_stats[2]):
                raise ValueError('Exact dirty-range verification did not execute successfully')
        if args.suite == 'finish' and not all(summary['verification']['finish']['stats']):
            raise ValueError('Buffer-only completion did not execute during verification')
        if args.suite == 'upload' and not all(summary['verification']['upload']['stats']):
            raise ValueError('SRT upload verification did not cover unused and required paths')
        print('VERIFIED ' + json.dumps(dict(resources=stats, predicates=pred)), flush=True)
        save()
        for resource, predicate, scratch, name in phases:
            if receipt['diagnostics_deadline_monotonic'] - time.monotonic() < args.seconds + 35:
                raise ValueError('Owned retention is ending; refusing a partial timing window')
            lookup = int(name.startswith('on')) if args.suite == 'lookup' else 1 if args.suite in ('trim', 'residency', 'finish', 'upload') else None
            trim = int(name.startswith('on')) if args.suite == 'trim' else 1 if args.suite in ('residency', 'finish', 'upload') else None
            residency = int(name.startswith('on')) if args.suite == 'residency' else 1 if args.suite in ('finish', 'upload') else None
            finish = int(name.startswith('on')) if args.suite == 'finish' else None
            upload = int(name.startswith('on')) if args.suite == 'upload' else None
            setting = control('control-' + name, resource, predicate, scratch=scratch, lookup=lookup, trim=trim, residency=residency, finish=finish, upload=upload)
            time.sleep(3)
            command = [sys.executable, str(ROOT / 'tools/local/quick-benchmark-game.py'),
                       '--pid', str(receipt['pid']), '--seconds', str(args.seconds),
                       '--camera-frames', '0', '--label', args.label + '-' + name, '--screenshots']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    timeout=args.seconds + 20)
            (args.output / (name + '.log')).write_text(result.stdout + result.stderr)
            result.check_returncode()
            measurement = json.loads(result.stdout)
            summary['phases'].append(dict(name=name, resource=resource, predicate=predicate, scratch=scratch, lookup=lookup, trim=trim, residency=residency, finish=finish, upload=upload,
                                          control=setting, measurement=measurement))
            save()
            print('MEASURED ' + json.dumps(dict(name=name, views=measurement['views'],
                                               evidence=measurement['evidence'])), flush=True)
        summary['status'] = 'completed_unverified'
    except BaseException as error:
        summary['status'] = 'failed'
        summary['error'] = repr(error)
        raise
    finally:
        try:
            summary['final_control'] = control('restore', 0, 0,
                                               scratch=0 if args.suite != 'resources' else None,
                                               lookup=0 if args.suite in ('lookup', 'trim', 'residency', 'finish', 'upload') else None,
                                               trim=0 if args.suite in ('trim', 'residency', 'finish', 'upload') else None,
                                               residency=0 if args.suite in ('residency', 'finish', 'upload') else None,
                                               finish=0 if args.suite == 'finish' else None,
                                               upload=0 if args.suite == 'upload' else None)
        except BaseException as error:
            summary['restore_error'] = repr(error)
            summary['status'] = 'failed'
        save()
    print('Sequence evidence: ' + str(args.output.resolve()), flush=True)
    if summary['status'] == 'failed':
        raise RuntimeError('The sequence could not restore the owned run; inspect its result')


if __name__ == '__main__':
    main()
