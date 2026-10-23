#!/usr/bin/env python3
"""Switch direct native resource output between measurements of an owned run."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('loaded', 'rejected', 'native_calls', 'verified_calls', 'failed_transactions', 'unavailable_calls')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', type=int, choices=(0, 1, 2))
    parser.add_argument('--predicate-mode', type=int, choices=(0, 1, 2),
                        help='New build only: original/native/verified condition selection')
    parser.add_argument('--scratch-mode', type=int, choices=(0, 1),
                        help='New build only: original/reusable preparation storage')
    parser.add_argument('--lookup-mode', type=int, choices=(0, 1, 2),
                        help='New build only: original/shortened/verified resource lookup')
    parser.add_argument('--render-cost-mode', type=int, choices=(0, 1, 2, 3),
                        help='Separate diagnostics: off/stages/readbacks/stages plus SRT timers')
    parser.add_argument('--trim-mode', type=int, choices=(0, 1),
                        help='New build only: eliminate redundant state clearing and remap allocation')
    parser.add_argument('--residency-mode', type=int, choices=(0, 1, 2),
                        help='New build only: original/exact-range/verified GPU ownership query')
    parser.add_argument('--finish-mode', type=int, choices=(0, 1),
                        help='New build only: original/buffer-only resource completion')
    parser.add_argument('--upload-mode', type=int, choices=(0, 1),
                        help='New build only: skip uploads for compiled stages with no SRT consumer')
    parser.add_argument('--pipeline-index-mode', type=int, choices=(0, 1, 2),
                        help='New build only: original/block-hashed/verified graphics pipeline index')
    parser.add_argument('--depth-store-mode', type=int, choices=(0, 1),
                        help='New build only: omit stores for wholly read-only depth/stencil attachments')
    parser.add_argument('--upload-placement-mode', type=int, choices=(0, 1, 2),
                        help='V12 only: small uploads device/host sequential/host cached')
    parser.add_argument('--copy-feedback-mode', type=int, choices=(0, 1, 2),
                        help='V13: original/completed copy mirrors/real GPU verification')
    parser.add_argument('--lod-copy-feedback-mode', type=int, choices=(0, 1),
                        help='Mirror the GPU-packed LOD report so its read can skip a full drain')
    parser.add_argument('--lod-stats-period', type=int, choices=tuple(range(1, 10)),
                        help='Produce a texture LOD feedback report once every N requests (1 keeps every one)')
    parser.add_argument('--diagnostics', type=int, choices=(0, 1))
    parser.add_argument('--reset-stats', action='store_true')
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    pid, identity, sha = receipt['pid'], str(receipt['pid_start_ticks']), receipt['binary_sha256']
    if type(pid) is not int or pid <= 1 or not identity.isdigit() or not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('Invalid owned-run identity')
    if receipt.get('status') not in ('running', 'measured', 'diagnostics-only'):
        raise ValueError('Owned run has ended')
    if (args.mode in (1, 2) or args.predicate_mode in (1, 2) or args.scratch_mode == 1
            or args.lookup_mode in (1, 2)
            or args.render_cost_mode in (1, 2, 3) or args.trim_mode == 1
            or args.residency_mode in (1, 2) or args.finish_mode == 1
            or args.copy_feedback_mode in (1, 2) or args.lod_copy_feedback_mode == 1
            or (args.lod_stats_period or 1) > 1 or args.upload_placement_mode in (1, 2) or args.upload_mode == 1 or args.pipeline_index_mode in (1, 2) or args.depth_store_mode == 1 or args.diagnostics == 1):
        if receipt.get('diagnostics_deadline_monotonic', 0) - time.monotonic() < 30:
            raise ValueError('Separate diagnostic stage must have at least 30 seconds left')
    library = Path(receipt['runtime_environment']['KYTY_SRT_AOT_LIBRARY']).resolve()
    with library.open('rb') as stream:
        library_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    script = args.output / 'control.gdb'
    script.write_text(f'''set pagination off
python
import gdb,pathlib,hashlib,json
p=pathlib.Path('/proc/{pid}')
assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=={identity!r}
assert hashlib.sha256(p.joinpath('exe').read_bytes()).hexdigest()=={sha!r}
assert {str(library)!r} in p.joinpath('maps').read_text()
assert hashlib.sha256(pathlib.Path({str(library)!r}).read_bytes()).hexdigest()=={library_sha!r}
inferior=gdb.selected_inferior()
mode_address=int(gdb.parse_and_eval('&kyty_local_srt_native_mode'))
diagnostic_address=int(gdb.parse_and_eval('&kyty_local_srt_native_diagnostics'))
stats_address=int(gdb.parse_and_eval('&kyty_local_srt_native_stats'))
old=int.from_bytes(bytes(inferior.read_memory(mode_address,4)),'little')
old_diagnostics=int.from_bytes(bytes(inferior.read_memory(diagnostic_address,4)),'little')
raw=bytes(inferior.read_memory(stats_address,48))
stats=[int.from_bytes(raw[i:i+8],'little') for i in range(0,48,8)]
mode={args.mode!r}
extras={{}}
for name,value in [('render_cost', {args.render_cost_mode!r}), ('trim', {args.trim_mode!r}), ('residency', {args.residency_mode!r}), ('finish', {args.finish_mode!r}), ('upload', {args.upload_mode!r}), ('pipeline_index', {args.pipeline_index_mode!r}), ('depth_store', {args.depth_store_mode!r}), ('upload_placement', {args.upload_placement_mode!r}), ('copy_feedback', {args.copy_feedback_mode!r}), ('lod_copy_feedback', {args.lod_copy_feedback_mode!r}), ('lod_stats', {args.lod_stats_period!r})]:
    if value is not None:
        symbol={{'render_cost':'kyty_local_render_cost_mode', 'trim':'kyty_local_preparation_trim_mode', 'residency':'kyty_local_buffer_residency_mode', 'finish':'kyty_local_resource_finish_mode', 'upload':'kyty_local_srt_upload_mode', 'pipeline_index':'kyty_local_pipeline_index_mode', 'depth_store':'kyty_local_depth_store_mode', 'upload_placement':'kyty_local_upload_placement_mode', 'copy_feedback':'kyty_local_copy_feedback_mode', 'lod_copy_feedback':'kyty_local_lod_copy_feedback_mode', 'lod_stats':'kyty_local_lod_stats_period'}}[name]
        address=int(gdb.parse_and_eval('&'+symbol))
        previous=int.from_bytes(bytes(inferior.read_memory(address,4)),'little')
        extras[name]=dict(address=address,old_mode=previous,mode=value)
        if name in ('residency','finish','upload','pipeline_index','depth_store','upload_placement','copy_feedback','lod_stats'):
            stats_symbol={{'residency':'kyty_local_buffer_residency_stats', 'finish':'kyty_local_resource_finish_stats', 'upload':'kyty_local_srt_upload_stats', 'pipeline_index':'kyty_local_pipeline_index_stats', 'depth_store':'kyty_local_depth_store_stats', 'upload_placement':'kyty_local_upload_placement_stats', 'copy_feedback':'kyty_local_copy_feedback_stats', 'lod_stats':'kyty_local_lod_stats_counts'}}[name]
            stats_bytes=128 if name == 'copy_feedback' else 96 if name == 'upload_placement' else 24 if name in ('finish','pipeline_index','lod_stats') else 32
            sa=int(gdb.parse_and_eval('&'+stats_symbol))
            raw_stats=bytes(inferior.read_memory(sa,stats_bytes))
            extras[name]['stats']=[int.from_bytes(raw_stats[i:i+8],'little') for i in range(0,stats_bytes,8)]
            extras[name]['stats_address']=sa
            extras[name]['stats_bytes']=stats_bytes
lookup_mode={args.lookup_mode!r}
lookup=None
if lookup_mode is not None:
    lookup_address=int(gdb.parse_and_eval('&kyty_local_preparation_lookup_mode'))
    lookup_stats_address=int(gdb.parse_and_eval('&kyty_local_preparation_lookup_stats'))
    lookup_old=int.from_bytes(bytes(inferior.read_memory(lookup_address,4)),'little')
    lookup_raw=bytes(inferior.read_memory(lookup_stats_address,24))
    lookup_stats=[int.from_bytes(lookup_raw[i:i+8],'little') for i in range(0,24,8)]
    lookup=dict(old_mode=lookup_old,mode=lookup_mode,stats=lookup_stats)
scratch_mode={args.scratch_mode!r}
scratch=None
if scratch_mode is not None:
    scratch_address=int(gdb.parse_and_eval('&kyty_local_preparation_scratch_mode'))
    scratch_old=int.from_bytes(bytes(inferior.read_memory(scratch_address,4)),'little')
    scratch=dict(old_mode=scratch_old,mode=scratch_mode)
predicate_mode={args.predicate_mode!r}
predicate=None
if predicate_mode is not None:
    predicate_address=int(gdb.parse_and_eval('&kyty_local_srt_predicate_mode'))
    predicate_stats_address=int(gdb.parse_and_eval('&kyty_local_srt_predicate_stats'))
    predicate_old=int.from_bytes(bytes(inferior.read_memory(predicate_address,4)),'little')
    predicate_raw=bytes(inferior.read_memory(predicate_stats_address,24))
    predicate_stats=[int.from_bytes(predicate_raw[i:i+8],'little') for i in range(0,24,8)]
    if predicate_mode: assert predicate_stats[0]>0
    predicate=dict(old_mode=predicate_old,mode=predicate_mode,stats=predicate_stats)
diagnostics={args.diagnostics!r}
if mode in (1,2): assert stats[0]>0
if mode==2 or predicate_mode==2 or lookup_mode==2 or {args.residency_mode!r}==2 or {args.pipeline_index_mode!r}==2 or {args.copy_feedback_mode!r}==2: diagnostics=1
if mode is not None:
    inferior.write_memory(mode_address,mode.to_bytes(4,'little'))
    assert bytes(inferior.read_memory(mode_address,4))==mode.to_bytes(4,'little')
if diagnostics is not None:
    inferior.write_memory(diagnostic_address,diagnostics.to_bytes(4,'little'))
    assert bytes(inferior.read_memory(diagnostic_address,4))==diagnostics.to_bytes(4,'little')
if predicate_mode is not None:
    inferior.write_memory(predicate_address,predicate_mode.to_bytes(4,'little'))
    assert bytes(inferior.read_memory(predicate_address,4))==predicate_mode.to_bytes(4,'little')
    if {args.reset_stats!r}: inferior.write_memory(predicate_stats_address+8,bytes(16))
if scratch_mode is not None:
    inferior.write_memory(scratch_address,scratch_mode.to_bytes(4,'little'))
    assert bytes(inferior.read_memory(scratch_address,4))==scratch_mode.to_bytes(4,'little')
if lookup_mode is not None:
    inferior.write_memory(lookup_address,lookup_mode.to_bytes(4,'little'))
    assert bytes(inferior.read_memory(lookup_address,4))==lookup_mode.to_bytes(4,'little')
    if {args.reset_stats!r}: inferior.write_memory(lookup_stats_address,bytes(24))
if {args.reset_stats!r}: inferior.write_memory(stats_address+16,bytes(32))
for extra in extras.values():
    address=extra.pop('address')
    sa=extra.pop('stats_address',None)
    stats_bytes=extra.pop('stats_bytes',0)
    if sa is not None and {args.reset_stats!r}: inferior.write_memory(sa,bytes(stats_bytes))
    value=extra['mode'].to_bytes(4,'little')
    inferior.write_memory(address,value)
    assert bytes(inferior.read_memory(address,4))==value
print('NATIVE_RESOURCE_CONTROL='+json.dumps(dict(old_mode=old,mode=old if mode is None else mode,
    old_diagnostics=old_diagnostics,diagnostics=old_diagnostics if diagnostics is None else diagnostics,stats=stats,predicate=predicate,scratch=scratch,lookup=lookup,**extras)))
end
detach
''')
    with (ROOT / '_Build/automation/input.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = subprocess.run(['sudo', '-n', 'gdb', '-q', '-nx', '-batch', '-iex',
            'set print thread-events off', '-p', str(pid), '-x', str(script)],
            capture_output=True, text=True, timeout=30)
    (args.output / 'control.log').write_text(completed.stdout + completed.stderr)
    (args.output / 'debugger-result.json').write_text(json.dumps({
        'returncode': completed.returncode,
        'pid': pid, 'pid_start_ticks': identity, 'binary_sha256': sha,
    }, indent=2) + '\n')
    records = re.findall(r'^NATIVE_RESOURCE_CONTROL=(.+)$', completed.stdout, re.M)
    if completed.returncode or len(records) != 1:
        raise RuntimeError(f'Native resource control failed (debugger exit {completed.returncode}, '
                           f'{len(records)} records): {args.output}')
    result = json.loads(records[0])
    result['stats'] = dict(zip(NAMES, result['stats'], strict=True))
    result.update(pid=pid, pid_start_ticks=identity, binary_sha256=sha,
                  library=str(library), library_sha256=library_sha, receipt=str(args.receipt.resolve()))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
