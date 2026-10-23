#!/usr/bin/env python3
"""Control a local Vulkan recording experiment outside its FPS/profile intervals."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('published replayed bytes chunks drains wait_ns worker_ns rejected direct max_chunk').split()


def control(args):
    receipt = json.loads(args.receipt.read_text())
    pid, identity, fingerprint = receipt['pid'], str(receipt['pid_start_ticks']), receipt['binary_sha256']
    if not isinstance(pid, int) or pid <= 1 or not identity.isdigit() or not re.fullmatch('[0-9a-f]{64}', fingerprint):
        raise ValueError('Invalid owned-run identity')
    if receipt.get('status') not in ('running', 'measured', 'diagnostics-only'):
        raise ValueError('This owned run has ended')
    if args.counter_query and fingerprint != 'f7ba2de61a4827ab3af902c22b70d014d43a7d8534c7d17ff4f6c7644ff4a13c':
        raise ValueError('The counter-query patch is only for the original frozen prototype')
    if args.mode or args.packet_mask is not None:
        deadline = receipt.get('diagnostics_deadline_monotonic')
        if deadline is None or deadline - time.monotonic() < 30:
            raise ValueError('The separate diagnostic stage is not ready or has less than 30 seconds left')
    args.output.mkdir(parents=True, exist_ok=False)
    source = f'''set pagination off
python
import gdb,pathlib,hashlib,json
p=pathlib.Path('/proc/{pid}')
assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=={identity!r}
assert hashlib.sha256(p.joinpath('exe').read_bytes()).hexdigest()=={fingerprint!r}
inferior=gdb.selected_inferior()
address=int(gdb.parse_and_eval('&kyty_local_vulkan_recording_mode'))
stats_address=int(gdb.parse_and_eval('&kyty_local_vulkan_recording_stats'))
direct_address=int(gdb.parse_and_eval('&kyty_local_vulkan_recording_direct'))
try:
    packet_address=int(gdb.parse_and_eval('&kyty_local_vulkan_packet_stats'))
except gdb.error:
    packet_address=None
old=int.from_bytes(bytes(inferior.read_memory(address,4)),'little')
def counters(address,count):
    raw=bytes(inferior.read_memory(address,count*8))
    return [int.from_bytes(raw[i:i+8],'little') for i in range(0,len(raw),8)]
stats=counters(stats_address,10)
direct=counters(direct_address,1024)
packets=counters(packet_address,4) if packet_address else None
packet_mask=None
if {args.packet_mask!r} is not None:
    mask_address=int(gdb.parse_and_eval('&kyty_local_vulkan_packet_mask'))
    packet_mask={args.packet_mask!r}
    inferior.write_memory(mask_address,packet_mask.to_bytes(4,'little'))
    assert bytes(inferior.read_memory(mask_address,4))==packet_mask.to_bytes(4,'little')
mode={args.mode!r}
if mode is not None:
    inferior.write_memory(address,mode.to_bytes(4,'little'))
    assert bytes(inferior.read_memory(address,4))==mode.to_bytes(4,'little')
if {args.reset_counters!r}:
    inferior.write_memory(stats_address,bytes(10*8))
    inferior.write_memory(direct_address,bytes(1024*8))
    if packet_address: inferior.write_memory(packet_address,bytes(4*8))
query_patch=[]
if {args.counter_query!r} is not None:
    for name in ['vkGetSemaphoreCounterValue','vkGetSemaphoreCounterValueKHR']:
        target=int(gdb.parse_and_eval("&'vk::detail::defaultDispatchLoaderDynamic'."+name))
        saved=int(gdb.parse_and_eval("'LocalVulkanRecording::(anonymous namespace)::original'."+name))
        wrapper=int(gdb.parse_and_eval("&'LocalVulkanRecording::(anonymous namespace)::Wrapped_"+name+"'"))
        previous=int.from_bytes(bytes(inferior.read_memory(target,8)),'little')
        if saved==0:
            assert previous==0
            continue
        assert previous in (saved,wrapper)
        value=saved if {args.counter_query!r}=='passthrough' else wrapper
        inferior.write_memory(target,value.to_bytes(8,'little'))
        assert bytes(inferior.read_memory(target,8))==value.to_bytes(8,'little')
        query_patch.append(dict(api=name,previous=previous,value=value,original=saved,wrapper=wrapper))
print('RECORDING_CONTROL='+json.dumps(dict(pid={pid},old_mode=old,mode=old if mode is None else mode,stats=stats,direct=direct,packets=packets,packet_mask=packet_mask,query_patch=query_patch)))
end
detach
'''
    script = args.output / 'control.gdb'
    script.write_text(source)
    with (ROOT/'_Build/automation/input.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(['sudo','-n','gdb','-q','-nx','-batch','-iex',
            'set print thread-events off','-p',str(pid),'-x',str(script)],
            capture_output=True,text=True,timeout=30)
    (args.output/'control.log').write_text(result.stdout+result.stderr)
    records = re.findall(r'^RECORDING_CONTROL=(.+)$',result.stdout,re.M)
    if result.returncode or len(records) != 1:
        raise RuntimeError(f'Recording control failed; see {args.output}')
    summary = json.loads(records[0])
    summary['stats'] = dict(zip(NAMES,summary['stats'],strict=True))
    if summary['packets'] is not None:
        summary['packets'] = dict(zip(('draws','state_reuse','bindings','inline_fallback'), summary['packets'], strict=True))
    if args.api_manifest:
        names = json.loads(args.api_manifest.read_text())['apis']
        summary['direct'] = {names[i] if i < len(names) else f'unknown_{i}': value
                             for i,value in enumerate(summary['direct']) if value}
    summary.update(receipt=str(args.receipt.resolve()), binary_sha256=fingerprint,
                   timestamp=datetime.datetime.now().isoformat(), reset=args.reset_counters)
    (args.output/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', type=int, choices=(0,1,2,3,4))
    parser.add_argument('--packet-mask', type=int, choices=(0,1,2,3), help='Bit 0 draws, bit 1 descriptors')
    parser.add_argument('--reset-counters', action='store_true')
    parser.add_argument('--api-manifest', type=Path)
    parser.add_argument('--counter-query', choices=('passthrough','ordered'),
                        help='Local x86-64 debugger experiment: bypass/restore the original prototype query drain')
    result = control(parser.parse_args())
    print(json.dumps({k:v for k,v in result.items() if k != 'direct'},indent=2))
