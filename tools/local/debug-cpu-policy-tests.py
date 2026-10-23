#!/usr/bin/env python3
"""Exercise selected-thread policy identity and actual affinity restoration."""
import os
from pathlib import Path
import threading

from debug_cpu_policy import pin_threads, start_ticks, thread_overrides_for_request


def main():
    allowed = set(os.sched_getaffinity(0))
    if len(allowed) < 3:
        raise RuntimeError('This integration fixture needs three allowed CPUs')
    render_cpu, candidate_cpu = sorted(allowed)[:2]
    event = threading.Event()
    selected = threading.Thread(target=event.wait)
    other = threading.Thread(target=event.wait)
    selected.start()
    other.start()
    pid = os.getpid()
    tasks = Path('/proc') / str(pid) / 'task'
    original = {int(p.name): os.sched_getaffinity(int(p.name)) for p in tasks.iterdir()}
    renderer = {'tid': pid, 'start_ticks': start_ticks(pid)}
    raw = {'tid': selected.native_id, 'start_ticks': start_ticks(selected.native_id),
           'cpus': [candidate_cpu]}
    try:
        for invalid in [dict(raw, cpus=[render_cpu]), dict(raw, cpus=[True]),
                        dict(raw, cpus=[]), dict(raw, start_ticks=None)]:
            try:
                thread_overrides_for_request({'thread_overrides': [invalid]}, allowed, render_cpu)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid override accepted')
        bad = thread_overrides_for_request(
            {'thread_overrides': [dict(raw, start_ticks='0')]}, allowed, render_cpu)
        try:
            pin_threads(pid, render_cpu, allowed - {render_cpu}, renderer, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Stale identity accepted')
        assert all(os.sched_getaffinity(tid) == mask for tid, mask in original.items())
        valid = thread_overrides_for_request({'thread_overrides': [raw]}, allowed, render_cpu)
        pin_threads(pid, render_cpu, allowed - {render_cpu}, renderer, valid)
        assert os.sched_getaffinity(pid) == {render_cpu}
        assert os.sched_getaffinity(selected.native_id) == {candidate_cpu}
        assert os.sched_getaffinity(other.native_id) == allowed - {render_cpu}
        pin_threads(pid, render_cpu, allowed - {render_cpu}, renderer)
        assert os.sched_getaffinity(selected.native_id) == allowed - {render_cpu}
        assert os.sched_getaffinity(other.native_id) == allowed - {render_cpu}
    finally:
        for tid, mask in original.items():
            os.sched_setaffinity(tid, mask)
        event.set()
        selected.join()
        other.join()
    print('Selected-thread masks, unrelated worker policy, stale identity rejection and restoration passed')


if __name__ == '__main__':
    main()
