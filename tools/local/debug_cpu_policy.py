"""PID-bound affinity policies shared by debug-run and its phase switch tool."""
import json
import os
from pathlib import Path


def start_ticks(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def parse_cpus(text):
    values = set()
    for group in text.split(','):
        ends = group.split('-')
        if len(ends) not in (1, 2) or any(not x.isdecimal() for x in ends):
            raise ValueError('CPUs must be numbers or ranges, such as 1-7,9')
        first, last = int(ends[0]), int(ends[-1])
        if not 0 <= first <= last < 4096:
            raise ValueError('Invalid CPU range')
        values.update(range(first, last + 1))
    return values


def validate_request(request, identity, allowed, renderer_cpu):
    if not isinstance(request, dict) or request.get('identity') != identity:
        raise ValueError('CPU policy belongs to a different process or executable')
    token = request.get('token')
    if not isinstance(token, str) or not 1 <= len(token) <= 128:
        raise ValueError('CPU policy needs a request token')
    values = request.get('worker_cpus')
    if (not isinstance(values, list) or not values or
            any(type(x) is not int for x in values) or len(set(values)) != len(values)):
        raise ValueError('CPU policy needs distinct worker CPU numbers')
    mask = set(values)
    if not mask <= set(allowed) - {renderer_cpu}:
        raise ValueError('Worker CPUs must be allowed and exclude the renderer CPU')
    return mask


def renderer_for_request(request, identity, allowed, default):
    if not isinstance(request, dict) or request.get('identity') != identity:
        raise ValueError('CPU policy belongs to a different process or executable')
    renderer_cpu = request.get('renderer_cpu', default)
    if type(renderer_cpu) is not int or renderer_cpu not in allowed:
        raise ValueError('Renderer CPU must be an allowed CPU number')
    return renderer_cpu


def thread_overrides_for_request(request, allowed, renderer_cpu):
    """Optional per-thread experiments; identity validation happens first."""
    values = request.get('thread_overrides', [])
    if not isinstance(values, list):
        raise ValueError('Thread overrides must be a list')
    result = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError('Each thread override needs an identity and CPU list')
        tid, started, cpus = value.get('tid'), value.get('start_ticks'), value.get('cpus')
        if type(tid) is not int or tid <= 0 or tid in result:
            raise ValueError('Thread overrides need distinct positive TIDs')
        if not isinstance(started, str) or not started.isdecimal():
            raise ValueError('Thread overrides need the observed thread start time')
        if (not isinstance(cpus, list) or not cpus or
                any(type(cpu) is not int for cpu in cpus) or len(set(cpus)) != len(cpus) or
                not set(cpus) <= set(allowed) - {renderer_cpu}):
            raise ValueError('Override CPUs must be distinct, allowed and exclude the renderer CPU')
        result[tid] = {'start_ticks': started, 'cpus': set(cpus)}
    return result


def pin_threads(pid, renderer_cpu, worker_cpus, renderer=None, thread_overrides=None):
    """Only operate on threads still in this child process's task directory."""
    tasks = Path('/proc') / str(pid) / 'task'
    try:
        names = {}
        for task in tasks.iterdir():
            try:
                names[int(task.name)] = (task / 'comm').read_text().strip()
            except FileNotFoundError:
                continue
    except FileNotFoundError:
        return 0, {}, {}
    if renderer is None:
        renderers = [tid for tid, name in names.items() if name == 'Kyty.Gpu']
    else:
        # Clean upstream builds do not carry the local Kyty.Gpu thread name.
        # The benchmark discovers this identity from the child's renderer TLS.
        if (not isinstance(renderer, dict) or type(renderer.get('tid')) is not int or
                renderer['tid'] not in names or
                start_ticks(renderer['tid']) != renderer.get('start_ticks')):
            raise ValueError('Renderer identity is absent, stale or outside the owned process')
        renderers = [renderer['tid']]
    if len(renderers) != 1:
        return 0, {}, names
    overrides = thread_overrides or {}
    # Validate the entire requested set before changing any affinity. A stale
    # name or a recycled TID must never select an unrelated guest/host worker.
    for tid, value in overrides.items():
        if (tid == renderers[0] or tid not in names or
                not (tasks / str(tid)).exists() or
                start_ticks(tid) != value['start_ticks']):
            raise ValueError('Override thread identity is absent, stale or is the renderer')
    updated, masks = 0, {}
    for tid in names:
        if not (tasks / str(tid)).exists():
            continue
        mask = ({renderer_cpu} if tid == renderers[0] else
                overrides[tid]['cpus'] if tid in overrides else worker_cpus)
        try:
            if os.sched_getaffinity(tid) != mask:
                os.sched_setaffinity(tid, mask)
                updated += 1
            masks[str(tid)] = sorted(os.sched_getaffinity(tid))
        except ProcessLookupError:
            pass
    return updated, masks, names
