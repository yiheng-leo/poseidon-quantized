"""Stage costs with sampled process-tree RSS and CUDA allocator peaks.

Wall/CPU times are exclusive of nested phases (e.g. optimization excludes
checkpoint IO). CUDA peaks are exact allocator high-water marks between phase
boundaries. CPU RSS is sampled, includes live child processes, and may count
shared pages more than once. No device-wide memory/utilization is attributed
to this process, and no unobserved pre-resume cost is invented.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import threading
import time
import uuid

import psutil
import torch

from .checkpoints import atomic_json

_ACTIVE = ContextVar('quantization_cost_tracker', default=None)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _empty(cuda):
    return {'wall_seconds': 0.0, 'cpu_seconds': 0.0, 'cpu_peak_rss_bytes': 0,
            'cpu_peak_utilization_percent': 0.0, 'calls': 0,
            'gpu_peak_allocated_bytes': 0 if cuda else None,
            'gpu_peak_reserved_bytes': 0 if cuda else None}


class CostTracker:
    def __init__(self, path, device, *, interval=0.1, resume=False):
        self.path, self.device = Path(path), torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.interval = float(interval)
        self.lock = threading.RLock()
        self.process = psutil.Process()
        self.stack = []
        self.last_cpu = {}
        self.last = time.perf_counter()
        self.last_flush = self.last
        if resume and self.path.exists():
            self.report = json.loads(self.path.read_text())
            for old in self.report['sessions']:
                if old['status'] == 'running':
                    old['status'] = 'interrupted_last_sample'
        else:
            self.report = {'schema_version': 1, 'sessions': [], 'definitions': {
                'time': 'exclusive wall/CPU seconds; checkpoint IO is a separate phase',
                'cpu_peak_rss_bytes': 'sampled sum of current process and live children RSS; shared pages may repeat',
                'cpu_peak_utilization_percent': 'sampled CPU time / wall time, 100% per logical core',
                'cpu_process_lifetime_peak_rss_bytes': 'OS high-water RSS for the parent process lifetime, excludes children; not phase-local',
                'gpu_peaks': 'PyTorch CUDA allocator allocated/reserved bytes on the selected device; includes resident model',
                'crash_coverage': 'last heartbeat/checkpoint sample; SIGKILL tail after last sample is unknown',
            }}
        self.session = {'id': uuid.uuid4().hex, 'started_at_utc': _utc(), 'ended_at_utc': None,
                        'status': 'running', 'sample_interval_seconds': interval, 'cpu_scope_complete': True,
                        'hardware': {'host': platform.node(), 'platform': platform.platform(),
                                     'cpu_logical_count': psutil.cpu_count(), 'device': str(device),
                                     'torch_version': str(torch.__version__)},
                        'total': _empty(self.cuda), 'phases': {}, 'counters': {}}
        if self.cuda:
            self.session['hardware'].update(gpu_name=torch.cuda.get_device_name(self.device),
                                           gpu_total_bytes=torch.cuda.get_device_properties(self.device).total_memory)
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.report['sessions'].append(self.session)
        if resume and not self.path.exists():
            self.report['unmeasured_history_before_first_session'] = True
        self._sample(initial=True)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name='quantization-costs', daemon=True)
        self.thread.start()
        self.flush()

    def _sample(self, initial=False):
        with self.lock:
            now = time.perf_counter()
            elapsed = 0.0 if initial else now - self.last
            rss, cpu_delta = 0, 0.0
            try:
                processes = [self.process, *self.process.children(recursive=True)]
            except (psutil.Error, OSError):
                processes = [self.process]
                self.session['cpu_scope_complete'] = False
                self.session['cpu_scope_note'] = 'Child-process enumeration denied; reporting accessible processes only'
            for process in processes:
                try:
                    with process.oneshot():
                        key = (process.pid, process.create_time())
                        rss += process.memory_info().rss
                        times = process.cpu_times()
                        cpu = times.user + times.system
                    previous = self.last_cpu.get(key, cpu if initial else 0.0)
                    cpu_delta += max(0.0, cpu - previous)
                    self.last_cpu[key] = cpu
                except (psutil.Error, OSError):
                    self.session['cpu_scope_complete'] = False
                    continue
            gpu_allocated = gpu_reserved = None
            if self.cuda:
                try:
                    gpu_allocated = torch.cuda.max_memory_allocated(self.device)
                    gpu_reserved = torch.cuda.max_memory_reserved(self.device)
                except RuntimeError as error:
                    self.session['gpu_monitor_error'] = str(error)
            targets = [self.session['total']]
            if self.stack:
                targets.append(self.session['phases'][self.stack[-1]])
            for target in targets:
                target['wall_seconds'] += elapsed
                target['cpu_seconds'] += cpu_delta
                target['cpu_peak_rss_bytes'] = max(target['cpu_peak_rss_bytes'], rss)
                if elapsed >= self.interval / 2:
                    target['cpu_peak_utilization_percent'] = max(
                        target['cpu_peak_utilization_percent'], 100 * cpu_delta / elapsed)
                if gpu_allocated is not None:
                    target['gpu_peak_allocated_bytes'] = max(target['gpu_peak_allocated_bytes'], gpu_allocated)
                    target['gpu_peak_reserved_bytes'] = max(target['gpu_peak_reserved_bytes'], gpu_reserved)
            self.last = now
            self.session['last_sample_at_utc'] = _utc()
            try:
                import resource
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                self.session['cpu_process_lifetime_peak_rss_bytes'] = int(peak if sys.platform == 'darwin' else peak * 1024)
            except (ImportError, OSError):
                self.session['cpu_process_lifetime_peak_rss_bytes'] = None

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                self._sample()
                if time.perf_counter() - self.last_flush >= 5:
                    self.flush()
            except Exception as error:
                # Monitoring failure must be visible without killing optimization.
                with self.lock:
                    self.session['monitor_error'] = f'{type(error).__name__}: {error}'

    @contextmanager
    def phase(self, name):
        with self.lock:
            nested_same = bool(self.stack and self.stack[-1] == name)
        if nested_same:
            yield
            return
        if self.cuda:
            torch.cuda.synchronize(self.device)
        with self.lock:
            self._sample()
            value = self.session['phases'].setdefault(name, _empty(self.cuda))
            value['calls'] += 1
            self.stack.append(name)
            if self.cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
        try:
            yield
        finally:
            if self.cuda:
                try:
                    torch.cuda.synchronize(self.device)
                except RuntimeError as error:
                    self.session['gpu_monitor_error'] = str(error)
            with self.lock:
                self._sample()
                assert self.stack.pop() == name
                if self.cuda:
                    torch.cuda.reset_peak_memory_stats(self.device)

    def add(self, name, value=1):
        with self.lock:
            counters = self.session['counters']
            counters[name] = counters.get(name, 0) + value

    def snapshot(self):
        with self.lock:
            if self.session['status'] == 'running':
                self._sample()
            sessions = self.report['sessions']
            cumulative = _empty(self.cuda)
            phases, counters = {}, {}
            for session in sessions:
                for name, value in session['counters'].items():
                    counters[name] = counters.get(name, 0) + value
                for key, value in session['total'].items():
                    if key.endswith('_bytes') or key.endswith('_percent'):
                        if value is not None:
                            cumulative[key] = max(cumulative.get(key) or 0, value)
                    elif isinstance(value, (int, float)):
                        cumulative[key] += value
                for name, stage in session['phases'].items():
                    combined = phases.setdefault(name, _empty(self.cuda))
                    for key, value in stage.items():
                        if key.endswith('_bytes') or key.endswith('_percent'):
                            if value is not None:
                                combined[key] = max(combined.get(key) or 0, value)
                        elif isinstance(value, (int, float)):
                            combined[key] += value
            self.report['cumulative'] = {**cumulative, 'phases': phases,
                                         'optimization_seconds': phases.get('optimization', {}).get('wall_seconds', 0.0),
                                         'cpu_scope_complete': all(item.get('cpu_scope_complete', True) for item in sessions),
                                         'counters': counters,
                                         'cpu_process_lifetime_peak_rss_bytes': max((item.get('cpu_process_lifetime_peak_rss_bytes') or 0) for item in sessions)}
            return json.loads(json.dumps(self.report))

    def flush(self):
        with self.lock:
            atomic_json(self.snapshot(), self.path)
            self.last_flush = time.perf_counter()

    def close(self, status):
        self.stop_event.set()
        self.thread.join(timeout=2)
        with self.lock:
            self._sample()
            self.session['status'] = status
            self.session['ended_at_utc'] = _utc()
            self.flush()


@contextmanager
def track(tracker):
    token = _ACTIVE.set(tracker)
    try:
        yield tracker
    finally:
        _ACTIVE.reset(token)


def phase(name):
    tracker = _ACTIVE.get()
    return tracker.phase(name) if tracker else nullcontext()


def counter(name, value=1):
    tracker = _ACTIVE.get()
    if tracker:
        tracker.add(name, value)


def flush():
    tracker = _ACTIVE.get()
    if tracker:
        tracker.flush()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Show recorded quantization time and memory costs.')
    parser.add_argument('path', help='Experiment directory or costs.json')
    parser.add_argument('--json', action='store_true', help='Print cumulative metrics as JSON')
    args = parser.parse_args(argv)
    path = Path(args.path)
    if path.is_dir():
        path /= 'costs.json'
    report = json.loads(path.read_text())
    summary = report['cumulative']
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    print(f'Recorded sessions: {len(report["sessions"])}')
    print('Exclusive phase costs; memory in GiB, GPU allocated/reserved are PyTorch allocator peaks.')
    print(f'{"phase":29} {"wall_s":>12} {"CPU_s":>12} {"CPU_RSS":>11} {"GPU_alloc":>11} {"GPU_resv":>11}')
    def gib(value):
        return '-' if value is None else f'{value / 1024**3:.3f}'
    for name, value in [*summary['phases'].items(), ('ALL_RECORDED_SESSIONS', summary)]:
        print(f'{name:29} {value["wall_seconds"]:12.3f} {value["cpu_seconds"]:12.3f} '
              f'{gib(value["cpu_peak_rss_bytes"]):>11} {gib(value["gpu_peak_allocated_bytes"]):>11} '
              f'{gib(value["gpu_peak_reserved_bytes"]):>11}')
    if not summary.get('cpu_scope_complete', True):
        print('CPU process access was restricted in at least one session; RSS covers accessible processes.')
    if report.get('unmeasured_history_before_first_session'):
        print('Work completed before cost recording was introduced has unknown historical cost.')


if __name__ == '__main__':
    main()
