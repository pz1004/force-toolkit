"""Fresh, manifest-bound paired performance experiments. No timing assertions in CI."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks._protocol import (
    SCHEMA, THREAD_ENV, EvidenceError, atomic_json, file_fingerprints, fresh_run,
    load_manifest, numerical_environment, probe_environment, read_records, validate_record,
    require_compiled_environment,
)
from benchmarks._statistics import summarize
from benchmarks._comparators import VARIANTS, ensure_prior_extracted, import_comparator, runtime_identity


def workloads(suite='full'):
    result = []
    for algorithm in ('force', 'mad', 'dual'):
        sizes = [(2000, 100, 10, 128), (6000, 200, 20, 512)] if algorithm == 'dual' else [(2000, 20, 0, 200), (5000, 50, 0, 200)]
        for n, p, k, calibration in sizes:
            for contaminated in (False, True):
                for detection in ((False, True) if algorithm == 'mad' else (False,)):
                    for chunk in (1, 256):
                        name = f'{algorithm}-{n}x{p}-' + ('dirty' if contaminated else 'clean') + ('-detect' if detection else '') + f'-b{chunk}'
                        result.append(dict(id=name, algorithm=algorithm, mode='stream', n=n, p=p, k=k,
                                           calibration=calibration, contaminated=contaminated, detection=detection, chunk=chunk))
    for algorithm in ('force', 'mad'):
        for n, p in ((2000, 20), (5000, 50)):
            for contaminated in (False, True):
                name = f'batch-{algorithm}-{n}x{p}-' + ('dirty' if contaminated else 'clean')
                result.append(dict(id=name, algorithm=algorithm, mode='batch', n=n, p=p, k=0,
                                   calibration=None, contaminated=contaminated, detection=False, chunk=None))
    if suite == 'revision':
        selected = {
            'force-2000x20-clean-b256', 'mad-2000x20-clean-b1',
            'mad-2000x20-dirty-detect-b256', 'mad-5000x50-clean-b256',
            'dual-2000x100-dirty-b256', 'dual-6000x200-clean-b1',
            'batch-force-5000x50-dirty', 'batch-mad-5000x50-dirty',
        }
        result = [item for item in result if item['id'] in selected]
    elif suite != 'full':
        raise ValueError('suite must be revision or full')
    return result


def make_data(spec, seed=20261108):
    import numpy as np
    rng = np.random.default_rng(seed + spec['n'] + spec['p'])
    latent = rng.normal(size=(spec['n'], 1))
    values = .6 * latent + .8 * rng.normal(size=(spec['n'], spec['p']))
    if spec['contaminated']:
        selected = rng.choice(spec['n'], spec['n'] // 10, replace=False)
        values[selected] += 15 * rng.standard_t(3, size=(len(selected), spec['p']))
    return np.ascontiguousarray(values)


def input_hash(values):
    return hashlib.sha256(memoryview(values).cast('B')).hexdigest()


def scheduled_jobs(design):
    jobs = []
    for session in range(design['sessions']):
        blocks = [(index, item['id'], seed_index, seed) for index, item in enumerate(design['workloads'])
                  for seed_index, seed in enumerate(design['seeds'])]
        random.Random(design['schedule_seed'] + session).shuffle(blocks)
        for block_index, (workload_index, name, seed_index, seed) in enumerate(blocks):
            for repeat in range(design['repetitions']):
                # Each three-repeat cell is a Latin rotation: every variant
                # occupies every position once. Reverse orientation in alternate
                # session/seed cells, balancing pairwise precedence to within one
                # over the declared 27 matched triplets per workload.
                order = list(VARIANTS)
                if (session * len(design['seeds']) + seed_index + workload_index) % 2:
                    order.reverse()
                rotation = repeat % len(order)
                order = order[rotation:] + order[:rotation]
                pair_id = f's{session:03d}--{name}--seed{seed}--r{repeat:03d}'
                for position, variant in enumerate(order):
                    jobs.append(dict(job_id=f'{pair_id}--{variant}', pair_id=pair_id,
                                     workload_id=name, session=session, seed=seed, repeat=repeat,
                                     variant=variant, order=position, block=block_index, position=len(jobs)))
    return jobs


def get_affinity(cpu):
    available = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    if cpu is not None:
        if available is None:
            raise EvidenceError('CPU pinning is not supported on this platform')
        if cpu not in available:
            raise EvidenceError(f'CPU {cpu} is outside the allowed affinity set')
        return [cpu]
    return available


@contextmanager
def run_lock(directory):
    path = Path(directory) / '.run.lock'
    try:
        stream = path.open('x')
    except FileExistsError as exc:
        raise EvidenceError(f'{path} exists; another run may be active. Inspect a stale lock before removing it.') from exc
    try:
        with stream:
            stream.write(json.dumps({'pid': os.getpid(), 'created_at': datetime.now(timezone.utc).isoformat()}))
        yield
    finally:
        path.unlink(missing_ok=True)


def _worker(directory, job_id):
    manifest = load_manifest(directory)
    job = next((item for item in manifest['jobs'] if item['job_id'] == job_id), None)
    if job is None:
        raise EvidenceError('Unknown worker job')
    cpu = manifest['design']['cpu']
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    if file_fingerprints() != manifest['files']:
        raise EvidenceError('Files changed before worker import')
    started = time.perf_counter()
    import numpy as np
    environment = numerical_environment(manifest['design']['cache_policy'])
    if environment != manifest['environment']:
        raise EvidenceError('Worker environment differs from run manifest')
    require_compiled_environment(environment)
    ft, imported_identity = import_comparator(directory, manifest, job['variant'])
    import_s = time.perf_counter() - started
    spec = next(item for item in manifest['design']['workloads'] if item['id'] == job['workload_id'])
    values = make_data(spec, job['seed'])
    observed_hash = input_hash(values)
    if observed_hash != manifest['inputs'][f"{spec['id']}@{job['seed']}"]:
        raise EvidenceError('Worker generated unexpected input bytes')

    def create():
        if spec['mode'] == 'batch':
            return ft.ForceEstimator() if spec['algorithm'] == 'force' else ft.MADForceEstimator(spec['p'])
        if spec['algorithm'] == 'force':
            return ft.StreamingForceEstimator(window_size=spec['calibration'])
        if spec['algorithm'] == 'mad':
            return ft.StreamingMADForceEstimator(spec['p'], use_drift_detection=spec['detection'])
        return ft.DualBoundForceEstimator(spec['p'], spec['k'], calibration_size=spec['calibration'])

    def consume(estimator, data):
        if spec['mode'] == 'batch':
            if estimator.fit(data) is not estimator:
                raise EvidenceError('fit must return self')
        elif spec['chunk'] == 1:
            for value in data:
                if estimator.update(value) is not estimator:
                    raise EvidenceError('update must return self')
        else:
            for offset in range(0, len(data), spec['chunk']):
                if estimator.update_batch(data[offset:offset + spec['chunk']]) is not estimator:
                    raise EvidenceError('update_batch must return self')

    started = time.perf_counter()
    warm = create()
    warm_size = min(len(values), 240 if spec['mode'] == 'batch' else spec['calibration'] + 40)
    consume(warm, values[:warm_size])
    warm.finalize()
    if spec['algorithm'] == 'mad' and spec['mode'] == 'stream':
        warm._moments.clear_dimension(0)
    first_use_s = time.perf_counter() - started
    del warm
    started = time.perf_counter()
    estimator = create()
    construction_s = time.perf_counter() - started
    calibration_s = processing_s = fit_s = None
    if spec['mode'] == 'batch':
        started = time.perf_counter(); consume(estimator, values); fit_s = time.perf_counter() - started
    else:
        started = time.perf_counter(); consume(estimator, values[:spec['calibration']]); calibration_s = time.perf_counter() - started
        started = time.perf_counter(); consume(estimator, values[spec['calibration']:]); processing_s = time.perf_counter() - started
    started = time.perf_counter(); output = estimator.finalize(); readout_s = time.perf_counter() - started
    scale_scatter_readout_s = scale_scatter_finite = None
    if spec['algorithm'] == 'dual':
        started = time.perf_counter()
        scale_scatter = estimator.get_scale_scatter()
        scale_scatter_readout_s = time.perf_counter() - started
        scale_scatter_finite = bool(np.isfinite(scale_scatter).all())
    diagnostics = estimator.get_diagnostics()
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin':
            peak *= 1024
        memory_measure = 'process_lifetime_peak_rss_including_import_warmup_and_diagnostics'
    except ImportError:
        import psutil
        memory = psutil.Process().memory_info()
        if not hasattr(memory, 'peak_wset'):
            raise EvidenceError('Platform does not expose a peak-process memory measure')
        peak = memory.peak_wset
        memory_measure = 'process_lifetime_peak_working_set'
    if file_fingerprints() != manifest['files']:
        raise EvidenceError('Files changed during worker measurement')
    final_identity = runtime_identity(ft, directory, manifest, job['variant'])
    # Delayed library imports are allowed, but all loaded module origins/bytes
    # are checked after processing as well as before it.
    record = dict(schema_version=SCHEMA, run_id=manifest['run_id'], manifest_sha256=manifest['manifest_sha256'],
                  job_id=job_id, job=job, recorded_at=datetime.now(timezone.utc).isoformat(),
                  runtime_identity=final_identity,
                  environment=environment, affinity=get_affinity(cpu), input_sha256=observed_hash,
                  import_s=import_s, first_use_s=first_use_s, construction_s=construction_s,
                  calibration_s=calibration_s, processing_s=processing_s, fit_s=fit_s, readout_s=readout_s,
                  scale_scatter_readout_s=scale_scatter_readout_s, scale_scatter_finite=scale_scatter_finite,
                  retained_state=diagnostics['state'], peak_process_bytes=int(peak), memory_measure=memory_measure,
                  returns_self=True, output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                  output_shape=list(output.shape), output_dtype=str(output.dtype), output_finite=bool(np.isfinite(output).all()))
    validate_record(record, manifest, job)
    print(json.dumps(record, allow_nan=False))


def worker(directory, job_id):
    manifest = load_manifest(directory)
    if manifest['design']['cache_policy'] == 'cold_per_process':
        with tempfile.TemporaryDirectory(prefix='force-toolkit-cold-') as cache:
            os.environ['NUMBA_CACHE_DIR'] = cache
            _worker(directory, job_id)
    else:
        job = next(item for item in manifest['jobs'] if item['job_id'] == job_id)
        cache = Path(directory).resolve() / '.numba-cache' / job['variant']
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = str(cache)
        _worker(directory, job_id)


def execute(directory, manifest, *, session=None):
    with run_lock(directory):
        ensure_prior_extracted(directory, manifest)
        records = read_records(directory, manifest)
        jobs = [job for job in manifest['jobs'] if session is None or job['session'] == session]
        missing = [job for job in jobs if job['job_id'] not in records]
        if not missing:
            print('Explicit resume found no pending measurements; no new benchmark was run.', flush=True)
        for job in missing:
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(directory), '--job', job['job_id']]
            completed = subprocess.run(command, cwd=ROOT, env={**os.environ, **THREAD_ENV}, capture_output=True, text=True)
            if completed.returncode:
                raise EvidenceError(f"Worker failed for {job['job_id']}:\n{completed.stderr}")
            try:
                record = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"Worker did not emit one complete record: {job['job_id']}") from exc
            validate_record(record, manifest, job)
            comparable = [row for row in records.values() if row['job']['workload_id'] == job['workload_id']
                          and row['job']['seed'] == job['seed']]
            if comparable and any(row['output_sha256'] != record['output_sha256'] for row in comparable):
                raise EvidenceError(f"Ordinary readout differs across variants/repeats: {job['workload_id']} seed={job['seed']}")
            # Validate all evidence, including cross-variant output agreement,
            # before publishing an updated summary. Each record is immutable.
            atomic_json(Path(directory) / 'records' / (job['job_id'] + '.json'), record)
            records[job['job_id']] = record
            if job['order'] == 2:
                print(f"session {job['session'] + 1}/{manifest['design']['sessions']} {job['workload_id']} seed={job['seed']} triplet={job['repeat'] + 1}/{manifest['design']['repetitions']}", flush=True)
        read_records(directory, manifest)
        return summarize(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--job', help=argparse.SUPPRESS)
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--output', type=Path, help='fresh directory, which must not exist')
    location.add_argument('--resume', type=Path, help='explicitly resume a matching schema-3 run')
    location.add_argument('--summarize', type=Path, help='validate raw evidence and recompute its summary')
    parser.add_argument('--suite', choices=('revision', 'full'), default=None)
    parser.add_argument('--filter', help='regular expression selecting workload IDs')
    parser.add_argument('--sessions', type=int, default=None)
    parser.add_argument('--seeds', default=None, help='comma-separated nonnegative input-generator seeds')
    parser.add_argument('--repetitions', type=int, default=None, help='matched triplet repeats per session/input-seed cell')
    parser.add_argument('--schedule-seed', type=int, default=None)
    parser.add_argument('--cpu', type=int, default=None)
    parser.add_argument('--cache', choices=('shared', 'cold'), default=None)
    parser.add_argument('--session', type=int, default=None, help='run only one zero-based session pass')
    args = parser.parse_args(argv)
    try:
        if args.worker is not None:
            return worker(args.worker, args.job)
        if args.summarize is not None:
            result = summarize(args.summarize)
            print(json.dumps({'complete': result['complete'], 'aggregate_streaming_mad': result['aggregate_streaming_mad']}))
            return
        if args.resume is not None:
            directory = args.resume.resolve()
            manifest = load_manifest(directory, check_current=True)
            design = manifest['design']
            requested = {'suite': args.suite, 'sessions': args.sessions, 'repetitions': args.repetitions,
                         'schedule_seed': args.schedule_seed, 'cpu': args.cpu,
                         'cache_policy': None if args.cache is None else ('shared_disk_cache' if args.cache == 'shared' else 'cold_per_process')}
            if args.seeds is not None:
                requested['seeds'] = [int(item) for item in args.seeds.split(',')]
            for key, value in requested.items():
                if value is not None and value != design[key]:
                    raise EvidenceError(f'Resume option {key} differs from manifest')
            if args.filter is not None:
                raise EvidenceError('Resume uses its complete declared workload selection; omit --filter')
            if get_affinity(design['cpu']) != design['affinity']:
                raise EvidenceError('Available CPU affinity differs from manifest')
        else:
            import re
            suite = args.suite or 'revision'
            selected = workloads(suite)
            if args.filter is not None:
                selected = [item for item in selected if re.search(args.filter, item['id'])]
            sessions = 3 if args.sessions is None else args.sessions
            repetitions = 3 if args.repetitions is None else args.repetitions
            seeds = [20261109, 20261110, 20261111] if args.seeds is None else [int(item) for item in args.seeds.split(',')]
            if not selected or sessions < 1 or repetitions < 1 or not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
                raise EvidenceError('Require selected workloads, positive sessions/repetitions, and unique nonnegative seeds')
            design = dict(suite=suite, workloads=selected, sessions=sessions, seeds=seeds, variants=list(VARIANTS),
                          repetitions=repetitions, schedule_seed=20261109 if args.schedule_seed is None else args.schedule_seed,
                          cpu=args.cpu, affinity=get_affinity(args.cpu),
                          cache_policy='cold_per_process' if args.cache == 'cold' else 'shared_disk_cache',
                          execution='serial fresh subprocesses; independently randomized session passes; Latin triplets',
                          comparisons={'acceleration': 'unoptimized integration / current',
                                       'revision_regression': 'optimized 0.1.0 wheel / current'},
                          readout_phases='finalize; separate Dual-Bound scale-scatter query; both outside processing/fit',
                          warmup='fresh estimator, first 240 batch rows or calibration+40 stream rows; then fresh measured estimator')
            environment = probe_environment(design['cache_policy'])
            require_compiled_environment(environment)
            inputs = {f"{item['id']}@{seed}": input_hash(make_data(item, seed)) for item in selected for seed in seeds}
            directory, manifest = fresh_run(design, scheduled_jobs(design), inputs, output=args.output, environment=environment)
        if args.session is not None and not 0 <= args.session < manifest['design']['sessions']:
            raise EvidenceError('--session is outside the declared session range')
        print(f'Run directory: {directory}', flush=True)
        print(f"Manifest: {manifest['manifest_sha256']}; {len(manifest['jobs'])} scheduled measurements", flush=True)
        result = execute(directory, manifest, session=args.session)
        print(json.dumps({'complete': result['complete'], 'aggregate_streaming_mad': result['aggregate_streaming_mad']}))
    except (EvidenceError, FileExistsError, ValueError) as exc:
        parser.exit(2, f'benchmark evidence error: {exc}\n')


if __name__ == '__main__':
    main()
