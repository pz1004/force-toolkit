"""Schema 4: frozen 0.1.2 versus candidate, balanced pairs, fresh serial workers.

Historical schemas and their auditors are unchanged. This module owns its new
manifest, record validation and crossed session/seed summary independently.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks._protocol import (THREAD_ENV, EvidenceError, atomic_json, digest,
    numerical_environment, probe_environment, require_compiled_environment)
from benchmarks.run import workloads, make_data, input_hash, get_affinity, run_lock

SCHEMA = 4
VARIANTS = ('baseline', 'candidate')


def files(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(path).rglob('*.py'))}


def identity(path, wheel=None):
    result = dict(package=str(Path(path).resolve()), files=files(path))
    result['tree_sha256'] = digest(result['files'])
    if wheel is not None:
        import zipfile
        result.update(wheel=str(Path(wheel).resolve()), wheel_sha256=hashlib.sha256(Path(wheel).read_bytes()).hexdigest())
        with zipfile.ZipFile(wheel) as archive:
            members = {name.removeprefix('force_toolkit/'): hashlib.sha256(archive.read(name)).hexdigest()
                       for name in archive.namelist() if name.startswith('force_toolkit/') and name.endswith('.py')}
        if members != result['files']:
            raise EvidenceError('Wheel Python bytes differ from the selected source comparator')
    return result


def harness_files():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        'benchmarks/revision020.py', 'benchmarks/run.py', 'benchmarks/_protocol.py',
        'benchmarks/_comparators.py', 'benchmarks/_statistics.py')}


def primary():
    selected = {w['id'] for w in workloads('revision')} | {
        'mad-2000x20-clean-b256', 'mad-2000x20-clean-detect-b256',
        'mad-2000x20-dirty-b256', 'mad-5000x50-clean-detect-b256',
        'mad-5000x50-dirty-b256', 'mad-5000x50-dirty-detect-b256',
        'batch-mad-2000x20-clean', 'dual-6000x200-clean-b256'}
    return [w for w in workloads() if w['id'] in selected]


def stress_workloads():
    template = next(w for w in workloads() if w['id'] == 'mad-2000x20-clean-detect-b256')
    result = []
    for kind in ('resets', 'long_window', 'guarded', 'callbacks', 'profiling', 'readout', 'latency'):
        item = dict(template, id='stress-mad-' + kind, stress=kind)
        if kind == 'resets':
            item['n'] = 10000
        if kind == 'long_window':
            item['n'] = 24000
        if kind == 'latency':
            item['chunk'] = 1
        result.append(item)
    for algorithm in ('force', 'dual'):
        item = dict(next(w for w in workloads('revision') if w['algorithm'] == algorithm and w['mode'] == 'stream'),
                    id='stress-' + algorithm + '-latency', stress='latency', chunk=1)
        result.append(item)
    return result


def scheduled_jobs(design):
    jobs = []
    for session in range(design['sessions']):
        rng = random.Random(design['schedule_seed'] + session)
        blocks = [(w['id'], seed) for w in design['workloads'] for seed in design['seeds']]
        rng.shuffle(blocks)
        for block, (name, seed) in enumerate(blocks):
            orientation = rng.randrange(2)
            for repeat in range(design['repetitions']):
                order = VARIANTS if (orientation + repeat) % 2 == 0 else VARIANTS[::-1]
                pair = f's{session:02d}--{name}--seed{seed}--r{repeat:02d}'
                for position, variant in enumerate(order):
                    jobs.append(dict(job_id=pair + '--' + variant, pair_id=pair, session=session,
                        workload_id=name, seed=seed, repeat=repeat, variant=variant, order=position,
                        block=block, position=len(jobs)))
    return jobs


def load(directory, current=False):
    m = json.loads((Path(directory) / 'manifest.json').read_text())
    if m.get('schema_version') != SCHEMA:
        raise EvidenceError('Use the historical schema auditor for earlier records')
    if m['manifest_sha256'] != digest({k: v for k, v in m.items() if k != 'manifest_sha256'}):
        raise EvidenceError('Manifest hash mismatch')
    if m['jobs'] != scheduled_jobs(m['design']):
        raise EvidenceError('Unbalanced or modified schedule')
    if current:
        if harness_files() != m['harness']:
            raise EvidenceError('Harness changed; start a fresh experiment')
        for item in m['comparators'].values():
            if identity(item['package'], item.get('wheel')) != item:
                raise EvidenceError('Comparator changed; start a fresh experiment')
        if probe_environment(m['design']['cache_policy']) != m['environment']:
            raise EvidenceError('Numerical environment changed')
    return m


def imported_identity(module, declared):
    package = Path(declared['package'])
    if Path(module.__file__).resolve() != package / '__init__.py' or files(package) != declared['files']:
        raise EvidenceError('Imported source origin or bytes differ')
    loaded = {}
    for name, value in list(sys.modules.items()):
        if name == 'force_toolkit' or name.startswith('force_toolkit.'):
            path = Path(value.__file__).resolve()
            if not path.is_relative_to(package):
                raise EvidenceError('Imported module escaped comparator')
            relative = str(path.relative_to(package))
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if declared['files'].get(relative) != actual:
                raise EvidenceError('Imported module bytes changed')
            loaded[relative] = actual
    return dict(package=str(package), tree_sha256=declared['tree_sha256'], loaded=loaded,
                version=module.__version__, wheel_sha256=declared.get('wheel_sha256'))


def validate(record, manifest):
    import math
    job = next((j for j in manifest['jobs'] if j['job_id'] == record.get('job_id')), None)
    if job is None or record.get('job') != job or record.get('schema_version') != SCHEMA:
        raise EvidenceError('Record job/schema mismatch')
    if record.get('manifest_sha256') != manifest['manifest_sha256']:
        raise EvidenceError('Record belongs to another manifest')
    if record['input_sha256'] != manifest['inputs'][job['workload_id'] + '@' + str(job['seed'])]:
        raise EvidenceError('Input bytes differ')
    if record['environment'] != manifest['environment'] or record['affinity'] != manifest['design']['affinity']:
        raise EvidenceError('Environment or affinity differs')
    require_compiled_environment(record['environment'])
    if any(pool['num_threads'] != 1 for pool in record['environment']['native_libraries']):
        raise EvidenceError('Numerical worker must use one thread')
    declared, actual = manifest['comparators'][job['variant']], record['runtime_identity']
    if (actual['package'] != declared['package'] or actual['tree_sha256'] != declared['tree_sha256']
            or actual['wheel_sha256'] != declared.get('wheel_sha256') or not actual['loaded']
            or any(declared['files'].get(p) != h for p, h in actual['loaded'].items())):
        raise EvidenceError('Imported identity mismatch')
    spec = next(w for w in manifest['design']['workloads'] if w['id'] == job['workload_id'])
    if record['output_shape'] != [spec['p'], spec['p']] or not record['output_finite'] or not record['returns_self']:
        raise EvidenceError('Invalid result')
    for name in ('import_s', 'first_use_s', 'construction_s', 'readout_s', 'startup_to_first_result_s'):
        value = record[name]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise EvidenceError('Invalid timing: ' + name)
    if manifest['design'].get('startup_full_result'):
        value = record.get('startup_to_full_result_s', 0)
        if not math.isfinite(value) or value <= record['startup_to_first_result_s']:
            raise EvidenceError('Invalid full-result startup timing')
    for name in ('fit_s',) if spec['mode'] == 'batch' else ('calibration_s', 'processing_s'):
        if not math.isfinite(record[name]) or record[name] <= 0:
            raise EvidenceError('Invalid processing timing')
    if record['peak_process_bytes'] <= 0 or record['retained_state']['bytes'] < 0:
        raise EvidenceError('Invalid memory accounting')
    return job


def records(directory, manifest):
    result, outputs = {}, {}
    for path in sorted((Path(directory) / 'records').iterdir()):
        if path.suffix != '.json':
            raise EvidenceError('Partial/unexpected record: ' + str(path))
        r = json.loads(path.read_text())
        j = validate(r, manifest)
        if path.stem != j['job_id'] or j['job_id'] in result:
            raise EvidenceError('Misnamed/duplicate record')
        key = j['workload_id'], j['seed']
        if key in outputs and outputs[key] != r['output_sha256']:
            raise EvidenceError('Ordinary result differs across paired implementations')
        outputs[key] = r['output_sha256']
        result[j['job_id']] = r
    return result


def create_estimator(ft, spec, variant):
    if spec['mode'] == 'batch':
        return ft.ForceEstimator() if spec['algorithm'] == 'force' else ft.MADForceEstimator(spec['p'])
    if spec['algorithm'] == 'force':
        return ft.StreamingForceEstimator(window_size=spec['calibration'])
    if spec['algorithm'] == 'dual':
        return ft.DualBoundForceEstimator(spec['p'], spec['k'], calibration_size=spec['calibration'])
    extra = {}
    kind = spec.get('stress')
    if kind == 'guarded':
        extra['tail_mode'] = 'guarded_asymmetric'
    if kind == 'callbacks':
        extra['diagnostic_sink'] = lambda event: None
    if kind == 'profiling' and variant == 'candidate':
        extra['collect_component_timing'] = True
    return ft.StreamingMADForceEstimator(spec['p'], use_drift_detection=spec['detection'], **extra)


def data(spec, seed):
    import numpy as np
    X = make_data(spec, seed)
    if spec.get('stress') == 'resets':
        X += ((np.arange(len(X)) // 1000) % 2)[:, None] * 20.
    elif spec.get('stress') == 'latency' and spec['algorithm'] == 'mad':
        X[1000:1500] += 20.
    return X


def consume(estimator, values, spec, latency=None):
    if spec['mode'] == 'batch':
        if estimator.fit(values) is not estimator:
            raise EvidenceError('fit return contract')
        return
    for offset in range(0, len(values), spec['chunk']):
        if latency is not None:
            before = estimator.get_diagnostics()
            previous_compressions = (estimator._engine._sketch.compressions
                if spec['algorithm'] == 'dual' and estimator._engine._sketch is not None else 0)
            start = time.perf_counter_ns()
        if spec['chunk'] == 1:
            result = estimator.update(values[offset])
        else:
            result = estimator.update_batch(values[offset:offset + spec['chunk']])
        if result is not estimator:
            raise EvidenceError('update return contract')
        if latency is not None:
            elapsed = time.perf_counter_ns() - start
            after = estimator.get_diagnostics()
            label = 'ordinary'
            if spec['algorithm'] == 'mad' and after['details']['total_resets'] > before['details']['total_resets']:
                label = 'alarm_reset'
            elif spec['algorithm'] == 'dual':
                engine = estimator._engine
                if engine.calibration_count == engine.calibration_size and engine.effective_n == 0:
                    label = 'calibration_completion'
                elif engine._sketch is not None and engine._sketch.compressions > previous_compressions:
                    label = 'compression'
            elif spec['algorithm'] == 'force' and estimator.window_count_ == 0:
                label = 'calibration_completion'
            latency.append(dict(row=after['observations'], phase=label, nanoseconds=elapsed))
        if spec.get('stress') == 'readout' and estimator.get_diagnostics()['ready']:
            estimator.finalize()


def _worker(directory, job_id):
    started_process = float(os.environ['FORCE_BENCH_PROCESS_STARTED'])
    manifest = load(directory)
    job = next(j for j in manifest['jobs'] if j['job_id'] == job_id)
    spec = next(w for w in manifest['design']['workloads'] if w['id'] == job['workload_id'])
    os.sched_setaffinity(0, set(manifest['design']['affinity']))
    start = time.perf_counter()
    import numpy as np
    environment = numerical_environment(manifest['design']['cache_policy'])
    if environment != manifest['environment']:
        raise EvidenceError('Worker numerical environment differs')
    require_compiled_environment(environment)
    declared = manifest['comparators'][job['variant']]
    sys.path.insert(0, str(Path(declared['package']).parent))
    ft = importlib.import_module('force_toolkit')
    imported_identity(ft, declared)
    import_s = time.perf_counter() - start
    X = data(spec, job['seed'])
    observed_hash = input_hash(X)
    latency = [] if spec.get('stress') == 'latency' else None
    start = time.perf_counter()
    warm = create_estimator(ft, spec, job['variant'])
    warm_size = 240 if spec['mode'] == 'batch' else spec['calibration'] + 40
    consume(warm, X[:warm_size], spec)
    warm.finalize()
    first_result = time.perf_counter() - started_process
    if spec['algorithm'] == 'mad' and spec['mode'] == 'stream':
        warm._moments.clear_dimension(0)
    first_use_s = time.perf_counter() - start
    del warm
    start = time.perf_counter()
    estimator = create_estimator(ft, spec, job['variant'])
    construction_s = time.perf_counter() - start
    calibration_s = processing_s = fit_s = None
    if spec['mode'] == 'batch':
        start = time.perf_counter(); consume(estimator, X, spec); fit_s = time.perf_counter() - start
    else:
        c = spec['calibration']
        start = time.perf_counter(); consume(estimator, X[:c], spec, latency); calibration_s = time.perf_counter() - start
        start = time.perf_counter(); consume(estimator, X[c:], spec, latency); processing_s = time.perf_counter() - start
    start = time.perf_counter(); output = estimator.finalize(); readout_s = time.perf_counter() - start
    full_result = time.perf_counter() - started_process
    scale_s = None
    if spec['algorithm'] == 'dual':
        start = time.perf_counter(); scale = estimator.get_scale_scatter(); scale_s = time.perf_counter() - start
        if not np.isfinite(scale).all():
            raise EvidenceError('Invalid scale scatter')
    diagnostics = estimator.get_diagnostics()
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    # Numerical workspace envelopes, separately labelled; not process RSS.
    p, n = spec['p'], spec['n']
    workspace = (8 * (6 * n * p + 2 * n) if spec['mode'] == 'batch' else
                 estimator._engine.calibration_peak_state_bytes_bound if spec['algorithm'] == 'dual' else
                 9 * max(spec['chunk'], spec['calibration']) * p +
                 (sum(a.nbytes for a in estimator._detector_state.arrays) if hasattr(estimator, '_detector_state') else 0))
    record = dict(schema_version=SCHEMA, manifest_sha256=manifest['manifest_sha256'],
        job_id=job_id, job=job, environment=environment, affinity=get_affinity(None),
        input_sha256=observed_hash, runtime_identity=imported_identity(ft, declared),
        import_s=import_s, first_use_s=first_use_s, startup_to_first_result_s=first_result,
        startup_to_full_result_s=full_result, construction_s=construction_s, calibration_s=calibration_s, processing_s=processing_s,
        fit_s=fit_s, readout_s=readout_s, scale_scatter_readout_s=scale_s,
        retained_state=diagnostics['state'], transient_numerical_workspace_envelope_bytes=workspace,
        workspace_scope='conservative phase array envelope; excludes outputs, Python/LLVM/BLAS overhead and source-owned input',
        peak_process_bytes=int(peak), memory_measure='process_lifetime_peak_RSS_including_import_compilation_diagnostics',
        output_sha256=input_hash(output), output_shape=list(output.shape), output_finite=bool(np.isfinite(output).all()),
        returns_self=True, latency=latency, recorded_at=datetime.now(timezone.utc).isoformat())
    validate(record, manifest)
    print(json.dumps(record, allow_nan=False))


def worker(directory, job_id):
    manifest = load(directory)
    if manifest['design']['cache_policy'] == 'cold_per_process':
        with tempfile.TemporaryDirectory(prefix='force020-cold-') as cache:
            os.environ['NUMBA_CACHE_DIR'] = cache
            _worker(directory, job_id)
    else:
        variant = next(j['variant'] for j in manifest['jobs'] if j['job_id'] == job_id)
        cache = Path(directory).resolve() / '.numba-cache' / variant
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = str(cache)
        _worker(directory, job_id)


def execute(directory):
    m = load(directory, current=True)
    with run_lock(directory):
        done = records(directory, m)
        for job in m['jobs']:
            if job['job_id'] in done:
                continue
            start = time.perf_counter()
            env = {**os.environ, **THREAD_ENV, 'FORCE_BENCH_PROCESS_STARTED': str(start)}
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(directory), job['job_id']],
                                    env=env, capture_output=True, text=True)
            if result.returncode:
                raise EvidenceError(result.stderr[-6000:])
            r = json.loads(result.stdout)
            r['process_startup_to_exit_s'] = time.perf_counter() - start
            validate(r, m)
            atomic_json(Path(directory) / 'records' / (job['job_id'] + '.json'), r)
            done[job['job_id']] = r
            if len(done) % 20 == 0:
                print(f'{len(done)}/{len(m["jobs"])} workers complete', flush=True)
        load(directory, current=True)
        records(directory, m)
    return summarize(directory)


def summarize(directory):
    m = load(directory)
    return summarize_observations(m, records(directory, m))


def summarize_observations(m, rr):
    """Aggregate validated records with the original crossed session/seed draws."""
    import numpy as np
    d = m['design']
    if len(rr) != len(m['jobs']):
        return dict(complete=False, records=len(rr), expected=len(m['jobs']))
    lookup = {(r['job']['workload_id'], r['job']['session'], r['job']['seed'], r['job']['repeat'], r['job']['variant']): r for r in rr.values()}
    logs = np.empty((len(d['workloads']), d['sessions'], len(d['seeds']), d['repetitions']))
    rng = np.random.default_rng(41000)
    sessions = rng.integers(0, d['sessions'], (10000, d['sessions']))
    seeds = rng.integers(0, len(d['seeds']), (10000, len(d['seeds'])))
    def result(values):
        means = values.mean(axis=(0, 3))
        draws = means[sessions[:, :, None], seeds[:, None, :]].mean(axis=(1, 2))
        ci = np.exp(np.quantile(draws, [.025, .975])).tolist() if d['sessions'] > 1 and len(d['seeds']) > 1 else None
        return dict(speedup=float(np.exp(values.mean())), exploratory_ci95=ci)
    configurations = []
    for index, spec in enumerate(d['workloads']):
        phase = 'fit_s' if spec['mode'] == 'batch' else 'processing_s'
        for session in range(d['sessions']):
            for seed_i, seed in enumerate(d['seeds']):
                for repeat in range(d['repetitions']):
                    pair = [lookup[spec['id'], session, seed, repeat, v][phase] for v in VARIANTS]
                    logs[index, session, seed_i, repeat] = np.log(pair[0] / pair[1])
        entry = dict(id=spec['id'], primary_phase=phase, **result(logs[index:index+1]), phases={})
        for variant in VARIANTS:
            chosen = [r for r in rr.values() if r['job']['workload_id'] == spec['id'] and r['job']['variant'] == variant]
            fields = ('import_s', 'first_use_s', 'startup_to_first_result_s', 'startup_to_full_result_s', 'construction_s', 'calibration_s',
                'processing_s', 'fit_s', 'readout_s', 'scale_scatter_readout_s', 'peak_process_bytes', 'transient_numerical_workspace_envelope_bytes')
            entry['phases'][variant] = {f: float(np.median([r[f] for r in chosen])) if chosen[0].get(f) is not None else None for f in fields}
            entry['phases'][variant]['retained_state_bytes'] = float(np.median([r['retained_state']['bytes'] for r in chosen]))
        configurations.append(entry)
    aggregates = {}
    for label, predicate in (
        ('streaming_mad', lambda w: w['algorithm'] == 'mad' and w['mode'] == 'stream'),
        ('stationary_mad', lambda w: w['algorithm'] == 'mad' and w['mode'] == 'batch'),
        ('force', lambda w: w['algorithm'] == 'force'),
        ('dual', lambda w: w['algorithm'] == 'dual')):
        indices = [i for i, w in enumerate(d['workloads']) if predicate(w)]
        if indices:
            aggregates[label] = result(logs[indices])
    mad = aggregates.get('streaming_mad')
    gates = dict(streaming_mad_target=bool(mad and mad['speedup'] >= 2 and mad['exploratory_ci95'] and mad['exploratory_ci95'][0] > 1.25),
                 every_primary_noninferior=all(r['exploratory_ci95'] and r['exploratory_ci95'][0] > 1/1.1 for r in configurations))
    return dict(schema_version=SCHEMA, manifest_sha256=m['manifest_sha256'], complete=True, measurements=len(rr),
                configurations=configurations, aggregates=aggregates, gates=gates,
                interval_scope='exploratory crossed session/seed percentile bootstrap; 10000 shared matched draws; equal configuration log weights; no universal or simultaneous coverage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=2)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--suite', choices=['screen', 'primary', 'stress', 'cold', 'ablation'], default='primary')
    parser.add_argument('--candidate', type=Path, default=ROOT / 'src/force_toolkit')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'tests/reference/force_toolkit_012')
    parser.add_argument('--candidate-wheel', type=Path)
    parser.add_argument('--cpu', type=int, default=min(os.sched_getaffinity(0)))
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker); return
    if args.audit:
        print(json.dumps(summarize(args.audit), indent=2)); return
    if args.resume:
        print(json.dumps(execute(args.resume), indent=2)); return
    suite = args.suite
    chosen = workloads() if suite == 'screen' else primary() if suite == 'primary' else stress_workloads() if suite == 'stress' else workloads('revision')
    if suite == 'cold':
        chosen = [w for w in chosen if w['id'] in {'force-2000x20-clean-b256', 'mad-2000x20-clean-b1', 'dual-2000x100-dirty-b256'}]
    design = dict(suite=suite, startup_full_result=True, workloads=chosen, sessions=5 if suite == 'primary' else 1,
        seeds=list(range(41001, 41006)) if suite == 'primary' else list(range(42001, 42004)) if suite == 'stress' else [41001],
        repetitions=6 if suite == 'cold' else 2, schedule_seed=41000,
        affinity=get_affinity(args.cpu), cache_policy='cold_per_process' if suite == 'cold' else 'shared_disk_cache')
    environment = probe_environment(design['cache_policy'])
    require_compiled_environment(environment)
    inputs = {w['id'] + '@' + str(seed): input_hash(data(w, seed)) for w in chosen for seed in design['seeds']}
    baseline = args.baseline.resolve()
    # Frozen reference is imported under the normal package name in isolated copies.
    if baseline.name != 'force_toolkit':
        import shutil
        isolated = ROOT / 'benchmarks/variants/baseline012/force_toolkit'
        if not isolated.exists():
            shutil.copytree(baseline, isolated, ignore=shutil.ignore_patterns('__pycache__'))
        if files(baseline) != files(isolated):
            raise EvidenceError('Isolated baseline differs')
        baseline = isolated
    prior_wheel = ROOT / 'benchmarks/artifacts/force_toolkit-0.1.2-py3-none-any.whl' if args.baseline == ROOT / 'tests/reference/force_toolkit_012' else None
    comparators = dict(baseline=identity(baseline, prior_wheel), candidate=identity(args.candidate, args.candidate_wheel))
    directory = args.output or ROOT / 'benchmarks/runs' / ('revision020-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'records').mkdir()
    m = dict(schema_version=SCHEMA, design=design, jobs=scheduled_jobs(design), environment=environment,
             comparators=comparators, harness=harness_files(), inputs=inputs,
             created_at=datetime.now(timezone.utc).isoformat())
    m['manifest_sha256'] = digest(m)
    atomic_json(directory / 'manifest.json', m)
    print(directory, flush=True)
    summary = execute(directory)
    atomic_json(directory / 'summary.json', summary)
    print(json.dumps(summary['aggregates'], indent=2), flush=True)


if __name__ == '__main__':
    main()
