"""Immutable run manifests and validated, atomic measurement records."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 3
THREAD_ENV = {key: '1' for key in (
    'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
)}


class EvidenceError(ValueError):
    """A run or measurement does not match its declared experiment."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_fingerprints(root=ROOT):
    result = {}
    for directory in ('src/force_toolkit', 'tests/reference/force_toolkit_baseline', 'benchmarks'):
        base = root / directory
        paths = base.glob('*.py') if directory == 'benchmarks' else base.rglob('*.py')
        for path in sorted(paths):
            if '__pycache__' not in path.parts and 'release-envs' not in path.parts:
                result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = root / 'pyproject.toml'
    result['pyproject.toml'] = hashlib.sha256(path.read_bytes()).hexdigest()
    from benchmarks._comparators import WHEEL_RELATIVE
    path = root / WHEEL_RELATIVE
    result[WHEEL_RELATIVE] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def numba_compilation_environment(cache_policy='shared_disk_cache'):
    """Effective CPU/JIT configuration, including file-based configuration.

    CACHE_DIR is normalized only because the runner sets a managed directory
    before import. The unused default thread capacity depends on affinity;
    the effective NUMBA_NUM_THREADS remains bound and fixed to one.
    """
    import numba
    from llvmlite import binding

    def plain(value):
        if type(value).__name__ == '_OptLevel':
            return {'value': int(value), 'is_opt_max': bool(value.is_opt_max)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return str(value)

    cache_identity = ('empty temporary directory per worker' if cache_policy == 'cold_per_process'
                      else 'per-run per-variant shared directory')
    configuration = {key: plain(value) for key, value in vars(numba.config).items()
                     if key.isupper() and not key.startswith('_')
                     and key not in ('CACHE_DIR', 'NUMBA_DEFAULT_NUM_THREADS')}
    configuration['CACHE_DIR'] = cache_identity
    overrides = {key: value for key, value in sorted(os.environ.items())
                 if key.startswith('NUMBA_') and key != 'NUMBA_CACHE_DIR'}
    return dict(effective=configuration, environment_overrides=overrides,
                cache_policy=cache_policy,
                host_cpu_name=str(binding.get_host_cpu_name()),
                host_cpu_features=binding.get_host_cpu_features().flatten())


def require_compiled_environment(environment):
    compiled = environment.get('numba_compilation', {}).get('effective', {})
    if compiled.get('DISABLE_JIT') != 0:
        raise EvidenceError('Acceleration evidence requires Numba JIT enabled (NUMBA_DISABLE_JIT=0)')


def numerical_environment(cache_policy='shared_disk_cache'):
    """Capture numerical-library identity in a process with the worker thread policy."""
    import numpy  # noqa: F401 -- load its native library before inspection
    import numba  # noqa: F401
    from threadpoolctl import threadpool_info

    versions = {}
    for name in ('numpy', 'numba', 'llvmlite', 'scipy', 'threadpoolctl', 'psutil'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    native = sorted(threadpool_info(), key=lambda x: (x.get('filepath', ''), x.get('prefix', '')))
    model = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                      if line.startswith('model name')), model)
    return {
        'python': sys.version, 'implementation': platform.python_implementation(),
        'executable': str(Path(sys.executable).absolute()), 'prefix': sys.prefix,
        'packages': versions, 'native_libraries': native,
        'machine': {'hostname': socket.gethostname(), 'platform': platform.platform(),
                    'architecture': platform.machine(), 'cpu_count': os.cpu_count(), 'cpu_model': model},
        'thread_environment': {key: os.environ.get(key) for key in THREAD_ENV},
        'numba_compilation': numba_compilation_environment(cache_policy),
    }


def probe_environment(cache_policy='shared_disk_cache'):
    code = ('import json; from benchmarks._protocol import numerical_environment; '
            'import sys; print(json.dumps(numerical_environment(sys.argv[1])))')
    result = subprocess.run([sys.executable, '-c', code, cache_policy], cwd=ROOT,
                            env={**os.environ, **THREAD_ENV}, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def atomic_json(path, value):
    """Publish one complete JSON object; never silently replace evidence."""
    path = Path(path)
    if path.exists():
        raise EvidenceError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def fresh_run(design, jobs, inputs, *, output=None, root=ROOT, environment=None):
    from benchmarks._comparators import comparator_identities, expected_package
    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%S.%fZ') + '-' + uuid.uuid4().hex[:8]
    directory = Path(output) if output is not None else root / 'benchmarks/runs' / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema_version': SCHEMA, 'run_id': run_id, 'created_at': now.isoformat(),
        'run_directory': str(directory.resolve()),
        'path_style': 'windows' if os.name == 'nt' else 'posix',
        'design': design, 'jobs': jobs, 'inputs': inputs,
        'project_root': str(Path(root).resolve()),
        'comparators': comparator_identities(root),
        'files': file_fingerprints(root),
        'environment': probe_environment(design['cache_policy']) if environment is None else environment,
    }
    for variant, implementation in manifest['comparators'].items():
        implementation['package_root'] = str(expected_package(directory, manifest, variant).resolve())
    manifest['manifest_sha256'] = digest(manifest)
    atomic_json(directory / 'manifest.json', manifest)
    (directory / 'records').mkdir()
    return directory, manifest


def load_manifest(directory, *, check_current=False, root=ROOT, environment=None):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / 'manifest.json').read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f'Cannot read a complete run manifest: {directory}') from exc
    if manifest.get('schema_version') != SCHEMA:
        raise EvidenceError('Historical schema-1/2 evidence cannot resume or be reinterpreted by schema 3; use its archived harness for audit')
    body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    if digest(body) != manifest.get('manifest_sha256'):
        raise EvidenceError('Manifest hash mismatch')
    if check_current:
        if directory.resolve() != Path(manifest['run_directory']).resolve():
            raise EvidenceError('Run directory moved; start a fresh experiment')
        if file_fingerprints(root) != manifest['files']:
            raise EvidenceError('Source, reference, project metadata, or harness changed; start a fresh run')
        current = probe_environment(manifest['design']['cache_policy']) if environment is None else environment
        if current != manifest['environment']:
            raise EvidenceError('Numerical environment or machine changed; start a fresh run')
    return manifest


def validate_record(record, manifest, expected_job=None):
    import math

    if record.get('schema_version') != SCHEMA:
        raise EvidenceError('Record schema mismatch')
    if record.get('run_id') != manifest['run_id'] or record.get('manifest_sha256') != manifest['manifest_sha256']:
        raise EvidenceError('Record belongs to another run or manifest')
    jobs = {job['job_id']: job for job in manifest['jobs']}
    job = jobs.get(record.get('job_id'))
    if job is None or (expected_job is not None and job != expected_job):
        raise EvidenceError('Unknown measurement identity')
    if record.get('job') != job:
        raise EvidenceError('Record configuration does not match its scheduled job')
    key = f"{job['workload_id']}@{job['seed']}"
    if record.get('input_sha256') != manifest['inputs'][key]:
        raise EvidenceError('Input hash mismatch')
    if record.get('environment') != manifest['environment']:
        raise EvidenceError('Record numerical environment differs from manifest')
    require_compiled_environment(record['environment'])
    implementation = manifest['comparators'][job['variant']]
    actual = record.get('runtime_identity', {})
    # Paths in historical evidence belong to the execution host. Auditing a
    # Linux run on Windows (or vice versa) must not reinterpret them using the
    # reader's drive, separators or filesystem. Import-time checks use real Path.
    path_class = {'posix': PurePosixPath, 'windows': PureWindowsPath}.get(manifest.get('path_style'))
    if path_class is None or 'package_root' not in implementation:
        raise EvidenceError('Missing declared execution path style or comparator path')
    expected_root = path_class(implementation['package_root'])
    if not expected_root.is_absolute():
        raise EvidenceError('Comparator path must be absolute in its declared path style')
    if (actual.get('variant') != job['variant'] or actual.get('kind') != implementation['kind']
            or actual.get('version') != implementation['version']
            or actual.get('source_tree_sha256') != implementation['source_tree_sha256']
            or path_class(actual.get('package_root', '')) != expected_root
            or path_class(actual.get('entrypoint', '')) != expected_root / '__init__.py'):
        raise EvidenceError('Record imported implementation identity differs from its declared comparator')
    loaded = actual.get('loaded_modules')
    if not isinstance(loaded, dict) or not loaded:
        raise EvidenceError('Missing imported module identity evidence')
    for value in loaded.values():
        path = path_class(value.get('path', ''))
        if not path.is_relative_to(expected_root):
            raise EvidenceError('Imported module escaped its declared comparator')
        if implementation['python_sha256'].get(path.relative_to(expected_root).as_posix()) != value.get('sha256'):
            raise EvidenceError('Imported module hash differs from declared comparator')
    if not record.get('returns_self', False):
        raise EvidenceError('Documented mutator return contract failed')
    shape = record.get('output_shape')
    p = next(w['p'] for w in manifest['design']['workloads'] if w['id'] == job['workload_id'])
    if shape != [p, p] or record.get('output_dtype') != 'float64' or not record.get('output_finite'):
        raise EvidenceError('Invalid matrix readout')
    if not isinstance(record.get('output_sha256'), str) or len(record['output_sha256']) != 64:
        raise EvidenceError('Invalid output fingerprint')
    expected_affinity = manifest['design']['affinity']
    if record.get('affinity') != expected_affinity:
        raise EvidenceError('Worker CPU affinity differs from declared policy')
    if any(pool.get('num_threads') != 1 for pool in record['environment']['native_libraries']):
        raise EvidenceError('Worker native thread count differs from the single-thread policy')
    for key in ('import_s', 'first_use_s', 'construction_s', 'readout_s'):
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise EvidenceError(f'Invalid {key}')
    mode = next(w['mode'] for w in manifest['design']['workloads'] if w['id'] == job['workload_id'])
    active = ('fit_s',) if mode == 'batch' else ('calibration_s', 'processing_s')
    inactive = ('calibration_s', 'processing_s') if mode == 'batch' else ('fit_s',)
    for key in active:
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise EvidenceError(f'Invalid {key}')
    if any(record.get(key) is not None for key in inactive):
        raise EvidenceError('Inapplicable phase reported as a timing')
    algorithm = next(w['algorithm'] for w in manifest['design']['workloads'] if w['id'] == job['workload_id'])
    scale_time = record.get('scale_scatter_readout_s')
    if algorithm == 'dual':
        if not isinstance(scale_time, (int, float)) or not math.isfinite(scale_time) or scale_time <= 0:
            raise EvidenceError('Invalid scale-scatter readout time')
        if not record.get('scale_scatter_finite'):
            raise EvidenceError('Invalid scale-scatter readout')
    elif scale_time is not None or record.get('scale_scatter_finite') is not None:
        raise EvidenceError('Scale-scatter readout is only applicable to Dual-Bound')
    if record.get('peak_process_bytes', 0) <= 0 or record.get('retained_state', {}).get('bytes', -1) < 0:
        raise EvidenceError('Invalid memory accounting')
    return job


def read_records(directory, manifest):
    directory = Path(directory) / 'records'
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix != '.json':
            raise EvidenceError(f'Partial or unexpected record file: {path.name}; investigate before resuming')
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise EvidenceError(f'Partial or corrupt record: {path.name}') from exc
        job = validate_record(record, manifest)
        if path.name != job['job_id'] + '.json' or job['job_id'] in result:
            raise EvidenceError('Duplicate or misnamed record')
        result[job['job_id']] = record
    # Each exact ordinary-data result must agree across both variants and repeated
    # sessions. Statistical/readiness metadata intentionally differs after revision.
    outputs = {}
    for record in result.values():
        job = record['job']
        key = (job['workload_id'], job['seed'])
        value = (record['output_sha256'], tuple(record['output_shape']), record['output_dtype'])
        if key in outputs and value != outputs[key]:
            raise EvidenceError(f'Ordinary readout differs across variants/repeats: {key}')
        outputs[key] = value
    return result
