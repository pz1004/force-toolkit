"""Evidence integrity and experiment design checks; no wall-clock thresholds."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from benchmarks import _protocol as protocol
from benchmarks._comparators import VARIANTS, WHEEL_RELATIVE, expected_package
from benchmarks._statistics import intervals, summarize
from benchmarks.acceptance import assess
from benchmarks.run import input_hash, make_data, run_lock, scheduled_jobs, workloads


def design(workload=None, *, sessions=3, seeds=(11, 22, 33), repetitions=3):
    return dict(suite='revision', workloads=[workload or workloads('revision')[1]],
                variants=list(VARIANTS),
                sessions=sessions, seeds=list(seeds), repetitions=repetitions, schedule_seed=42,
                cpu=None, affinity=[0], cache_policy='shared_disk_cache')


def create_run(tmp_path, *, selected=None):
    root = tmp_path / 'project'
    root.mkdir(exist_ok=True)
    (root / 'pyproject.toml').write_text('[project]\nname="fixture"\nversion="0.1.2"\n')
    source = root / 'src/force_toolkit'
    source.mkdir(parents=True, exist_ok=True)
    (source / '__init__.py').write_text('# source\n')
    reference = root / 'tests/reference/force_toolkit_baseline'
    reference.mkdir(parents=True, exist_ok=True)
    (reference / '__init__.py').write_text('# reference\n')
    artifact = root / WHEEL_RELATIVE
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / WHEEL_RELATIVE, artifact)
    plan = selected or design()
    environment = {'native_libraries': [{'num_threads': 1}], 'python': 'fixture',
                   'numba_compilation': {'effective': {'DISABLE_JIT': 0}}}
    inputs = {f"{w['id']}@{seed}": 'a' * 64 for w in plan['workloads'] for seed in plan['seeds']}
    directory, manifest = protocol.fresh_run(plan, scheduled_jobs(plan), inputs, root=root, environment=environment)
    return directory, manifest, root, environment


def record(manifest, job, *, speedup=2, prior_speedup=1):
    workload = next(w for w in manifest['design']['workloads'] if w['id'] == job['workload_id'])
    elapsed = {'baseline': speedup, 'optimized_010': prior_speedup, 'accelerated': 1.0}[job['variant']]
    batch = workload['mode'] == 'batch'
    implementation = manifest['comparators'][job['variant']]
    package = expected_package(manifest['run_directory'], manifest, job['variant']).resolve()
    identity = dict(variant=job['variant'], kind=implementation['kind'], version=implementation['version'],
                    source_tree_sha256=implementation['source_tree_sha256'], package_root=str(package),
                    entrypoint=str(package / '__init__.py'),
                    loaded_modules={'fixture': {'path': str(package / '__init__.py'),
                                               'sha256': implementation['python_sha256']['__init__.py']}})
    return dict(schema_version=protocol.SCHEMA, run_id=manifest['run_id'], manifest_sha256=manifest['manifest_sha256'],
                runtime_identity=identity,
                job_id=job['job_id'], job=job, input_sha256=manifest['inputs'][f"{workload['id']}@{job['seed']}"],
                environment=manifest['environment'], affinity=manifest['design']['affinity'],
                returns_self=True, output_shape=[workload['p'], workload['p']], output_dtype='float64',
                output_finite=True, output_sha256='f' * 64, import_s=.2, first_use_s=.3,
                construction_s=.01, readout_s=.001, calibration_s=None if batch else .5,
                processing_s=None if batch else elapsed, fit_s=elapsed if batch else None,
                scale_scatter_readout_s=.002 if workload['algorithm'] == 'dual' else None,
                scale_scatter_finite=True if workload['algorithm'] == 'dual' else None,
                peak_process_bytes=1000, retained_state={'bytes': 100, 'accounting': 'fixture arrays'},
                memory_measure='fixture_peak')


def write_records(directory, manifest, *, speedup=2, prior_speedup=1):
    for job in manifest['jobs']:
        protocol.atomic_json(directory / 'records' / (job['job_id'] + '.json'), record(manifest, job, speedup=speedup, prior_speedup=prior_speedup))


def test_revision_workloads_and_balanced_serial_schedule():
    plan = design()
    plan['workloads'] = workloads('revision')
    jobs = scheduled_jobs(plan)
    assert len(plan['workloads']) == 8
    assert len(workloads('full')) == 40
    assert sum(w['mode'] == 'batch' for w in plan['workloads']) == 2
    assert len(jobs) == 8 * 3 * 3 * 3 * 3
    assert jobs == scheduled_jobs(plan)
    assert len({job['job_id'] for job in jobs}) == len(jobs)
    for first, second, third in zip(jobs[::3], jobs[1::3], jobs[2::3]):
        assert first['pair_id'] == second['pair_id'] == third['pair_id']
        assert [first['order'], second['order'], third['order']] == [0, 1, 2]
        assert {first['variant'], second['variant'], third['variant']} == set(VARIANTS)
    for workload in plan['workloads']:
        for position in range(3):
            values = [j['variant'] for j in jobs if j['workload_id'] == workload['id'] and j['order'] == position]
            assert all(values.count(variant) == 9 for variant in VARIANTS)
    orders = [[(j['workload_id'], j['seed']) for j in jobs if j['session'] == s and j['repeat'] == 0 and j['order'] == 0]
              for s in range(3)]
    assert len({tuple(order) for order in orders}) == 3


def test_hashed_inputs_reproducible_seeded_and_exact_ten_percent_contamination():
    spec = dict(n=100, p=4, contaminated=False)
    clean = make_data(spec, 17)
    assert input_hash(clean) == input_hash(make_data(spec, 17))
    assert input_hash(clean) != input_hash(make_data(spec, 18))
    dirty = make_data({**spec, 'contaminated': True}, 17)
    assert np.count_nonzero(np.any(clean != dirty, axis=1)) == 10


def test_fresh_directories_never_silently_resume(tmp_path):
    first, manifest, root, environment = create_run(tmp_path)
    second, _ = protocol.fresh_run(manifest['design'], manifest['jobs'], manifest['inputs'], root=root, environment=environment)
    assert first != second
    assert protocol.load_manifest(first, check_current=True, root=root, environment=environment) == manifest
    with pytest.raises(FileExistsError):
        protocol.fresh_run(manifest['design'], manifest['jobs'], manifest['inputs'], output=first, root=root, environment=environment)


@pytest.mark.parametrize('change', ['source', 'environment', 'manifest'])
def test_resume_rejects_changed_provenance(tmp_path, change):
    directory, manifest, root, environment = create_run(tmp_path)
    if change == 'source':
        (root / 'src/force_toolkit/__init__.py').write_text('# revised\n')
    elif change == 'environment':
        environment = {**environment, 'python': 'different'}
    else:
        manifest['design']['repetitions'] = 999
        (directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(protocol.EvidenceError):
        protocol.load_manifest(directory, check_current=True, root=root, environment=environment)


@pytest.mark.parametrize('changes', [
    {'run_id': 'stale'}, {'manifest_sha256': 'stale'}, {'input_sha256': 'stale'},
    {'environment': {'python': 'other'}}, {'returns_self': False}, {'affinity': [9]},
    {'processing_s': float('nan')}, {'fit_s': 1}, {'output_finite': False}, {'peak_process_bytes': 0},
])
def test_record_provenance_and_measurement_validation(tmp_path, changes):
    _, manifest, _, _ = create_run(tmp_path)
    entry = record(manifest, manifest['jobs'][0])
    entry.update(changes)
    with pytest.raises(protocol.EvidenceError):
        protocol.validate_record(entry, manifest)


@pytest.mark.parametrize('kind', ['partial_json', 'temporary', 'misnamed', 'different_output'])
def test_partial_duplicate_or_differential_failures_are_not_skipped(tmp_path, kind):
    directory, manifest, _, _ = create_run(tmp_path)
    first = manifest['jobs'][0]
    protocol.atomic_json(directory / 'records' / (first['job_id'] + '.json'), record(manifest, first))
    if kind == 'partial_json':
        (directory / 'records/broken.json').write_text('{')
    elif kind == 'temporary':
        (directory / 'records/pending.json.tmp-fixture').write_text('{}')
    elif kind == 'misnamed':
        protocol.atomic_json(directory / 'records/duplicate.json', record(manifest, first))
    else:
        second = manifest['jobs'][1]
        changed = record(manifest, second)
        changed['output_sha256'] = 'e' * 64
        protocol.atomic_json(directory / 'records' / (second['job_id'] + '.json'), changed)
    with pytest.raises(protocol.EvidenceError):
        protocol.read_records(directory, manifest)


def test_record_cannot_be_overwritten_and_controller_cannot_overlap(tmp_path):
    directory, _, _, _ = create_run(tmp_path)
    destination = directory / 'one.json'
    protocol.atomic_json(destination, {'a': 1})
    with pytest.raises(protocol.EvidenceError):
        protocol.atomic_json(destination, {'a': 2})
    assert json.loads(destination.read_text()) == {'a': 1}
    with run_lock(directory):
        with pytest.raises(protocol.EvidenceError):
            with run_lock(directory):
                pytest.fail('overlapping measurement controller acquired lock')
    assert not (directory / '.run.lock').exists()


def test_grouped_bootstrap_retains_configuration_blocks_and_session_effects():
    a = np.log(np.arange(1, 28).reshape(3, 3, 3))
    balanced = np.stack([a, -a])
    summary = intervals(balanced, resamples=1000)
    assert summary['speedup'] == pytest.approx(1)
    assert summary['conditional_fixed_design_ci95'][0] < 1 < summary['conditional_fixed_design_ci95'][1]
    assert summary['session_seed_cluster_ci95'] == pytest.approx([1, 1])
    assert summary['session_speedups'] == pytest.approx([1, 1, 1])
    assert summary['exploratory']
    relabeled = balanced.copy()
    relabeled[0] = relabeled[0, :, :, ::-1]
    assert intervals(relabeled, resamples=1000)['conditional_fixed_design_ci95'] == summary['conditional_fixed_design_ci95']
    session_effects = np.log(np.array([1, 2, 4]))[None, :, None, None] * np.ones((1, 3, 3, 3))
    result = intervals(session_effects, resamples=1000)
    assert result['session_speedups'] == pytest.approx([1, 2, 4])
    assert result['session_seed_cluster_ci95'][0] < result['speedup'] < result['session_seed_cluster_ci95'][1]
    once = intervals(np.zeros((1, 1, 1, 1)))
    assert once['conditional_fixed_design_ci95'] is None
    assert once['session_seed_cluster_ci95'] is None


def test_summary_and_assessment_recompute_from_validated_raw_evidence(tmp_path):
    directory, manifest, _, _ = create_run(tmp_path)
    write_records(directory, manifest, speedup=2)
    (directory / 'summary.json').write_text('{"complete": false, "forged": true}')
    summary = summarize(directory)
    assert summary['complete']
    assert summary['aggregate_streaming_mad']['speedup'] == pytest.approx(2)
    assert summary['aggregate_streaming_mad']['session_seed_cluster_ci95'] == pytest.approx([2, 2])
    acceptance = assess(directory)
    assert acceptance['performance_criterion_supported']
    assert acceptance['assessment'].endswith('exploratory')
    assert not acceptance['historical_data_reused']
    assert 'forged' not in json.loads((directory / 'summary.json').read_text())
    (directory / 'records' / (manifest['jobs'][0]['job_id'] + '.json')).unlink()
    incomplete = assess(directory)
    assert not incomplete['complete']
    assert not incomplete['performance_criterion_supported']


def test_two_pass_uses_fit_phase_and_never_enters_streaming_mad_aggregate(tmp_path):
    batch = next(w for w in workloads('revision') if w['mode'] == 'batch' and w['algorithm'] == 'mad')
    directory, manifest, _, _ = create_run(tmp_path, selected=design(batch))
    write_records(directory, manifest, speedup=.5, prior_speedup=.5)
    summary = summarize(directory, write=False)
    assert summary['aggregate_streaming_mad'] is None
    assert summary['results'][0]['primary_phase'] == 'fit_s'
    assert summary['results'][0]['acceleration']['speedup'] == pytest.approx(.5)
    assert summary['results'][0]['revision_regression']['speedup'] == pytest.approx(.5)
    assert summary['results'][0]['regression_assessment'] == 'regression_over_10pct_supported'
