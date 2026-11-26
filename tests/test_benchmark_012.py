"""Compiler identity, preserved-wheel execution and revision comparisons."""
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import _protocol as protocol
from benchmarks._comparators import (
    VARIANTS, WHEEL_RELATIVE, WHEEL_SHA256, ensure_prior_extracted,
)
from benchmarks._statistics import matched_intervals, summarize
from benchmarks.acceptance import assess
from benchmarks.run import scheduled_jobs
from tests.test_benchmark_revision import create_run, design, record, write_records


def compilation_environment(overrides=None, cache_policy='shared_disk_cache'):
    # This probe needs only core dependencies, not threadpoolctl/psutil extras.
    environment = {key: value for key, value in os.environ.items() if not key.startswith('NUMBA_')}
    environment.update(protocol.THREAD_ENV)
    environment.update(overrides or {})
    code = ('import json,sys; from benchmarks._protocol import numba_compilation_environment; '
            'print(json.dumps(numba_compilation_environment(sys.argv[1])))')
    result = subprocess.run([sys.executable, '-c', code, cache_policy],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize('overrides', [
    {'NUMBA_DISABLE_JIT': '1'}, {'NUMBA_CPU_NAME': 'generic'},
    {'NUMBA_OPT': '0'}, {'NUMBA_OPT': 'max'}, {'NUMBA_BOUNDSCHECK': '1'},
])
def test_effective_compiler_changes_cannot_resume(tmp_path, overrides):
    original = compilation_environment()
    changed = compilation_environment(overrides)
    assert original != changed
    directory, manifest, root, environment = create_run(tmp_path)
    environment['numba_compilation'] = original
    manifest['environment'] = deepcopy(environment)
    manifest['manifest_sha256'] = protocol.digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(protocol.EvidenceError, match='environment'):
        protocol.load_manifest(directory, check_current=True, root=root,
                               environment={**environment, 'numba_compilation': changed})
    if overrides.get('NUMBA_DISABLE_JIT') == '1':
        with pytest.raises(protocol.EvidenceError, match='JIT enabled'):
            protocol.require_compiled_environment({'numba_compilation': changed})


def test_managed_cache_paths_do_not_disguise_semantic_configuration():
    left = compilation_environment({'NUMBA_CACHE_DIR': '/tmp/first-managed-cache'}, 'cold_per_process')
    right = compilation_environment({'NUMBA_CACHE_DIR': '/tmp/second-managed-cache'}, 'cold_per_process')
    assert left == right
    assert left != compilation_environment({'NUMBA_CACHE_DIR': '/tmp/first-managed-cache'})


def test_latin_triplets_balance_positions_and_pairwise_precedence():
    plan = design()
    jobs = scheduled_jobs(plan)
    groups = [jobs[start:start + 3] for start in range(0, len(jobs), 3)]
    for left_index, left in enumerate(VARIANTS):
        for right in VARIANTS[left_index + 1:]:
            left_first = sum(next(j['order'] for j in group if j['variant'] == left)
                             < next(j['order'] for j in group if j['variant'] == right)
                             for group in groups)
            assert abs(left_first - (len(groups) - left_first)) == 1
    for session in range(3):
        for seed in plan['seeds']:
            cell = [j for j in jobs if j['session'] == session and j['seed'] == seed]
            for variant in VARIANTS:
                assert sorted(j['order'] for j in cell if j['variant'] == variant) == [0, 1, 2]


def test_both_contrasts_resample_complete_cycles_together():
    values = np.log(np.arange(1, 55).reshape(1, 3, 3, 6))
    first, second = matched_intervals(np.stack([values, -values]), resamples=2000)
    np.testing.assert_allclose(second['conditional_fixed_design_ci95'],
                               1 / np.array(first['conditional_fixed_design_ci95'][::-1]))
    np.testing.assert_allclose(second['session_seed_cluster_ci95'],
                               1 / np.array(first['session_seed_cluster_ci95'][::-1]))
    one_cycle = matched_intervals(np.stack([values[..., :3], -values[..., :3]]), resamples=100)
    assert one_cycle[0]['conditional_fixed_design_ci95'] is None
    assert one_cycle[0]['conditional_cycles_per_cell'] == 1
    # A position effect cancels within each balanced three-triplet cycle.
    balanced = np.tile(np.log([.5, 1., 2.]), 2).reshape(1, 1, 1, 1, 6)
    assert matched_intervals(balanced, resamples=100)[0]['conditional_fixed_design_ci95'] == pytest.approx([1, 1])


def test_acceleration_does_not_hide_regression_against_optimized_release(tmp_path):
    directory, manifest, _, _ = create_run(tmp_path)
    write_records(directory, manifest, speedup=2, prior_speedup=.5)
    result = assess(directory)
    assert result['aggregate_streaming_mad']['speedup'] == pytest.approx(2)
    assert result['aggregate_revision_streaming_mad']['speedup'] == pytest.approx(.5)
    assert result['acceleration_criterion_supported']
    assert not result['revision_nonregression_criterion_supported']
    assert not result['performance_criterion_supported']
    assert result['regressions_over_10pct_supported'] == [manifest['design']['workloads'][0]['id']]


def test_imported_identity_must_match_variant_not_just_version(tmp_path):
    _, manifest, _, _ = create_run(tmp_path)
    prior_job = next(j for j in manifest['jobs'] if j['variant'] == 'optimized_010')
    candidate_job = next(j for j in manifest['jobs'] if j['variant'] == 'accelerated')
    entry = record(manifest, prior_job)
    entry['runtime_identity'] = record(manifest, candidate_job)['runtime_identity']
    with pytest.raises(protocol.EvidenceError, match='implementation identity'):
        protocol.validate_record(entry, manifest)


def test_preserved_wheel_is_bound_and_imported_from_its_own_extraction(tmp_path):
    directory, manifest, root, _ = create_run(tmp_path)
    package = ensure_prior_extracted(directory, manifest)
    assert manifest['comparators']['optimized_010']['wheel_sha256'] == WHEEL_SHA256
    code = ('import json,sys; from benchmarks._protocol import load_manifest; '
            'from benchmarks._comparators import import_comparator; '
            'm=load_manifest(sys.argv[1]); f,i=import_comparator(sys.argv[1],m,"optimized_010"); '
            'assert f.ForceEstimator.__module__ == "force_toolkit.api"; print(json.dumps(i))')
    result = subprocess.run([sys.executable, '-c', code, str(directory)],
                            cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, **protocol.THREAD_ENV},
                            capture_output=True, text=True, check=True)
    identity = json.loads(result.stdout)
    assert identity['version'] == '0.1.0'
    assert identity['package_root'] == str(package)
    assert len(identity['loaded_modules']) > 5
    # An existing extraction is verified, not silently reused or repaired.
    (package / 'api.py').write_text('# changed\n')
    with pytest.raises(protocol.EvidenceError, match='Extracted prior-wheel'):
        ensure_prior_extracted(directory, manifest)
    artifact = root / WHEEL_RELATIVE
    artifact.write_bytes(artifact.read_bytes() + b'changed')
    with pytest.raises(protocol.EvidenceError, match='checksum'):
        ensure_prior_extracted(directory, manifest)


def test_relocated_evidence_can_be_audited_without_original_checkout(tmp_path):
    directory, manifest, root, _ = create_run(tmp_path)
    write_records(directory, manifest)
    relocated = tmp_path / 'relocated-evidence'
    shutil.move(directory, relocated)
    shutil.rmtree(root)
    result = summarize(relocated, write=False)
    assert result['complete'] and result['measurements'] == 81
    with pytest.raises(protocol.EvidenceError, match='directory moved'):
        protocol.load_manifest(relocated, check_current=True)


def test_old_schema_is_not_silently_upgraded(tmp_path):
    directory, manifest, _, _ = create_run(tmp_path)
    manifest['schema_version'] = 2
    manifest['manifest_sha256'] = protocol.digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(protocol.EvidenceError, match='Historical schema'):
        protocol.load_manifest(directory)


@pytest.mark.parametrize('style,path_class,prefix', [
    ('windows', PureWindowsPath, 'C:/recorded-project'),
    ('posix', PurePosixPath, '/recorded-project'),
])
def test_recorded_origins_are_audited_in_the_execution_hosts_path_style(tmp_path, monkeypatch, style, path_class, prefix):
    _, manifest, _, _ = create_run(tmp_path)
    job = next(j for j in manifest['jobs'] if j['variant'] == 'accelerated')
    entry = record(manifest, job)
    package = path_class(prefix) / 'src' / 'force_toolkit'
    manifest['path_style'] = style
    manifest['comparators'][job['variant']]['package_root'] = str(package)
    identity = entry['runtime_identity']
    identity['package_root'] = str(package)
    identity['entrypoint'] = str(package / '__init__.py')
    identity['loaded_modules']['fixture']['path'] = str(package / '__init__.py')
    def forbidden(*args, **kwargs):
        raise AssertionError('Historical identity validation used the reader filesystem')
    monkeypatch.setattr(protocol, 'Path', forbidden)
    assert protocol.validate_record(entry, manifest) == job
    identity['loaded_modules']['fixture']['path'] = str(path_class(prefix) / 'escaped.py')
    with pytest.raises(protocol.EvidenceError, match='escaped'):
        protocol.validate_record(entry, manifest)
