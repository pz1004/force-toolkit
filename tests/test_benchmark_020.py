"""The new protocol is independent of historical schema-2/3 auditing."""
from collections import Counter, defaultdict
from copy import deepcopy
import numpy as np
import pytest
from benchmarks import revision020 as bench


def test_predeclared_primary_design_is_1600_balanced_workers():
    design = dict(workloads=bench.primary(), sessions=5, seeds=list(range(41001,41006)),
                  repetitions=2, schedule_seed=41000)
    jobs = bench.scheduled_jobs(design)
    assert len(design['workloads']) == 16 and len(jobs) == 1600
    cells = defaultdict(list)
    for job in jobs:
        if job['order'] == 0:
            cells[job['workload_id'], job['session'], job['seed']].append(job['variant'])
    assert len(cells) == 400
    assert all(Counter(first) == {'baseline':1,'candidate':1} for first in cells.values())
    assert jobs == bench.scheduled_jobs(deepcopy(design))
    assert len({j['job_id'] for j in jobs}) == len(jobs)


def test_screen_grid_and_stress_separation():
    assert len(bench.workloads()) == 40
    assert {w['id'] for w in bench.stress_workloads()}.isdisjoint(w['id'] for w in bench.primary())
    for spec in bench.primary():
        assert bench.input_hash(bench.data(spec,41001)) == bench.input_hash(bench.data(spec,41001))
    latency = [w for w in bench.stress_workloads() if w.get('stress') == 'latency']
    assert {w['algorithm'] for w in latency} == {'mad','force','dual'}
    assert all(w['chunk'] == 1 for w in latency)


def test_source_identity_detects_wheel_mismatch(tmp_path):
    import zipfile
    package = tmp_path/'force_toolkit'; package.mkdir()
    (package/'__init__.py').write_text("__version__ = '0.2.0'\n")
    wheel = tmp_path/'mismatch.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('force_toolkit/__init__.py', "__version__ = '0.1.2'\n")
    with pytest.raises(bench.EvidenceError, match='Wheel Python bytes'):
        bench.identity(package, wheel)
