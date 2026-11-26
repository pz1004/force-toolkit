"""Frozen compatibility references; independent statistical oracles live separately."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import force_toolkit as actual
from tests.reference import force_toolkit_baseline as reference
from force_toolkit._mad.common import PairwiseMoments
from tests.reference.force_toolkit_baseline._mad.common import PairwiseMoments as ReferenceMoments
from force_toolkit._mad.adwin import ADWINDetector
from tests.reference.force_toolkit_baseline._mad.adwin import ADWINDetector as ReferenceDetector
from tests.test_api import assert_tree


def assert_numerical(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def assert_compatible_diagnostics(actual, reference):
    """Check every old diagnostic field; additions are explicitly enumerated."""
    additions = {
        'pair_status_codes', 'pair_status_code_labels', 'pair_validity',
        'scale_construction', 'lifecycle_status', 'failure',
        'observations_attempted', 'observations_committed', 'detector_scope', 'component_timing_enabled',
    }
    assert set(actual) - set(reference) <= {'any_pair_estimable', 'all_pairs_estimable'}
    assert set(actual['details']) - set(reference['details']) <= additions
    # 0.2.0 changes owned storage, not statistical diagnostics.
    def comparable(value):
        if isinstance(value, dict):
            return {k: comparable(v) for k, v in value.items() if k not in {'state_bytes', 'state'}}
        if isinstance(value, list):
            return [comparable(v) for v in value]
        return value
    for key, expected in reference.items():
        if key == 'details':
            assert_tree(comparable({name: actual[key][name] for name in expected}), comparable(expected))
        elif key != 'state':
            assert_tree(actual[key], expected)
    if actual['algorithm'] == 'MAD-FORCE' and actual['ready']:
        p = actual['details']['n_features']
        assert actual['state']['bytes'] == reference['state']['bytes'] + p * p
    elif actual['algorithm'] not in {'Streaming MAD-FORCE'}:
        assert_tree(actual['state'], reference['state'])


def test_reference_is_frozen():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'provenance/baseline-sha256.json').read_text())
    base = root/'tests/reference/force_toolkit_baseline'
    assert set(manifest) == {p.relative_to(base).as_posix() for p in base.rglob('*.py')}
    for path, digest in manifest.items():
        assert hashlib.sha256((base/path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('scale', [1., 1e-200, 1e200, np.nextafter(0.,1.)])
def test_pair_moment_trajectory_and_incident_resets(scale):
    rng=np.random.default_rng(919)
    X=rng.integers(-5,6,size=(130,5)).astype(float)*scale
    masks=rng.random(X.shape) > .25
    a,b=PairwiseMoments(5),ReferenceMoments(5)
    for i,(row,mask) in enumerate(zip(X,masks)):
        for estimator in (a,b):
            estimator.update(row,mask)
            if i in (5,29,40,99):
                estimator.clear_dimension(i%5)
        for name,value in vars(a).items():
            if isinstance(value,np.ndarray):
                # Strict binary arithmetic is expected to agree bit for bit here.
                np.testing.assert_array_equal(value,getattr(b,name))
        assert_numerical(a.correlation(),b.correlation())


def test_pair_failure_leaves_same_partial_state_and_invalidates():
    a,b=PairwiseMoments(3),ReferenceMoments(3)
    for obj in (a,b):
        obj.count[1,2]=2**53
        with pytest.raises(FloatingPointError):
            obj.update(np.ones(3),np.ones(3,dtype=bool))
        with pytest.raises(RuntimeError):
            obj.correlation()
    for name,value in vars(a).items():
        if isinstance(value,np.ndarray):
            np.testing.assert_array_equal(value,getattr(b,name))


@pytest.mark.parametrize('cap', [None, 256])
def test_detector_candidate_and_cut_trajectory(cap):
    rng=np.random.default_rng(91)
    bits=np.r_[np.ones(400),np.zeros(100),rng.integers(0,2,600),np.ones(300)]
    a,b=ADWINDetector(max_window=cap),ReferenceDetector(max_window=cap)
    for i,bit in enumerate(bits):
        assert a.update(bit,allow_detection=i>50)==b.update(bit,allow_detection=i>50)
        da, db = a.diagnostics(), b.diagnostics()
        # CPython key-sharing dictionary sizes depend on prior allocations of
        # each separately imported bucket class, not its statistical state.
        assert da.pop('state_bytes') > 0 and db.pop('state_bytes') > 0
        assert_tree(da, db)
        assert [[vars(x) for x in row] for row in a._rows] == [[vars(x) for x in row] for row in b._rows]


@pytest.mark.parametrize('detect', [False,True])
@pytest.mark.parametrize('tail', ['none','guarded_asymmetric'])
def test_mad_every_row_acceptance_readiness_alarms_resets(detect,tail):
    rng=np.random.default_rng(7)
    X=np.vstack([rng.normal(size=(1000,4)),rng.normal(loc=15,size=(100,4)),rng.normal(size=(100,4))])
    events_a,events_b=[],[]
    options=dict(use_drift_detection=detect,tail_mode=tail)
    a=actual.StreamingMADForceEstimator(4,**options,diagnostic_sink=events_a.append)
    b=reference.StreamingMADForceEstimator(4,**options,diagnostic_sink=events_b.append)
    for x in X:
        a.update(x);b.update(x)
        assert events_a == events_b
        events_a.clear();events_b.clear()
        for name in ('n_jk','acceptance_total','acceptance_sum','reset_counts','post_reset_obs'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        assert [a._dimension_ready(j) for j in range(4)] == [b._dimension_ready(j) for j in range(4)]
        if np.any(a.n_jk):
            assert_numerical(a.finalize(),b.finalize())
    assert_compatible_diagnostics(a.get_diagnostics(),b.get_diagnostics())
    if detect:
        assert a.reset_counts.sum() > 0


@pytest.mark.parametrize('chunk', [1,7,64,1024])
def test_dual_every_boundary_and_subspace(chunk):
    rng=np.random.default_rng(18)
    X=rng.normal(size=(1100,9));X[::10]*=30
    a=actual.DualBoundForceEstimator(9,3,calibration_size=17,epoch_size=83)
    b=reference.DualBoundForceEstimator(9,3,calibration_size=17,epoch_size=83)
    # Check both individual boundaries and arbitrary partitions on the same data.
    checkpoints=sorted(set(range(1,len(X)+1,chunk)) | {17,18,22,23,83,84,100,101,len(X)})
    start=0
    for end in checkpoints:
        a.update_batch(X[start:end]);b.update_batch(X[start:end]);start=end
        assert_compatible_diagnostics(a.get_diagnostics(),b.get_diagnostics())
        if a.get_diagnostics()['ready']:
            assert_numerical(a.finalize(),b.finalize())
            assert_numerical(a.get_scale_scatter(),b.get_scale_scatter())
            assert_numerical(a.transform(X[0]),b.transform(X[0]))
            u,v=a.get_subspace(),b.get_subspace()
            assert_numerical(u@u.T,v@v.T)
        if a._engine._sketch is not None:
            assert_numerical(a._engine._sketch.B,b._engine._sketch.B)


@pytest.mark.parametrize('chunk', [1,7,64,1024])
def test_force_stream_and_two_pass_frozen_equivalence(chunk):
    X=np.random.default_rng(12).normal(size=(1031,4));X[::10]*=20
    a=actual.StreamingForceEstimator(window_size=50)
    b=reference.StreamingForceEstimator(window_size=50)
    for start in range(0,len(X),chunk):
        a.update_batch(X[start:start+chunk]);b.update_batch(X[start:start+chunk])
        assert_compatible_diagnostics(a.get_diagnostics(),b.get_diagnostics())
        if a.get_diagnostics()['ready']:
            assert_numerical(a.finalize(),b.finalize())
    for name,kwargs in [('ForceEstimator',{}),('MADForceEstimator',{'p':4})]:
        first=getattr(actual,name)(**kwargs).fit(X)
        second=getattr(reference,name)(**kwargs).fit(X)
        assert_numerical(first.finalize(),second.finalize())
        assert_compatible_diagnostics(first.get_diagnostics(),second.get_diagnostics())


@pytest.mark.parametrize('failure', ['calibration', 'compression'])
def test_dual_chunk_arithmetic_failure_preserves_original_commit_boundary(monkeypatch, failure):
    X = np.random.default_rng(77).normal(size=(30, 4))
    a = actual.DualBoundForceEstimator(4, 2, calibration_size=5)
    b = reference.DualBoundForceEstimator(4, 2, calibration_size=5)
    start = 0
    if failure == 'compression':
        a.update_batch(X[:5]); b.update_batch(X[:5])
        start = 5
    def fail():
        raise FloatingPointError('injected arithmetic failure')
    for estimator in (a, b):
        target = estimator._engine if failure == 'calibration' else estimator._engine._sketch
        monkeypatch.setattr(target, '_calibrate' if failure == 'calibration' else '_compress', fail)
        with pytest.raises(FloatingPointError, match='injected arithmetic failure'):
            estimator.update_batch(X[start:])
    assert_tree(a._engine.get_diagnostics(), b._engine.get_diagnostics())
    expected_seen = 5 if failure == 'calibration' else 9
    assert a.get_diagnostics()['details']['observations_attempted'] == expected_seen
    assert a.get_diagnostics()['observations'] == expected_seen - 1
    assert not a.get_diagnostics()['ready']
    if failure == 'calibration':
        np.testing.assert_array_equal(a._engine._calibration, b._engine._calibration)
    else:
        np.testing.assert_array_equal(a._engine._sketch.B, b._engine._sketch.B)
        assert a._engine._sketch.next_row == b._engine._sketch.next_row == 4
    for estimator in (a, b):
        estimator.reset()
        assert not estimator.get_diagnostics()['ready']
