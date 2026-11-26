"""0.1.2 compatibility is separate from the retained independent scientific tests."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import force_toolkit as ft
from tests.reference import force_toolkit_012 as ref


def compatible(a, b):
    if isinstance(b, dict):
        for key in b:
            if key not in {'state', 'state_bytes', 'component_timing_s'}:
                compatible(a[key], b[key])
    elif isinstance(b, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compatible(x, y)
    elif isinstance(b, np.ndarray):
        if b.dtype.kind in 'biu':
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
    elif isinstance(b, float):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
    else:
        assert a == b


def test_012_reference_and_wheel_remain_frozen():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'provenance/comparator-0.1.2.json').read_text())
    reference = root / 'tests/reference/force_toolkit_012'
    actual = {p.relative_to(reference).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in reference.rglob('*.py')}
    assert actual == manifest['files']
    wheel = root / 'benchmarks/artifacts/force_toolkit-0.1.2-py3-none-any.whl'
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == manifest['wheel_sha256']


@pytest.mark.parametrize('chunk', [1, 7, 64, 256, 1024])
@pytest.mark.parametrize('timing', [False, True])
@pytest.mark.parametrize('tail', ['none', 'guarded_asymmetric'])
def test_012_complete_stream_trajectory(chunk, timing, tail):
    rng = np.random.default_rng(6102)
    values = rng.standard_t(4, size=(2100, 3))
    values[700:1400, 0] += 20
    values[1400:, 1] *= 10
    values[::23, 2] = 0
    values = np.asfortranarray(values)  # exercise noncontiguous public input
    ae, be = [], []
    options = dict(n_initial_warm=17, min_reset_obs=9, tail_mode=tail,
                   acceptance_trace_eval_points=[1, 5, 6, 17, 18, 709, 1024],
                   diagnostic_change_points=[700], detector_max_buckets=2)
    a = ft.StreamingMADForceEstimator(3, collect_component_timing=timing, diagnostic_sink=ae.append, **options)
    b = ref.StreamingMADForceEstimator(3, diagnostic_sink=be.append, **options)
    alias = a.n_jk
    for offset in range(0, len(values), chunk):
        a.update_batch(values[offset:offset + chunk])
        b.update_batch(values[offset:offset + chunk])
        compatible(a.get_diagnostics(), b.get_diagnostics())
        np.testing.assert_array_equal(a.n_jk, b.n_jk)
        for x, y in zip(a.streaming_mad, b.streaming_mad):
            for s in ('stage1', 'stage2'):
                xq, yq = getattr(x, s), getattr(y, s)
                assert (xq.count, xq.initialized) == (yq.count, yq.initialized)
                if xq.initialized:
                    np.testing.assert_array_equal(xq.q, yq.q)
        assert a.n_jk is alias
    assert ae == be
    assert a.reset_counts.sum() > 0
    np.testing.assert_array_equal(a.finalize(), b.finalize())
    a.reset()
    assert a.collect_component_timing is timing
    details = a.get_diagnostics()['details']
    assert details['component_timing_enabled'] is timing
    assert (details['component_timing_s'] is None) is (not timing)


@pytest.mark.parametrize('cap', [None, 64, 257])
def test_native_no_sink_growth_cuts_and_candidate_order(cap):
    from force_toolkit._mad._detector import DetectorState
    from force_toolkit._mad.adwin import ADWINDetector
    from tests.reference.force_toolkit_012._mad.adwin import ADWINDetector as Reference
    original = ADWINDetector(max_window=cap, max_buckets=2)
    state = DetectorState([original], levels=1)
    a, b = state.adapters[0], Reference(max_window=cap, max_buckets=2)
    values = np.r_[np.ones(6000), np.tile(np.r_[np.ones(70), np.zeros(40)], 10)]
    for time, bit in enumerate(values, 1):
        assert a.update(bit, time_index=time) == b.update(bit, time_index=time)
        if time % 61 == 0 or a.alarm_count:
            compatible(a.diagnostics(), b.diagnostics())
            assert [[vars(bucket) for bucket in row] for row in a._rows] == [[vars(bucket) for bucket in row] for row in b._rows]
    if cap is None:
        assert state.buckets.shape[1] > 1
    # Full native controller, no per-row history/callback returns.
    X = np.column_stack([values, -values])
    a = ft.StreamingMADForceEstimator(2, detector_max_window=cap, detector_max_buckets=2).fit(X)
    b = ref.StreamingMADForceEstimator(2, detector_max_window=cap, detector_max_buckets=2).fit(X)
    compatible(a.get_diagnostics(), b.get_diagnostics())
    np.testing.assert_array_equal(a.n_jk, b.n_jk)


@pytest.mark.parametrize('kind', ['constant', 'ties', 'skew', 'huge_offset', 'tiny', 'neighbors'])
@pytest.mark.parametrize('detection', [False, True])
def test_native_extremes_bootstrap_reseed_and_stationary(kind, detection):
    rng = np.random.default_rng(711)
    X = rng.normal(size=(313, 3))
    if kind == 'constant':
        X.fill(1e308)  # (a+b)/2 overflows, NumPy quantile does not
    elif kind == 'ties':
        X = np.round(X)
    elif kind == 'skew':
        X = np.exp(X) * [1e-100, 1., 1e100]
    elif kind == 'huge_offset':
        X = 1e100 + np.round(10 * X) * np.spacing(1e100)
    elif kind == 'tiny':
        X = np.round(X * 10) * np.nextafter(0., 1.)
    elif kind == 'neighbors':
        X[:6] = 0.
        X[6:] = np.nextafter(np.round(X[6:]), np.where(X[6:] > 0, np.inf, -np.inf))
    options = dict(n_initial_warm=5, use_drift_detection=detection)
    a, b = ft.StreamingMADForceEstimator(3, **options), ref.StreamingMADForceEstimator(3, **options)
    for start, end in zip([0, 1, 2, 4, 5, 6, 7, 32, 33, 201], [1, 2, 4, 5, 6, 7, 32, 33, 201, 313]):
        a.update_batch(X[start:end]); b.update_batch(X[start:end])
        compatible(a.get_diagnostics(), b.get_diagnostics())
    a._selective_reset(0); b._selective_reset(0)
    a.update_batch(X[:40]); b.update_batch(X[:40])
    compatible(a.get_diagnostics(), b.get_diagnostics())
    for tail in ('none', 'guarded_asymmetric'):
        a = ft.MADForceEstimator(3, tail_mode=tail).fit(X)
        b = ref.MADForceEstimator(3, tail_mode=tail).fit(X)
        compatible(a.get_diagnostics(), b.get_diagnostics())
        np.testing.assert_array_equal(a.finalize(), b.finalize())


@pytest.mark.parametrize('phase,j', [(1, 1), (2, 0), (4, 1), (7, 0), (8, 0)])
@pytest.mark.parametrize('exception', [FloatingPointError, KeyboardInterrupt, SystemExit])
def test_native_failure_after_earlier_rows_commit(phase, j, exception):
    a = ft.StreamingMADForceEstimator(2, n_initial_warm=0).fit(np.ones((40, 2)))
    a._native_fault[:] = (43, phase, j)
    def fail():
        raise exception('native phase interruption')
    a._native_failure = fail
    with pytest.raises(exception):
        a.update_batch(np.ones((10, 2)))
    details = a.get_diagnostics()
    assert details['observations'] == 42
    assert details['details']['observations_attempted'] == 43
    assert not details['ready']
    with pytest.raises(ft.FailedStateError):
        a.update([1., 1.])
    a.reset().fit(np.ones((40, 2)))
    assert a.get_diagnostics()['observations'] == 40


def test_real_native_pair_exception_and_bound_continuation():
    a = ft.StreamingMADForceEstimator(2, n_initial_warm=0).fit(np.ones((40, 2)))
    a._native_fault[:] = (43, 7, 0)
    def poison_pair():
        a._native_fault.fill(-1)
        a.n_jk[0, 1] = 2**53
    a._native_failure = poison_pair
    with pytest.raises(FloatingPointError, match='Pair count'):
        a.update_batch(np.ones((10, 2)))
    assert a.get_diagnostics()['observations'] == 42
    tiny = np.nextafter(0., 1.)
    X = np.tile(np.array([[-3., -1.], [1., 2.], [3., -2.], [-1., 1.]]) * tiny, (100, 1))
    options = dict(n_initial_warm=0, lam=1e308, use_drift_detection=False)
    a = ft.StreamingMADForceEstimator(2, **options).fit(X)
    b = ref.StreamingMADForceEstimator(2, **options).fit(X)
    compatible(a.get_diagnostics(), b.get_diagnostics())
    np.testing.assert_array_equal(a.finalize(), b.finalize())


def test_callback_replacement_during_delivery_and_late_tail_change():
    X = np.random.default_rng(184).normal(size=(600, 3))
    logs = []
    a = ft.StreamingMADForceEstimator(3)
    def first(event):
        logs.append(('first', event))
        assert a.get_diagnostics()['observations'] == event['t']
        a.diagnostic_sink = lambda event: logs.append(('second', event))
    a.diagnostic_sink = first
    a.update_batch(X[:300])
    assert logs[0][0] == 'first' and all(label == 'second' for label, _ in logs[1:])
    b = ref.StreamingMADForceEstimator(3).fit(X[:300])
    a.diagnostic_sink = None
    a.tail_mode = b.tail_mode = 'guarded_asymmetric'
    a.update_batch(X[300:]); b.update_batch(X[300:])
    compatible(a.get_diagnostics(), b.get_diagnostics())
    np.testing.assert_array_equal(a.finalize(), b.finalize())


def test_timing_validation_and_shared_tracker_adapters():
    with pytest.raises(ValueError, match='collect_component_timing'):
        ft.StreamingMADForceEstimator(2, collect_component_timing='yes')
    a = ft.StreamingMADForceEstimator(2)
    for j in range(2):
        assert np.shares_memory(a.streaming_mad[j].stage1.q, a._tracker_state.q)
        assert np.shares_memory(a.p2_q99[j]._counters, a._tracker_state.counts)
    a.fit(np.ones((250, 2)))
    before = a.finalize()
    a.collect_component_timing = True
    a.update_batch(np.ones((20, 2)))
    np.testing.assert_array_equal(a.finalize(), before)


def test_dual_degenerate_cutoff_invariant_eigenspace():
    # The second selected eigenvalue is tied. No arbitrary unique projector is required.
    a = ft.DualBoundForceEstimator(4, 4, calibration_size=8).fit(
        np.random.default_rng(99).normal(size=(20, 4)))
    target = np.eye(4)
    target[0, 1] = target[1, 0] = .5
    a._engine._sketch.B.fill(0.)
    a._engine._sketch.B[:4] = np.linalg.cholesky(target).T
    actual = a.finalize()
    np.testing.assert_allclose(actual, target, rtol=1e-12, atol=1e-14)
    vectors = a.get_subspace(2)
    assert vectors.shape == (4, 2)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(2), rtol=1e-12, atol=1e-14)
    selected = vectors.T @ actual @ vectors
    np.testing.assert_allclose(actual @ vectors, vectors @ selected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(selected), [1., 1.5], rtol=1e-12, atol=1e-14)
    eigenvalues, basis = np.linalg.eigh(actual)
    np.testing.assert_allclose(basis[:, :1].T @ vectors, 0., atol=1e-14)
    leading = basis[:, -1]
    np.testing.assert_allclose(vectors @ vectors.T @ leading, leading, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('phase', ['tracker', 'bounds', 'detector', 'pair'])
def test_profiled_failure_after_committed_prefix(monkeypatch, phase):
    a = ft.StreamingMADForceEstimator(2, n_initial_warm=0, collect_component_timing=True).fit(np.ones((40, 2)))
    target, method = {'tracker': (a.streaming_mad[1], 'update'),
                      'bounds': (a, '_tail_multipliers'),
                      'detector': (a.adwin[1], 'update'),
                      'pair': (a._moments, 'update')}[phase]
    original = getattr(target, method)
    def interrupt(*args, **kwargs):
        if a._obs_count == 43:
            raise KeyboardInterrupt('profiled phase')
        return original(*args, **kwargs)
    monkeypatch.setattr(target, method, interrupt)
    with pytest.raises(KeyboardInterrupt):
        a.update_batch(np.ones((10, 2)))
    d = a.get_diagnostics()
    assert d['observations'] == 42 and d['details']['observations_attempted'] == 43
    assert not d['ready']
    with pytest.raises(ft.FailedStateError):
        a.finalize()


@pytest.mark.parametrize('mode', ['native', 'profiled', 'reference012'])
def test_callback_catching_readout_failure_stops_before_next_attempt(mode):
    constructor = ref.StreamingMADForceEstimator if mode == 'reference012' else ft.StreamingMADForceEstimator
    options = {} if mode == 'reference012' else {'collect_component_timing': mode == 'profiled'}
    a = constructor(2, n_initial_warm=0, **options).fit(np.random.default_rng(50).normal(size=(40, 2)))
    delivered = []
    def sink(event):
        delivered.append(event)
        # Exercise the real correlation range guard and its invalidation path.
        a._moments.co_moment[0, 1] = np.inf
        with pytest.raises(FloatingPointError, match='Non-finite correlation moments'):
            a.finalize()
        a.diagnostic_sink = None
    a.diagnostic_sink = sink
    exception = ref.FailedStateError if mode == 'reference012' else ft.FailedStateError
    with pytest.raises(exception):
        a.update_batch(np.ones((10, 2)))
    assert len(delivered) == 1
    diagnostics = a.get_diagnostics()
    assert diagnostics['observations'] == 41
    assert diagnostics['details']['observations_attempted'] == 41
    assert not diagnostics['ready']
