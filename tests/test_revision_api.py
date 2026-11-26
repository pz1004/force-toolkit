"""Public contracts for numerical failures, estimability and stationary state."""
import numpy as np
import pytest
import force_toolkit as ft
from force_toolkit import api
from tests.reference import force_toolkit_baseline as frozen


def test_stationary_constructor_and_fit_do_not_allocate_online_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('stationary path allocated online state')
    monkeypatch.setattr(api, '_MAD', forbidden)
    large = ft.MADForceEstimator(5000)
    assert large.get_diagnostics()['state']['bytes'] == 0
    X = np.random.default_rng(8).normal(size=(80, 4))
    for tail in ('none', 'guarded_asymmetric'):
        actual = ft.MADForceEstimator(4, tail_mode=tail).fit(X)
        reference = frozen.MADForceEstimator(4, tail_mode=tail).fit(X)
        np.testing.assert_array_equal(actual.finalize(), reference.finalize())
        np.testing.assert_array_equal(actual.get_diagnostics()['details']['accepted_counts'],
                                      reference.get_diagnostics()['details']['accepted_counts'])


@pytest.mark.parametrize('method', ['fit_stationary', 'fit_streaming'])
def test_retired_legacy_methods_fail_before_mutation(method):
    X = np.random.default_rng(3).normal(size=(60, 2))
    est = ft.StreamingMADForceEstimator(2, n_initial_warm=0).fit(X)
    before = est.finalize()
    with pytest.raises(AttributeError, match='Use .*Estimator'):
        getattr(est, method)(object())
    assert est.get_diagnostics()['observations'] == len(X)
    np.testing.assert_array_equal(before, est.finalize())


@pytest.mark.parametrize('factory', [
    lambda: ft.MADForceEstimator(3),
    lambda: ft.StreamingMADForceEstimator(3, n_initial_warm=0, use_drift_detection=False),
])
def test_estimability_distinguishes_constants_mixed_and_true_zero(factory):
    constant = factory().fit(np.ones((80, 3)))
    d = constant.get_diagnostics()
    assert d['ready'] and not d['any_pair_estimable'] and not d['all_pairs_estimable']
    np.testing.assert_array_equal(np.diag(d['details']['pair_status_codes']), 3)
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_array_equal(d['details']['pair_status_codes'][off], 2)
    d['details']['pair_status_codes'].fill(0)
    assert not constant.get_diagnostics()['any_pair_estimable']
    X = np.column_stack([np.ones(120), np.tile([-1., -1., 1., 1.], 30),
                         np.tile([-1., 1., -1., 1.], 30)])
    mixed = factory().fit(X)
    d = mixed.get_diagnostics()
    assert d['any_pair_estimable'] and not d['all_pairs_estimable']
    assert d['details']['pair_status_codes'][0, 1] == 2
    assert d['details']['pair_status_codes'][1, 2] == 0
    if isinstance(mixed, ft.MADForceEstimator):
        assert mixed.finalize()[1, 2] == 0


def test_single_accepted_pair_is_available_but_not_estimable():
    est = ft.StreamingMADForceEstimator(2, n_initial_warm=20, use_drift_detection=False)
    est.fit(np.ones((21, 2)))
    d = est.get_diagnostics()
    assert d['ready'] and not d['any_pair_estimable']
    assert d['details']['pair_status_codes'][0, 1] == 1
    assert d['details']['pair_status_code_labels'][1] == 'insufficient_observations'


def test_mad_real_internal_overflow_poisoning_and_reset():
    rng = np.random.default_rng(7)
    X = np.column_stack([-1e308 + rng.integers(-8, 9, 300) * 1e292,
                         rng.normal(size=300)])
    est = ft.StreamingMADForceEstimator(2, use_drift_detection=False).fit(X)
    with pytest.raises(ValueError, match='finite'):
        est.update([1e308, 0.])
    d = est.get_diagnostics()
    assert not d['ready'] and not d['any_pair_estimable']
    assert d['observations'] == d['details']['observations_committed'] == 300
    assert d['details']['observations_attempted'] == 301
    assert d['details']['lifecycle_status'] == 'failed'
    for call in (est.finalize, lambda: est.update([0., 0.]), lambda: est.update_batch([[0., 0.]])):
        with pytest.raises(ft.FailedStateError):
            call()
    est.reset().fit(np.random.default_rng(8).normal(size=(300, 2)))
    assert est.get_diagnostics()['ready']


@pytest.mark.parametrize('phase', ['tracker', 'acceptance', 'detector', 'pair'])
def test_mad_core_failures_invalidate_whole_estimator(monkeypatch, phase):
    est = ft.StreamingMADForceEstimator(2, n_initial_warm=0).fit(np.ones((40, 2)))
    def fail(*a, **k):
        raise FloatingPointError('injected')
    est._native_fault[:] = (41, {'tracker': 1, 'acceptance': 2, 'detector': 4, 'pair': 7}[phase], 0)
    monkeypatch.setattr(est, '_native_failure', fail)
    with pytest.raises(FloatingPointError):
        est.update([1., 1.])
    d = est.get_diagnostics()
    assert d['observations'] == 40 and d['details']['observations_attempted'] == 41
    assert not d['ready']
    with pytest.raises(ft.FailedStateError):
        est.finalize()


@pytest.mark.parametrize('operation', ['update', 'update_batch', 'fit', 'reset'])
def test_reentrant_mad_mutation_does_not_poison_committed_row(operation):
    est = ft.StreamingMADForceEstimator(2, n_initial_warm=0)
    def sink(event):
        if event['kind'] == 'acceptance':
            assert est.get_diagnostics()['observations'] == event['t']
            if operation == 'reset':
                est.reset()
            elif operation == 'update':
                est.update([1., 1.])
            else:
                getattr(est, operation)([[1., 1.]])
    est.diagnostic_sink = sink
    with pytest.raises(ft.DiagnosticDeliveryError) as error:
        est.update([1., 1.])
    assert isinstance(error.value.__cause__, RuntimeError)
    d = est.get_diagnostics()
    assert d['observations'] == d['details']['observations_attempted'] == 1
    assert d['details']['lifecycle_status'] != 'failed'
    est.diagnostic_sink = None
    est.update([1., 1.])
    assert est.get_diagnostics()['observations'] == 2


def test_reset_delivery_failure_leaves_healthy_empty_state():
    est = ft.StreamingMADForceEstimator(2).fit(np.ones((201, 2)))
    def sink(event):
        assert event['kind'] == 'stream_reset'
        d = est.get_diagnostics()
        assert d['observations'] == d['details']['observations_attempted'] == 0
        assert d['details']['failure'] is None
        raise OSError('notification failed')
    est.diagnostic_sink = sink
    with pytest.raises(ft.DiagnosticDeliveryError):
        est.reset()
    est.diagnostic_sink = None
    est.update([1., 1.])
    assert est.get_diagnostics()['observations'] == 1


@pytest.mark.parametrize('failure', ['calibration', 'compression', 'epoch'])
@pytest.mark.parametrize('chunk', [1, 7, 64])
@pytest.mark.parametrize('error_type', [FloatingPointError, KeyboardInterrupt])
def test_dual_core_failure_commit_counters_and_readout_guard(monkeypatch, failure, chunk, error_type):
    X = np.random.default_rng(77).normal(size=(30, 4))
    est = ft.DualBoundForceEstimator(4, 2, calibration_size=5, epoch_size=12)
    def fail(*a, **k):
        raise error_type('injected')
    start = 0
    if failure == 'calibration':
        target, method, attempted, committed = est._engine, '_calibrate', 5, 4
    elif failure == 'compression':
        est.update_batch(X[:5]); start = 5
        target, method, attempted, committed = est._engine._sketch, '_compress', 9, 8
    else:
        est.update_batch(X[:12]); start = 12
        target, method, attempted, committed = est._engine, '_start_next_epoch', 13, 12
    monkeypatch.setattr(target, method, fail)
    with pytest.raises(error_type):
        for offset in range(start, len(X), chunk):
            est.update_batch(X[offset:offset + chunk])
    d = est.get_diagnostics()
    assert d['details']['observations_attempted'] == attempted
    assert d['observations'] == d['details']['observations_committed'] == committed
    assert not d['ready']
    for call in (est.finalize, est.get_scale_scatter, est.get_subspace,
                 lambda: est.transform(X[0]), lambda: est.update(X[0]),
                 lambda: est.update_batch(X)):
        with pytest.raises(ft.FailedStateError): call()
    est.reset()
    assert est.get_diagnostics()['observations'] == 0
    assert est.get_diagnostics()['details']['failure'] is None


def test_invalid_inputs_do_not_poison_healthy_streams():
    for est in (ft.StreamingMADForceEstimator(2), ft.DualBoundForceEstimator(2, 2, calibration_size=5)):
        est.fit(np.ones((210, 2)))
        for values in ([[1., 2.], [np.nan, 2.]], [[1., 2., 3.]]):
            with pytest.raises(ValueError): est.update_batch(values)
        assert est.get_diagnostics()['ready']
        est.finalize()


@pytest.mark.parametrize('algorithm', ['mad', 'dual'])
def test_core_keyboard_interrupt_invalidates_partial_stream(monkeypatch, algorithm):
    values = np.random.default_rng(61).normal(size=(40, 2))
    if algorithm == 'mad':
        est = ft.StreamingMADForceEstimator(2, n_initial_warm=0).fit(values)
        est._native_fault[:] = (41, 1, 1)
        target, method = est, '_native_failure'
    else:
        est = ft.DualBoundForceEstimator(2, 2, calibration_size=5).fit(values)
        target, method = est._engine._sketch, 'update'
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(target, method, interrupt)
    with pytest.raises(KeyboardInterrupt):
        est.update([1., 2.])
    details = est.get_diagnostics()
    assert not details['ready'] and details['observations'] == 40
    assert details['details']['observations_attempted'] == 41
    with pytest.raises(ft.FailedStateError):
        est.finalize()


def test_callback_keyboard_interrupt_leaves_committed_row_healthy():
    est = ft.StreamingMADForceEstimator(2, n_initial_warm=0)
    def interrupt(event):
        raise KeyboardInterrupt()
    est.diagnostic_sink = interrupt
    with pytest.raises(KeyboardInterrupt):
        est.update([1., 1.])
    d = est.get_diagnostics()
    assert d['observations'] == 1
    assert d['details']['failure'] is None
    est.diagnostic_sink = None
    est.update([2., 2.])
    assert est.get_diagnostics()['observations'] == 2


@pytest.mark.parametrize('kind', ['acceptance', 'detector_cut', 'alarm'])
@pytest.mark.parametrize('prefix_size', [0, 137])
def test_callback_batch_failure_can_resume_from_committed_count(kind, prefix_size):
    baseline = np.tile([-2., -1., 1., 2.], 250)
    shift = np.tile([-2., -1., 1., 2.], 25) + 20
    values = np.column_stack([np.r_[baseline, shift]] * 2)
    reference = ft.StreamingMADForceEstimator(2).fit(values)
    def sink(event):
        if event['kind'] == kind:
            raise OSError('delivery failed')
    # Configure the sink before construction so nested detector events are wired.
    # Delay failures until the new chunk, even when the stream already has rows.
    enabled = False
    def delayed_sink(event):
        if enabled:
            sink(event)
    est = ft.StreamingMADForceEstimator(2, diagnostic_sink=delayed_sink)
    if prefix_size:
        est.update_batch(values[:prefix_size])
    before = est.get_diagnostics()['observations']
    chunk = values[prefix_size:]
    enabled = True
    with pytest.raises(ft.DiagnosticDeliveryError):
        est.update_batch(chunk)
    committed = est.get_diagnostics()['observations']
    assert before < committed <= len(values)
    assert est.get_diagnostics()['details']['failure'] is None
    est.diagnostic_sink = None
    est.update_batch(chunk[committed - before:])
    np.testing.assert_array_equal(est.finalize(), reference.finalize())
    np.testing.assert_array_equal(est.n_jk, reference.n_jk)
    assert est.get_diagnostics()['details']['reset_counts'] == reference.get_diagnostics()['details']['reset_counts']


def test_selective_reset_invalidates_only_incident_pair_estimates():
    est = ft.StreamingMADForceEstimator(3, n_initial_warm=0, use_drift_detection=False)
    est.fit(np.random.default_rng(1).normal(size=(100, 3)))
    assert est.get_diagnostics()['all_pairs_estimable']
    # Exercise the same moment operation used by a detector-triggered reset.
    est._moments.clear_dimension(0)
    d = est.get_diagnostics()
    assert d['any_pair_estimable'] and not d['all_pairs_estimable']
    np.testing.assert_array_equal(d['details']['pair_status_codes'][0, 1:], 1)
    assert d['details']['pair_status_codes'][1, 2] == 0
