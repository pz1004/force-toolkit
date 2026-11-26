"""Mutation boundaries, standardization range and live callback configuration."""
from collections import Counter
from decimal import Decimal, localcontext

import numpy as np
import pytest

from force_toolkit import (
    DiagnosticDeliveryError, DualBoundForceEstimator, FailedStateError,
    StreamingForceEstimator, StreamingMADForceEstimator,
)
from force_toolkit._dual.estimator import _standardize
from force_toolkit._force import streaming as force_streaming


@pytest.mark.parametrize('phase', ['initialization', 'after_pairs', 'quantile', 'window'])
@pytest.mark.parametrize('exception', [KeyboardInterrupt, SystemExit])
def test_force_interrupt_invalidates_partial_state_and_reset_recovers(monkeypatch, phase, exception):
    values = np.random.default_rng(44).normal(size=(180, 3))
    estimator = StreamingForceEstimator(window_size=50)

    def interrupt(*args, **kwargs):
        raise exception('injected processing interruption')

    if phase == 'initialization':
        monkeypatch.setattr(force_streaming, 'PairMoments', interrupt)
        chunk = values[:10]
    else:
        prefix = 145 if phase == 'window' else 120
        estimator.fit(values[:prefix])
        chunk = values[prefix:prefix + 10]
        if phase == 'window':
            monkeypatch.setattr(force_streaming, '_compute_bounds_from_quantiles', interrupt)
        elif phase == 'after_pairs':
            before_counts = estimator._pair_moments.accepted_counts()
            monkeypatch.setattr(estimator._quantile_state, 'update_batch', interrupt)
        else:
            update = estimator._quantile_state.update_batch
            def partial_quantile_update(part):
                update(part[:3])
                interrupt()
            monkeypatch.setattr(estimator._quantile_state, 'update_batch', partial_quantile_update)

    with pytest.raises(exception):
        estimator.update_batch(chunk)
    if phase == 'after_pairs':
        assert np.any(estimator._pair_moments.accepted_counts() > before_counts)
        assert estimator.n_samples_seen_ == 120
    assert estimator.status_ == 'failed'
    for call in (estimator.finalize, estimator.get_diagnostics,
                 lambda: estimator.update(values[0]),
                 lambda: estimator.update_batch(values[:3]),
                 lambda: estimator.accepted_counts_):
        with pytest.raises(RuntimeError, match='invalid after a processing failure'):
            call()
    monkeypatch.undo()
    assert estimator.reset().fit(values) is estimator
    expected = StreamingForceEstimator(window_size=50).fit(values)
    np.testing.assert_array_equal(estimator.finalize(), expected.finalize())
    np.testing.assert_array_equal(estimator.accepted_counts_, expected.accepted_counts_)


def test_force_preflight_and_readonly_interrupt_do_not_invalidate(monkeypatch):
    values = np.random.default_rng(9).normal(size=(120, 3))
    estimator = StreamingForceEstimator(window_size=50).fit(values)
    before = estimator.finalize()
    bad_chunk = values[:4].copy()
    bad_chunk[-1, 0] = np.nan
    with pytest.raises(ValueError):
        estimator.update_batch(bad_chunk)
    assert estimator.n_samples_seen_ == len(values)
    assert estimator.status_ == 'estimating'
    np.testing.assert_array_equal(estimator.finalize(), before)

    def interrupt_readout():
        raise KeyboardInterrupt('readout interrupted')
    monkeypatch.setattr(estimator._pair_moments, 'correlation', interrupt_readout)
    with pytest.raises(KeyboardInterrupt):
        estimator.finalize()
    assert estimator.status_ == 'estimating'
    monkeypatch.undo()
    np.testing.assert_array_equal(estimator.finalize(), before)


def _large_calibration(size=5):
    return np.column_stack([np.arange(-(size // 2), size // 2 + 1, dtype=float)] * 2) * 1e200


@pytest.mark.parametrize('chunk_size', [1, 7, 64])
@pytest.mark.parametrize('phase', ['initial_calibration', 'estimation', 'scheduled_calibration'])
def test_dual_underflow_fails_explicitly_and_blocks_partial_stream(phase, chunk_size):
    bad_calibration = np.column_stack([
        np.array([-2e200, -1e200, -1e-150, 0., 1e-150, 1e200, 2e200])
    ] * 2)
    if phase == 'initial_calibration':
        estimator = DualBoundForceEstimator(2, 2, calibration_size=7)
        values, attempted, committed = bad_calibration, 7, 6
    elif phase == 'estimation':
        estimator = DualBoundForceEstimator(2, 2, calibration_size=5)
        estimator.update_batch(_large_calibration())
        values = np.array([[1., 2.], [2., 1.], [1., -1.]]) * 1e-150
        attempted, committed = 6, 5
    else:
        estimator = DualBoundForceEstimator(2, 2, calibration_size=7, epoch_size=10)
        estimator.update_batch(np.vstack([_large_calibration(7), [[1., 2.], [2., 1.], [1., -1.]]]))
        assert estimator.get_diagnostics()['ready']
        values, attempted, committed = bad_calibration, 17, 16

    with pytest.raises(FloatingPointError, match='nonzero coordinate to zero'):
        for offset in range(0, len(values), chunk_size):
            estimator.update_batch(values[offset:offset + chunk_size])
    diagnostics = estimator.get_diagnostics()
    assert not diagnostics['ready']
    assert diagnostics['observations'] == committed
    assert diagnostics['details']['observations_attempted'] == attempted
    assert diagnostics['details']['lifecycle_status'] == 'failed'
    for call in (estimator.finalize, estimator.get_scale_scatter, estimator.get_subspace,
                 lambda: estimator.update([1., 2.]), lambda: estimator.transform([1., 2.])):
        with pytest.raises(FailedStateError):
            call()
    estimator.reset()
    assert estimator.get_diagnostics()['observations'] == 0
    assert estimator.get_diagnostics()['details']['failure'] is None
    estimator.fit(np.random.default_rng(1).normal(size=(30, 2)))
    assert estimator.get_diagnostics()['ready']


def test_dual_query_underflow_does_not_invalidate_existing_estimate():
    estimator = DualBoundForceEstimator(2, 2, calibration_size=5).fit(
        np.vstack([_large_calibration(), [[1., 2.], [2., 1.]]])
    )
    before = estimator.get_diagnostics()
    matrix = estimator.finalize()
    with pytest.raises(FloatingPointError, match='nonzero coordinate to zero'):
        estimator.transform([1e-150, 2e-150])
    assert estimator.get_diagnostics() == before
    np.testing.assert_array_equal(estimator.finalize(), matrix)
    estimator.update([1., -1.])
    assert estimator.get_diagnostics()['observations'] == before['observations'] + 1


def test_dual_centered_zero_and_representable_subnormals_remain_valid():
    tiny = np.nextafter(0., 1.)
    # Inexact subnormal rounding is permitted when the quotient stays nonzero.
    with np.errstate(under='raise'):
        actual = _standardize(np.array([3 * tiny, -3 * tiny, 0.]), np.zeros(3), np.full(3, 2.))
    np.testing.assert_array_equal(actual, [2 * tiny, -2 * tiny, 0.])
    estimator = DualBoundForceEstimator(2, 2, calibration_size=5).fit(_large_calibration())
    np.testing.assert_array_equal(estimator.transform([0., -0.]), [0., 0.])
    transformed = estimator.transform([1e-120, 2e-120])
    assert np.all(transformed > 0.)
    assert np.all(transformed < np.finfo(float).tiny)
    estimator.update([1e-120, 2e-120])
    assert estimator.get_diagnostics()['ready']
    np.testing.assert_allclose(np.diag(estimator.finalize()), 1., rtol=1e-14, atol=0.)


@pytest.mark.parametrize('external_underflow_policy', ['raise', 'ignore', 'warn'])
def test_dual_range_loss_policy_does_not_depend_on_numpy_seterr(external_underflow_policy):
    with np.errstate(under=external_underflow_policy):
        with pytest.raises(FloatingPointError, match='nonzero coordinate to zero'):
            _standardize(np.array([1e-150, 0.]), np.zeros(2), np.full(2, 1e200))


def _explicit_mask_standardize(values, location, scale):
    """Frozen pre-optimization range policy for differential verification."""
    with np.errstate(all='ignore'):
        difference = values - location
        standardized = difference / scale
        if np.any(np.isinf(difference)):
            alternative = values / scale - location / scale
            standardized = np.where(np.isinf(difference), alternative, standardized)
    if np.isnan(standardized).any():
        raise FloatingPointError('undefined')
    if np.any((standardized == 0.) & (values != location)):
        raise FloatingPointError('lost coordinate')
    return standardized


@pytest.mark.parametrize('matrix', [False, True])
def test_dual_underflow_retry_matches_explicit_mask_over_exponent_grid(matrix):
    rng = np.random.default_rng(805)
    shape = (4, 8) if matrix else (8,)
    def sample(shape, *, signed=True):
        with np.errstate(under='ignore'):
            values = np.ldexp(rng.uniform(.5, 1., size=shape),
                              rng.integers(-1073, 1025, size=shape))
        return values * rng.choice([-1., 1.], size=shape) if signed else values
    for _ in range(200):
        values, location, scale = sample(shape), sample((8,)), sample((8,), signed=False)
        values[..., 0] = location[0]  # Include genuine centered zeros.
        try:
            expected = _explicit_mask_standardize(values, location, scale)
        except FloatingPointError:
            with pytest.raises(FloatingPointError, match='nonzero coordinate to zero'):
                _standardize(values, location, scale)
        else:
            np.testing.assert_array_equal(_standardize(values, location, scale), expected)


@pytest.mark.parametrize('small', [1e-150, 1e-120, 0.])
@pytest.mark.parametrize('external_policy', ['raise', 'ignore', 'warn'])
def test_dual_mixed_overflow_fallback_and_tiny_coordinate(small, external_policy):
    largest = np.finfo(float).max
    values = np.array([largest, small, -largest])
    location = np.array([-largest, 0., largest])
    scale = np.array([largest, 1e200, largest])
    with np.errstate(all=external_policy):
        before = np.geterr()
        if small == 1e-150:
            with pytest.raises(FloatingPointError, match='nonzero coordinate to zero'):
                _standardize(values, location, scale)
        else:
            result = _standardize(values, location, scale)
            np.testing.assert_array_equal(result[[0, 2]], [2., -2.])
            with localcontext() as context:
                context.prec = 2000
                expected = float(Decimal.from_float(small) / Decimal.from_float(1e200))
            assert result[1] == expected
        assert np.geterr() == before


@pytest.mark.parametrize('numerator,denominator', [
    (np.nextafter(0., 1.), 2.),
    (-np.nextafter(0., 1.), 2.),
    (3 * np.nextafter(0., 1.), 2.),
    (-3 * np.nextafter(0., 1.), 2.),
    (np.nextafter(0., 1.), 1.),
])
def test_dual_smallest_quotients_follow_independent_rounding_oracle(numerator, denominator):
    with localcontext() as context:
        context.prec = 2000
        exact = Decimal.from_float(numerator) / Decimal.from_float(denominator)
        expected = float(exact)
    values, location, scale = np.array([numerator]), np.zeros(1), np.array([denominator])
    if expected == 0.:
        with pytest.raises(FloatingPointError, match='nonzero coordinate to zero'):
            _standardize(values, location, scale)
    else:
        np.testing.assert_array_equal(_standardize(values, location, scale), [expected])


def _drift_stream():
    marginal = np.r_[np.tile([-2., -1., 1., 2.], 250), np.tile([-2., -1., 1., 2.], 25) + 20]
    return np.column_stack([marginal, marginal])


def _observation_events(events):
    return [event for event in events if event['kind'] != 'stream_reset']


def test_mad_late_sink_receives_same_ordered_cuts_alarms_and_acceptances():
    values = _drift_stream()
    expected_events, actual_events = [], []
    reference = StreamingMADForceEstimator(2, diagnostic_sink=expected_events.append).fit(values)
    observed = StreamingMADForceEstimator(2)
    observed.diagnostic_sink = actual_events.append
    observed.update_batch(values)
    assert actual_events == _observation_events(expected_events)
    kinds = Counter(event['kind'] for event in actual_events)
    assert kinds == {'acceptance': 1100, 'detector_cut': 2, 'alarm': 2}
    np.testing.assert_array_equal(observed.finalize(), reference.finalize())
    np.testing.assert_array_equal(observed.n_jk, reference.n_jk)


def test_mad_sink_replacement_removal_and_reenable_respects_live_configuration():
    values = _drift_stream()
    expected_events = []
    reference = StreamingMADForceEstimator(2, diagnostic_sink=expected_events.append).fit(values)
    expected_events = _observation_events(expected_events)
    observed = StreamingMADForceEstimator(2)
    first, second, resumed = [], [], []
    observed.diagnostic_sink = first.append
    observed.update_batch(values[:300])
    observed.diagnostic_sink = second.append
    observed.update_batch(values[300:600])
    observed.diagnostic_sink = None
    observed.update_batch(values[600:900])
    observed.diagnostic_sink = resumed.append
    observed.update_batch(values[900:])
    def event_time(event):
        return event.get('t', event.get('time'))
    assert first == [event for event in expected_events if event_time(event) <= 300]
    assert second == [event for event in expected_events if 300 < event_time(event) <= 600]
    assert resumed == [event for event in expected_events if event_time(event) > 900]
    assert Counter(event['kind'] for event in resumed)['detector_cut'] == 2
    np.testing.assert_array_equal(observed.finalize(), reference.finalize())
    np.testing.assert_array_equal(observed.n_jk, reference.n_jk)


@pytest.mark.parametrize('reentrant', [False, True])
def test_mad_late_cut_callback_failure_commits_and_can_resume(reentrant):
    values = _drift_stream()
    reference = StreamingMADForceEstimator(2).fit(values)
    observed = StreamingMADForceEstimator(2).fit(values[:900])
    before = observed.get_diagnostics()['observations']
    delivered = []
    def sink(event):
        delivered.append(event)
        if event['kind'] == 'detector_cut':
            if reentrant:
                observed.update([1., 2.])
            raise OSError('cut logging unavailable')
    observed.diagnostic_sink = sink
    with pytest.raises(DiagnosticDeliveryError) as error:
        observed.update_batch(values[900:])
    assert isinstance(error.value.__cause__, RuntimeError if reentrant else OSError)
    committed = observed.get_diagnostics()['observations']
    assert committed > before
    assert observed.get_diagnostics()['details']['failure'] is None
    assert any(event['kind'] == 'acceptance' and event['t'] == committed for event in delivered)
    observed.diagnostic_sink = None
    observed.update_batch(values[committed:])
    np.testing.assert_array_equal(observed.finalize(), reference.finalize())
    np.testing.assert_array_equal(observed.n_jk, reference.n_jk)
