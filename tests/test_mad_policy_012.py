"""Independent range oracles and positive-unit equivariance for MAD policies."""
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from force_toolkit import MADForceEstimator, StreamingMADForceEstimator
from force_toolkit._mad.common import inclusive_bounds
from force_toolkit._mad.streaming_mad import StreamingMAD
from force_toolkit._mad.ter import compute_asymmetric_ter, compute_guarded_asymmetric_ter
from tests.reference.force_toolkit_baseline._mad.common import inclusive_bounds as old_bounds
from tests.reference.force_toolkit_baseline._mad.ter import compute_guarded_asymmetric_ter as old_ter


def decimal_endpoint(location, scale, lam, multiplier, direction):
    with localcontext() as context:
        context.prec = 4000
        value = (Decimal.from_float(float(location)) + direction
                 * Decimal.from_float(float(scale)) * Decimal.from_float(float(lam))
                 * Decimal.from_float(float(multiplier)))
        largest = Decimal.from_float(np.finfo(float).max)
        if value > largest:
            return math.inf
        if value < -largest:
            return -math.inf
        return float(value)


@pytest.mark.parametrize('location,scale,lam,lower_mult,upper_mult', [
    (-1e308, 7e307, 3., 1., 1.),
    (1e308, 7e307, 3., 1., 1.),
    (.5, 1e-308, 1e308, 3., 2.),
    (0., 1e308, 1e-300, 1e-100, 2e-100),
    (0., 1e308, 1e-320, 2., 3.),
    (np.nextafter(0., 1.), np.nextafter(0., 1.), .5, 1., 1.),
    (np.finfo(float).max, 0., np.finfo(float).max, 3., 3.),
    (np.finfo(float).max, np.finfo(float).max, 3., 1., 1.),
])
@pytest.mark.parametrize('shape', ['scalar', 'vector', 'zero_dim_multiplier'])
def test_range_safe_bounds_match_independent_decimal(location, scale, lam, lower_mult, upper_mult, shape):
    expected = [decimal_endpoint(location, scale, lam, lower_mult, -1),
                decimal_endpoint(location, scale, lam, upper_mult, 1)]
    if shape == 'vector':
        location, scale = np.array([location]), np.array([scale])
    elif shape == 'zero_dim_multiplier':
        lower_mult, upper_mult = np.array(lower_mult), np.array(upper_mult)
    with np.errstate(all='raise'):
        actual = inclusive_bounds(location, scale, lam, lower_mult, upper_mult)
    for result, value in zip(actual, expected):
        np.testing.assert_array_equal(result, value if shape != 'vector' else [value])


def test_broadcast_bounds_and_ordinary_arithmetic_remain_compatible():
    rng = np.random.default_rng(77)
    location = rng.normal(size=(3, 1))
    scale = rng.uniform(.1, 2., size=4)
    multipliers = np.array([1., 1.1, 1.2, 1.3])
    actual = inclusive_bounds(location, scale, 3., multipliers, multipliers[::-1])
    expected = old_bounds(location, scale, 3., multipliers, multipliers[::-1])
    for observed, reference in zip(actual, expected):
        np.testing.assert_array_equal(observed, reference)
    for j in range(3):
        for k in range(4):
            scalar = inclusive_bounds(location[j, 0], scale[k], 3., multipliers[k], multipliers[::-1][k])
            for position in range(2):
                assert scalar[position] == actual[position][j, k]
    for observed, reference in zip(inclusive_bounds(1., 2., multipliers),
                                   old_bounds(1., 2., multipliers)):
        np.testing.assert_array_equal(observed, reference)


def test_reported_bound_overflow_rejects_the_outlier_with_finite_endpoint():
    x = np.r_[[0., 1.5e308, 0., 0., 0.], np.tile([-1e308, -.6e308, -.2e308], 10)]
    values = np.column_stack([x, np.arange(len(x))])
    tracker = StreamingMAD()
    for value in x:
        tracker.update(float(value))
    lam = (np.finfo(float).max / tracker.scale()) * 1.01
    expected_upper = decimal_endpoint(tracker.location(), tracker.scale(), lam, 1., 1)
    estimator = MADForceEstimator(2, lam=lam).fit(values)
    lower, upper = estimator.get_diagnostics()['details']['thresholds']
    assert upper[0] == expected_upper < x[1]
    accepted = (values >= lower).all(axis=1) & (values <= upper).all(axis=1)
    assert accepted.sum() == 34
    assert estimator.get_diagnostics()['details']['accepted_counts'][0, 1] == 34
    reference = np.corrcoef((values[accepted] / [1e308, 1.]).T)
    np.testing.assert_allclose(estimator.finalize(), reference, rtol=1e-12, atol=1e-14)
    with np.errstate(over='ignore'):
        assert np.isinf(old_bounds(tracker.location(), tracker.scale(), lam)[1])


@pytest.mark.parametrize('q01,q50,q99,scale', [
    (-2., 0., 3., 1.),
    (-1.6e308, -1.5e308, 1.5e308, 1e308),
    (-1e308, 0., 1e308, 1e308),
    (-1e-320, 0., 3e-320, 1e-320),
    (-np.nextafter(0., 1.), 0., 3*np.nextafter(0., 1.), np.nextafter(0., 1.)),
])
def test_tail_ratios_match_independent_decimal_across_float_range(q01, q50, q99, scale):
    with localcontext() as context:
        context.prec = 4000
        denominator = Decimal.from_float(2.326) * Decimal.from_float(float(scale))
        expected = tuple(min(3., max(1., float(abs(Decimal.from_float(float(a))
                              - Decimal.from_float(float(b))) / denominator)))
                         for a, b in ((q99, q50), (q50, q01)))
    np.testing.assert_allclose(compute_asymmetric_ter(q01, q50, q99, scale), expected,
                               rtol=2e-15, atol=0.)


def test_zero_scale_keeps_collapsed_bounds_and_ordinary_guard_policy():
    assert compute_asymmetric_ter(-1., 0., 1., 0.) == (1., 1.)
    assert compute_guarded_asymmetric_ter(-1., 0., 1., 0.) == (1., 1., False)
    for args in ((-2., 0., 3., 1.), (-2., 0., 2., 1.), (-2., 0., 8., 1.)):
        assert compute_guarded_asymmetric_ter(*args) == old_ter(*args)
    # The changed behavior is intentional only where the dimensional gate was active.
    args = (-2e-12, 0., 3e-12, 1e-12)
    assert old_ter(*args) == (1., 1., False)
    assert compute_guarded_asymmetric_ter(*args)[2]


def skewed_input(n=3000):
    rng = np.random.default_rng(42)
    values = rng.normal(size=(n, 2))
    values[:, 1] = .6 * values[:, 0] + .8 * values[:, 1]
    values[values > 0] *= 1.4
    return values


@pytest.mark.parametrize('scales', [
    [1e-12, 1e-12], [1e-12, 1e12], [2.**-600, 2.**600],
])
def test_stationary_guarded_acceptance_is_equivariant_under_positive_units(scales):
    values = skewed_input()
    reference = MADForceEstimator(2, tail_mode='guarded_asymmetric').fit(values)
    actual = MADForceEstimator(2, tail_mode='guarded_asymmetric').fit(values * scales)
    a, b = actual.get_diagnostics()['details'], reference.get_diagnostics()['details']
    np.testing.assert_array_equal(a['accepted_counts'], b['accepted_counts'])
    for endpoint, expected in zip(a['thresholds'], b['thresholds']):
        np.testing.assert_allclose(endpoint / scales, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual.finalize(), reference.finalize(), rtol=1e-12, atol=1e-14)
    assert b['accepted_counts'][0, 1] == 2994


@pytest.mark.parametrize('scales', [[1e-12, 1e12], [2.**-600, 2.**600]])
@pytest.mark.parametrize('chunk', [1, 7, 64, 1024])
def test_streaming_guarded_decisions_resets_and_counts_are_unit_equivariant(scales, chunk):
    values = np.vstack([skewed_input(1000), skewed_input(100) + 20., skewed_input(100)])
    events_a, events_b = [], []
    actual = StreamingMADForceEstimator(2, tail_mode='guarded_asymmetric', diagnostic_sink=events_a.append)
    reference = StreamingMADForceEstimator(2, tail_mode='guarded_asymmetric', diagnostic_sink=events_b.append)
    for offset in range(0, len(values), chunk):
        actual.update_batch(values[offset:offset + chunk] * scales)
        reference.update_batch(values[offset:offset + chunk])
        np.testing.assert_array_equal(actual.n_jk, reference.n_jk)
        np.testing.assert_array_equal(actual.reset_counts, reference.reset_counts)
        assert actual.get_diagnostics()['ready'] == reference.get_diagnostics()['ready']
    assert events_a == events_b
    assert actual.reset_counts.sum() > 0
    np.testing.assert_allclose(actual.finalize(), reference.finalize(), rtol=1e-12, atol=1e-14)
