"""Pearson conformance on representable data with difficult origins and units."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from force_toolkit._mad.common import PairwiseMoments, covariance_to_correlation, trimmed_correlation
from force_toolkit._mad.drift_mad_force import DriftMADForce


def decimal_pearson(pair):
    """Independent raw-sum identity, evaluated with ample decimal precision."""
    with localcontext() as context:
        context.prec = 1500
        x, y = ([Decimal.from_float(float(value)) for value in column] for column in pair.T)
        n = Decimal(len(x))
        sx, sy = sum(x), sum(y)
        xx = n * sum(value * value for value in x) - sx * sx
        yy = n * sum(value * value for value in y) - sy * sy
        xy = n * sum(a * b for a, b in zip(x, y)) - sx * sy
        return 0.0 if xx == 0 or yy == 0 else float(xy / (xx * yy).sqrt())


def accumulate(x, accepted=None):
    moments = PairwiseMoments(x.shape[1])
    if accepted is None:
        accepted = np.ones(x.shape, dtype=bool)
    for row, mask in zip(x, accepted):
        moments.update(row, mask)
    return moments


@pytest.mark.parametrize("scales", [[1e-200] * 3, [1e200] * 3, [1e-200, 1e200, 1e-150]])
def test_accepted_pair_pearson_matches_decimal_in_extreme_units(scales):
    base = np.array([
        [-4, 1, 3], [-3, 0, -2], [-2, -1, -1], [-1, 3, 4],
        [0, -2, 1], [1, 4, 0], [2, 2, -3], [3, -3, 2], [4, 1, -4],
    ], dtype=float)
    lower, upper = np.array([-3, -2, -3]), np.array([3, 3, 3])
    accepted = (base >= lower) & (base <= upper)
    x = base * scales
    moments = accumulate(x, accepted)
    expected = np.eye(3)
    for j in range(3):
        for k in range(j + 1, 3):
            mask = accepted[:, j] & accepted[:, k]
            expected[j, k] = expected[k, j] = decimal_pearson(x[mask][:, [j, k]])
            assert moments.count[j, k] == np.count_nonzero(mask)
    np.testing.assert_allclose(moments.correlation(), expected, atol=2e-15)
    np.testing.assert_allclose(
        trimmed_correlation(x, lower * scales, upper * scales), expected, atol=2e-15
    )


@pytest.mark.parametrize("offset", [0.0, 1e16])
def test_mad_streaming_retains_accepted_pair_pearson_after_large_translation(offset):
    # The first 200 rows are warm-up. The accepted sample is one (8, 8)
    # followed by N balanced corners, giving rho = 16 / (N + 17) exactly.
    body = np.tile([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]], (2500, 1))
    x = np.vstack([body[:200], [8.0, 8.0], body]) + offset
    estimator = DriftMADForce(2, use_drift_detection=False)
    actual = estimator.fit_streaming(x)
    assert estimator.n_jk[0, 1] == len(body) + 1
    assert actual[0, 1] == pytest.approx(16 / (len(body) + 17), abs=2e-14)


def test_batch_large_offset_matches_exact_represented_input():
    base = np.array([[-8, 2], [-4, 6], [0, -2], [2, 4], [6, 0], [8, 8]], dtype=float)
    x = base + 1e16
    expected = decimal_pearson(x)
    result = trimmed_correlation(x, [-np.inf] * 2, [np.inf] * 2)
    assert result[0, 1] == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("method", ["fit_stationary", "fit_streaming"])
def test_mad_force_nonperfect_correlation_survives_tiny_units(method):
    rng = np.random.default_rng(10)
    x = rng.normal(size=1000)
    data = np.column_stack([x, 0.8 * x + 0.6 * rng.normal(size=1000)])
    reference_estimator = DriftMADForce(2, use_drift_detection=False)
    actual_estimator = DriftMADForce(2, use_drift_detection=False)
    reference = getattr(reference_estimator, method)(data)
    actual = getattr(actual_estimator, method)(data * 1e-200)
    assert 0.7 < reference[0, 1] < 0.9
    np.testing.assert_allclose(actual, reference, atol=2e-14)
    if method == "fit_streaming":
        np.testing.assert_array_equal(actual_estimator.n_jk, reference_estimator.n_jk)


@pytest.mark.parametrize("reverse", [False, True])
def test_unequal_covariance_units_preserve_representable_tiny_correlation(reverse):
    covariance = np.array([[1e300, 1e-200], [1e-200, 1e-300]])
    if reverse:
        covariance = covariance[::-1, ::-1]
    with localcontext() as context:
        context.prec = 100
        expected = float(Decimal.from_float(1e-200) / (
            Decimal.from_float(1e300) * Decimal.from_float(1e-300)
        ).sqrt())
    actual = covariance_to_correlation(covariance)[0, 1]
    assert actual == pytest.approx(expected, rel=2e-15, abs=0.0)


@pytest.mark.parametrize("scales", [
    [np.finfo(float).max / 2, np.finfo(float).max / 2],
    [np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)],
    [np.finfo(float).max / 2, np.nextafter(0.0, 1.0)],
])
def test_finite_endpoint_and_subnormal_observations_match_decimal(scales):
    x = np.array([[-2., -1.], [-1., -1.], [0., 0.], [1., 2.], [2., 1.]]) * scales
    expected = decimal_pearson(x)
    assert accumulate(x).correlation()[0, 1] == pytest.approx(expected, abs=2e-15)
    assert trimmed_correlation(x, [-np.inf] * 2, [np.inf] * 2)[0, 1] == pytest.approx(
        expected, abs=2e-15
    )


def test_late_extreme_observation_rescales_existing_pair_state():
    x = np.array([[0., 0.], [1e-300, 1e-300], [-1e-300, -1e-300],
                  [1e300, 1e-300], [-1e300, -1e-300]])
    moments = PairwiseMoments(2)
    for end, row in enumerate(x, start=1):
        moments.update(row, np.ones(2, dtype=bool))
        if end >= 2:
            assert moments.correlation()[0, 1] == pytest.approx(decimal_pearson(x[:end]), abs=2e-15)


def test_selective_reset_clears_origins_and_scales_without_replacing_aliases():
    before = np.array([[1., 2., -2.], [2., 3., 1.], [-1., -1., 3.]]) * 1e200
    after = np.array([[0., 0., 0.], [2., 3., 1.], [-1., 2., 3.]]) * 1e-200
    moments = accumulate(before)
    arrays = {name: value for name, value in vars(moments).items() if isinstance(value, np.ndarray)}
    unrelated = {name: value[0, 2] for name, value in arrays.items()}
    moments.clear_dimension(1)
    for name, array in arrays.items():
        assert getattr(moments, name) is array
        assert array[0, 2] == unrelated[name]
    assert moments.state_bytes == sum(array.nbytes for array in arrays.values())
    assert PairwiseMoments(6).state_bytes == 4 * moments.state_bytes
    for row in after:
        moments.update(row, np.ones(3, dtype=bool))
    actual = moments.correlation()
    for j, k in [(0, 1), (1, 2)]:
        assert moments.count[j, k] == len(after)
        assert actual[j, k] == pytest.approx(decimal_pearson(after[:, [j, k]]), abs=2e-15)
    assert moments.count[0, 2] == len(before) + len(after)
    assert actual[0, 2] == pytest.approx(decimal_pearson(np.vstack([before, after])[:, [0, 2]]), abs=2e-15)


@pytest.mark.parametrize("covariance", [
    [[-1., 0.], [0., 1.]],
    [[1., 2.], [2., 1.]],
])
def test_invalid_moments_are_not_clipped_or_reported_as_zero(covariance):
    with pytest.raises(FloatingPointError):
        covariance_to_correlation(covariance)


def test_failed_arithmetic_cannot_finalize_partially_updated_pairs():
    moments = accumulate(np.array([[0., 0., 0.], [1., 2., 3.]]))
    moments.count[1, 2] = 2**53
    with pytest.raises(FloatingPointError, match="count"):
        moments.update(np.array([2., 3., 4.]), np.ones(3, dtype=bool))
    with pytest.raises(RuntimeError, match="recreate"):
        moments.correlation()
    with pytest.raises(RuntimeError, match="recreate"):
        moments.update(np.array([2., 3., 4.]), np.ones(3, dtype=bool))
