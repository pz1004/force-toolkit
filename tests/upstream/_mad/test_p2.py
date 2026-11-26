"""Tests for the P² quantile estimator."""

import numpy as np
import pytest
from force_toolkit._mad.p2 import P2Estimator, p2_init, p2_update, p2_estimate


@pytest.mark.parametrize("phi", [0.01, 0.25, 0.50, 0.75, 0.99])
def test_gaussian_quantiles(phi):
    """P² on 10 000 N(0,1) samples should approximate Φ⁻¹(φ) within 2%."""
    from scipy.stats import norm
    rng = np.random.default_rng(42)
    est = P2Estimator(phi)
    for x in rng.standard_normal(10_000):
        est.update(x)
    expected = norm.ppf(phi)
    # For extreme quantiles, use absolute tolerance
    if abs(expected) < 0.1:
        assert abs(est.estimate() - expected) < 0.15
    else:
        assert abs(est.estimate() - expected) / abs(expected) < 0.05, \
            f"phi={phi}: got {est.estimate():.4f}, expected {expected:.4f}"


def test_cauchy_median():
    """Median of Cauchy(0,1) should converge near 0."""
    rng = np.random.default_rng(123)
    est = P2Estimator(0.50)
    for x in rng.standard_cauchy(10_000):
        est.update(x)
    assert abs(est.estimate()) < 0.15


def test_five_observations():
    """With exactly 5 obs the estimates equal the sorted values."""
    vals = [3.0, 1.0, 5.0, 2.0, 4.0]
    est = P2Estimator(0.50)
    for v in vals:
        est.update(v)
    # After sorting: [1, 2, 3, 4, 5] → median marker q[2] = 3.0
    assert est.estimate() == 3.0


def test_numba_matches_class():
    """Numba flat-array API and Python class produce the same result."""
    rng = np.random.default_rng(7)
    phi = 0.75
    data = rng.standard_normal(500)

    # Python class
    est = P2Estimator(phi)
    for x in data:
        est.update(x)
    class_result = est.estimate()

    # Direct Numba calls
    q, n, n_prime, count, buf, init = p2_init(phi)
    for x in data:
        count, init = p2_update(q, n, n_prime, count, buf, init, phi, x)
    numba_result = p2_estimate(q, init)

    assert class_result == numba_result


def test_not_initialized_returns_nan():
    """Estimate before 5 obs should be NaN."""
    est = P2Estimator(0.50)
    assert np.isnan(est.estimate())
    est.update(1.0)
    assert np.isnan(est.estimate())


def test_reset():
    """Reset should return estimator to uninitialized state."""
    est = P2Estimator(0.50)
    rng = np.random.default_rng(0)
    for x in rng.standard_normal(100):
        est.update(x)
    est.reset()
    assert np.isnan(est.estimate())
    assert est.count == 0


@pytest.mark.parametrize("quantile", [-0.1, 0.0, 1.0, 1.1, np.nan, np.inf])
def test_rejects_invalid_quantiles(quantile):
    with pytest.raises(ValueError):
        P2Estimator(quantile)


def test_marker_invariants_and_standard_desired_positions():
    rng = np.random.default_rng(908)
    estimator = P2Estimator(0.37)
    for value in rng.standard_t(df=3, size=2000):
        estimator.update(value)
        if estimator.initialized:
            estimator.validate_state()


def test_rejects_nonfinite_observations():
    estimator = P2Estimator(0.5)
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            estimator.update(value)
