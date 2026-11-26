"""Independent range/cancellation oracles for reconstructed Dual-Bound scatter."""
from copy import deepcopy
from decimal import Decimal, localcontext

import numpy as np
import pytest

from force_toolkit import DualBoundForceEstimator
from force_toolkit._dual.estimator import _scale_reconstructed_gram


def decimal_scatter(buffer, scale, denominator):
    """Independent high-precision accumulation, without the production Fraction path."""
    with localcontext() as context:
        context.prec = 5000
        result = np.empty((buffer.shape[1], buffer.shape[1]))
        for j in range(buffer.shape[1]):
            for k in range(buffer.shape[1]):
                total = sum(
                    Decimal.from_float(float(row[j])) * Decimal.from_float(float(row[k]))
                    for row in buffer
                )
                total *= Decimal.from_float(float(scale[j]))
                total *= Decimal.from_float(float(scale[k]))
                result[j, k] = float(total / Decimal(int(denominator)))
        return result


def test_huge_calibration_small_estimation_recovers_representable_scatter():
    calibration = np.array([-2., -1., 0., 1., 2.])[:, None] * np.array([[1e200, 1e200]])
    tail = np.array([[1., 2.], [2., 1.], [1., -1.], [-1., 1.]])
    estimator = DualBoundForceEstimator(2, 2, calibration_size=5).fit(
        np.vstack((calibration, tail))
    )
    engine = estimator._engine
    expected = decimal_scatter(engine._sketch.B, engine.scale, engine.effective_n)
    np.testing.assert_allclose(expected, [[1.75, .5], [.5, 1.75]], rtol=2e-14, atol=0.)
    np.testing.assert_array_equal(estimator.get_scale_scatter(), expected)
    assert engine._sketch.compressions == 1  # k=p preserves every singular value.


@pytest.mark.parametrize('buffer,scale,denominator', [
    (np.array([[1e154, 0.], [1e-170, 1e154]]), np.ones(2), 2),
    (np.array([[1e-200, 1e200], [2e-200, -1e200]]), np.array([1e200, 1e-200]), 2),
    (np.array([[2e154]]), np.ones(1), 4),  # Squaring first would overflow.
    (np.array([[2.]]), np.array([1e308]), 10**310),  # Even B*scale overflows first.
    (np.array([[1e200, 1e-200], [1e200, -1e-200]]), np.array([1e-200, 1e200]), 2),
    (np.array([[2.0**-537]]), np.ones(1), 1),  # Smallest representable subnormal.
])
def test_extreme_products_match_independent_decimal_oracle(buffer, scale, denominator):
    expected = decimal_scatter(buffer, scale, denominator)
    actual = _scale_reconstructed_gram(buffer, scale, denominator)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()


def test_mixed_units_preserve_small_cross_term_and_large_diagonals():
    buffer = np.array([[1e154, 0.], [1e-170, 1e154]])
    result = _scale_reconstructed_gram(buffer, np.ones(2), 2)
    assert result[0, 1] == pytest.approx(5e-17, rel=2e-15, abs=0.)
    np.testing.assert_allclose(np.diag(result), [5e307, 5e307], rtol=2e-15)


def test_ordinary_fast_path_and_near_cancellation_resolution(monkeypatch):
    import force_toolkit._dual.estimator as implementation
    calls = []
    exact = implementation._exact_scatter_entries
    def recording(*args):
        calls.append(True)
        return exact(*args)
    monkeypatch.setattr(implementation, '_exact_scatter_entries', recording)
    rng = np.random.default_rng(812)
    buffer = rng.normal(size=(20, 4))
    scale = np.array([.25, 2., 3., 1.5])
    np.testing.assert_allclose(
        _scale_reconstructed_gram(buffer, scale, 37),
        decimal_scatter(buffer, scale, 37), rtol=2e-14, atol=1e-15,
    )
    assert not calls
    # The exact cross term survives; ordinary summation can lose it between
    # two much larger terms even though all operands have modest exponents.
    buffer = np.array([[2.0**30, 2.0**30], [2.0**-30, 2.0**-30], [2.0**30, -2.0**30]])
    result = _scale_reconstructed_gram(buffer, np.ones(2), 3)
    expected = decimal_scatter(buffer, np.ones(2), 3)
    np.testing.assert_array_equal(result, expected)
    assert calls
    assert result[0, 1] > 0.


def test_exact_zero_is_allowed_even_for_extreme_units():
    buffer = np.array([[1e200, 1e-200, 0.], [1e200, -1e-200, 0.]])
    scale = np.array([1e-200, 1e200, 1e308])
    result = _scale_reconstructed_gram(buffer, scale, 2)
    np.testing.assert_array_equal(result, decimal_scatter(buffer, scale, 2))
    assert result[0, 1] == 0.
    np.testing.assert_array_equal(result[2], np.zeros(3))
    np.testing.assert_array_equal(_scale_reconstructed_gram(np.zeros((3, 2)), np.full(2, 1e308), 1), np.zeros((2, 2)))


@pytest.mark.parametrize('buffer,denominator', [
    (np.array([[1e200]]), 1),
    (np.array([[1e-200]]), 1),
    (np.array([[1., 0.], [np.nextafter(0., 1.), 1.]]), 2),
])
def test_true_final_range_failure_is_explicit(buffer, denominator):
    with pytest.raises(FloatingPointError, match='range'):
        _scale_reconstructed_gram(buffer, np.ones(buffer.shape[1]), denominator)


def test_readout_overflow_leaves_normalized_estimate_and_state_usable():
    calibration = np.array([-2., -1., 0., 1., 2.])[:, None] * np.array([[1e200, 1e200]])
    tail = np.array([[1., 2.], [2., 1.], [1., -1.], [-1., 1.]]) * 1e200
    estimator = DualBoundForceEstimator(2, 2, calibration_size=5).fit(np.vstack((calibration, tail)))
    diagnostics = deepcopy(estimator.get_diagnostics())
    normalized = estimator.finalize()
    buffer = estimator._engine._sketch.B.copy()
    with pytest.raises(FloatingPointError, match='range'):
        estimator.get_scale_scatter()
    assert estimator.get_diagnostics() == diagnostics
    np.testing.assert_array_equal(estimator.finalize(), normalized)
    np.testing.assert_array_equal(estimator._engine._sketch.B, buffer)
    assert estimator.get_diagnostics()['ready']


def test_scheduled_recalibration_uses_old_served_scale_without_mutation():
    rng = np.random.default_rng(918)
    estimator = DualBoundForceEstimator(3, 3, calibration_size=5, epoch_size=12).fit(rng.normal(size=(12, 3)))
    expected = estimator.get_scale_scatter()
    estimator.update_batch(rng.normal(loc=10, scale=2, size=(5, 3)))
    before = deepcopy(estimator.get_diagnostics())
    np.testing.assert_array_equal(estimator.get_scale_scatter(), expected)
    np.testing.assert_array_equal(estimator.get_scale_scatter(), expected)
    assert estimator.get_diagnostics() == before
    assert estimator.get_diagnostics()['details']['serving_epoch'] == 0
