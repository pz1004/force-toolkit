"""Focused invariants for the corrected paper implementation."""

import numpy as np
import pytest

from force_toolkit._mad.common import PairwiseMoments, trimmed_correlation
from force_toolkit._mad.drift_mad_force import DriftMADForce


def test_pairwise_moments_match_independent_centered_reference_with_large_offset():
    rng = np.random.default_rng(4)
    x = 1e12 + rng.standard_normal((500, 3))
    accepted = rng.random((500, 3)) > 0.15
    moments = PairwiseMoments(3, epsilon=1e-12)
    for row, mask in zip(x, accepted):
        moments.update(row, mask)
    actual = moments.correlation()
    expected = trimmed_correlation(
        x,
        np.array([-np.inf, -np.inf, -np.inf]),
        np.array([np.inf, np.inf, np.inf]),
        epsilon=1e-12,
    )
    # Build the truly pair-masked independent result.
    for j in range(3):
        for k in range(j + 1, 3):
            mask = accepted[:, j] & accepted[:, k]
            centered_j = x[mask, j] - np.mean(x[mask, j])
            centered_k = x[mask, k] - np.mean(x[mask, k])
            expected[j, k] = expected[k, j] = np.dot(
                centered_j, centered_k
            ) / np.sqrt(np.dot(centered_j, centered_j) * np.dot(centered_k, centered_k))
    np.testing.assert_allclose(actual, expected, atol=2e-5)


def test_degenerate_scale_uses_collapsed_inclusive_bounds():
    x = np.column_stack(
        [
            np.ones(20),
            np.linspace(-1.0, 1.0, 20),
            np.linspace(-1.0, 1.0, 20) ** 2,
        ]
    )
    result = DriftMADForce(3, use_drift_detection=False).fit_stationary(x)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(np.diag(result), np.ones(3))
    assert result[0, 1] == 0.0
    assert result[0, 2] == 0.0


def test_finalize_matches_correlation_accessor():
    estimator = DriftMADForce(
        3, use_drift_detection=False, n_initial_warm=0, min_reset_obs=0
    )
    for row in np.random.default_rng(2).standard_normal((50, 3)):
        estimator.update(row)
    np.testing.assert_array_equal(
        estimator.finalize(), estimator.get_correlation_matrix()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p": 1},
        {"p": 3, "lam": 0},
        {"p": 3, "detector_min_window": 10, "detector_min_subwindow": 6},
        {"p": 3, "detector_max_window": 10, "detector_min_window": 32},
        {"p": 3, "reset_buffer_size": 4},
        {"p": 3, "epsilon": 0},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        DriftMADForce(**kwargs)


