"""Distinguish numerical conformance, approximation and detector scope."""
import numpy as np
import pytest

from force_toolkit import MADForceEstimator, StreamingMADForceEstimator
from studies.mad_characterization import (
    accepted_pair_reference, final_center_bounds, make_data, ordered_data,
)


def test_marginal_detector_has_no_signal_for_correlation_only_change():
    # Every marginal acceptance bit is one. The post-warm-up sample contains
    # 400 positively related rows and 600 negatively related rows, with equal
    # means and variances, hence historical correlation exactly -0.2.
    x = np.tile([-2.0, -1.0, 1.0, 2.0], 300)
    y = x.copy()
    y[600:] *= -1
    data = np.column_stack([x, y])
    estimator = StreamingMADForceEstimator(2).fit(data)
    details = estimator.get_diagnostics()["details"]
    assert details["alarm_count"] == 0
    assert details["reset_counts"] == [0, 0]
    assert details["acceptance_rates"] == [1.0, 1.0]
    assert estimator.finalize()[0, 1] == pytest.approx(-0.2, abs=1e-12)
    assert np.corrcoef(data[600:].T)[0, 1] == pytest.approx(-1.0, abs=1e-14)


def test_designed_marginal_change_produces_a_post_change_alarm():
    # This is a deterministic positive control, not a power/calibration claim.
    baseline = np.tile([-2.0, -1.0, 1.0, 2.0], 250)
    shifted = np.tile([-2.0, -1.0, 1.0, 2.0], 25) + 20
    data = np.column_stack([np.r_[baseline, shifted]] * 2)
    estimator = StreamingMADForceEstimator(2).fit(data)
    details = estimator.get_diagnostics()["details"]
    assert details["alarm_count"] >= 1
    assert all(alarm["t"] > len(baseline) for alarm in details["alarm_locations"])
    assert all(count >= 1 for count in details["reset_counts"])


@pytest.mark.parametrize("distribution", ["gaussian", "lognormal", "student_t3", "contaminated_gaussian"])
def test_stationary_readout_matches_its_actual_accepted_sample(distribution):
    values = make_data(distribution, seed=19, n=150, p=4)
    estimator = MADForceEstimator(4).fit(values)
    lower, upper = estimator.get_diagnostics()["details"]["thresholds"]
    accepted = (values >= lower) & (values <= upper)
    reference, _, _ = accepted_pair_reference(values, accepted)
    np.testing.assert_allclose(estimator.finalize(), reference, rtol=1e-12, atol=1e-14)


def test_exact_final_center_comparator_is_permutation_invariant():
    values = make_data("contaminated_gaussian", seed=19, n=150, p=4)
    lower, upper, scale = final_center_bounds(values)
    expected, valid, counts = accepted_pair_reference(values, (values >= lower) & (values <= upper))
    for order in ("ascending_first", "descending_first", "shuffle_0"):
        reordered = ordered_data(values, order, seed=19)
        other_lower, other_upper, other_scale = final_center_bounds(reordered)
        np.testing.assert_array_equal(lower, other_lower)
        np.testing.assert_array_equal(upper, other_upper)
        np.testing.assert_array_equal(scale, other_scale)
        actual, actual_valid, actual_counts = accepted_pair_reference(
            reordered, (reordered >= other_lower) & (reordered <= other_upper))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(actual_valid, valid)
        np.testing.assert_array_equal(actual_counts, counts)


def test_exact_previous_prefix_deviations_differ_from_final_center_definition():
    from studies.mad_characterization import exact_prefix_bounds
    # Exact arithmetic, no P² approximation: deviations from the previous
    # prefix median are [1, 1.5, 2, 2.5, 3], whose median is 2; final-center
    # deviations have median 1.5.
    values = np.column_stack([np.arange(6, dtype=float)] * 2)
    prefix_lower, prefix_upper = exact_prefix_bounds(values)
    final_lower, final_upper, _ = final_center_bounds(values)
    np.testing.assert_allclose((prefix_upper - prefix_lower) / 6, 1.4826 * 2)
    np.testing.assert_allclose((final_upper - final_lower) / 6, 1.4826 * 1.5)
