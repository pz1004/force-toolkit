"""Detector timing and reset-path verification."""

import numpy as np

from force_toolkit._mad.adwin import ADWINDetector
from force_toolkit._mad.drift_mad_force import DriftMADForce


def test_equation_nine_uses_absolute_time_not_retained_window_length():
    detector = ADWINDetector(max_window=32, min_window=16, min_subwindow=8)
    for time_index in range(1, 101):
        detector.update(1.0, time_index=time_index, allow_detection=False)
    assert detector.count <= 32
    assert detector.total_time == 100
    expected = detector.delta / (np.log2(100) + 1.0)
    retained_clock = detector.delta / (np.log2(detector.count) + 1.0)
    assert expected != retained_clock
    diagnostics = detector.diagnostics()
    assert diagnostics["total_time"] == 100
    assert diagnostics["retained_window_length"] == detector.count


def test_warmup_ingests_bits_but_suppresses_alarms():
    estimator = DriftMADForce(
        2,
        n_initial_warm=20,
        detector_min_window=8,
        detector_min_subwindow=4,
        detector_max_window=64,
        adwin_delta=0.5,
        min_reset_obs=0,
    )
    for value in [0.0] * 10 + [100.0] * 10:
        estimator.update(np.array([value, value]))
    assert estimator.get_diagnostics()["alarm_count"] == 0
    assert all(detector.total_time == 20 for detector in estimator.adwin)


def test_cut_retains_suffix_and_selective_reset_does_not_clear_detector():
    detector = ADWINDetector(
        delta=0.9,
        max_window=256,
        max_buckets=20,
        min_window=16,
        min_subwindow=4,
    )
    detected = False
    for value in [1.0] * 40 + [0.0] * 40:
        if detector.update(value):
            detected = True
            break
    assert detected
    suffix_count = detector.count
    total_time = detector.total_time
    assert suffix_count > 0

    estimator = DriftMADForce(
        2,
        n_initial_warm=0,
        min_reset_obs=3,
        reset_buffer_size=5,
        detector_min_window=8,
        detector_min_subwindow=4,
    )
    for row in np.random.default_rng(2).standard_normal((20, 2)):
        estimator.update(row)
    estimator.adwin[0] = detector
    estimator._selective_reset(0)
    assert estimator.adwin[0].count == suffix_count
    assert estimator.adwin[0].total_time == total_time
    assert not estimator._dimension_ready(0)


def test_default_reset_gate_requires_32_subsequent_observations_and_diagnostics():
    estimator = DriftMADForce(
        2,
        use_drift_detection=False,
        n_initial_warm=0,
        reset_buffer_size=32,
    )
    rng = np.random.default_rng(11)
    for row in rng.standard_normal((50, 2)):
        estimator.update(row)

    estimator._selective_reset(0)
    for row in rng.standard_normal((31, 2)):
        estimator.update(row)
    assert not estimator._dimension_ready(0)

    estimator.update(rng.standard_normal(2))
    diagnostics = estimator.get_diagnostics()
    assert estimator._dimension_ready(0)
    assert diagnostics["post_reset_observations"][0] == 32
    assert diagnostics["reset_ready"][0]
    assert diagnostics["state_bytes"] > 0
    detector_diagnostics = diagnostics["detector"]["per_dimension"]
    assert all("retained_window_length" in item for item in detector_diagnostics)
    assert all("total_time" in item for item in detector_diagnostics)
