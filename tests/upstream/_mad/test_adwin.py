"""Tests for the ADWIN drift detector."""

import numpy as np
import pytest
from force_toolkit._mad.adwin import ADWINDetector


def test_detects_mean_shift():
    """Feed N(0,1) then N(3,1) — drift should be detected within ~300 obs of change."""
    rng = np.random.default_rng(42)
    det = ADWINDetector(delta=0.002)
    drift_at = None
    change_point = 5000
    for t in range(10_000):
        if t < change_point:
            x = rng.standard_normal()
        else:
            x = rng.standard_normal() + 3.0
        if det.update(x) and t >= change_point and drift_at is None:
            drift_at = t
    assert drift_at is not None, "Drift was not detected"
    assert drift_at - change_point < 300, f"Detection too slow: {drift_at - change_point} obs"


def test_no_false_alarm_stationary_binary():
    """10 000 stationary binary (acceptance rate 0.9) — at most 3 false alarms.

    In the actual algorithm ADWIN monitors binary acceptance indicators,
    so we test with Bernoulli data (the relevant use case).
    """
    rng = np.random.default_rng(123)
    det = ADWINDetector(delta=0.002)
    alarms = 0
    for _ in range(10_000):
        x = 1.0 if rng.random() < 0.9 else 0.0
        if det.update(x):
            alarms += 1
    assert alarms <= 3, f"Too many false alarms: {alarms}"


def test_detects_acceptance_rate_drop():
    """Mean shift from 0.8 to 0.3 (mimics acceptance rate drop)."""
    rng = np.random.default_rng(7)
    det = ADWINDetector(delta=0.002)
    detected = False
    for t in range(6000):
        x = 0.8 + rng.standard_normal() * 0.1 if t < 3000 else 0.3 + rng.standard_normal() * 0.1
        if det.update(x) and t >= 3000:
            detected = True
            break
    assert detected, "Failed to detect acceptance rate drop"


def test_reset():
    """After reset, detector state is clean."""
    det = ADWINDetector()
    rng = np.random.default_rng(0)
    for _ in range(100):
        det.update(rng.standard_normal())
    det.reset()
    assert det._count == 0
    assert det.bucket_count == 0
    assert det.n_rows == 0


def test_bucket_memory_is_bounded():
    """Bucket compression keeps memory sublinear in the window length."""
    rng = np.random.default_rng(1234)
    det = ADWINDetector(max_window=10_000, max_buckets=5)
    for _ in range(20_000):
        det.update(rng.standard_normal())
    assert det._count <= det.max_window
    assert det.bucket_count <= det.max_buckets * max(det.n_rows, 1)
