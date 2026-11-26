"""Tests for the two-stage streaming MAD estimator."""

import numpy as np
import pytest
from force_toolkit._mad.streaming_mad import StreamingMAD


def test_gaussian_location_and_scale():
    """50 000 N(5, 4) → location ≈ 5, scale ≈ 2 (sd)."""
    rng = np.random.default_rng(42)
    mad = StreamingMAD()
    for x in rng.normal(5.0, 2.0, 50_000):
        mad.update(x)
    assert abs(mad.location() - 5.0) / 5.0 < 0.01, f"location={mad.location()}"
    assert abs(mad.scale() - 2.0) / 2.0 < 0.10, f"scale={mad.scale()}"


def test_cauchy_location_and_mad():
    """50 000 Cauchy(0,1) → location ≈ 0, MAD ≈ 1."""
    rng = np.random.default_rng(123)
    mad = StreamingMAD()
    for x in rng.standard_cauchy(50_000):
        mad.update(x)
    assert abs(mad.location()) < 0.15, f"location={mad.location()}"
    assert abs(mad.raw_mad() - 1.0) < 0.15, f"raw_mad={mad.raw_mad()}"


def test_survives_28pct_contamination():
    """10 000 clean + 4 000 outliers at 1000 (28.6 %) — survives above IQR's 25 %."""
    rng = np.random.default_rng(7)
    mad = StreamingMAD()
    data = np.concatenate([rng.standard_normal(10_000), np.full(4_000, 1000.0)])
    rng.shuffle(data)
    for x in data:
        mad.update(x)
    assert abs(mad.location()) < 1.0, f"location={mad.location()}"
    assert abs(mad.scale() - 1.0) < 1.0, f"scale={mad.scale()}"


def test_not_initialized_returns_nan():
    """Before 5 obs, both location and scale are NaN."""
    mad = StreamingMAD()
    assert np.isnan(mad.location())
    assert np.isnan(mad.scale())


def test_reset():
    """Reset clears both stages."""
    mad = StreamingMAD()
    rng = np.random.default_rng(0)
    for x in rng.standard_normal(200):
        mad.update(x)
    mad.reset()
    assert np.isnan(mad.location())


def test_equation_two_starts_at_second_observation():
    mad = StreamingMAD()
    for value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        mad.update(value)
    assert mad.deviation_count == 5
    assert mad.initialized
    # Deviations from exact bootstrap/P² stale medians:
    # 1, 1.5, 2, 2.5, 3.
    assert mad.raw_mad() == pytest.approx(2.0)


def test_bootstrap_location_remains_publicly_uninitialized():
    mad = StreamingMAD()
    mad.update(1.0)
    assert np.isnan(mad.location())
    assert mad.stage1.provisional_estimate() == 1.0
