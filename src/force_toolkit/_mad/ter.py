"""Tail expansion ratio helpers for Drift-MAD FORCE."""

import math
import numpy as np
from numba import njit

EPS = 1e-10
GAUSSIAN_Q99 = 2.326  # Φ⁻¹(0.99)


@njit(cache=True)
def _capped_tail_ratio(left, right, sigma_mad, ter_max):
    """Evaluate |left-right| / (2.326*sigma) without a dimensional floor."""
    difference = abs(left - right)
    reference = GAUSSIAN_Q99 * sigma_mad
    # Retain the original operation order in the ordinary range. The exceptional
    # path also avoids quantizing a subnormal numerator/reference before division.
    tiny = np.finfo(np.float64).tiny
    if (np.isfinite(difference) and np.isfinite(reference)
            and reference >= tiny and (difference == 0.0 or difference >= tiny)):
        return min(ter_max, max(1.0, difference / reference))
    if difference == 0.0:
        return 1.0
    if np.isfinite(difference):
        numerator, exponent = math.frexp(difference)
    else:
        # A finite-endpoint subtraction can overflow only for opposite signs.
        _, left_exponent = math.frexp(left)
        _, right_exponent = math.frexp(right)
        shift = max(left_exponent, right_exponent)
        scaled = abs(math.ldexp(left, -shift) - math.ldexp(right, -shift))
        numerator, local_exponent = math.frexp(scaled)
        exponent = shift + local_exponent
    denominator, denominator_exponent = math.frexp(sigma_mad)
    fraction, local_exponent = math.frexp(numerator / denominator / GAUSSIAN_Q99)
    exponent += local_exponent - denominator_exponent
    cap_fraction, cap_exponent = math.frexp(ter_max)
    if exponent > cap_exponent or (exponent == cap_exponent and fraction >= cap_fraction):
        return ter_max
    if exponent <= 0:
        return 1.0
    return max(1.0, math.ldexp(fraction, exponent))


@njit(cache=True)
def compute_asymmetric_ter(q01, q50, q99, sigma_mad, ter_max=3.0):
    """Compute capped asymmetric TER multipliers.

    Parameters
    ----------
    q01, q50, q99 : float
        Streaming quantile estimates (1st, 50th, 99th percentiles).
    sigma_mad : float
        Robust scale from streaming MAD (1.4826 * MAD).
    ter_max : float
        Cap to prevent unbounded expansion.

    Returns
    -------
    ter_plus, ter_minus : float
        Upper-tail and lower-tail TER multipliers (each >= 1.0).
    """
    if (not np.isfinite(q01) or not np.isfinite(q50) or not np.isfinite(q99)
            or not np.isfinite(sigma_mad) or sigma_mad < 0.0
            or not np.isfinite(ter_max) or ter_max < 1.0):
        raise ValueError("TER inputs must be finite, with nonnegative scale and ter_max >= 1")
    if sigma_mad == 0.0:
        # No scale information: retain the collapsed-bound convention.
        return 1.0, 1.0

    ter_plus = _capped_tail_ratio(q99, q50, sigma_mad, ter_max)
    ter_minus = _capped_tail_ratio(q50, q01, sigma_mad, ter_max)
    return ter_plus, ter_minus


def compute_guarded_asymmetric_ter(
    q01,
    q50,
    q99,
    sigma_mad,
    ter_max=3.0,
    activate_threshold=1.10,
    opposite_tail_max=1.05,
    extreme_tail_max=1.50,
    imbalance_min=1.15,
    imbalance_max=1.75,
):
    """Return a conservative asymmetric TER or fall back to no expansion.

    The guarded mode widens only a moderately heavier tail. Extreme or unstable
    imbalances are treated as likely contamination and fall back to TER = 1.
    """
    ter_plus, ter_minus = compute_asymmetric_ter(q01, q50, q99, sigma_mad, ter_max)

    dominant = max(ter_plus, ter_minus)
    opposite = min(ter_plus, ter_minus)
    imbalance = dominant / opposite
    guard_ok = (
        dominant >= activate_threshold
        and dominant <= extreme_tail_max
        and opposite <= opposite_tail_max
        and imbalance >= imbalance_min
        and imbalance <= imbalance_max
    )
    if not guard_ok:
        return 1.0, 1.0, False

    if ter_plus >= ter_minus:
        return ter_plus, 1.0, True
    return 1.0, ter_minus, True


@njit(cache=True)
def compute_symmetric_ter(q01, q50, q99):
    """Compute FORCE's original symmetric TER.

    TER = max(1, |q99 - q50| / |q50 - q01|)
    """
    upper = abs(q99 - q50)
    lower = abs(q50 - q01)
    if lower < EPS:
        return 1.0
    return max(1.0, upper / lower)
