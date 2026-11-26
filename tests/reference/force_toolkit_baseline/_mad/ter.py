"""Tail expansion ratio helpers for Drift-MAD FORCE."""

import numpy as np
from numba import njit

EPS = 1e-10
GAUSSIAN_Q99 = 2.326  # Φ⁻¹(0.99)


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
    c_ref = GAUSSIAN_Q99 * sigma_mad
    if c_ref < EPS:
        return 1.0, 1.0

    ter_plus = min(ter_max, max(1.0, abs(q99 - q50) / c_ref))
    ter_minus = min(ter_max, max(1.0, abs(q50 - q01) / c_ref))
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
    if opposite < EPS:
        return 1.0, 1.0, False

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
