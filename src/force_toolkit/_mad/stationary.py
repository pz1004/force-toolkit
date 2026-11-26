"""Stationary MAD bounds without allocating online pair or detector state.

This deliberately retains the previous-prefix residual construction. It is not
an exact final-center MAD; see docs/statistical-contracts.md for its audit.
"""
import numpy as np
from .common import validate_dimension, validate_positive_finite, inclusive_bounds
from .streaming_mad import StreamingMAD
from .p2 import P2Estimator
from .ter import compute_guarded_asymmetric_ter


def validate_parameters(p, lam, tail_mode, ter_max, epsilon):
    if tail_mode not in {'none', 'guarded_asymmetric'}:
        raise ValueError("tail_mode must be 'none' or 'guarded_asymmetric'")
    p = validate_dimension(p)
    lam = validate_positive_finite('lam', lam)
    ter_max = validate_positive_finite('ter_max', ter_max)
    if ter_max < 1:
        raise ValueError('ter_max must be at least 1')
    epsilon = validate_positive_finite('epsilon', epsilon)
    return p, lam, tail_mode, ter_max, epsilon


def stationary_bounds(X, *, lam, tail_mode, ter_max):
    p = X.shape[1]
    from ._state import TrackerState
    from ._controller import tracker_pass, BOOTSTRAP
    state = TrackerState(p)
    cursor = np.zeros(2, dtype=np.int64)
    # Bound uninterrupted native work, including on large stationary fits.
    for offset in range(0, len(X), 256):
        part = X[offset:offset + 256]
        cursor.fill(0)
        while tracker_pass(part, cursor, *state.arrays) == BOOTSTRAP:
            state.bootstrap(cursor[1], part[cursor[0], cursor[1]])
            cursor[1] += 1
    trackers, q01, q99 = state.mad, state.low, state.high
    location = np.array([tracker.location() for tracker in trackers])
    scale = np.array([tracker.scale() for tracker in trackers])
    lower_mult, upper_mult = np.ones(p), np.ones(p)
    if tail_mode == 'guarded_asymmetric':
        for j in range(p):
            low, high = q01[j].estimate(), q99[j].estimate()
            if np.isfinite(low) and np.isfinite(high):
                upper_mult[j], lower_mult[j], _ = compute_guarded_asymmetric_ter(
                    low, location[j], high, scale[j], ter_max)
    return inclusive_bounds(location, scale, lam, lower_mult, upper_mult)
