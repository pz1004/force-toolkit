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
    trackers = [StreamingMAD() for _ in range(p)]
    q01 = [P2Estimator(.01) for _ in range(p)]
    q99 = [P2Estimator(.99) for _ in range(p)]
    for row in X:
        for j in range(p):
            value = float(row[j])
            trackers[j].update(value)
            q01[j].update(value)
            q99[j].update(value)
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
