"""Strict Numba translations of MAD pair loops; see frozen reference/provenance.

Dense state, j-then-k traversal and binary rescaling are unchanged. No fastmath,
parallelism or reassociation. Mutation stays in place to preserve count aliases.
"""
import math
import numpy as np
from numba import njit

@njit(cache=True, fastmath=False)
def _centered_coordinate(value: float, origin: float, exponent: int) -> tuple[float, int]:
    """Center one finite observation, increasing its binary scale as needed."""
    difference = value - origin
    if math.isfinite(difference):
        if difference == 0.0:
            return 0.0, exponent
        _, required = math.frexp(difference)
        updated = max(exponent, required)
        return math.ldexp(difference, -updated), updated
    # Opposite finite endpoints may have an unrepresentable raw difference.
    updated = max(exponent, math.frexp(value)[1] + 1, math.frexp(origin)[1] + 1)
    return math.ldexp(value, -updated) - math.ldexp(origin, -updated), updated

@njit(cache=True, fastmath=False)
def _correlation_from_moments(cross: float, variance_j: float, variance_k: float) -> float:
    """Normalize finite moments through mantissas and binary exponents.

    This avoids both overflowing products and prematurely underflowing ratios
    when coordinate units differ. Zero variance uses the public undefined
    correlation convention; invalid arithmetic is reported explicitly.
    """
    cross, variance_j, variance_k = float(cross), float(variance_j), float(variance_k)
    if not (math.isfinite(cross) and math.isfinite(variance_j) and math.isfinite(variance_k)):
        raise FloatingPointError("Non-finite correlation moments; rescale the observations")
    if variance_j < 0.0 or variance_k < 0.0:
        raise FloatingPointError("Negative variance cannot define correlation")
    if variance_j == 0.0 or variance_k == 0.0 or cross == 0.0:
        return 0.0
    mantissa_j, exponent_j = math.frexp(variance_j)
    mantissa_k, exponent_k = math.frexp(variance_k)
    mantissa_cross, exponent_cross = math.frexp(cross)
    exponent = exponent_j + exponent_k
    denominator = math.sqrt(mantissa_j * mantissa_k * (2.0 if exponent % 2 else 1.0))
    value = math.ldexp(mantissa_cross / denominator, exponent_cross - exponent // 2)
    if not math.isfinite(value):
        raise FloatingPointError("Correlation normalization overflowed; check covariance moments")
    if abs(value) > 1.0 + 1e-12:
        raise FloatingPointError("Correlation moments violate the correlation interval")
    return max(-1.0, min(1.0, value))

@njit(cache=True, fastmath=False)
def update_pair(j, k, xj, xk, count, mean_j, mean_k, m2_j, m2_k, co_moment, _origin_j, _origin_k, _exponent_j, _exponent_k):
    n_old = int(count[j, k])
    if n_old >= 2**53:
        raise FloatingPointError("Pair count exceeds the exactly representable update range")
    if n_old == 0:
        _origin_j[j, k], _origin_k[j, k] = xj, xk
        count[j, k] = 1
        return
    centered_j, exponent_j = _centered_coordinate(
        xj, float(_origin_j[j, k]), int(_exponent_j[j, k])
    )
    centered_k, exponent_k = _centered_coordinate(
        xk, float(_origin_k[j, k]), int(_exponent_k[j, k])
    )
    shift_j = int(_exponent_j[j, k]) - exponent_j
    shift_k = int(_exponent_k[j, k]) - exponent_k
    previous_mean_j = math.ldexp(float(mean_j[j, k]), shift_j)
    previous_mean_k = math.ldexp(float(mean_k[j, k]), shift_k)
    previous_m2_j = math.ldexp(float(m2_j[j, k]), 2 * shift_j)
    previous_m2_k = math.ldexp(float(m2_k[j, k]), 2 * shift_k)
    previous_cross = math.ldexp(float(co_moment[j, k]), shift_j + shift_k)
    n_new = n_old + 1
    dx, dy = centered_j - previous_mean_j, centered_k - previous_mean_k
    new_mean_j = previous_mean_j + dx / n_new
    new_mean_k = previous_mean_k + dy / n_new
    new_m2_j = previous_m2_j + dx * (centered_j - new_mean_j)
    new_m2_k = previous_m2_k + dy * (centered_k - new_mean_k)
    cross = previous_cross + dx * (centered_k - new_mean_k)
    if not (math.isfinite(new_mean_j) and math.isfinite(new_mean_k) and math.isfinite(new_m2_j) and math.isfinite(new_m2_k) and math.isfinite(cross)):
        raise FloatingPointError("Pair moments exceeded the supported arithmetic range")
    if new_m2_j < 0.0 or new_m2_k < 0.0:
        raise FloatingPointError("Pair variance became negative through arithmetic error")
    if (new_m2_j == 0.0 and (previous_m2_j > 0.0 or centered_j != 0.0)) or (
        new_m2_k == 0.0 and (previous_m2_k > 0.0 or centered_k != 0.0)
    ):
        raise FloatingPointError("A positive pair variance was lost")
    _exponent_j[j, k], _exponent_k[j, k] = exponent_j, exponent_k
    mean_j[j, k], mean_k[j, k] = new_mean_j, new_mean_k
    m2_j[j, k], m2_k[j, k], co_moment[j, k] = new_m2_j, new_m2_k, cross
    count[j, k] = n_new

@njit(cache=True, fastmath=False)
def update(x, mask, count, mean_j, mean_k, m2_j, m2_k, co_moment, _origin_j, _origin_k, _exponent_j, _exponent_k):
    for j in range(count.shape[0]):
        if not mask[j]:
            continue
        for k in range(j + 1, count.shape[0]):
            if mask[k]:
                xj, xk = float(x[j]), float(x[k])
                if not math.isfinite(xj) or not math.isfinite(xk):
                    raise ValueError("Pair observations must be finite")
                update_pair(j, k, xj, xk, count, mean_j, mean_k, m2_j, m2_k, co_moment, _origin_j, _origin_k, _exponent_j, _exponent_k)

@njit(cache=True, fastmath=False)
def clear_dimension(j, count, mean_j, mean_k, m2_j, m2_k, co_moment, _origin_j, _origin_k, _exponent_j, _exponent_k):
    for k in range(count.shape[0]):
        if k == j:
            continue
        jj, kk = min(j, k), max(j, k)
        count[jj, kk] = 0
        mean_j[jj, kk] = 0.0
        mean_k[jj, kk] = 0.0
        m2_j[jj, kk] = 0.0
        m2_k[jj, kk] = 0.0
        co_moment[jj, kk] = 0.0
        _origin_j[jj, kk] = 0.0
        _origin_k[jj, kk] = 0.0
        _exponent_j[jj, kk] = -1074
        _exponent_k[jj, kk] = -1074

@njit(cache=True, fastmath=False)
def correlation(count, m2_j, m2_k, co_moment):
    p = count.shape[0]
    result = np.eye(p, dtype=np.float64)
    for j in range(p):
        for k in range(j + 1, p):
            if count[j, k] < 2:
                continue
            result[j, k] = result[k, j] = _correlation_from_moments(
                co_moment[j, k], m2_j[j, k], m2_k[j, k])
    return result
