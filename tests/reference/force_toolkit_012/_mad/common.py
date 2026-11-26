"""Shared numerical and validation helpers for correlation estimators."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from fractions import Fraction
import math
import sys

import numpy as np
from . import _kernels


def validate_dimension(p: int) -> int:
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)) or int(p) < 2:
        raise ValueError("p must be an integer greater than or equal to 2")
    return int(p)


def validate_positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def validate_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def validate_positive_int(name: str, value: int) -> int:
    result = validate_nonnegative_int(name, value)
    if result == 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def validate_vector(x: np.ndarray, p: int, *, name: str = "x") -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != p:
        raise ValueError(f"{name} must have shape ({p},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_matrix(
    X: np.ndarray,
    p: int,
    *,
    min_rows: int = 1,
    name: str = "X",
) -> np.ndarray:
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[1] != p:
        raise ValueError(f"{name} must have exactly {p} columns")
    if array.shape[0] < min_rows:
        raise ValueError(f"{name} must have at least {min_rows} rows")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _exact_bound_endpoint(location, scale, lam, multiplier, direction):
    """Rare exact-dyadic endpoint; saturate only the final out-of-range value."""
    endpoint = (Fraction.from_float(location) + direction * Fraction.from_float(lam)
                * Fraction.from_float(multiplier) * Fraction.from_float(scale))
    largest = Fraction.from_float(sys.float_info.max)
    if endpoint > largest:
        return math.inf
    if endpoint < -largest:
        return -math.inf
    return float(endpoint)


def _bound_endpoint(location, scale, lam, multiplier, direction):
    if scale == 0.0 or lam == 0.0 or multiplier == 0.0:
        return location
    first = lam * multiplier
    width = first * scale
    endpoint = location + direction * width
    if (not math.isfinite(first) or not math.isfinite(width) or not math.isfinite(endpoint)
            or abs(first) < sys.float_info.min or abs(width) < sys.float_info.min):
        return _exact_bound_endpoint(location, scale, lam, multiplier, direction)
    return endpoint


def inclusive_bounds(
    location: np.ndarray | float,
    scale: np.ndarray | float,
    lam: float,
    lower_multiplier: np.ndarray | float = 1.0,
    upper_multiplier: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive float64 endpoints with range-safe exceptional evaluation.

    Ordinary arithmetic retains its previous ordering. A zero scale collapses
    both bounds. If width intermediates overflow or lose range, exact arithmetic
    evaluates the final endpoint; an endpoint outside float64 saturates to its
    signed infinity. Representable endpoint rounding is allowed.
    """
    location_array = np.asarray(location, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    # Streaming calls are scalar. Avoid allocating several temporary arrays for
    # their range checks while retaining the vector/broadcasting stationary path.
    if (location_array.ndim == 0 and scale_array.ndim == 0
            and np.isscalar(lam) and np.isscalar(lower_multiplier) and np.isscalar(upper_multiplier)):
        loc, sigma, factor = float(location_array), float(scale_array), float(lam)
        lower_mult, upper_mult = float(lower_multiplier), float(upper_multiplier)
        if not math.isfinite(loc) or not math.isfinite(sigma):
            raise ValueError("location and scale must be finite")
        if sigma < 0.0:
            raise ValueError("scale must be non-negative")
        if (not math.isfinite(factor) or not math.isfinite(lower_mult)
                or not math.isfinite(upper_mult)):
            raise ValueError("bound factors must be finite")
        return (np.float64(_bound_endpoint(loc, sigma, factor, lower_mult, -1)),
                np.float64(_bound_endpoint(loc, sigma, factor, upper_mult, 1)))
    if not np.isfinite(location_array).all() or not np.isfinite(scale_array).all():
        raise ValueError("location and scale must be finite")
    if np.any(scale_array < 0.0):
        raise ValueError("scale must be non-negative")
    loc, sigma, factor, lower_mult, upper_mult = np.broadcast_arrays(
        location_array, scale_array, np.asarray(lam, dtype=np.float64),
        np.asarray(lower_multiplier, dtype=np.float64),
        np.asarray(upper_multiplier, dtype=np.float64))
    if (not np.isfinite(factor).all() or not np.isfinite(lower_mult).all()
            or not np.isfinite(upper_mult).all()):
        raise ValueError("bound factors must be finite")
    results = []
    for multiplier, direction in ((lower_mult, -1), (upper_mult, 1)):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            first = factor * multiplier
            width = first * sigma
            endpoint = loc + direction * width
        zero_width = (sigma == 0.0) | (factor == 0.0) | (multiplier == 0.0)
        endpoint = np.where(zero_width, loc, endpoint)
        exceptional = ~zero_width & (
            ~np.isfinite(first) | ~np.isfinite(width) | ~np.isfinite(endpoint)
            | (np.abs(first) < sys.float_info.min) | (np.abs(width) < sys.float_info.min))
        for index in np.flatnonzero(exceptional):
            endpoint.flat[index] = _exact_bound_endpoint(
                float(loc.flat[index]), float(sigma.flat[index]), float(factor.flat[index]),
                float(multiplier.flat[index]), direction)
        results.append(endpoint)
    return results[0], results[1]


class PairwiseMoments:
    """Stable pair-specific online covariance state.

    Each upper-triangular cell contains moments for the rows accepted by both
    coordinates in that pair. This is the sufficient-statistic architecture
    described by MAD-FORCE, implemented with Welford-style updates. Each pair
    uses its first accepted observation as origin and adaptive powers of two
    as coordinate scales. ``mean_j``, ``mean_k`` and the second moments refer
    to these centered, scaled coordinates; Pearson correlation is unchanged.
    ``epsilon`` is a compatibility argument, never a variance floor.
    """

    def __init__(self, p: int, epsilon: float = 1e-10):
        self.p = validate_dimension(p)
        self.epsilon = validate_positive_finite("epsilon", epsilon)
        shape = (self.p, self.p)
        self.count = np.zeros(shape, dtype=np.int64)
        self.mean_j = np.zeros(shape, dtype=np.float64)
        self.mean_k = np.zeros(shape, dtype=np.float64)
        self.m2_j = np.zeros(shape, dtype=np.float64)
        self.m2_k = np.zeros(shape, dtype=np.float64)
        self.co_moment = np.zeros(shape, dtype=np.float64)
        self._origin_j = np.zeros(shape, dtype=np.float64)
        self._origin_k = np.zeros(shape, dtype=np.float64)
        # The smallest nonzero float has frexp exponent -1073.
        self._exponent_j = np.full(shape, -1074, dtype=np.int64)
        self._exponent_k = np.full(shape, -1074, dtype=np.int64)
        self._failed = False

    def _ensure_valid(self) -> None:
        if self._failed:
            raise RuntimeError("Pair-moment arithmetic failed; recreate the estimator state")

    def update_pair(self, j: int, k: int, xj: float, xk: float) -> None:
        self._ensure_valid()
        xj, xk = float(xj), float(xk)
        if not math.isfinite(xj) or not math.isfinite(xk):
            raise ValueError("Pair observations must be finite")
        try:
            self._update_pair(j, k, xj, xk)
        except (FloatingPointError, OverflowError) as error:
            # An observation may already have updated another pair. Do not
            # allow a partially updated stream to produce a plausible result.
            self._failed = True
            raise FloatingPointError(str(error)) from error

    def _arrays(self):
        return (self.count, self.mean_j, self.mean_k, self.m2_j, self.m2_k, self.co_moment, self._origin_j, self._origin_k, self._exponent_j, self._exponent_k,)

    def _update_pair(self, j, k, xj, xk):
        _kernels.update_pair(j, k, xj, xk, *self._arrays())

    def update(self, x, accepted, ready=None):
        self._ensure_valid()
        mask = accepted if ready is None else accepted & ready
        try:
            _kernels.update(x, mask, *self._arrays())
        except (FloatingPointError, OverflowError) as error:
            self._failed = True
            raise FloatingPointError(str(error)) from error

    def clear_dimension(self, j):
        self._ensure_valid()
        _kernels.clear_dimension(j, *self._arrays())

    def correlation(self):
        self._ensure_valid()
        try:
            return _kernels.correlation(self.count, self.m2_j, self.m2_k, self.co_moment)
        except FloatingPointError:
            self._failed = True
            raise

    @property
    def state_bytes(self) -> int:
        arrays = (
            self.count,
            self.mean_j,
            self.mean_k,
            self.m2_j,
            self.m2_k,
            self.co_moment,
            self._origin_j,
            self._origin_k,
            self._exponent_j,
            self._exponent_k,
        )
        return int(sum(array.nbytes for array in arrays))


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


def _scaled_origin_differences(values: np.ndarray, *, origin: float | None = None) -> np.ndarray:
    """Translate and binary-scale a finite vector without changing Pearson.

    Centering at an observed value preserves small, representable differences
    around large offsets. The default origin is the first observation; robust
    callers can supply a central observation instead. Scaling before products
    protects very small units.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite one-dimensional array")
    origin = float(values[0]) if origin is None else float(origin)
    if not math.isfinite(origin):
        raise ValueError("The centering origin must be finite")
    with np.errstate(over="ignore"):
        differences = values - origin
    if np.isfinite(differences).all():
        maximum = float(np.max(np.abs(differences)))
        if maximum == 0.0:
            return differences
        return np.ldexp(differences, -math.frexp(maximum)[1])
    exponent = math.frexp(max(float(np.max(np.abs(values))), abs(origin)))[1] + 1
    scaled = np.ldexp(values, -exponent)
    return scaled - math.ldexp(origin, -exponent)


def _correlation_from_moments(cross: float, variance_j: float, variance_k: float) -> float:
    """Normalize finite moments through mantissas and binary exponents.

    This avoids both overflowing products and prematurely underflowing ratios
    when coordinate units differ. Zero variance uses the public undefined
    correlation convention; invalid arithmetic is reported explicitly.
    """
    cross, variance_j, variance_k = float(cross), float(variance_j), float(variance_k)
    if not all(math.isfinite(value) for value in (cross, variance_j, variance_k)):
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
    try:
        value = math.ldexp(mantissa_cross / denominator, exponent_cross - exponent // 2)
    except OverflowError as error:
        raise FloatingPointError("Correlation normalization overflowed; check covariance moments") from error
    if not math.isfinite(value):
        raise FloatingPointError("Correlation normalization overflowed; check covariance moments")
    if abs(value) > 1.0 + 1e-12:
        raise FloatingPointError("Correlation moments violate the correlation interval")
    return max(-1.0, min(1.0, value))


def trimmed_correlation(
    X: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    epsilon: float = 1e-10,
    return_diagnostics: bool = False,
) -> np.ndarray:
    """Pairwise Pearson correlation after inclusive coordinate-wise trimming.

    ``epsilon`` is retained for API compatibility, not used as a variance floor.
    """
    validate_positive_finite("epsilon", epsilon)
    X = np.asarray(X, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    p = X.shape[1]
    accepted = (X >= lower) & (X <= upper)
    result = np.eye(p, dtype=np.float64)
    counts = np.zeros((p, p), dtype=np.int64) if return_diagnostics else None
    codes = np.full((p, p), 1, dtype=np.int8) if return_diagnostics else None
    if return_diagnostics:
        np.fill_diagonal(codes, 3)  # unit diagonal is a presentation convention
        np.fill_diagonal(counts, accepted.sum(axis=0))
    for j in range(p):
        for k in range(j + 1, p):
            mask = accepted[:, j] & accepted[:, k]
            count = np.count_nonzero(mask)
            if return_diagnostics:
                counts[j, k] = counts[k, j] = count
            if count < 2:
                continue
            xj = _scaled_origin_differences(X[mask, j])
            xk = _scaled_origin_differences(X[mask, k])
            centered_j = xj - np.mean(xj)
            centered_k = xk - np.mean(xk)
            norm_j = float(np.dot(centered_j, centered_j))
            norm_k = float(np.dot(centered_k, centered_k))
            result[j, k] = result[k, j] = _correlation_from_moments(
                np.dot(centered_j, centered_k), norm_j, norm_k
            )
            if return_diagnostics:
                codes[j, k] = codes[k, j] = 0 if norm_j > 0 and norm_k > 0 else 2
    if return_diagnostics:
        return result, counts, codes
    return result


def covariance_to_correlation(covariance: np.ndarray, *, epsilon: float = 1e-10) -> np.ndarray:
    """Normalize covariance with zero undefined off-diagonals and unit diagonal.

    ``epsilon`` is a validated compatibility argument, not a variance floor.
    """
    validate_positive_finite("epsilon", epsilon)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must contain only finite values")
    variances = np.diag(covariance)
    result = np.eye(covariance.shape[0], dtype=np.float64)
    for j in range(covariance.shape[0]):
        for k in range(j + 1, covariance.shape[0]):
            left, right = covariance[j, k], covariance[k, j]
            # Avoid overflow in a+b and preserve identical subnormal values.
            cross = left if left == right else 0.5 * left + 0.5 * right
            result[j, k] = result[k, j] = _correlation_from_moments(
                cross, variances[j], variances[k]
            )
    return result


def explicit_state_bytes(values: Iterable[object]) -> int:
    """Count reachable owned state, including containers, once per identity.

    NumPy views count their headers and share their base allocation. This is
    an explicit object-state estimate, not allocator overhead or process RSS.
    Callers should pass owned state rather than external diagnostic sinks.
    """
    total = 0
    seen: set[int] = set()
    stack = list(values)
    while stack:
        value = stack.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(value, np.ndarray):
            total += sys.getsizeof(value)
            if value.base is not None:
                stack.append(value.base)
            if value.dtype.hasobject:
                stack.extend(value.flat)
        elif isinstance(value, dict):
            total += sys.getsizeof(value)
            stack.extend(value.keys())
            stack.extend(value.values())
            if hasattr(value, "__dict__"):
                stack.append(vars(value))
        elif isinstance(value, (list, tuple, set, frozenset, deque)):
            total += sys.getsizeof(value)
            stack.extend(value)
            if hasattr(value, "__dict__"):
                stack.append(vars(value))
        elif hasattr(value, "__dict__"):
            total += sys.getsizeof(value)
            stack.append(vars(value))
        else:
            total += sys.getsizeof(value)
    return int(total)
