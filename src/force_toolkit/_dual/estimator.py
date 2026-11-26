"""Dual-bound robust streaming correlation sketch."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from typing import Any

import numpy as np

from ._fd import FrequentDirections


from .._exceptions import NotFittedError


@dataclass
class _ServingState:
    sketch: FrequentDirections
    mean: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    parallel_radius: float
    residual_radius: float
    effective_n: int
    epoch_index: int


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _positive(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _median(values: np.ndarray) -> np.ndarray:
    """Axis-zero empirical median with the paper's even-sample convention.

    Evaluate the midpoint without overflowing when both central order
    statistics are large and have the same sign.
    """
    values = np.asarray(values, dtype=float)
    middle = len(values) // 2
    indices = (middle,) if len(values) % 2 else (middle - 1, middle)
    ordered = np.partition(values, indices, axis=0)
    if len(values) % 2:
        return ordered[middle].copy()
    lower, upper = ordered[middle - 1], ordered[middle]
    with np.errstate(over="ignore", invalid="ignore"):
        midpoint = (lower + upper) / 2.0
        fallback = lower + (upper - lower) / 2.0
    return np.where(np.isfinite(midpoint), midpoint, fallback)


def _finite_ulp(values: np.ndarray) -> np.ndarray:
    """Larger finite adjacent binary64 gap, including the largest float."""
    magnitude = np.abs(values)
    with np.errstate(over="ignore"):
        upward = np.nextafter(magnitude, np.inf) - magnitude
    downward = magnitude - np.nextafter(magnitude, -np.inf)
    return np.maximum(np.where(np.isfinite(upward), upward, 0.0), downward)


def _standardize(
    values: np.ndarray, location: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    # Infinities from the final quotient are safely handled by marginal
    # clipping. An overflowing subtraction may nevertheless have a finite
    # quotient, so evaluate that exceptional branch after division instead.
    try:
        # NumPy's underflow flag enters the exceptional range check without
        # another coordinate-mask scan on ordinary rows. The inputs here are
        # the validated float64 arrays owned by the calibration/transform path.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="raise"):
            difference = values - location
            standardized = difference / scale
            if np.any(np.isinf(difference)):
                alternative = values / scale - location / scale
                standardized = np.where(np.isinf(difference), alternative, standardized)
    except FloatingPointError:
        # Underflow also signals an inexact but nonzero subnormal. Re-evaluate
        # the same arithmetic, then reject only coordinates actually lost to
        # zero. Exact centered zeros and retained subnormals remain valid.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            difference = values - location
            standardized = difference / scale
            if np.any(np.isinf(difference)):
                alternative = values / scale - location / scale
                standardized = np.where(np.isinf(difference), alternative, standardized)
        if np.any((standardized == 0.0) & (values != location)):
            raise FloatingPointError("standardization rounded a nonzero coordinate to zero") from None
    if np.isnan(standardized).any():
        raise FloatingPointError("standardization produced an undefined value")
    return standardized


def _robust_upper_radius(values: np.ndarray, multiplier: float, floor: float) -> float:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError("calibration component norms are not finite")
    median = float(_median(values))
    mad = float(_median(np.abs(values - median)))
    radius = max(floor, median + multiplier * 1.4826 * mad)
    if not np.isfinite(radius):
        raise FloatingPointError("calibration radius is not representable")
    return float(radius)


def _column_normalized_factor(buffer: np.ndarray) -> np.ndarray:
    """Paper Equation (3), with exact zero columns retained as zero.

    Scale each column before computing its norm to avoid squared-norm
    overflow or underflow. This needs only O(pk) workspace and never forms
    the p-by-p scatter or introduces an estimation-count scale factor.
    """
    if not np.isfinite(buffer).all():
        raise FloatingPointError("sketch contains non-finite values")
    largest = np.max(np.abs(buffer), axis=0)
    positive = largest > 0.0
    factor = np.zeros_like(buffer)
    factor[:, positive] = buffer[:, positive] / largest[positive]
    factor[:, positive] /= np.linalg.norm(factor[:, positive], axis=0)
    return factor


def _exact_scatter_entries(buffer, scale, denominator, pairs, result):
    """Evaluate selected entries as exact rational expressions of binary64 state.

    This exceptional path is portable: it does not rely on ``longdouble``
    having a wider exponent range than binary64. Scale factors are applied
    only after the exact dot product, and division precedes float conversion.
    """
    columns = [
        [Fraction.from_float(float(value)) for value in buffer[:, column]]
        for column in range(buffer.shape[1])
    ]
    scales = [Fraction.from_float(float(value)) for value in scale]
    for left, right in pairs:
        dot = sum(
            (a * b for a, b in zip(columns[left], columns[right])),
            Fraction(0),
        )
        exact = dot * scales[left] * scales[right] / denominator
        try:
            value = float(exact)
        except OverflowError as error:
            raise FloatingPointError(
                "scale scatter exceeds the finite float64 range"
            ) from error
        if not np.isfinite(value):
            raise FloatingPointError("scale scatter exceeds the finite float64 range")
        if value == 0.0 and exact:
            raise FloatingPointError(
                "a nonzero scale-scatter entry underflows the float64 range"
            )
        result[left, right] = result[right, left] = value
    return result


def _scale_reconstructed_gram(buffer, scale, denominator):
    """Return ``diag(scale) @ buffer.T @ buffer @ diag(scale) / n`` safely.

    Ordinary magnitudes retain a float64 matrix-product fast path. Conservative
    exponent bounds keep all its intermediates normal and finite; cross terms
    near cancellation are resolved exactly. Extreme magnitudes use exact
    rational products and sums followed by one binary64 conversion per entry.

    Rounding to a representable binary64 value is permitted. A final overflow,
    or a mathematically nonzero entry rounding to zero, raises explicitly. The
    input arrays and estimator state are never modified by this readout.
    """
    buffer = np.asarray(buffer, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if buffer.ndim != 2 or scale.shape != (buffer.shape[1],):
        raise ValueError("scale-scatter inputs have incompatible shapes")
    if not np.isfinite(buffer).all() or not np.isfinite(scale).all():
        raise FloatingPointError("scale-scatter inputs must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("scale-scatter scales must be positive")
    denominator = _integer("denominator", denominator, minimum=1)
    p = buffer.shape[1]
    nonzero = np.abs(buffer[buffer != 0.0])
    if not nonzero.size:
        return np.zeros((p, p), dtype=np.float64)

    # These deliberately conservative bounds also keep a nonzero exact dot
    # product (including cancellation of binary64 operands) above underflow
    # after the bounded scales and denominator are applied.
    ordinary = (
        denominator <= 2**53
        and buffer.shape[0] <= 2**31
        and nonzero.min() >= 2.0**-128
        and nonzero.max() <= 2.0**128
        and scale.min() >= 2.0**-128
        and scale.max() <= 2.0**128
    )
    if not ordinary:
        pairs = ((left, right) for left in range(p) for right in range(left, p))
        return _exact_scatter_entries(
            buffer, scale, denominator, pairs, np.empty((p, p), dtype=np.float64)
        )

    gram = buffer.T @ buffer / denominator
    result = scale[:, None] * gram * scale[None, :]
    norms = np.sqrt(np.diag(result))
    # Cauchy-Schwarz bounds the sum of absolute dot-product terms. The factor
    # covers accumulation plus division/scaling roundoff with a wide margin.
    error_bound = (8.0 * (len(buffer) + 3) * np.finfo(float).eps) * (
        norms[:, None] * norms[None, :]
    )
    active = norms > 0.0
    uncertain = (
        (np.abs(result) <= error_bound)
        & active[:, None]
        & active[None, :]
    )
    left, right = np.triu_indices(p, 1)
    result[right, left] = result[left, right]
    pairs = list(zip(left[uncertain[left, right]], right[uncertain[left, right]]))
    if pairs:
        _exact_scatter_entries(buffer, scale, denominator, pairs, result)
    return result


class DualBoundFORCE:
    """MAD-calibrated FD estimator with conditional row-contribution bounds.

    Calibration observations set parameters and are excluded from the
    estimation denominator.  With scheduled epochs, the preceding completed
    estimate remains serviceable during the next calibration block.  Epoch
    diagnostics retain only the newest 64 completed epochs so that optional
    scheduling cannot create stream-length-dependent estimator state.
    """

    def __init__(
        self,
        p: int,
        k: int,
        *,
        calibration_size: int = 512,
        epoch_size: int | None = None,
        marginal_lambda: float = 3.0,
        parallel_lambda: float = 3.0,
        residual_lambda: float = 3.0,
        scale_floor: float = 1e-8,
        epsilon: float = 1e-10,
    ):
        self.p = _integer("p", p, minimum=1)
        self.k = _integer("k", k, minimum=1)
        if self.k > self.p:
            raise ValueError("k must not exceed p")
        self.calibration_size = _integer(
            "calibration_size", calibration_size, minimum=5
        )
        if epoch_size is not None:
            epoch_size = _integer("epoch_size", epoch_size, minimum=1)
            if epoch_size <= self.calibration_size:
                raise ValueError("epoch_size must exceed calibration_size")
        self.epoch_size = epoch_size
        self.marginal_lambda = _positive("marginal_lambda", marginal_lambda)
        self.parallel_lambda = _positive("parallel_lambda", parallel_lambda)
        self.residual_lambda = _positive("residual_lambda", residual_lambda)
        self.scale_floor = _positive("scale_floor", scale_floor)
        self.epsilon = _positive("epsilon", epsilon)
        # Section 3's numerical sensitivity analysis varies the radial
        # regularizer while holding singular-value rank selection fixed.
        self._rank_tolerance = 1e-10
        self.reset()

    def reset(self) -> "DualBoundFORCE":
        self.total_seen = 0
        self._attempted_total = 0
        self._committed_total = 0
        self.epoch_index = 0
        self.epoch_seen = 0
        self.phase = "calibrating"
        self.calibration_count = 0
        self._calibration: np.ndarray | None = np.empty(
            (self.calibration_size, self.p), dtype=float
        )
        self.location: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.basis: np.ndarray | None = None
        self.parallel_radius: float | None = None
        self.residual_radius: float | None = None
        self._sketch: FrequentDirections | None = None
        self._mean = np.zeros(self.p, dtype=float)
        self.effective_n = 0
        self._last_completed: _ServingState | None = None
        self.marginal_clipped_coordinates = 0
        self.parallel_clipped_rows = 0
        self.residual_clipped_rows = 0
        self.epoch_history: list[dict[str, Any]] = []
        self._epoch_history_limit = 64
        # Conservative simultaneous-array envelope for calibration: the
        # resident raw block, a working copy, standardized/parallel/residual
        # workspaces and one transient C-by-p expression (6Cp); the 2k-by-p
        # preliminary sketch plus p-by-k basis (3kp); the C-by-k projection;
        # and coordinate/norm vectors.  A scheduled recalibration can also
        # retain the preceding serving sketch, basis, mean, location, and
        # scale (an additional 3kp + 3p).  This is an explicit numerical-state
        # bound, not process RSS; stationary initial calibration is cheaper.
        self.calibration_peak_state_bytes_bound = int(
            8
            * (
                6 * self.calibration_size * self.p
                + 6 * self.k * self.p
                + self.calibration_size * self.k
                + 7 * self.p
                + 2 * self.calibration_size
            )
        )
        return self

    def _validate_row(self, row: np.ndarray) -> np.ndarray:
        value = np.asarray(row, dtype=float)
        if value.shape != (self.p,):
            raise ValueError(f"row must have shape ({self.p},)")
        if not np.isfinite(value).all():
            raise ValueError("row contains non-finite values")
        return value

    def _calibrate(self) -> None:
        assert self._calibration is not None
        values = self._calibration.copy()
        location = _median(values)
        with np.errstate(over="ignore", invalid="ignore"):
            raw_mad = _median(np.abs(values - location))
            scale = 1.4826 * raw_mad
        scale = np.maximum(
            scale,
            np.maximum(self.scale_floor, _finite_ulp(location)),
        )
        if not np.isfinite(location).all() or not np.isfinite(scale).all():
            raise FloatingPointError(
                "calibration location or scale is not representable"
            )
        standardized = np.clip(
            _standardize(values, location, scale),
            -self.marginal_lambda,
            self.marginal_lambda,
        )
        preliminary = FrequentDirections(self.p, self.k, epsilon=self._rank_tolerance)
        for row in standardized:
            preliminary.update(row)
        basis = preliminary.basis(self.k)
        if basis.shape[1]:
            parallel = standardized @ basis @ basis.T
        else:
            parallel = np.zeros_like(standardized)
        residual = standardized - parallel
        parallel_norms = np.linalg.norm(parallel, axis=1)
        residual_norms = np.linalg.norm(residual, axis=1)

        self.location = location
        self.scale = scale
        self.basis = basis
        self.parallel_radius = _robust_upper_radius(
            parallel_norms, self.parallel_lambda, self.epsilon
        )
        self.residual_radius = _robust_upper_radius(
            residual_norms, self.residual_lambda, self.epsilon
        )
        self._sketch = FrequentDirections(self.p, self.k, epsilon=self._rank_tolerance)
        self._mean.fill(0.0)
        self.effective_n = 0
        self._calibration = None
        self.phase = "estimating"

    def _transform_with(
        self,
        row: np.ndarray,
        *,
        location: np.ndarray,
        scale: np.ndarray,
        basis: np.ndarray,
        parallel_radius: float,
        residual_radius: float,
        count_diagnostics: bool,
    ) -> np.ndarray:
        standardized_raw = _standardize(row, location, scale)
        standardized = np.clip(
            standardized_raw, -self.marginal_lambda, self.marginal_lambda
        )
        if count_diagnostics:
            self.marginal_clipped_coordinates += int(
                np.count_nonzero(standardized != standardized_raw)
            )
        if basis.shape[1]:
            parallel = basis @ (basis.T @ standardized)
        else:
            parallel = np.zeros(self.p, dtype=float)
        residual = standardized - parallel
        parallel_norm = float(np.linalg.norm(parallel))
        residual_norm = float(np.linalg.norm(residual))
        # Apply the same continuous epsilon-regularized radial map at every
        # norm.  A former ``norm <= epsilon`` identity shortcut was harmless
        # for the fitted nondegenerate study radii, but made the map
        # discontinuous when a radius attained its numerical floor and thus
        # prevented the global nonexpansiveness result used in the calibration
        # stability lemma.
        parallel_factor = min(1.0, parallel_radius / (parallel_norm + self.epsilon))
        residual_factor = min(1.0, residual_radius / (residual_norm + self.epsilon))
        if count_diagnostics:
            self.parallel_clipped_rows += int(
                parallel_norm > 0.0 and parallel_factor < 1.0
            )
            self.residual_clipped_rows += int(
                residual_norm > 0.0 and residual_factor < 1.0
            )
        return parallel_factor * parallel + residual_factor * residual

    def _current_transform(self, row: np.ndarray, *, diagnostics: bool) -> np.ndarray:
        if self.phase == "estimating" and self.location is not None:
            assert self.scale is not None and self.basis is not None
            assert self.parallel_radius is not None and self.residual_radius is not None
            return self._transform_with(
                row,
                location=self.location,
                scale=self.scale,
                basis=self.basis,
                parallel_radius=self.parallel_radius,
                residual_radius=self.residual_radius,
                count_diagnostics=diagnostics,
            )
        if self._last_completed is not None:
            state = self._last_completed
            return self._transform_with(
                row,
                location=state.location,
                scale=state.scale,
                basis=state.basis,
                parallel_radius=state.parallel_radius,
                residual_radius=state.residual_radius,
                count_diagnostics=False,
            )
        raise NotFittedError("initial calibration has not completed")

    def transform(self, row: np.ndarray) -> np.ndarray:
        """Transform a query with the parameters of the estimate being served."""
        value = self._validate_row(row)
        # During scheduled recalibration, keep query preprocessing paired with
        # the estimate still being served until the new epoch has one
        # estimation observation. Initial calibration has no prior serving
        # state, so its fitted map is usable immediately.
        if self._last_completed is not None and self.effective_n == 0:
            state = self._last_completed
            return self._transform_with(
                value,
                location=state.location,
                scale=state.scale,
                basis=state.basis,
                parallel_radius=state.parallel_radius,
                residual_radius=state.residual_radius,
                count_diagnostics=False,
            )
        if self.phase == "estimating" and self.location is not None:
            assert self.scale is not None and self.basis is not None
            assert self.parallel_radius is not None and self.residual_radius is not None
            return self._transform_with(
                value,
                location=self.location,
                scale=self.scale,
                basis=self.basis,
                parallel_radius=self.parallel_radius,
                residual_radius=self.residual_radius,
                count_diagnostics=False,
            )
        if self._last_completed is not None:
            state = self._last_completed
            return self._transform_with(
                value,
                location=state.location,
                scale=state.scale,
                basis=state.basis,
                parallel_radius=state.parallel_radius,
                residual_radius=state.residual_radius,
                count_diagnostics=False,
            )
        raise NotFittedError("initial calibration has not completed")

    def _snapshot_current(self) -> None:
        if self._sketch is None or self.effective_n <= 0:
            return
        assert self.location is not None and self.scale is not None
        assert self.basis is not None
        assert self.parallel_radius is not None and self.residual_radius is not None
        self._last_completed = _ServingState(
            sketch=self._sketch,
            mean=self._mean.copy(),
            location=self.location.copy(),
            scale=self.scale.copy(),
            basis=self.basis.copy(),
            parallel_radius=float(self.parallel_radius),
            residual_radius=float(self.residual_radius),
            effective_n=int(self.effective_n),
            epoch_index=int(self.epoch_index),
        )
        self.epoch_history.append(
            {
                "epoch_index": int(self.epoch_index),
                "effective_n": int(self.effective_n),
                "parallel_radius": float(self.parallel_radius),
                "residual_radius": float(self.residual_radius),
            }
        )
        if len(self.epoch_history) > self._epoch_history_limit:
            del self.epoch_history[: -self._epoch_history_limit]

    def _start_next_epoch(self) -> None:
        self._snapshot_current()
        self.epoch_index += 1
        self.epoch_seen = 0
        self.phase = "calibrating"
        self.calibration_count = 0
        self._calibration = np.empty((self.calibration_size, self.p), dtype=float)
        self.location = None
        self.scale = None
        self.basis = None
        self.parallel_radius = None
        self.residual_radius = None
        self._sketch = None
        self._mean.fill(0.0)
        self.effective_n = 0

    def update(self, row: np.ndarray) -> "DualBoundFORCE":
        return self._update_trusted(self._validate_row(row))

    def _update_trusted(self, value):
        """Consume an already validated finite float64 row, preserving schedule."""
        self._attempted_total += 1
        if self.epoch_size is not None and self.epoch_seen == self.epoch_size:
            self._start_next_epoch()

        self.total_seen += 1
        self.epoch_seen += 1
        if self.phase == "calibrating":
            assert self._calibration is not None
            self._calibration[self.calibration_count] = value
            self.calibration_count += 1
            if self.calibration_count == self.calibration_size:
                self._calibrate()
            self._committed_total += 1
            return self

        transformed = self._current_transform(value, diagnostics=True)
        if self.effective_n == 0:
            self._last_completed = None
        self.effective_n += 1
        delta = transformed - self._mean
        self._mean += delta / self.effective_n
        assert self._sketch is not None
        # The target is the calibration-centered transformed scatter
        # E[z z^T], not a second re-centered covariance.  Direct insertion is
        # also what makes the replacement term at most epsilon_row M^2, where
        # M^2 is the smaller simultaneous marginal/radial row envelope.
        self._sketch.update(transformed)
        self._committed_total += 1
        return self

    def update_batch(self, matrix):
        from .._validation import matrix as validate_matrix
        values = validate_matrix(matrix, self.p)
        return self._update_batch_trusted(values)

    def _update_batch_trusted(self, values):
        """Consume a fully validated chunk; retain per-row commit boundaries."""
        offset = 0
        while offset < len(values):
            if self.epoch_size is not None and self.epoch_seen == self.epoch_size:
                try:
                    self._start_next_epoch()
                except BaseException:
                    self._attempted_total += 1
                    raise
            if self.phase == "calibrating" and self.calibration_count < self.calibration_size:
                size = min(len(values) - offset, self.calibration_size - self.calibration_count)
                assert self._calibration is not None
                self._calibration[self.calibration_count:self.calibration_count + size] = values[offset:offset + size]
                self._attempted_total += size
                self.calibration_count += size
                self.total_seen += size
                self.epoch_seen += size
                offset += size
                if self.calibration_count == self.calibration_size:
                    # The terminal row commits only after calibration succeeds.
                    self._committed_total += size - 1
                    self._calibrate()
                    self._committed_total += 1
                else:
                    self._committed_total += size
            else:
                # Row order and the FD compression points stay unchanged.
                self._update_trusted(values[offset])
                offset += 1
        return self

    def fit(self, matrix: np.ndarray) -> "DualBoundFORCE":
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.p:
            raise ValueError(f"matrix must have shape (n, {self.p})")
        if not np.isfinite(values).all():
            raise ValueError("matrix contains non-finite values")
        for row in values:
            self.update(row)
        return self

    def _serving(self) -> _ServingState:
        if (
            self.phase == "estimating"
            and self._sketch is not None
            and self.effective_n > 0
        ):
            assert self.location is not None and self.scale is not None
            assert self.basis is not None
            assert self.parallel_radius is not None and self.residual_radius is not None
            return _ServingState(
                self._sketch,
                self._mean,
                self.location,
                self.scale,
                self.basis,
                self.parallel_radius,
                self.residual_radius,
                self.effective_n,
                self.epoch_index,
            )
        if self._last_completed is not None:
            return self._last_completed
        raise NotFittedError("no estimation observations are available")

    def _gram(self) -> tuple[np.ndarray, _ServingState]:
        state = self._serving()
        return state.sketch.gram(state.effective_n), state

    def get_correlation(self) -> np.ndarray:
        """Return the normalized transformed second moment (paper Eq. 3).

        Zero sketch columns remain zero. The diagonal on nonzero columns
        equals one up to roundoff; no entry clipping or diagonal reset is
        applied. Transformed observations are not mean-recentered.
        """
        factor = _column_normalized_factor(self._serving().sketch.B)
        gram = factor.T @ factor
        return (gram + gram.T) / 2.0

    def get_scale_scatter(self) -> np.ndarray:
        """Return scale-reconstructed robust scatter, not classical covariance.

        Extreme magnitudes are evaluated without premature product underflow
        or overflow. A nonzero final entry that rounds to zero, or a final
        overflow, raises ``FloatingPointError`` without changing the stream.
        """
        state = self._serving()
        return _scale_reconstructed_gram(
            state.sketch.B, state.scale, state.effective_n
        )

    def get_subspace(self, rank: int | None = None) -> np.ndarray:
        """Return leading directions of the served transformed correlation."""
        state = self._serving()
        requested = self.k if rank is None else _integer("rank", rank, minimum=1)
        if requested > self.k:
            raise ValueError("rank must not exceed sketch rank k")
        scaled_sketch = _column_normalized_factor(state.sketch.B)
        if not np.any(scaled_sketch):
            return np.zeros((self.p, 0), dtype=float)
        _, singular_values, right_vectors = np.linalg.svd(
            scaled_sketch, full_matrices=False
        )
        used = min(requested, int(np.sum(singular_values > self._rank_tolerance)))
        return right_vectors[:used].T.copy()

    def finalize(self) -> np.ndarray:
        return self.get_correlation()

    @property
    def state_bytes(self) -> int:
        arrays = [self._mean]
        if self._calibration is not None:
            arrays.append(self._calibration)
        for value in (self.location, self.scale, self.basis):
            if value is not None:
                arrays.append(value)
        if self._sketch is not None:
            arrays.append(self._sketch.B)
        if self._last_completed is not None:
            arrays.extend(
                [
                    self._last_completed.sketch.B,
                    self._last_completed.mean,
                    self._last_completed.location,
                    self._last_completed.scale,
                    self._last_completed.basis,
                ]
            )
        return int(sum(value.nbytes for value in arrays))

    def get_diagnostics(self) -> dict[str, Any]:
        serving_epoch = None
        serving_n = 0
        try:
            serving = self._serving()
            serving_epoch = int(serving.epoch_index)
            serving_n = int(serving.effective_n)
        except NotFittedError:
            pass
        return {
            "phase": self.phase,
            "ready": serving_epoch is not None,
            "total_seen": int(self.total_seen),
            "epoch_index": int(self.epoch_index),
            "epoch_seen": int(self.epoch_seen),
            "calibration_count": int(self.calibration_count),
            "effective_n": int(self.effective_n),
            "serving_epoch": serving_epoch,
            "serving_n": serving_n,
            "basis_rank": 0 if self.basis is None else int(self.basis.shape[1]),
            "numerical_rank_tolerance": self._rank_tolerance,
            "normalization": "column_normalized_factor_gram",
            "parallel_radius": self.parallel_radius,
            "residual_radius": self.residual_radius,
            "marginal_clipped_coordinates": int(self.marginal_clipped_coordinates),
            "parallel_clipped_rows": int(self.parallel_clipped_rows),
            "residual_clipped_rows": int(self.residual_clipped_rows),
            "state_bytes": self.state_bytes,
            "calibration_peak_state_bytes_bound": int(
                self.calibration_peak_state_bytes_bound
            ),
            "epoch_history": list(self.epoch_history),
        }
