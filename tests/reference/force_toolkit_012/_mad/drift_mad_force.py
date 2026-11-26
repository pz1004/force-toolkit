"""Paper-faithful MAD-FORCE with optional bounded-memory drift handling."""

from __future__ import annotations

from collections import deque
import sys
import time

import numpy as np

from .adwin import ADWINDetector
from .common import (
    PairwiseMoments,
    explicit_state_bytes,
    inclusive_bounds,
    trimmed_correlation,
    validate_dimension,
    validate_matrix,
    validate_nonnegative_int,
    validate_positive_finite,
    validate_positive_int,
    validate_vector,
)
from .diagnostics import (
    BoundedHistory, diagnostic_points, diagnostic_transaction, queue_diagnostic,
    validate_diagnostics,
)
from .p2 import P2Estimator
from .streaming_mad import StreamingMAD
from .ter import compute_guarded_asymmetric_ter


class DriftMADForce:
    """Streaming MAD-FORCE estimator implementing Equations (1)-(9).

    The default ``tail_mode="none"`` is the symmetric MAD-FORCE method used
    in the paper. Drift detection is optional; ``fit_stationary`` always uses
    the paper's two-pass stationary helper.
    """

    def __init__(
        self,
        p: int,
        lam: float = 3.0,
        adwin_delta: float = 0.002,
        detector_delta: float | None = None,
        detector_max_buckets: int = 5,
        detector_min_window: int = 32,
        detector_min_subwindow: int = 8,
        ter_max: float = 3.0,
        use_asymmetric_ter=None,
        use_drift_detection: bool = True,
        n_initial_warm: int = 200,
        tail_mode: str = "none",
        reset_buffer_size: int = 32,
        min_reset_obs: int = 32,
        acceptance_trace_eval_points: list[int] | None = None,
        diagnostic_change_points: list[int] | None = None,
        detector_max_window: int | None = None,
        epsilon: float = 1e-10,
        diagnostic_history_size: int = 128,
        diagnostic_sink=None,
    ):
        if use_asymmetric_ter is not None:
            if not isinstance(use_asymmetric_ter, (bool, np.bool_)):
                raise ValueError("use_asymmetric_ter must be boolean or None")
            tail_mode = "guarded_asymmetric" if use_asymmetric_ter else "none"
        if tail_mode not in {"none", "guarded_asymmetric"}:
            raise ValueError("tail_mode must be 'none' or 'guarded_asymmetric'")
        if not isinstance(use_drift_detection, (bool, np.bool_)):
            raise ValueError("use_drift_detection must be boolean")

        self.p = validate_dimension(p)
        self.lam = validate_positive_finite("lam", lam)
        resolved_delta = adwin_delta if detector_delta is None else detector_delta
        self.adwin_delta = float(resolved_delta)
        if not np.isfinite(self.adwin_delta) or not 0.0 < self.adwin_delta < 1.0:
            raise ValueError("detector delta must be finite and strictly between 0 and 1")
        self.detector_max_window = (
            None if detector_max_window is None else validate_positive_int(
                "detector_max_window", detector_max_window
            )
        )
        self.detector_max_buckets = validate_positive_int(
            "detector_max_buckets", detector_max_buckets
        )
        self.detector_min_window = validate_positive_int(
            "detector_min_window", detector_min_window
        )
        self.detector_min_subwindow = validate_positive_int(
            "detector_min_subwindow", detector_min_subwindow
        )
        if self.detector_max_buckets < 2:
            raise ValueError("detector_max_buckets must be at least 2")
        if self.detector_min_window < 2 * self.detector_min_subwindow:
            raise ValueError("detector_min_window must be at least twice detector_min_subwindow")
        if self.detector_max_window is not None and self.detector_max_window < self.detector_min_window:
            raise ValueError("detector_max_window must be at least detector_min_window")
        self.ter_max = validate_positive_finite("ter_max", ter_max)
        if self.ter_max < 1.0:
            raise ValueError("ter_max must be at least 1")
        self.use_drift_detection = bool(use_drift_detection)
        self.n_initial_warm = validate_nonnegative_int("n_initial_warm", n_initial_warm)
        self.tail_mode = tail_mode
        self.reset_buffer_size = validate_positive_int(
            "reset_buffer_size", reset_buffer_size
        )
        if self.reset_buffer_size < 5:
            raise ValueError("reset_buffer_size must be at least 5")
        self.min_reset_obs = validate_nonnegative_int("min_reset_obs", min_reset_obs)
        self.epsilon = validate_positive_finite("epsilon", epsilon)
        self.diagnostic_history_size, self.diagnostic_sink = validate_diagnostics(
            diagnostic_history_size, diagnostic_sink
        )
        self.acceptance_trace_eval_points = diagnostic_points(
            "acceptance_trace_eval_points", acceptance_trace_eval_points,
            limit=self.diagnostic_history_size,
        )
        self.acceptance_trace_eval_points_set = set(self.acceptance_trace_eval_points)
        self.diagnostic_change_points = diagnostic_points(
            "diagnostic_change_points", diagnostic_change_points,
            limit=self.diagnostic_history_size,
        )
        self._init_state()

    def _init_state(self) -> None:
        with diagnostic_transaction(self):
            self._initialize_state()

    def _initialize_state(self) -> None:
        self.streaming_mad = [StreamingMAD() for _ in range(self.p)]
        self.p2_q01 = [P2Estimator(0.01) for _ in range(self.p)]
        self.p2_q99 = [P2Estimator(0.99) for _ in range(self.p)]
        self.adwin = [
            ADWINDetector(
                delta=self.adwin_delta,
                max_window=self.detector_max_window,
                max_buckets=self.detector_max_buckets,
                min_window=self.detector_min_window,
                min_subwindow=self.detector_min_subwindow,
                diagnostic_history_size=self.diagnostic_history_size,
                # Always forward to the parent: its current sink may be
                # attached, replaced, removed or re-enabled after construction.
                diagnostic_sink=(
                    lambda event, dimension=j: queue_diagnostic(
                        self, {**event, "dimension": dimension}
                    )
                ),
            )
            for j in range(self.p)
        ]
        self.recent_values = [
            deque(maxlen=self.reset_buffer_size) for _ in range(self.p)
        ]
        # The post-reset gate applies only after a selective reset.
        self.post_reset_obs = np.full(self.p, self.min_reset_obs, dtype=np.int64)
        self.acceptance_total = np.zeros(self.p, dtype=np.int64)
        self.acceptance_sum = np.zeros(self.p, dtype=np.int64)
        self.reset_counts = np.zeros(self.p, dtype=np.int64)
        self.alarm_locations = BoundedHistory(self.diagnostic_history_size)
        self.component_timing = {
            "mad_update_s": 0.0,
            "acceptance_check_s": 0.0,
            "detector_update_s": 0.0,
            "pair_update_s": 0.0,
        }
        self.acceptance_trace = BoundedHistory(self.diagnostic_history_size)
        self._trace_acceptance_sum = np.zeros(self.p, dtype=np.float64)
        self._trace_acceptance_count = 0
        self._pre_change_acceptance_sum = np.zeros(self.p, dtype=np.float64)
        self._pre_change_acceptance_count = 0
        self._post_change_acceptance_sum = np.zeros(self.p, dtype=np.float64)
        self._post_change_acceptance_count = 0
        self._moments = PairwiseMoments(self.p, self.epsilon)
        # Backward-compatible read-only-by-convention alias used by old tests.
        self.n_jk = self._moments.count
        self._obs_count = 0
        queue_diagnostic(self, {"kind": "stream_reset"})

    def _tail_multipliers(
        self, q01: float, location: float, q99: float, scale: float
    ) -> tuple[float, float]:
        if self.tail_mode == "none":
            return 1.0, 1.0
        if not np.isfinite(q01) or not np.isfinite(q99):
            return 1.0, 1.0
        upper, lower, _ = compute_guarded_asymmetric_ter(
            q01, location, q99, scale, self.ter_max
        )
        return float(upper), float(lower)

    def _tracker_ready(self, j: int) -> bool:
        if not self.streaming_mad[j].initialized:
            return False
        if self.tail_mode == "guarded_asymmetric":
            return self.p2_q01[j].initialized and self.p2_q99[j].initialized
        return True

    def _dimension_ready(self, j: int) -> bool:
        return self._tracker_ready(j) and self.post_reset_obs[j] >= self.min_reset_obs

    def _record_acceptance_diagnostics(self, accepted: np.ndarray) -> None:
        if self.diagnostic_sink is not None:
            queue_diagnostic(self, {
                "kind": "acceptance", "t": self._obs_count,
                "accepted": accepted.tolist(),
            })
        accepted_float = accepted.astype(np.float64)
        if self.diagnostic_change_points:
            if self._obs_count <= self.diagnostic_change_points[0]:
                self._pre_change_acceptance_sum += accepted_float
                self._pre_change_acceptance_count += 1
            else:
                self._post_change_acceptance_sum += accepted_float
                self._post_change_acceptance_count += 1
        if self.acceptance_trace_eval_points_set:
            self._trace_acceptance_sum += accepted_float
            self._trace_acceptance_count += 1
            if self._obs_count in self.acceptance_trace_eval_points_set:
                per_dimension = (
                    self._trace_acceptance_sum / max(self._trace_acceptance_count, 1)
                )
                self.acceptance_trace.append(
                    {
                        "t": int(self._obs_count),
                        "mean_acceptance": float(np.mean(per_dimension)),
                        "per_dimension_acceptance": per_dimension.tolist(),
                    }
                )
                self._trace_acceptance_sum.fill(0.0)
                self._trace_acceptance_count = 0

    def update(self, x: np.ndarray) -> None:
        """Process one finite observation, then deliver diagnostic events.

        A ``DiagnosticDeliveryError`` means the observation is fully committed
        and must not be retried. Calls from a diagnostic callback are rejected.
        """
        with diagnostic_transaction(self):
            self._update(x)

    def _update(self, x: np.ndarray) -> None:
        x = validate_vector(x, self.p)
        self._obs_count += 1
        for j in range(self.p):
            self.recent_values[j].append(float(x[j]))

        accepted = np.ones(self.p, dtype=bool)
        reset_this_step = np.zeros(self.p, dtype=bool)
        for j in range(self.p):
            xj = float(x[j])
            started = time.perf_counter()
            self.streaming_mad[j].update(xj)
            self.p2_q01[j].update(xj)
            self.p2_q99[j].update(xj)
            self.component_timing["mad_update_s"] += time.perf_counter() - started

            started = time.perf_counter()
            if self.streaming_mad[j].initialized:
                location = self.streaming_mad[j].location()
                scale = self.streaming_mad[j].scale()
                upper_mult, lower_mult = self._tail_multipliers(
                    self.p2_q01[j].estimate(),
                    location,
                    self.p2_q99[j].estimate(),
                    scale,
                )
                lower, upper = inclusive_bounds(
                    location, scale, self.lam, lower_mult, upper_mult
                )
                accepted[j] = bool(lower <= xj <= upper)
            self.component_timing["acceptance_check_s"] += time.perf_counter() - started

            self.acceptance_total[j] += 1
            self.acceptance_sum[j] += int(accepted[j])
            # Count this fresh observation before checking the post-reset gate.
            # Replayed buffer values and the alarm observation do not count.
            if self.reset_counts[j] > 0 and self.post_reset_obs[j] < self.min_reset_obs:
                self.post_reset_obs[j] += 1
            if self.use_drift_detection:
                acceptance_bit = float(accepted[j])
                started = time.perf_counter()
                detected = self.adwin[j].update(
                    acceptance_bit,
                    time_index=self._obs_count,
                    allow_detection=(
                        self._obs_count > self.n_initial_warm
                        and self.post_reset_obs[j] >= self.min_reset_obs
                    ),
                )
                self.component_timing["detector_update_s"] += time.perf_counter() - started
                if detected:
                    self._selective_reset(j)
                    reset_this_step[j] = True
                    alarm = {
                        "t": int(self._obs_count),
                        "dimension": int(j),
                        "acceptance": acceptance_bit,
                    }
                    self.alarm_locations.append(alarm)
                    queue_diagnostic(self, {"kind": "alarm", **alarm})

        self._record_acceptance_diagnostics(accepted)
        if self._obs_count <= self.n_initial_warm:
            return

        started = time.perf_counter()
        ready = np.array(
            [
                not reset_this_step[j] and self._dimension_ready(j)
                for j in range(self.p)
            ],
            dtype=bool,
        )
        self._moments.update(x, accepted, ready)
        self.component_timing["pair_update_s"] += time.perf_counter() - started

    def _selective_reset(self, j: int) -> None:
        """Reseed coordinate trackers while retaining the ADWIN suffix."""
        seed_values = list(self.recent_values[j]) or None
        self.streaming_mad[j].reset(seed_values=seed_values)
        self.p2_q01[j].reset(seed_values=seed_values)
        self.p2_q99[j].reset(seed_values=seed_values)
        # Algorithm 2 drops the old prefix but keeps the retained suffix. The
        # detector has already performed that cut, so it must not be reset here.
        self.post_reset_obs[j] = 0
        self.reset_counts[j] += 1
        self._moments.clear_dimension(j)

    def get_correlation_matrix(self) -> np.ndarray:
        return self._moments.correlation()

    def finalize(self) -> np.ndarray:
        """Return the current correlation matrix (paper deployment accessor)."""
        return self.get_correlation_matrix()

    def _stationary_bounds(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        trackers = [StreamingMAD() for _ in range(self.p)]
        q01 = [P2Estimator(0.01) for _ in range(self.p)]
        q99 = [P2Estimator(0.99) for _ in range(self.p)]
        for row in X:
            for j in range(self.p):
                value = float(row[j])
                trackers[j].update(value)
                q01[j].update(value)
                q99[j].update(value)
        location = np.array([tracker.location() for tracker in trackers])
        scale = np.array([tracker.scale() for tracker in trackers])
        lower_mult = np.ones(self.p)
        upper_mult = np.ones(self.p)
        if self.tail_mode == "guarded_asymmetric":
            for j in range(self.p):
                upper_mult[j], lower_mult[j] = self._tail_multipliers(
                    q01[j].estimate(), location[j], q99[j].estimate(), scale[j]
                )
        return inclusive_bounds(
            location, scale, self.lam, lower_mult, upper_mult
        )

    def fit_stationary(self, X: np.ndarray) -> np.ndarray:
        """Paper two-pass stationary helper."""
        X = validate_matrix(X, self.p, min_rows=6)
        lower, upper = self._stationary_bounds(X)
        return trimmed_correlation(X, lower, upper, epsilon=self.epsilon)

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for :meth:`fit_stationary`."""
        return self.fit_stationary(X)

    def fit_streaming(self, X: np.ndarray) -> np.ndarray:
        X = validate_matrix(X, self.p, min_rows=1)
        self.reset()
        for row in X:
            self.update(row)
        return self.finalize()

    def reset(self) -> None:
        self._init_state()

    @staticmethod
    def _p2_state_bytes(estimator: P2Estimator) -> int:
        return int(
            estimator.q.nbytes
            + estimator.n.nbytes
            + estimator.n_prime.nbytes
            + estimator._buf.nbytes
        )

    @property
    def state_bytes(self) -> int:
        tracker_bytes = 0
        for tracker in self.streaming_mad:
            tracker_bytes += self._p2_state_bytes(tracker.stage1)
            tracker_bytes += self._p2_state_bytes(tracker.stage2)
        tracker_bytes += sum(self._p2_state_bytes(item) for item in self.p2_q01)
        tracker_bytes += sum(self._p2_state_bytes(item) for item in self.p2_q99)
        array_bytes = sum(
            array.nbytes
            for array in (
                self.post_reset_obs,
                self.acceptance_total,
                self.acceptance_sum,
                self.reset_counts,
                self._trace_acceptance_sum,
                self._pre_change_acceptance_sum,
                self._post_change_acceptance_sum,
            )
        )
        detector_bytes = sum(detector.state_bytes for detector in self.adwin)
        recent_bytes = sum(
            sys.getsizeof(values) + len(values) * sys.getsizeof(0.0)
            for values in self.recent_values
        )
        return int(
            tracker_bytes
            + array_bytes
            + detector_bytes
            + recent_bytes
            + self._moments.state_bytes
            + explicit_state_bytes([
                self.alarm_locations, self.acceptance_trace,
                self.acceptance_trace_eval_points,
                self.acceptance_trace_eval_points_set,
                self.diagnostic_change_points, self.component_timing,
                self._diagnostic_pending,
            ])
        )

    def get_diagnostics(self) -> dict:
        detector_diagnostics = [detector.diagnostics() for detector in self.adwin]
        acceptance_rates = [
            float(self.acceptance_sum[j] / max(self.acceptance_total[j], 1))
            for j in range(self.p)
        ]
        pre_dimension = None
        post_dimension = None
        pre_mean = None
        post_mean = None
        if self._pre_change_acceptance_count:
            values = (
                self._pre_change_acceptance_sum / self._pre_change_acceptance_count
            )
            pre_dimension = values.tolist()
            pre_mean = float(np.mean(values))
        if self._post_change_acceptance_count:
            values = (
                self._post_change_acceptance_sum / self._post_change_acceptance_count
            )
            post_dimension = values.tolist()
            post_mean = float(np.mean(values))
        return {
            "obs_count": int(self._obs_count),
            "use_drift_detection": self.use_drift_detection,
            "tail_mode": self.tail_mode,
            "reset_counts": self.reset_counts.astype(int).tolist(),
            "post_reset_observations": self.post_reset_obs.astype(int).tolist(),
            "reset_ready": [self._dimension_ready(j) for j in range(self.p)],
            "total_resets": int(self.reset_counts.sum()),
            "alarm_count": self.alarm_locations.total_count,
            "alarm_locations": list(self.alarm_locations),
            "alarm_history_truncated": self.alarm_locations.truncated,
            "diagnostic_history_size": self.diagnostic_history_size,
            "diagnostic_history_source": "bounded_estimator",
            "acceptance_rates": acceptance_rates,
            "acceptance_summary": {
                "change_points": list(self.diagnostic_change_points),
                "pre_change_mean": pre_mean,
                "post_change_mean": post_mean,
                "pre_change_per_dimension": pre_dimension,
                "post_change_per_dimension": post_dimension,
                "trace": list(self.acceptance_trace),
                "trace_history_truncated": self.acceptance_trace.truncated,
            },
            "component_timing_s": dict(self.component_timing),
            "state_bytes": self.state_bytes,
            "detector": {
                "delta": self.adwin_delta,
                "max_window": self.detector_max_window,
                "window_policy": "cut_only" if self.detector_max_window is None else "capped",
                "max_buckets": self.detector_max_buckets,
                "min_window": self.detector_min_window,
                "min_subwindow": self.detector_min_subwindow,
                "per_dimension": detector_diagnostics,
            },
        }
