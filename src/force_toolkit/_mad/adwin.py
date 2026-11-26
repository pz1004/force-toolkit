"""
Bucketed ADWIN-style drift detector.

This keeps a compressed bucket summary rather than every observation. A finite
window cap gives bounded deployment state; ``max_window=None`` implements the
paper's cut-only window, whose state grows as O(log W). The detector keeps bucket
boundaries as candidate cut points and uses an empirical-Bernstein-style mean
shift test.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .common import explicit_state_bytes, validate_positive_int
from .diagnostics import (
    BoundedHistory, diagnostic_transaction, queue_diagnostic, validate_diagnostics,
)


@dataclass
class _Bucket:
    """Compressed summary of a contiguous slice of the stream."""

    width: int
    total: float
    total_sq: float


class ADWINDetector:
    """Per-dimension acceptance-rate drift detector with an explicit window policy.

    Parameters
    ----------
    delta : float
        Nominal cut-threshold parameter (smaller makes cuts harder). This is
        not a calibrated lifetime or across-coordinate false-alarm probability
        for the adaptive acceptance bits produced by MAD-FORCE.
    max_window : int or None
        Upper bound enforced by expiring whole buckets, or None for Algorithm 2
        prefix removal only after a successful change test.
    max_buckets : int
        Maximum number of buckets kept per row before compression.
    min_window : int
        Minimum window length before testing for drift.
    min_subwindow : int
        Minimum size for each side of a candidate cut.
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_window: int | None = None,
        max_buckets: int = 5,
        min_window: int = 32,
        min_subwindow: int = 8,
        diagnostic_history_size: int = 128,
        diagnostic_sink=None,
    ):
        delta = float(delta)
        if not np.isfinite(delta) or not 0.0 < delta < 1.0:
            raise ValueError("delta must be finite and strictly between 0 and 1")
        for name, value in (
            ("max_buckets", max_buckets),
            ("min_window", min_window),
            ("min_subwindow", min_subwindow),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if int(max_buckets) < 2:
            raise ValueError("max_buckets must be at least 2")
        if int(min_window) < 2 * int(min_subwindow):
            raise ValueError("min_window must be at least twice min_subwindow")
        if max_window is not None:
            max_window = validate_positive_int("max_window", max_window)
        if max_window is not None and max_window < int(min_window):
            raise ValueError("max_window must be greater than or equal to min_window")
        self.delta = delta
        self.max_window = max_window
        self.diagnostic_history_size, self.diagnostic_sink = validate_diagnostics(
            diagnostic_history_size, diagnostic_sink
        )
        self.max_buckets = int(max_buckets)
        self.min_window = int(min_window)
        self.min_subwindow = int(min_subwindow)
        self._rows: list[deque[_Bucket]] = []
        self._count: int = 0
        self._total_time: int = 0
        self._total: float = 0.0
        self._total_sq: float = 0.0
        self._last_cut_threshold: float | None = None
        self._last_cut_diff: float | None = None
        self._alarm_count: int = 0
        self._cut_history = BoundedHistory(self.diagnostic_history_size)
        self._expired_count = 0
        self._diagnostic_active = False
        self._diagnostic_pending = []

    @property
    def bucket_count(self) -> int:
        """Total number of buckets currently retained."""
        return sum(len(row) for row in self._rows)

    @property
    def n_rows(self) -> int:
        """Number of bucket rows currently in use."""
        return len(self._rows)

    @property
    def count(self) -> int:
        """Effective window length currently retained."""
        return self._count

    @property
    def total_time(self) -> int:
        """Absolute observation time used in the Equation (9) schedule."""
        return self._total_time

    @property
    def total(self) -> float:
        """Sum over the retained window."""
        return self._total

    @property
    def total_sq(self) -> float:
        """Squared-sum over the retained window."""
        return self._total_sq

    @property
    def last_cut_threshold(self) -> float | None:
        """Threshold used by the most recent successful detection."""
        return self._last_cut_threshold

    @property
    def last_cut_diff(self) -> float | None:
        """Observed mean difference from the most recent successful detection."""
        return self._last_cut_diff

    @property
    def alarm_count(self) -> int:
        """Number of detected drifts since the last reset."""
        return self._alarm_count

    @property
    def cut_history(self) -> list[dict[str, float | int]]:
        """Successful cut summaries kept for diagnostics and reporting."""
        return list(self._cut_history)

    def update(
        self,
        x: float,
        *,
        time_index: int | None = None,
        allow_detection: bool = True,
    ) -> bool:
        """Add one scalar value and optionally test for drift.

        ``time_index`` is the absolute stream time in Equation (9). It is
        intentionally separate from the compressed retained-window length.
        Diagnostic callbacks run after the cut is committed. If delivery
        raises ``DiagnosticDeliveryError``, do not retry this observation;
        inspect ``alarm_count``/``cut_history`` for the committed alarm.
        """
        with diagnostic_transaction(self):
            return self._update(x, time_index=time_index, allow_detection=allow_detection)

    def _update(self, x: float, *, time_index: int | None,
                allow_detection: bool) -> bool:
        x = float(x)
        if not np.isfinite(x):
            raise ValueError("ADWIN observations must be finite")
        if time_index is None:
            self._total_time += 1
        else:
            if (
                isinstance(time_index, bool)
                or not isinstance(time_index, (int, np.integer))
                or int(time_index) <= self._total_time
            ):
                raise ValueError("time_index must be a strictly increasing positive integer")
            self._total_time = int(time_index)
        self._insert(x)
        self._trim_to_window()
        if not allow_detection or self._count < self.min_window:
            return False
        return self._detect_change()

    def reset(self) -> None:
        """Clear all state."""
        if self._diagnostic_active:
            raise RuntimeError("update/reset cannot be called from a diagnostic callback")
        self._rows = []
        self._count = 0
        self._total_time = 0
        self._total = 0.0
        self._total_sq = 0.0
        self._last_cut_threshold = None
        self._last_cut_diff = None
        self._alarm_count = 0
        self._cut_history = BoundedHistory(self.diagnostic_history_size)
        self._expired_count = 0

    def _insert(self, x: float) -> None:
        if not self._rows:
            self._rows.append(deque())
        self._rows[0].append(_Bucket(width=1, total=x, total_sq=x * x))
        self._count += 1
        self._total += x
        self._total_sq += x * x
        self._compress_rows()

    def _compress_rows(self) -> None:
        level = 0
        while level < len(self._rows):
            row = self._rows[level]
            if len(row) <= self.max_buckets:
                level += 1
                continue

            if level + 1 == len(self._rows):
                self._rows.append(deque())

            oldest = row.popleft()
            second_oldest = row.popleft()
            merged = self._merge_buckets(oldest, second_oldest)
            self._rows[level + 1].append(merged)

    def _merge_buckets(self, first: _Bucket, second: _Bucket) -> _Bucket:
        return _Bucket(
            width=first.width + second.width,
            total=first.total + second.total,
            total_sq=first.total_sq + second.total_sq,
        )

    def _iter_buckets_oldest_first(self):
        # Older data lives in higher rows; within each row buckets are oldest->newest.
        for level in range(len(self._rows) - 1, -1, -1):
            row = self._rows[level]
            for bucket in row:
                yield bucket

    def _detect_change(self) -> bool:
        prefix_count = 0
        prefix_total = 0.0

        buckets = list(self._iter_buckets_oldest_first())
        if len(buckets) < 2:
            return False

        var, dd = self._scan_constants()
        for bucket in buckets[:-1]:
            prefix_count += bucket.width
            prefix_total += bucket.total
            suffix_count = self._count - prefix_count

            if prefix_count < self.min_subwindow or suffix_count < self.min_subwindow:
                continue

            mu0 = prefix_total / prefix_count
            mu1 = (self._total - prefix_total) / suffix_count
            diff = abs(mu0 - mu1)
            threshold = self._threshold_with_constants(prefix_count, suffix_count, var, dd)

            if diff > threshold:
                self._last_cut_threshold = threshold
                self._last_cut_diff = diff
                self._alarm_count += 1
                cut = {
                    'time': self._total_time,
                    'count': self._count,
                    'prefix_count': prefix_count,
                    'suffix_count': suffix_count,
                    'diff': diff,
                    'threshold': threshold,
                }
                self._cut_history.append(cut)
                self._drop_oldest(prefix_count)
                queue_diagnostic(self, {"kind": "detector_cut", **cut})
                return True

        return False

    def _scan_constants(self):
        mean = self._total / max(self._count, 1)
        var = max(self._total_sq / max(self._count, 1) - mean * mean, 0.0)
        # log(2 / delta_t), evaluated without clipping small confidence values
        # or underflowing delta_t. A constant acceptance window has variance 0.
        dd = (
            math.log(2.0) - math.log(self.delta)
            + math.log(math.log2(max(self._total_time, 2)) + 1.0)
        )
        return var, dd

    @staticmethod
    def _threshold_with_constants(n0, n1, var, dd):
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        return math.sqrt(2.0 * var * dd / m) + (2.0 * dd) / (3.0 * m)

    def _cut_threshold(self, n0: int, n1: int) -> float:
        return self._threshold_with_constants(n0, n1, *self._scan_constants())

    def _trim_to_window(self) -> None:
        if self.max_window is None or self._count <= self.max_window:
            return

        excess = self._count - self.max_window
        removed = 0
        for level in range(len(self._rows) - 1, -1, -1):
            row = self._rows[level]
            while row and removed < excess:
                removed += row[0].width
                bucket = row.popleft()
                self._count -= bucket.width
                self._total -= bucket.total
                self._total_sq -= bucket.total_sq
            if removed >= excess:
                break
        self._prune_empty_rows()
        self._expired_count += removed

    def _drop_oldest(self, width: int) -> None:
        remaining = width
        for level in range(len(self._rows) - 1, -1, -1):
            row = self._rows[level]
            while row and remaining > 0:
                bucket = row[0]
                if bucket.width > remaining:
                    raise RuntimeError("Bucket boundary mismatch during ADWIN cut")
                bucket = row.popleft()
                remaining -= bucket.width
                self._count -= bucket.width
                self._total -= bucket.total
                self._total_sq -= bucket.total_sq
            if remaining == 0:
                break

        if remaining != 0:
            raise RuntimeError("Failed to remove the requested ADWIN prefix")
        self._prune_empty_rows()

    def _prune_empty_rows(self) -> None:
        while self._rows and not self._rows[-1]:
            self._rows.pop()

    def diagnostics(self) -> dict:
        """Expose detector state for experiment logging and theorem alignment."""
        return {
            'delta': self.delta,
            'max_window': self.max_window,
            'window_policy': 'cut_only' if self.max_window is None else 'capped',
            'expired_observations': self._expired_count,
            'max_buckets': self.max_buckets,
            'min_window': self.min_window,
            'min_subwindow': self.min_subwindow,
            'count': self._count,
            'retained_window_length': self._count,
            'total_time': self._total_time,
            'bucket_count': self.bucket_count,
            'n_rows': self.n_rows,
            'alarm_count': self._alarm_count,
            'last_cut_threshold': self._last_cut_threshold,
            'last_cut_diff': self._last_cut_diff,
            'cut_history': list(self._cut_history),
            'cut_history_truncated': self._cut_history.truncated,
            'diagnostic_history_size': self.diagnostic_history_size,
            'state_bytes': self.state_bytes,
        }

    @property
    def state_bytes(self) -> int:
        """Explicit retained bucket and diagnostic storage, excluding the sink."""
        return explicit_state_bytes([self._rows, self._cut_history, self._diagnostic_pending])
