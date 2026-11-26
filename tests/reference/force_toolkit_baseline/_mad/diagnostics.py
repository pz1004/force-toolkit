"""Bounded estimator histories and optional experiment-owned diagnostics.

Full experiment logs are intentionally owned by ``DiagnosticRecorder``, not by
the estimator. A recorder can use memory proportional to the requested output;
that storage is separate from the streaming estimator's memory contract.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from .common import validate_nonnegative_int, validate_positive_int


class BoundedHistory(deque):
    """A fixed-capacity history with a lifetime append count."""

    def __init__(self, capacity: int):
        super().__init__(maxlen=capacity)
        self.total_count = 0

    def append(self, value) -> None:
        self.total_count += 1
        super().append(value)

    @property
    def truncated(self) -> bool:
        return self.total_count > len(self)

    def __reduce__(self):
        return (_restore_history, (self.maxlen, list(self), self.total_count))


def _restore_history(capacity, values, total_count):
    history = BoundedHistory(capacity)
    deque.extend(history, values)
    history.total_count = total_count
    return history


def validate_diagnostics(history_size, sink):
    history_size = validate_nonnegative_int("diagnostic_history_size", history_size)
    if sink is not None and not callable(sink):
        raise ValueError("diagnostic_sink must be callable or None")
    return history_size, sink


def diagnostic_points(name, points, *, limit=None):
    """Validate a schedule, limiting estimator-owned schedules explicitly."""
    values = set()
    for value in (() if points is None else points):
        values.add(validate_positive_int(name, value))
        if limit is not None and len(values) > limit:
            raise ValueError(
                f"{name} exceeds diagnostic_history_size; use an external "
                "DiagnosticRecorder for a longer schedule"
            )
    return sorted(values)


def emit_diagnostic(sink, event: dict) -> None:
    if sink is not None:
        # A sink receives no references to mutable estimator state.
        sink(deepcopy(event))


class DiagnosticDeliveryError(RuntimeError):
    """A sink failed after the estimator operation had already committed.

    The observation/reset must not be retried. All pending events have been
    attempted once; ``__cause__`` is the first callback exception. This error
    concerns notification delivery only, not a rollback of estimator state.
    """

    state_committed = True

    def __init__(self, failed_events: int):
        self.failed_events = failed_events
        super().__init__(
            f"{failed_events} diagnostic callback(s) failed after state was committed; "
            "do not retry the observation or reset"
        )


@contextmanager
def diagnostic_transaction(owner):
    """Deliver notifications only after a complete estimator operation.

    This is an event boundary, not a rollback mechanism for core errors.
    An update buffers at most one cut and alarm per coordinate plus acceptance
    (2p+1 events), and releases them on every exit. Reentrant mutations from
    a callback are rejected before they can change the estimator.
    """
    if getattr(owner, "_diagnostic_active", False):
        raise RuntimeError("update/reset cannot be called from a diagnostic callback")
    owner._diagnostic_active = True
    owner._diagnostic_pending = pending = []
    try:
        yield
        first_error = None
        failed_events = 0
        for event in pending:
            try:
                emit_diagnostic(owner.diagnostic_sink, event)
            except Exception as error:
                failed_events += 1
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise DiagnosticDeliveryError(failed_events) from first_error
    finally:
        pending.clear()
        owner._diagnostic_active = False


def queue_diagnostic(owner, event: dict) -> None:
    """Queue an event for the owner's current operation, when a sink exists."""
    if owner.diagnostic_sink is None:
        return
    if getattr(owner, "_diagnostic_active", False):
        owner._diagnostic_pending.append(event)
    else:
        # Legacy subclasses call shared diagnostic helpers without wrapping
        # their historical update path. Preserve that compatibility behavior.
        with diagnostic_transaction(owner):
            owner._diagnostic_pending.append(event)


class DiagnosticRecorder:
    """Collect complete experiment logs through ``diagnostic_sink``.

    Pass ``recorder.record`` to an estimator, then use
    ``recorder.enrich(estimator.get_diagnostics())``. Acceptance observations
    are aggregated online; only requested trace points and alarm/cut records
    are retained. All schedules and complete histories belong to this object.
    """

    def __init__(self, eval_points=None, change_points=None):
        self.eval_points = diagnostic_points("eval_points", eval_points)
        self._eval_points_set = set(self.eval_points)
        self.change_points = diagnostic_points("change_points", change_points)
        self.reset()

    def reset(self):
        self.alarm_locations = []
        self.cut_history = []
        self.acceptance_trace = []
        self._trace_sum = None
        self._trace_count = 0
        self._pre_sum = None
        self._pre_count = 0
        self._post_sum = None
        self._post_count = 0

    def record(self, event: dict) -> None:
        kind = event["kind"]
        if kind == "stream_reset":
            self.reset()
        elif kind == "alarm":
            self.alarm_locations.append({k: v for k, v in event.items() if k != "kind"})
        elif kind == "detector_cut":
            self.cut_history.append({k: v for k, v in event.items() if k != "kind"})
        elif kind == "acceptance":
            t = event["t"]
            values = np.asarray(event["accepted"], dtype=float)
            if self._trace_sum is None:
                self._trace_sum = np.zeros_like(values)
                self._pre_sum = np.zeros_like(values)
                self._post_sum = np.zeros_like(values)
            if self.change_points:
                if t <= self.change_points[0]:
                    self._pre_sum += values
                    self._pre_count += 1
                else:
                    self._post_sum += values
                    self._post_count += 1
            if self._eval_points_set:
                self._trace_sum += values
                self._trace_count += 1
                if t in self._eval_points_set:
                    rates = self._trace_sum / self._trace_count
                    self.acceptance_trace.append({
                        "t": int(t),
                        "mean_acceptance": float(np.mean(rates)),
                        "per_dimension_acceptance": rates.tolist(),
                    })
                    self._trace_sum.fill(0.0)
                    self._trace_count = 0

    def enrich(self, diagnostics: dict) -> dict:
        """Return complete logs without changing estimator-owned diagnostics."""
        output = deepcopy(diagnostics)
        pre = self._pre_sum / self._pre_count if self._pre_count else None
        post = self._post_sum / self._post_count if self._post_count else None
        output["alarm_locations"] = deepcopy(self.alarm_locations)
        output["alarm_history_truncated"] = False
        output["acceptance_summary"] = {
            "change_points": list(self.change_points),
            "pre_change_mean": None if pre is None else float(np.mean(pre)),
            "post_change_mean": None if post is None else float(np.mean(post)),
            "pre_change_per_dimension": None if pre is None else pre.tolist(),
            "post_change_per_dimension": None if post is None else post.tolist(),
            "trace": deepcopy(self.acceptance_trace),
            "trace_history_truncated": False,
        }
        for dimension, detector in enumerate(output.get("detector", {}).get("per_dimension", [])):
            detector["cut_history"] = [
                {k: v for k, v in event.items() if k != "dimension"}
                for event in self.cut_history if event.get("dimension") == dimension
            ]
            detector["cut_history_truncated"] = False
        output["diagnostic_history_source"] = "external_recorder"
        return output
