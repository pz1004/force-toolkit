"""A logging failure must not change Algorithm 2 or consume half an observation."""

import numpy as np
import pytest

from force_toolkit._mad.adwin import ADWINDetector
from force_toolkit._mad.diagnostics import DiagnosticDeliveryError
from force_toolkit._mad.drift_mad_force import DriftMADForce


def _assert_same_core(observed, reference):
    assert observed._obs_count == reference._obs_count
    for name in ("acceptance_total", "acceptance_sum", "reset_counts", "post_reset_obs"):
        np.testing.assert_array_equal(getattr(observed, name), getattr(reference, name))
    for name in ("count", "mean_j", "mean_k", "m2_j", "m2_k", "co_moment"):
        np.testing.assert_array_equal(
            getattr(observed._moments, name), getattr(reference._moments, name)
        )
    for actual, expected in zip(observed.adwin, reference.adwin):
        assert actual._rows == expected._rows
        assert actual.alarm_count == expected.alarm_count
        assert actual.total_time == expected.total_time
    np.testing.assert_array_equal(observed.finalize(), reference.finalize())


@pytest.mark.parametrize("estimator_type", [DriftMADForce])
@pytest.mark.parametrize("failed_kind", ["detector_cut", "alarm", "acceptance"])
def test_failed_sink_commits_full_observation_and_stream_can_continue(estimator_type, failed_kind):
    rng = np.random.default_rng(7)
    stream = np.vstack([rng.normal(size=(1000, 2)), rng.normal(loc=15, size=(100, 2))])
    attempted = []
    failed_at = None
    observed = None
    reference = estimator_type(2, detector_max_window=None)

    def sink(event):
        nonlocal failed_at
        attempted.append(event)
        if event["kind"] == "stream_reset":
            return
        # Every callback sees all coordinates and pairs at the same time.
        _assert_same_core(observed, reference)
        event_time = event.get("t", event.get("time"))
        if failed_at is None and event["kind"] == failed_kind and event_time >= 1008:
            failed_at = event_time
            raise OSError("diagnostic output unavailable")

    observed = estimator_type(2, detector_max_window=None, diagnostic_sink=sink)
    failures = []
    for row in stream:
        reference.update(row)
        try:
            observed.update(row)
        except DiagnosticDeliveryError as error:
            failures.append(error)
            assert error.state_committed
            assert error.failed_events == 1
            assert isinstance(error.__cause__, OSError)
            _assert_same_core(observed, reference)
        assert observed._diagnostic_pending == []

    assert len(failures) == 1
    assert failed_at == 1008
    assert observed.reset_counts.tolist() == [1, 1]
    # A failed cut/alarm must not suppress delivery of this observation's later events.
    assert any(event["kind"] == "acceptance" and event["t"] == failed_at for event in attempted)
    _assert_same_core(observed, reference)


def test_standalone_detector_callback_failure_preserves_committed_cut():
    failed = False

    def sink(event):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("cut notification failed")

    observed = ADWINDetector(max_window=None, diagnostic_sink=sink)
    reference = ADWINDetector(max_window=None)
    errors = 0
    for bit in np.r_[np.ones(400), np.zeros(100)]:
        alarm = reference.update(bit)
        try:
            assert observed.update(bit) == alarm
        except DiagnosticDeliveryError as error:
            errors += 1
            assert alarm and error.state_committed
            assert isinstance(error.__cause__, OSError)
        assert observed._rows == reference._rows
        assert observed.count == reference.count
        assert observed.total_time == reference.total_time
        assert observed.alarm_count == reference.alarm_count
        assert observed.cut_history == reference.cut_history
        assert observed._diagnostic_pending == []
    assert errors == 1


@pytest.mark.parametrize("estimator_type", [DriftMADForce])
@pytest.mark.parametrize("operation", ["update", "reset"])
def test_callback_cannot_reenter_estimator_mutation(estimator_type, operation):
    attempted = False
    observed = None

    def sink(event):
        nonlocal attempted
        if event["kind"] == "acceptance" and not attempted:
            attempted = True
            if operation == "update":
                observed.update(np.ones(2))
            elif isinstance(observed, DriftMADForce):
                observed.reset()
            else:
                observed.fit_streaming(np.ones((1, 2)))

    observed = estimator_type(2, diagnostic_sink=sink)
    with pytest.raises(DiagnosticDeliveryError) as raised:
        observed.update(np.ones(2))
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "diagnostic callback" in str(raised.value.__cause__)
    assert observed._obs_count == 1
    np.testing.assert_array_equal(observed.acceptance_total, [1, 1])
    observed.update(np.ones(2))
    assert observed._obs_count == 2


@pytest.mark.parametrize("estimator_type", [DriftMADForce])
def test_stream_reset_callback_sees_initialized_state_even_when_it_fails(estimator_type):
    observed = estimator_type(2)
    observed.update(np.ones(2))

    def sink(event):
        assert event["kind"] == "stream_reset"
        assert observed._obs_count == 0
        assert observed.n_jk.sum() == 0
        assert observed.reset_counts.sum() == 0
        raise OSError("reset notification failed")

    # Constructing a new fit stream and explicit reset use the same event boundary.
    observed.diagnostic_sink = sink
    with pytest.raises(DiagnosticDeliveryError):
        if isinstance(observed, DriftMADForce):
            observed.reset()
        else:
            observed.fit_streaming(np.ones((1, 2)))
    assert observed._obs_count == 0
    assert observed._diagnostic_pending == []
    observed.diagnostic_sink = None
    observed.update(np.ones(2))
    assert observed._obs_count == 1


@pytest.mark.parametrize("operation", ["update", "reset"])
def test_detector_callback_cannot_reenter_mutation(operation):
    detector = None

    def sink(event):
        if operation == "update":
            detector.update(0.0)
        else:
            detector.reset()

    detector = ADWINDetector(max_window=None, diagnostic_sink=sink)
    for time, bit in enumerate(np.r_[np.ones(400), np.zeros(100)], start=1):
        try:
            detector.update(bit)
        except DiagnosticDeliveryError as error:
            assert isinstance(error.__cause__, RuntimeError)
            assert detector.total_time == time
            assert detector.alarm_count == 1
            break
    else:
        pytest.fail("expected a change notification")


def test_core_validation_errors_are_not_wrapped_as_delivery_failures():
    estimator = DriftMADForce(2, diagnostic_sink=lambda event: None)
    with pytest.raises(ValueError):
        estimator.update(np.array([np.nan, 1.0]))
    assert estimator._obs_count == 0
    estimator.update(np.ones(2))
    assert estimator._obs_count == 1
