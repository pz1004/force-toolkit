"""Fresh-observation detector gates from the paper's deployment recipe."""

import numpy as np
import pytest

from force_toolkit._mad.drift_mad_force import DriftMADForce


@pytest.mark.parametrize("estimator_type", [DriftMADForce])
@pytest.mark.parametrize("gate", [0, 1, 3, 32])
def test_cut_tests_and_pairs_resume_at_fresh_observation_boundary(
    estimator_type, gate, monkeypatch
):
    estimator = estimator_type(3, n_initial_warm=0, min_reset_obs=gate)
    for _ in range(40):
        estimator.update(np.ones(3))
    detector = estimator.adwin[0]
    retained_count, retained_total = detector.count, detector.total
    unaffected_pairs = estimator.n_jk[1, 2]
    estimator._selective_reset(0)
    # Reseeding the trackers does not replay bits or satisfy the fresh gate.
    assert detector.count == retained_count
    assert detector.total == retained_total
    assert estimator.post_reset_obs[0] == 0
    tested_at = [[], []]

    def cut_test(dimension):
        tested_at[dimension].append(estimator.adwin[dimension].total_time)
        return False

    for dimension in (0, 1):
        monkeypatch.setattr(
            estimator.adwin[dimension], "_detect_change",
            lambda dimension=dimension: cut_test(dimension),
        )

    first_eligible = max(1, gate)
    for fresh in range(1, first_eligible + 2):
        estimator.update(np.ones(3))
        assert detector.total_time == 40 + fresh
        assert detector.count == retained_count + fresh
        assert detector.total == retained_total + fresh
        assert estimator.post_reset_obs[0] == min(fresh, gate)
        assert estimator.n_jk[0, 1] == max(0, fresh - first_eligible + 1)
        assert estimator.n_jk[0, 2] == estimator.n_jk[0, 1]
        assert estimator.n_jk[1, 2] == unaffected_pairs + fresh

    assert tested_at[0] == [40 + first_eligible, 41 + first_eligible]
    assert tested_at[1] == list(range(41, 42 + first_eligible))
    assert estimator.reset_counts.tolist() == [1, 0, 0]


@pytest.mark.parametrize("estimator_type", [DriftMADForce])
def test_post_reset_readiness_does_not_bypass_initial_warmup(estimator_type, monkeypatch):
    estimator = estimator_type(2, n_initial_warm=50, min_reset_obs=3)
    for _ in range(40):
        estimator.update(np.ones(2))
    estimator._selective_reset(0)
    tested_at = []

    def cut_test():
        tested_at.append(estimator.adwin[0].total_time)
        return False

    monkeypatch.setattr(estimator.adwin[0], "_detect_change", cut_test)
    for _ in range(10):
        estimator.update(np.ones(2))
    assert estimator.post_reset_obs[0] == 3
    assert estimator.n_jk[0, 1] == 0
    assert tested_at == []
    estimator.update(np.ones(2))
    assert tested_at == [51]
    assert estimator.n_jk[0, 1] == 1


@pytest.mark.parametrize("estimator_type", [DriftMADForce])
def test_real_detector_ingests_bits_during_cooldown_and_spaces_resets(estimator_type):
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal((i % 2) * 30, 1, 100) for i in range(6)])
    events = []
    options = {} if estimator_type is DriftMADForce else {"use_symmetric_ter": False}
    estimator = estimator_type(2, diagnostic_sink=events.append, **options)
    last_alarm = None
    alarm_times = []
    cooldown_observations = 0
    detector = estimator.adwin[0]

    for time, value in enumerate(values, start=1):
        count, total = detector.count, detector.total
        reset_count = estimator.reset_counts[0]
        estimator.update(np.array([value, value]))
        assert detector.total_time == time
        acceptance = events[-1]
        assert acceptance["kind"] == "acceptance"
        if last_alarm is not None and time - last_alarm < 32:
            cooldown_observations += 1
            assert estimator.reset_counts[0] == reset_count
            assert detector.count == count + 1
            assert detector.total == total + acceptance["accepted"][0]
        if estimator.reset_counts[0] > reset_count:
            last_alarm = time
            alarm_times.append(time)
            assert estimator.post_reset_obs[0] == 0
            assert estimator.n_jk[0, 1] == 0

    assert len(alarm_times) >= 2
    assert min(np.diff(alarm_times)) >= 32
    assert cooldown_observations >= 31
    if estimator_type is DriftMADForce:
        # The original review case reset at 201 and again at 202.
        assert alarm_times[0] == 201
        assert 202 not in alarm_times
