import copy
import numpy as np
import pytest
import force_toolkit as ft

FACTORIES = [
    lambda: ft.ForceEstimator(),
    lambda: ft.StreamingForceEstimator(window_size=50),
    lambda: ft.MADForceEstimator(4),
    lambda: ft.StreamingMADForceEstimator(4, n_initial_warm=20),
    lambda: ft.DualBoundForceEstimator(4, 2, calibration_size=16, epoch_size=71),
]


def assert_tree(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            if key != 'component_timing_s':
                assert_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('factory', FACTORIES)
def test_fresh_fit_reset_snapshots_and_validation(factory):
    X = np.random.default_rng(191).normal(size=(330, 4))
    estimator = factory()
    with pytest.raises(ft.NotFittedError):
        estimator.finalize()
    assert not estimator.get_diagnostics()['ready']
    assert estimator.fit(X) is estimator
    expected = factory().fit(X[40:])
    assert estimator.fit(X[40:]) is estimator
    np.testing.assert_array_equal(estimator.finalize(), expected.finalize())
    assert_tree(estimator.get_diagnostics(), expected.get_diagnostics())
    saved = estimator.finalize()
    snapshot = estimator.finalize()
    snapshot.fill(17)
    diagnostics = estimator.get_diagnostics()
    diagnostics['details'].clear()
    np.testing.assert_array_equal(estimator.finalize(), saved)
    assert estimator.get_diagnostics()['details']
    assert estimator.reset() is estimator
    assert not estimator.get_diagnostics()['ready']
    with pytest.raises(ft.NotFittedError):
        estimator.finalize()
    estimator.fit(X)
    np.testing.assert_array_equal(estimator.finalize(), factory().fit(X).finalize())


@pytest.mark.parametrize('factory', [FACTORIES[1], FACTORIES[3], FACTORIES[4]])
@pytest.mark.parametrize('chunk', [1, 7, 64, 1024])
def test_chunks_and_invalid_chunks_are_atomic(factory, chunk):
    X = np.random.default_rng(918).normal(size=(1031, 4))
    X[501:650, 0] += 15
    expected = factory().fit(X)
    estimator = factory()
    for start in range(0, len(X), chunk):
        assert estimator.update_batch(X[start:start+chunk]) is estimator
    np.testing.assert_array_equal(estimator.finalize(), expected.finalize())
    assert_tree(estimator.get_diagnostics(), expected.get_diagnostics())
    before = copy.deepcopy(estimator.get_diagnostics())
    for invalid in [np.r_[X[:3], [[np.nan]*4]], X[:3, :3], X[:2].astype(complex) + 1j]:
        with pytest.raises(ValueError):
            estimator.update_batch(invalid)
        assert_tree(estimator.get_diagnostics(), before)
    assert estimator.update_batch(np.empty((0, 4))) is estimator
    assert_tree(estimator.get_diagnostics(), before)


def test_replay_and_invalid_replay_preserve_published_snapshot():
    X = np.random.default_rng(4).normal(size=(1031, 4))
    expected = ft.ForceEstimator().fit(X)
    estimator = ft.ForceEstimator()
    for chunk in (1, 7, 64, 1024):
        assert estimator.fit_replayable(lambda: iter(X), chunk_size=chunk) is estimator
        np.testing.assert_array_equal(estimator.finalize(), expected.finalize())
    source = iter(X)
    with pytest.raises(ValueError):
        estimator.fit_replayable(lambda: source)
    np.testing.assert_array_equal(estimator.finalize(), expected.finalize())


def test_boundaries_and_statistical_target():
    X = np.random.default_rng(4).normal(size=(80, 4))
    force = ft.StreamingForceEstimator(window_size=50).fit(X[:50])
    dual = ft.DualBoundForceEstimator(4, 2, calibration_size=16).fit(X[:16])
    mad = ft.StreamingMADForceEstimator(4, n_initial_warm=20).fit(X[:20])
    for estimator, next_row in [(force, X[50]), (dual, X[16]), (mad, X[20])]:
        with pytest.raises(ft.NotFittedError):
            estimator.finalize()
        assert estimator.update(next_row) is estimator
        assert estimator.get_diagnostics()['ready']
    assert not hasattr(dual, 'get_correlation')
    assert 'scatter' in dual.get_diagnostics()['statistical_target']
    for estimator in (force, mad):
        np.testing.assert_array_equal(estimator.finalize(), estimator.get_correlation())


def test_mad_chunk_callback_failure_commits_row_and_stops_chunk():
    events = []
    def sink(event):
        events.append(event)
        if event.get('t') == 51:
            raise OSError('unavailable')
    X = np.random.default_rng(4).normal(size=(80, 4))
    estimator = ft.StreamingMADForceEstimator(4, n_initial_warm=10, diagnostic_sink=sink)
    with pytest.raises(ft.DiagnosticDeliveryError) as error:
        estimator.update_batch(X)
    assert error.value.state_committed
    assert estimator.get_diagnostics()['observations'] == 51
    reference = ft.StreamingMADForceEstimator(4, n_initial_warm=10).fit(X[:51])
    np.testing.assert_array_equal(estimator.finalize(), reference.finalize())
    estimator.update_batch(X[51:])
    reference.update_batch(X[51:])
    np.testing.assert_array_equal(estimator.finalize(), reference.finalize())
