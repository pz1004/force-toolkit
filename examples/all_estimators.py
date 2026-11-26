"""Run with: python examples/all_estimators.py (no optional dependencies)."""
import numpy as np
from force_toolkit import (
    ForceEstimator, StreamingForceEstimator, MADForceEstimator,
    StreamingMADForceEstimator, DualBoundForceEstimator, NotFittedError,
)

X = np.random.default_rng(42).normal(size=(1200, 6))
X[::10] *= 20

for estimator in (
    ForceEstimator(),
    StreamingForceEstimator(window_size=100),
    MADForceEstimator(p=6),
    StreamingMADForceEstimator(p=6, use_drift_detection=True),
    DualBoundForceEstimator(p=6, k=3, calibration_size=128),
):
    try:
        estimator.finalize()
    except NotFittedError:
        pass
    else:
        raise AssertionError('Unfitted readout must fail')
    assert estimator.fit(X) is estimator
    estimate = estimator.finalize()
    assert estimate.shape == (6, 6) and np.isfinite(estimate).all()
    diagnostics = estimator.get_diagnostics()
    print(diagnostics['algorithm'], diagnostics['statistical_target'], diagnostics['ready'])
    estimate.fill(123)  # snapshots do not mutate retained state
    assert not np.any(estimator.finalize() == 123)
    if hasattr(estimator, 'update_batch'):
        estimator.reset()
        for start in range(0, len(X), 7):
            estimator.update_batch(X[start:start + 7])
        assert estimator.get_diagnostics()['observations'] == len(X)

replayed = ForceEstimator().fit_replayable(lambda: iter(X), chunk_size=64)
np.testing.assert_array_equal(replayed.get_correlation(), ForceEstimator().fit(X).get_correlation())
sketch = DualBoundForceEstimator(6, 3, calibration_size=128, epoch_size=400).fit(X)
assert sketch.get_normalized_scatter().shape == (6, 6)
assert sketch.get_scale_scatter().shape == (6, 6)
assert sketch.transform(X[0]).shape == (6,)
basis = sketch.get_subspace(rank=2)
assert basis.shape[0] == 6
