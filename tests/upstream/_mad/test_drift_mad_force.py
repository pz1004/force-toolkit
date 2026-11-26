"""Tests for the Drift-MAD FORCE algorithm."""

import numpy as np
import pytest
from force_toolkit._mad.drift_mad_force import DriftMADForce


def _make_data(p, N, R_clean, epsilon=0.0, seed=42):
    """Generate contaminated multivariate normal data."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(R_clean)
    n_clean = int(N * (1 - epsilon))
    n_contam = N - n_clean
    X_clean = rng.standard_normal((n_clean, p)) @ L.T
    X_contam = rng.standard_cauchy((n_contam, p)) * 10.0
    X = np.vstack([X_clean, X_contam])
    perm = rng.permutation(N)
    return X[perm]


def _block_corr(p, n_blocks=5, within=0.6, between=0.05):
    block_size = p // n_blocks
    R = np.full((p, p), between)
    for b in range(n_blocks):
        s, e = b * block_size, min((b + 1) * block_size, p)
        R[s:e, s:e] = within
    np.fill_diagonal(R, 1.0)
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def _frobenius_rmse(R_est, R_true):
    p = R_est.shape[0]
    mask = np.triu(np.ones((p, p), dtype=bool), k=1)
    diff = R_est[mask] - R_true[mask]
    return np.sqrt(np.mean(diff ** 2))


def test_clean_data():
    """On clean data, Drift-MAD FORCE should produce reasonable correlation."""
    p, N = 10, 5000
    R_true = _block_corr(p, n_blocks=2)
    X = _make_data(p, N, R_true, epsilon=0.0)
    est = DriftMADForce(p)
    R_est = est.fit(X)
    rmse = _frobenius_rmse(R_est, R_true)
    assert rmse < 0.15, f"RMSE on clean data too high: {rmse:.4f}"


def test_robust_at_30pct_contamination():
    """At 30% Cauchy contamination, Drift-MAD FORCE stays robust (RMSE < 0.3).

    This tests above the 25% IQR breakdown point, validating MAD's advantage.
    We disable drift detection since this is stationary contamination.
    """
    p, N = 10, 10_000
    R_true = _block_corr(p, n_blocks=2)
    X = _make_data(p, N, R_true, epsilon=0.30)
    dm = DriftMADForce(p, use_drift_detection=False)
    R_dm = dm.fit(X)
    rmse_dm = _frobenius_rmse(R_dm, R_true)
    assert rmse_dm < 0.45, f"Drift-MAD FORCE RMSE too high at 30%: {rmse_dm:.4f}"


def test_ablation_no_drift():
    """Drift-MAD FORCE with drift detection disabled still works."""
    p, N = 5, 3000
    R_true = _block_corr(p, n_blocks=1, within=0.5)
    X = _make_data(p, N, R_true, epsilon=0.10)
    est = DriftMADForce(p, use_drift_detection=False)
    R_est = est.fit(X)
    rmse = _frobenius_rmse(R_est, R_true)
    assert rmse < 0.3, f"RMSE={rmse:.4f}"


def test_ablation_no_asymmetric_ter():
    """Drift-MAD FORCE with symmetric TER still works."""
    p, N = 5, 3000
    R_true = _block_corr(p, n_blocks=1, within=0.5)
    X = _make_data(p, N, R_true, epsilon=0.10)
    est = DriftMADForce(p, use_asymmetric_ter=False)
    R_est = est.fit(X)
    rmse = _frobenius_rmse(R_est, R_true)
    assert rmse < 0.3, f"RMSE={rmse:.4f}"


def test_fit_stationary_alias():
    """fit() remains an alias of fit_stationary()."""
    p, N = 6, 4000
    R_true = _block_corr(p, n_blocks=2, within=0.4)
    X = _make_data(p, N, R_true, epsilon=0.15, seed=99)
    est = DriftMADForce(p, use_drift_detection=False, tail_mode='none')
    np.testing.assert_allclose(est.fit(X), est.fit_stationary(X))


def test_selective_reset_reseeds_without_corrupting_unaffected_pairs():
    """Selective reset clears only affected pairs and reseeds the dimension promptly."""
    est = DriftMADForce(
        3,
        use_drift_detection=False,
        tail_mode='none',
        n_initial_warm=0,
        reset_buffer_size=12,
        min_reset_obs=12,
    )
    rng = np.random.default_rng(5)
    for _ in range(80):
        est.update(rng.standard_normal(3))

    unaffected_before = est.n_jk[1, 2]
    assert unaffected_before > 0

    est._selective_reset(0)

    assert not est._dimension_ready(0)
    assert est.n_jk[0, 1] == 0
    assert est.n_jk[0, 2] == 0
    assert est.n_jk[1, 2] == unaffected_before

    for _ in range(11):
        est.update(rng.standard_normal(3))
        assert not est._dimension_ready(0)
    est.update(rng.standard_normal(3))
    assert est._dimension_ready(0)


def test_guarded_asymmetric_tail_falls_back_on_extreme_imbalance():
    """Extreme one-sided contamination should fall back to the no-TER path."""
    p = 4
    rng = np.random.default_rng(17)
    X_clean = rng.standard_normal((4000, p))
    X_contam = np.zeros((800, p))
    X_contam[:, 0] = 1_000.0
    X = np.vstack([X_clean, X_contam])

    est_none = DriftMADForce(p, use_drift_detection=False, tail_mode='none')
    est_guarded = DriftMADForce(
        p, use_drift_detection=False, tail_mode='guarded_asymmetric'
    )

    R_none = est_none.fit_stationary(X)
    R_guarded = est_guarded.fit_stationary(X)
    np.testing.assert_allclose(R_guarded, R_none)


def test_reset():
    """After reset, correlation should be identity (no data)."""
    est = DriftMADForce(5)
    rng = np.random.default_rng(0)
    for _ in range(100):
        est.update(rng.standard_normal(5))
    est.reset()
    R = est.get_correlation_matrix()
    np.testing.assert_array_equal(R, np.eye(5))
