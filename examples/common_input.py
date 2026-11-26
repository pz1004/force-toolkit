"""Compare three readouts for features 0 and 1 of the same seeded matrix."""
import hashlib

import numpy as np

from force_toolkit import DualBoundForceEstimator, ForceEstimator, MADForceEstimator


def main():
    values = np.random.default_rng(0).normal(size=(2_000, 20))
    values[:, 1] = 0.8 * values[:, 0] + 0.6 * values[:, 1]
    assert hashlib.sha256(values.astype("<f8", copy=False).tobytes()).hexdigest() == (
        "6869b0584252e08cc76a84ef5c01366ce66944dfc0989eee6d37935fde3a8f4a"
    )

    for model, expected_count, expected_entry in (
        (ForceEstimator(), 1988, 0.79807449),
        (MADForceEstimator(p=20), 1985, 0.79767977),
    ):
        entry = model.fit(values).get_correlation()[0, 1]
        diagnostics = model.get_diagnostics()
        details = diagnostics["details"]
        lower, upper = details["thresholds"]
        pair = values[:, :2]
        accepted = np.all((lower[:2] <= pair) & (pair <= upper[:2]), axis=1)
        count = np.count_nonzero(accepted)
        assert details["pair_validity"][0, 1]
        assert count == details["accepted_counts"][0, 1] == expected_count
        np.testing.assert_allclose(entry, np.corrcoef(pair[accepted].T)[0, 1], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(entry, expected_entry, rtol=0, atol=5e-9)
        print(f'{diagnostics["algorithm"]} Pearson[0,1] = {entry:.8f} ({count} jointly accepted rows)')

    sketch = DualBoundForceEstimator(
        p=20, k=10, calibration_size=512,
        marginal_lambda=2, parallel_lambda=4, residual_lambda=2,
    ).fit(values)
    entry = sketch.get_normalized_scatter()[0, 1]
    count = sketch.get_diagnostics()["details"]["serving_n"]
    assert count == len(values) - 512 == 1488
    np.testing.assert_allclose(entry, 0.9873888335953267, rtol=1e-10, atol=1e-12)
    print(f'Dual-Bound normalized transformed scatter[0,1] = {entry:.8f} ({count} estimation rows)')


if __name__ == "__main__":
    main()
