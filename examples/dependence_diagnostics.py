"""Change the dependence at row 600 while keeping both marginal ranges fixed."""
import numpy as np

from force_toolkit import StreamingMADForceEstimator


def main():
    x = np.tile([-2., -1., 1., 2.], 300)
    y = x.copy()
    y[600:] *= -1
    values = np.column_stack((x, y))

    model = StreamingMADForceEstimator(p=2, n_initial_warm=200).fit(values)
    details = model.get_diagnostics()["details"]
    retained = model.get_correlation()[0, 1]
    current = np.corrcoef(values[600:].T)[0, 1]
    assert details["alarm_count"] == 0 and details["acceptance_rates"] == [1., 1.]
    assert details["pair_validity"][0, 1]
    np.testing.assert_allclose([retained, current], [-0.2, -1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(retained, np.corrcoef(values[200:].T)[0, 1], atol=1e-12, rtol=0)
    print(f'Marginal acceptance alarms: {details["alarm_count"]}')
    print(f'Retained accepted-pair Pearson correlation (rows 200–1199): {retained:.3f}')
    print(f'Current-regime Pearson correlation (rows 600–1199): {current:.3f}')


if __name__ == "__main__":
    main()
