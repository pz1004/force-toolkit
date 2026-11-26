# force-toolkit

Which observations contributed to a correlation estimate? FORCE and MAD-FORCE compute Pearson correlation separately for each feature pair, using the rows that pass both features’ bounds. Dual-Bound FORCE uses a calibration prefix to transform incoming rows and estimates normalized transformed scatter with a low-rank sketch. All five estimators accept NumPy arrays and run in the same Python environment.

## Install and try version 0.2.1

With Git and Python 3.11–3.13 installed, run these commands from an empty directory on Linux or macOS. On Windows PowerShell, replace the activation line with `.\.venv\Scripts\Activate.ps1`.

```sh
git clone --branch v0.2.1 --depth 1 https://github.com/pz1004/force-toolkit.git
cd force-toolkit
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/common_input.py
```

The example generates 2,000 rows and 20 features locally. For features 0 and 1 it prints:

| Estimator | Readout | Value | Contributing rows |
|---|---|---:|---|
| Stationary FORCE | Accepted-pair Pearson | 0.79807449 | 1,988 jointly accepted |
| Stationary MAD-FORCE | Accepted-pair Pearson | 0.79767977 | 1,985 jointly accepted |
| Dual-Bound FORCE | Normalized transformed scatter | 0.98738883 | 1,488 after calibration |

The script checks each Pearson value against NumPy on that estimator’s accepted rows. Dual-Bound uses the first 512 rows to calibrate its transformation, then sketches the remaining rows with rank 10. Its entry measures alignment of the transformed sketch columns. [Example and parameters](examples/common_input.py)

### Using the wheel

In an activated environment, install the release wheel directly:

```sh
python -m pip install https://github.com/pz1004/force-toolkit/releases/download/v0.2.1/force_toolkit-0.2.1-py3-none-any.whl
```

This example runs from any directory after installation. It prints the number of jointly accepted rows and their correlation for one pair.

```python
import numpy as np
from force_toolkit import ForceEstimator

X = np.random.default_rng(42).normal(size=(1200, 6))
fit = ForceEstimator().fit(X)
details = fit.get_diagnostics()["details"]
if details["pair_validity"][0, 1]:
    print(int(details["accepted_counts"][0, 1]), fit.get_correlation()[0, 1])
```

NumPy and Numba are the direct runtime dependencies. Numba compiles kernels on first use. The downloadable source archive includes the example scripts. [Installation and development](docs/installation.md) · [Release downloads](https://github.com/pz1004/force-toolkit/releases/tag/v0.2.1)

## Choose an estimator

| Estimator | Quantity returned | Data used for estimation |
|---|---|---|
| `ForceEstimator()` | Final-bound accepted-pair Pearson | At least five rows, two passes with final quantile bounds |
| `StreamingForceEstimator(window_size=1000)` | Historical accepted-pair Pearson | Rows after the first window, using the preceding window’s bounds |
| `MADForceEstimator(p)` | Final-bound accepted-pair Pearson | At least six rows, two passes with final median-based bounds |
| `StreamingMADForceEstimator(p)` | Historical accepted-pair Pearson with coordinate-specific resets | Jointly accepted rows after warm-up (default 200), since the last reset involving either coordinate |
| `DualBoundForceEstimator(p, k)` | Normalized transformed scatter | Rows after the calibration prefix (default 512), sketch rank `k` |

Supply a dense, finite real matrix shaped `(rows, features)`, or one row of shape `(features,)` for online updates. Keep feature count and order consistent. `fit` starts fresh, `update_batch` continues a stream, and `finalize` returns a snapshot.

For Pearson readouts, use `pair_validity` before interpreting an entry. Zero can be a placeholder for too few observations or zero variance. Because different pairs can use different rows, the assembled matrix can have a negative eigenvalue. Check positive semidefiniteness before using it in a procedure that requires it. [Statistical contracts](docs/statistical-contracts.md) · [API and architecture](docs/migration.md)

## What a streaming MAD alarm measures

Run `python examples/dependence_diagnostics.py` from the source tree. All 1,000 post-warm-up rows pass both marginal bounds, so the detector registers zero alarms even as the correlation reverses. The first 400 retained rows have correlation +1 and the next 600 have correlation −1. Both segments contain complete cycles with identical marginal means and variances. In this example the pooled correlation is therefore (400 − 600)/1,000 = −0.2, while the current-regime correlation is −1. A companion control shifting both marginals produces two alarms. [Example](examples/dependence_diagnostics.py) · [Control results](studies/results/mad-characterization/REPORT.md)

## Validation, citation and support

Run `python tools/report_020.py` from the source tree to reconstruct the archived **0.2.0 versus 0.1.2** performance comparison. Version 0.2.1 has its own installed-wheel tests and MAD characterization. [Validation](docs/validation.md) · [Benchmark protocol](benchmarks/README.md) · [Release notes](docs/release-0.2.1.md)

Cite Sooyoung Jang, *force-toolkit*, version 0.2.1, and the method paper for each estimator you use. For support, contact **syjang@hanbat.ac.kr** or open an issue. [Citation](CITATION.cff) · [Method papers and upstream revisions](provenance/README.md) · [Issue tracker](https://github.com/pz1004/force-toolkit/issues) · [MIT license](Licence.txt)
