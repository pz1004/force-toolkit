# Statistical characterization

Run from an installed checkout, using only the library's NumPy and Numba
dependencies:

```bash
python -m pip install -e .
python studies/mad_characterization.py --output studies/results/my-mad-run
```

The output directory must be new or empty. The script never resumes a previous
run. Defaults are 500 rows, four features, seeds 0–9, four synthetic
distributions, and six orders of each generated data set. To specify these
settings explicitly:

```bash
python studies/mad_characterization.py --output studies/results/my-explicit-run \
  --n 500 --p 4 --seeds 0 1 2 3 4 5 6 7 8 9
```

The four distributions are correlated Gaussian, its exponentiated skewed
version, correlated Student t with three degrees of freedom, and Gaussian with
10% of rows receiving a heavy-tailed perturbation. Each data set is evaluated
in generated order, sorted and reverse-sorted by the first feature, and three
fixed random permutations. No sibling-project data or research files are used.

The study makes three distinct comparisons:

1. Production readout against an independent NumPy Pearson calculation on the
   *same accepted rows*. This is a numerical correctness check at
   `rtol=1e-12, atol=1e-14`.
2. Production causal P² bounds against exact final-center median/MAD bounds.
   Bound differences, acceptance disagreements, pair-count differences and
   correlation differences characterize two different trimming procedures.
   The exact comparator is not population truth.
3. A study-only exact previous-prefix comparator against exact final-center
   MAD. Replacing both quantile approximations with exact quantiles reveals
   that previous-prefix centering itself can produce order sensitivity. This
   quadratic-time comparator does not alter the production estimator.

Two deterministic controls illustrate the marginal detector's scope. One
changes correlation from +1 to −1 while keeping every marginal acceptance bit
constant; the other shifts both marginals beyond their acceptance bounds.
They test an absence of detector signal and one designed sensitivity case;
they do not estimate general detection power or false-alarm probability.

Each run writes configuration, environment and executed-source SHA-256 hashes,
input hashes and per-case metrics, a JSON summary, and a Markdown report. The
source fingerprint covers the *imported* library, so an installed wheel can be
characterized without substituting checkout source. A source change during a
run aborts the run. NumPy/Numba version or platform changes may change random
draws or floating-point details; compare the retained input hashes before
treating a new run as the same experiment.

The [recorded 240-case report](results/mad-characterization/REPORT.md) retains
its configuration and raw evidence in the same directory.

The results are descriptive for a fixed finite grid. Permutations of one data
set are dependent; seeds from different distribution cases also reuse the
same generator seed numbers. There are no confidence intervals or population
accuracy claims. See [the statistical contracts](../docs/statistical-contracts.md)
for interpretation.

The files under `results/mad-characterization` are the original **0.1.1** study. The **0.2.0** rerun and the **0.2.1** installed-wheel rerun have version-specific provenance. The release evidence bundle supplies the 0.2.1 outputs under `mad-0.2.1/` with the wheel SHA-256 and comparisons to both earlier runs. [0.2.0 validation record](../provenance/validation-0.2.0.json) · [Validation](../docs/validation.md)
