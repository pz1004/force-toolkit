# Statistical contracts and limitations

`finalize()` returns a numerical snapshot when its lifecycle permits a readout.
Snapshot availability is distinct from whether a particular correlation is
estimable, and neither establishes statistical accuracy or uncertainty.

## Pairwise targets

`ForceEstimator` and `MADForceEstimator` report Pearson correlation computed
separately for each pair, using the rows accepted by both coordinates' final
bounds. The bounds are estimated from the fitted data. Changing the bounds
changes the accepted sample and therefore the reported quantity.

`StreamingForceEstimator` accumulates accepted pairs using previous-window
bounds. It retains historical accepted observations; its matrix does not
automatically describe only the latest window. `StreamingMADForceEstimator`
uses adaptive bounds and clears the pairs incident to a coordinate when that
coordinate resets. Each pair therefore describes its own accepted historical
sample since the most recent reset affecting that pair, excluding warm-up.
Different pairs can have different counts, time spans and acceptance histories.

Pairwise deletion and trimming do not guarantee a positive semidefinite
matrix. A unit diagonal is a presentation convention; it does not establish
nonzero marginal variance. The estimators retain their zero off-diagonal
placeholders for insufficient samples and zero variance. They do not provide
standard errors, confidence intervals, a generic test of correlation, or
unqualified estimates of the untrimmed population correlation. Ordinary
Pearson inference cannot be transferred to these data-selected and possibly
adaptively retained samples without additional assumptions and analysis.

## Availability and estimability

All pairwise estimators expose `ready`, `any_pair_estimable`, and
`all_pairs_estimable` at the top level of `get_diagnostics()`. `ready` describes
snapshot availability. The other two flags summarize only off-diagonal pairs.
Before fitting they are false. A failed streaming MAD instance reports false
estimability flags; failed streaming FORCE retains its upstream behavior of
rejecting diagnostics and readouts with `RuntimeError` until reset.
A ready snapshot may have no estimable pairs.

`details["pair_status_codes"]` is an independent square array, or `None` when
no valid pair assessment exists. Off-diagonal codes are shared across the
pairwise estimators. MAD uses code 3 on the diagonal; FORCE retains its original
diagonal codes 0–2, which describe its retained marginal moments.

| Code | Meaning |
|---|---|
| 0 | At least two jointly accepted observations and positive retained variance for both coordinates |
| 1 | Fewer than two jointly accepted observations |
| 2 | At least two jointly accepted observations but zero retained variance in one or both coordinates |
| 3 | MAD diagonal: not an off-diagonal correlation assessment |

`details["pair_validity"]` provides the corresponding Boolean off-diagonal
mask. This mask and both aggregate estimability flags always ignore the
diagonal, including FORCE's marginal status codes. An off-diagonal code of
zero establishes numerical estimability under the retained
moment calculations, not adequate sample size, independence, conditioning or
scientific reliability. Very small positive variance remains positive; no
new variance floor is imposed.

```python
import numpy as np
from force_toolkit import MADForceEstimator

estimator = MADForceEstimator(2).fit(np.ones((8, 2)))
diagnostics = estimator.get_diagnostics()
assert diagnostics["ready"]
assert not diagnostics["any_pair_estimable"]
assert diagnostics["details"]["pair_status_codes"][0, 1] == 2
assert estimator.finalize()[0, 1] == 0.0  # placeholder, not evidence of independence
```

## What MAD means here

The inherited MAD-FORCE scale construction feeds absolute deviations from the
*previous prefix's* P² median estimate into a second P² median tracker. The
first row has no previous prefix and supplies no such deviation. The final
scale uses the historical multiplier 1.4826; stationary fitting fixes the
resulting final bounds for its second pass.

This is not the exact sample MAD around the final sample median. There are two
sources of discrepancy: approximate quantile tracking and the use of a changing
previous-prefix center. Even replacing the trackers with exact medians leaves
a different finite-sample scale. For `[0, 1, 2, 3, 4, 5]`, exact previous-prefix
deviations are `[1, 1.5, 2, 2.5, 3]`, with median 2; final-center absolute
deviations have median 1.5. Reordering data can affect the causal construction.

The [reproducible study](../studies/README.md) characterizes this difference
over clean Gaussian, skewed, heavy-tailed and contaminated data, using ten
seeds and six orders of each data set. It reports bound differences,
acceptance disagreement and resulting correlation differences separately
from a numerical oracle on the same accepted sample. Exact final-center MAD
is an alternative procedure, not a known population truth. Sorted orders are
deliberate stress cases, and permutations are dependent observations.

The [recorded 240-case grid](../studies/results/mad-characterization/REPORT.md)
found maximum pair-correlation differences from
the exact final-center comparator up to 0.0541 in generated/shuffled cases
and 0.3038 in sorted contaminated cases. These finite-grid results support
documenting order sensitivity; they are not error bounds or estimates of
general performance. Exact previous-prefix comparisons in the report show
that quantile approximation alone does not explain all differences.

That characterization was recorded with version 0.1.1 and `tail_mode='none'`.
It does not validate the guarded-tail policy. Revision 0.1.2 adds dedicated
unit-conversion checks for guarded tails: the former fixed dimensional cutoff
could change accepted observations under positive rescaling. The corrected
policy retains dimensionless guard thresholds and the zero-scale convention.
These tests establish specified numerical behavior, not general robustness or
inferential guarantees.

## Drift detection scope

The MAD detector monitors changes in marginal acceptance indicators. Its
`delta` parameter enters an ADWIN-style threshold calculation; this library
does not establish a simultaneous or lifetime false-alarm guarantee for the
adaptive multi-coordinate procedure. In particular, a change in dependence
can leave every marginal acceptance indicator unchanged. No alarm in that
case means that this monitored signal did not change.

The deterministic negative control switches correlation from +1 to −1 while
all acceptance indicators remain one. It produces zero alarms. After warm-up,
the retained sample has correlation −0.2, while the most recent regime has
sample correlation −1. The historical result is consistent with the estimator's
target. A separate deterministic marginal-shift control produces alarms and
coordinate resets. That positive result demonstrates one designed case,
not general sensitivity, delay or false-alarm calibration.

## Dual-Bound target

`DualBoundForceEstimator` applies a calibrated row transformation and sketches
its second moment without recentering on the estimation sample's mean.
`get_normalized_scatter()` normalizes that sketch by its diagonal. It is not
accepted-pair Pearson correlation. `get_scale_scatter()` expresses the sketched
transformed second moment in the calibration scale's coordinate units; it
does not recover the raw data's sample covariance or undo clipping and
sketch compression. Calibration and estimation phases, and the served epoch,
remain part of the interpretation.

Scale-scatter reconstruction evaluates representable results of the retained
binary64 sketch without intermediate overflow or underflow. It cannot recover
information already lost before sketch insertion. Revision 0.1.2 explicitly
rejects nonzero standardized coordinates that round to zero; a core ingestion
failure invalidates the stream, while a failed read-only transformation query
leaves the existing estimate usable. Nonzero subnormal quotients remain valid.
A final scale-scatter result outside
the finite float64 range, or a nonzero entry that rounds to zero, raises
`FloatingPointError`. Ordinary representable rounding is allowed. This readout
failure does not mutate the stream, so a normalized-scatter readout can still
be available. Numerical representability is separate from statistical
adequacy of the calibration and sketch rank.
