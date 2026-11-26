# API and migration

The numerical state machines stay separate. FORCE and MAD-FORCE retain their
own P² trackers, bounds, numerical safeguards, and statistical policies.

| Earlier call | Unified call |
|---|---|
| `force.ForceEstimator(...).fit(X)` returns a matrix | `ForceEstimator(...).fit(X).finalize()` |
| FORCE replay fitting returns a matrix | `ForceEstimator(...).fit_replayable(factory).get_correlation()` |
| `DriftMADForce(p).fit(X)` is stationary | `MADForceEstimator(p).fit(X).get_correlation()` |
| `DriftMADForce(p).fit_streaming(X)` | `StreamingMADForceEstimator(p).fit(X).get_correlation()` |
| Online MAD `update(row)` returns `None` | `StreamingMADForceEstimator.update(row)` returns `self` |
| `DualBoundFORCE.fit(X)` appends to state | `DualBoundForceEstimator.update_batch(X)` appends; `fit(X)` starts fresh |
| Dual-Bound `get_correlation()` | `get_normalized_scatter()` |

`ForceEstimator` preserves `lambda_scale=3`, `exact_cutover=5`, `use_ter=True`,
`ter_max=None`, `epsilon=1e-10`, and keyword-only `exact_prefix=0`.
`fit_replayable(source_factory, chunk_size=4096)` preserves checks for repeated
content, row order, feature count and length. The factory must return a fresh
iterable for each pass. Failed two-pass fits retain the previously published
estimate and diagnostics; successful fits never append observations.

`StreamingForceEstimator` preserves `window_size=1000`, `lambda_scale=3`,
`use_ter=True`, `ter_max=None`, and `epsilon=1e-10`. The first window calibrates;
its rows are excluded from the moments. Subsequent windows accumulate against
the preceding window's bounds. `fit()` may infer a new feature count; continued
updates must match the established count. Arithmetic failures invalidate the
stream until reset. Invalid input chunks leave it unchanged.

`MADForceEstimator(p, lam=3, *, tail_mode='none', ter_max=3, epsilon=1e-10)`
selects the stationary two-pass procedure. It has no detector or online
updates. It retains final bounds, accepted counts, and the matrix snapshot.

`StreamingMADForceEstimator` preserves all original `DriftMADForce` constructor
parameters/defaults, including `n_initial_warm=200`, `use_drift_detection=True`,
`tail_mode='none'`, `reset_buffer_size=32`, `min_reset_obs=32`,
`adwin_delta=.002`, `detector_delta=None`, `detector_max_window=None`,
`detector_max_buckets=5`, `detector_min_window=32`, `detector_min_subwindow=8`,
and bounded diagnostic history size 128. `detector_max_window=None` keeps the
cut-only detector, whose retained bucket count can grow logarithmically with
stream length. The nominal detector delta is not a calibrated lifetime or
across-coordinate false-alarm probability. Optional guarded asymmetric tails
and diagnostic callbacks remain available.

MAD streaming readiness requires at least one retained accepted pair observation.
An initial identity placeholder no longer counts as a fitted estimate. A
selective reset clears incident pairs and may make the estimate unavailable if
all pair counts vanish. Pair counts remain dense, upper triangular, with the
original `n_jk` alias intact. First-row availability does not establish positive
variance or sufficient samples for every pair.

MAD callbacks run after the full row (all dimensions and pair moments) commits.
A `DiagnosticDeliveryError` has `state_committed=True`. Do not retry that row.
For a callback-delivery failure in a chunk, the change in
`get_diagnostics()['observations']` identifies that chunk's consumed prefix;
the count itself is cumulative. Resume with the unconsumed suffix. Mutation
from a callback is rejected.
Reset notifications follow the same commit rule. Core arithmetic errors can
leave partial state and invalidate the public stream until reset; chunks are
not arithmetic rollback transactions.

`DualBoundForceEstimator(p, k, *, calibration_size=512, epoch_size=None,
marginal_lambda=3, parallel_lambda=3, residual_lambda=3, scale_floor=1e-8,
epsilon=1e-10)` preserves the calibrated transform, fixed SVD implementation,
and exact Frequent Directions compression schedule. Calibration rows do not
enter estimation. During scheduled recalibration, the previous estimate and
its query transform remain serviceable until the next epoch's first estimation
row. `transform(row)` is available once initial calibration completes, before
an estimate necessarily exists. `get_subspace(rank=None)` returns directions
of the normalized scatter; compare subspaces by their projectors because SVD
vectors can change sign. Scale scatter is a robust transformed second moment,
not classical centered covariance. A failed core calibration, standardization
or sketch computation invalidates the public stream until reset. Query and
readout failures have the separate rules described below.

All readouts and diagnostics are independent snapshots. State byte labels
explicitly distinguish numerical array storage from MAD's selected Python
container accounting. Neither is process RSS or a claim to measure all object
memory. Returned p-by-p matrices require additional memory. No estimator is
safe for concurrent mutation of the same instance or input array.

## Revision 0.1.1

`ready` continues to mean that a matrix snapshot is available, including the
original undefined-pair placeholders. Pairwise diagnostics now separately expose
`any_pair_estimable` and `all_pairs_estimable`, evaluated only over off-diagonal
pairs. `details.pair_validity` is a symmetric boolean mask with a false diagonal.
Both MAD classes provide `details.pair_status_codes`: 0 estimable, 1 insufficient
observations, 2 zero variance, 3 diagonal not applicable. Their unit diagonal is
an output convention. Streaming MAD retains its upper-triangular count alias;
stationary MAD counts include marginal counts on the diagonal.

The public `StreamingMADForceEstimator.fit_stationary` and `fit_streaming`
legacy methods now raise an actionable `AttributeError` before touching state.
Use `MADForceEstimator.fit` for stationary fitting or
`StreamingMADForceEstimator.fit` for online fitting. Stationary construction
validates scalar settings without allocating online moments or drift detectors.
Its bound computation keeps the same previous-prefix residual definition and
numerical ordering as 0.1.0.

Incomplete core processing in public streaming MAD or Dual-Bound now invalidates
the stream. Subsequent updates, transformations and estimate readouts raise
`FailedStateError` until `reset()` or a new successful fresh `fit()`. Diagnostics
remain available: top-level `observations` counts fully committed rows, while
`details.observations_attempted` includes the failed row. Internal diagnostic
counters such as MAD's `obs_count` or Dual-Bound's `total_seen` describe partial
engine state and need not equal the public committed count after failure.
`details.lifecycle_status` and `details.failure` identify the failed state.
Rejected invalid input and rejected reentrant mutation do not poison a stream.

A `DiagnosticDeliveryError` still reports a fully committed operation. Its row
is included in the public committed count before callbacks execute, and reset
callbacks see a healthy empty state even if notification fails. Do not retry a
committed row. Resume a failed callback chunk using the unconsumed suffix.
Core-failure chunks require reset, not retry or continuation from partial state.
An interrupt during core mutation also invalidates the stream. An interrupt
raised by a callback propagates after commit without invalidating it; pending
notifications may then remain undelivered, and the committed count still
identifies the consumed prefix.

For an existing stream, save `before = estimator.get_diagnostics()['observations']`
before `update_batch(chunk)`. After `DiagnosticDeliveryError`, compute
`consumed = estimator.get_diagnostics()['observations'] - before`. Once the
notification problem is resolved, continue with `update_batch(chunk[consumed:])`.
This procedure applies only to committed callback failures, not failed core
processing. Preserve the original chunk until resumption is complete.

Dual-Bound scale-scatter reconstruction now avoids premature overflow/underflow
and resolves extreme-unit products with exact rational arithmetic on the stored
binary64 sketch. Ordinary values use a bounded floating-point fast path;
near-cancellation cross terms can also require exact evaluation. The final
binary64 result may round normally; a nonzero entry rounding to zero, or final
overflow, raises `FloatingPointError`. Such a readout error does not invalidate
an otherwise usable normalized-scatter estimate. This exact exceptional path is
slower and is not the estimator's default normalized-scatter readout.

See `statistical-contracts.md` for the causal MAD approximation and the precise
scope of marginal acceptance drift detection. Existing 0.1.0 performance results
are historical; they do not certify this revision's implementation.

## Revision 0.1.2

Streaming FORCE now invalidates its state when initialization or processing is
interrupted, including `KeyboardInterrupt` and `SystemExit`. Its established
`RuntimeError`/`status_ == 'failed'` convention remains; use `reset()` or a fresh
`fit()` to recover. Invalid input and interruption of a purely observational
readout do not invalidate an otherwise healthy stream.

Guarded asymmetric MAD tails no longer use a fixed dimensional epsilon to
disable tail expansion for small measurement units. Zero scale still means no
expansion; dimensionless activation and imbalance thresholds are unchanged.
Positive unit conversions should preserve the intended trimming policy when
the represented inputs retain their information. Rounding at acceptance ties,
subnormal quantization and tracker range limits still constrain numerical
equivalence. Small-unit guarded-mode results may intentionally differ from
0.1.0/0.1.1. The default `tail_mode='none'` is unaffected by this policy change.

MAD bound evaluation retains ordinary arithmetic order but evaluates exceptional
endpoints without premature overflow/underflow. A genuinely out-of-range final
bound can remain infinite; an intermediate overflow cannot widen a finite bound.
This can change accepted samples at extreme scales and factors.

Dual-Bound rejects a nonzero centered coordinate that standardizes to zero
through underflow. Calibration, recalibration or estimation ingestion then raises
`FloatingPointError` and invalidates the stream until reset. A failed standalone
`transform()` query raises without changing a previously valid estimate. Exact
centered zeros and representable nonzero subnormal quotients remain supported.
This is explicit failure rather than recovery of unrepresentable standardized
values; sketch storage has not been redesigned. Exact scale-scatter reconstruction
protects arithmetic on the sketch values that were actually retained.

Assigning a new callable to `StreamingMADForceEstimator.diagnostic_sink` now
affects detector-cut events as well as alarms and acceptance events. Assign
`None` to disable delivery and another callable to resume it. Only future events
are delivered; disabled periods are not replayed. Existing callback commit and
resumption rules still apply.

Benchmark schema 3 binds effective Numba settings and uses a separately verified
optimized 0.1.0 wheel as a third variant. Acceleration versus the unoptimized
reference and regression versus optimized 0.1.0 are different reported contrasts.
Historical schema-2 runs are preserved; they cannot resume as schema-3 evidence.

## Revision 0.2.0

`StreamingMADForceEstimator(..., collect_component_timing=False)` adds a
keyword-only option. Component profiling is now disabled by default:
`details['component_timing_s']` is `None` and
`details['component_timing_enabled']` is `False`. Pass `True` to retain the four
component timing totals. `reset()` preserves this option. Profiling uses the
Python controller over the same numerical state and changes execution cost;
profile percentages cannot predict attainable throughput speedups.

The default controller compiles sequential tracker, acceptance, pair and
bucket processing with Numba. All four quantile trackers remain active, even
with symmetric tails. Bootstrap still uses the existing NumPy quantile
interpolation. Exceptional bounds still use exact arithmetic. Detector rings
grow as needed and do not impose an implicit window cap. Without a sink,
native calls process at most 256 rows before returning to Python, sometimes
fewer for bootstrap, exact bounds, history or resets. This offers signal
processing opportunities, without a fixed interrupt-latency guarantee.

With a sink, each row commits before its events are delivered and before the
next row starts. A callback may query diagnostics, replace/remove the sink, or
raise; the existing event order, committed-prefix and reentrancy rules apply.
Escaping core errors and interrupts invalidate the stream; earlier rows stay
committed. Invalid final rows still reject an entire public chunk before any
mutation. There is no public backend-selection setting.

State accounting now includes owned contiguous tracker/counter/recent-value
arrays, allocated detector ring capacity and controller work arrays. Selected
Python histories remain counted separately; view headers and other uncounted
object overhead remain excluded. Byte counts intentionally differ from 0.1.2.
The proposed FORCE validation and Dual-Bound workspace optimizations were not
accepted after development measurements; their 0.1.2 execution paths remain. These are not process RSS measurements. Public transformations and readouts remain snapshots.

This is an implementation revision. It establishes no new breakdown,
convergence-rate, drift-detection, or XGBoost-superiority guarantee. The current
dense online pair traversal costs O(N p²); this is not a universal lower bound
on alternative formulations. Stationary MAD accelerates only its tracker pass;
its final-bound pair-correlation calculation remains unchanged.

## How an observation reaches a readout

The five classes in `api.py` adapt three numerical implementations. A fresh `fit` publishes a new estimate. Online `update` and `update_batch` advance the existing state. Each public readout returns a snapshot, and diagnostics report the observation count, readiness, target and family-specific details. [Public API](../src/force_toolkit/api.py)

* **FORCE** learns marginal quantiles and constructs acceptance bounds. Batch fitting replays the observations with final bounds. Streaming uses the previous completed window to screen the next rows. The moment accumulator updates each pair on its jointly accepted observations. The readout uses the pair’s own means and variances. [Batch](../src/force_toolkit/_force/core.py) · [Streaming](../src/force_toolkit/_force/streaming.py) · [Pair moments](../src/force_toolkit/_force/_moments.py)
* **MAD-FORCE** tracks the median and absolute deviations from preceding median estimates. The stationary tracker pass freezes final bounds for the second pass. The online controller processes each row in sequence, passing coordinate acceptance bits to the enabled detectors and clearing incident pairs on reset. Compiled Numba loops accelerate the tracker, detector and pair updates. Pair counts and validity accompany the Pearson readout. [Stationary bounds](../src/force_toolkit/_mad/stationary.py) · [Controller](../src/force_toolkit/_mad/_controller.py) · [Detector](../src/force_toolkit/_mad/_detector.py)
* **Dual-Bound FORCE** fixes marginal scaling, a reference subspace and parallel/residual bounds on the calibration prefix. Each subsequent row passes through that map into the Frequent Directions sketch. The readout normalizes the sketched transformed second moment, and `serving_n` reports its estimation count. [Transform and readouts](../src/force_toolkit/_dual/estimator.py) · [Sketch](../src/force_toolkit/_dual/_fd.py)
