"""Common lifecycle and explicit statistical targets for the FORCE family."""
from copy import deepcopy
import numpy as np

from ._exceptions import NotFittedError, FailedStateError
from ._validation import matrix, row
from ._force.core import ForceEstimator as _Force, _as_float_matrix
from ._force.streaming import StreamingForceEstimator as _StreamingForce
from ._force._input import DEFAULT_CHUNK_SIZE
from ._mad.drift_mad_force import DriftMADForce as _MAD
from ._mad.common import trimmed_correlation
from ._mad.stationary import validate_parameters, stationary_bounds
from ._mad.diagnostics import DiagnosticDeliveryError
from ._validity import PAIR_STATUS_LABELS, estimability, streaming_mad_codes
from ._dual.estimator import DualBoundFORCE as _Dual


def _diagnostics(algorithm, target, ready, observations, size, accounting, details):
    result = {
        'algorithm': algorithm,
        'statistical_target': target,
        'ready': bool(ready),
        'observations': int(observations),
        'state': {'bytes': int(size), 'accounting': accounting,
                  'excludes': 'input, returned snapshots, transient workspace, JIT code and external callback storage'},
        'details': details,
    }
    if 'pair_status_codes' in details:
        codes = details['pair_status_codes']
        result.update(estimability(codes))
        details = dict(details)
        details['pair_validity'] = (None if codes is None else
            (codes == 0) & ~np.eye(len(codes), dtype=bool))
        result['details'] = details
    return deepcopy(result)


class ForceEstimator(_Force):
    """Two-pass accepted-pair Pearson correlation using final FORCE bounds.

    Constructor parameters and defaults match FORCE 1.1.0. Fitting returns
    self; finalize/get_correlation return independent matrix snapshots.
    """
    def fit(self, X):
        result = super().fit(X)
        self._result = result
        return self

    def fit_replayable(self, source_factory, chunk_size=DEFAULT_CHUNK_SIZE):
        """Fit two identical ordered traversals, returning self."""
        result = super().fit_replayable(source_factory, chunk_size)
        self._result = result
        return self

    def reset(self):
        _Force.__init__(self, self.lambda_scale, self.exact_cutover, self.use_ter,
                        self.ter_max, self.epsilon, exact_prefix=self.exact_prefix)
        self._result = None
        return self

    def finalize(self):
        result = getattr(self, '_result', None)
        if result is None:
            raise NotFittedError('Call fit or fit_replayable before requesting an estimate')
        return result.copy()

    get_correlation = finalize

    def get_diagnostics(self):
        result = getattr(self, '_result', None)
        return _diagnostics('FORCE', 'final_bound_accepted_pair_pearson', result is not None,
            self.n_samples_ or 0, self.state_nbytes_ + (0 if result is None else result.nbytes),
            'retained numerical array bytes; excludes Python object overhead',
            dict(n_features=self.n_features_in_, execution_mode=self.execution_mode_,
                 quantiles=self.quantiles, thresholds=self.thresholds,
                 accepted_counts=self.accepted_counts_, pair_status_codes=self.pair_status_codes_,
                 bound_status=self.bound_status_))


class StreamingForceEstimator(_StreamingForce):
    """Previous-window bounds; cumulative historical accepted-pair Pearson.

    The first window calibrates. An estimate exists after the next row, even
    if some pairs have insufficient samples; inspect diagnostics for validity.
    """
    def fit(self, X):
        values = self._numeric_array(X)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError('Expected a matrix with at least two features')
        self.reset()
        return self.update_batch(values)

    def reset(self):
        _StreamingForce.__init__(self, self.window_size, self.lambda_scale,
                                self.use_ter, self.ter_max, self.epsilon)
        return self

    def finalize(self):
        result = super().finalize()
        if result is None:
            raise NotFittedError('No post-calibration estimation observation is available')
        return result.copy()

    get_correlation = finalize

    def get_diagnostics(self, *, include_status_labels=False):
        details = super().get_diagnostics(include_status_labels=include_status_labels)
        return _diagnostics('Streaming FORCE', 'historical_accepted_pair_pearson',
            self.estimation_count_ > 0, self.n_samples_seen_, self.state_nbytes_,
            'retained numerical array bytes; excludes Python object overhead', details)


class MADForceEstimator:
    """Stationary two-pass MAD-FORCE, separate from the online state machine."""
    def __init__(self, p, lam=3.0, *, tail_mode='none', ter_max=3.0, epsilon=1e-10):
        self.p, self.lam, self.tail_mode, self.ter_max, self.epsilon = validate_parameters(
            p, lam, tail_mode, ter_max, epsilon)
        self.reset()

    def reset(self):
        self._result = self._thresholds = self._counts = self._pair_codes = None
        self.n_samples_seen_ = 0
        return self

    def fit(self, X):
        values = matrix(X, self.p, min_rows=6)
        lower, upper = stationary_bounds(values, lam=self.lam,
                                         tail_mode=self.tail_mode, ter_max=self.ter_max)
        result, counts, codes = trimmed_correlation(
            values, lower, upper, epsilon=self.epsilon, return_diagnostics=True)
        self._result, self._thresholds, self._counts = result, (lower, upper), counts
        self._pair_codes = codes
        self.n_samples_seen_ = len(values)
        return self

    def finalize(self):
        if self._result is None:
            raise NotFittedError('Call fit before requesting an estimate')
        return self._result.copy()

    get_correlation = finalize

    def get_diagnostics(self):
        arrays = [] if self._result is None else [self._result, self._counts, self._pair_codes, *self._thresholds]
        return _diagnostics('MAD-FORCE', 'final_bound_accepted_pair_pearson', self._result is not None,
            self.n_samples_seen_, sum(a.nbytes for a in arrays),
            'retained numerical array bytes; excludes Python object overhead',
            dict(n_features=self.p, tail_mode=self.tail_mode, thresholds=self._thresholds,
                 accepted_counts=self._counts, minimum_fit_rows=6,
                 pair_status_codes=self._pair_codes, pair_status_code_labels=PAIR_STATUS_LABELS,
                 scale_construction='P2_median_of_previous_prefix_absolute_residuals'))


class StreamingMADForceEstimator(_MAD):
    """Online MAD-FORCE with optional ADWIN-style acceptance drift detection.

    All original constructor defaults are preserved, including 200 warm-up
    observations. Ready means at least one accepted pair observation is retained;
    zero-variance/insufficient pairs still use zero off-diagonal placeholders.
    Callback errors occur after committing the current row. Do not retry it.
    """
    def _initialize_state(self):
        # This runs inside the upstream reset transaction, before notification.
        self._failure = None
        self._committed_observations = 0
        try:
            super()._initialize_state()
        except BaseException as error:
            self._failure = f'{type(error).__name__}: {error}'
            raise

    def _check_mutation(self, *, allow_failed=False):
        if getattr(self, '_diagnostic_active', False):
            raise RuntimeError('update/reset cannot be called from a diagnostic callback')
        if not allow_failed:
            self._ensure_healthy()

    def _ensure_healthy(self):
        if self._failure is not None or self._moments._failed:
            raise FailedStateError('MAD stream has partially failed; call reset() before continuing')

    def _update(self, x):
        # Invoked inside diagnostic_transaction: callbacks see a committed row.
        try:
            super()._update(x)
            self._committed_observations += 1
        except BaseException as error:
            self._failure = f'{type(error).__name__}: {error}'
            raise

    def fit(self, X):
        self._check_mutation(allow_failed=True)
        values = matrix(X, self.p)
        self.reset()
        return self.update_batch(values)

    def update(self, x):
        self._check_mutation()
        values = row(x, self.p)
        try:
            super().update(values)
        except DiagnosticDeliveryError:
            raise  # the full row has already committed
        except Exception as error:
            self._failure = f'{type(error).__name__}: {error}'
            raise
        return self

    def update_batch(self, X):
        self._check_mutation()
        values = matrix(X, self.p)
        for value in values:
            self.update(value)
        return self

    def reset(self):
        self._check_mutation(allow_failed=True)
        try:
            super().reset()
        except DiagnosticDeliveryError:
            raise
        except Exception as error:
            self._failure = f'{type(error).__name__}: {error}'
            raise
        return self

    def fit_stationary(self, X):
        raise AttributeError('Use MADForceEstimator(p).fit(X) for stationary fitting')

    def fit_streaming(self, X):
        raise AttributeError('Use StreamingMADForceEstimator(p).fit(X); read with finalize()')

    def finalize(self):
        self._ensure_healthy()
        if not np.any(self.n_jk):
            raise NotFittedError('No accepted pair observation is retained after warm-up/reset')
        return self._moments.correlation()

    get_correlation = finalize
    get_correlation_matrix = finalize

    def get_diagnostics(self):
        details = super().get_diagnostics()
        details['accepted_counts_upper_triangle'] = self.n_jk.copy()
        failed = self._failure is not None or self._moments._failed
        details.update(
            lifecycle_status='failed' if failed else ('ready' if np.any(self.n_jk) else 'warming'),
            failure=self._failure, observations_attempted=self._obs_count,
            observations_committed=self._committed_observations,
            pair_status_codes=None if failed else streaming_mad_codes(self._moments),
            pair_status_code_labels=PAIR_STATUS_LABELS,
            detector_scope='marginal_acceptance_changes_only',
            scale_construction='P2_median_of_previous_prefix_absolute_residuals')
        return _diagnostics('Streaming MAD-FORCE', 'adaptive_accepted_pair_pearson_since_incident_reset',
            not failed and np.any(self.n_jk), self._committed_observations, self.state_bytes,
            'explicit numerical arrays and selected Python containers; excludes uncounted object overhead', details)


class DualBoundForceEstimator:
    """Calibrated transformed second-moment sketch, not Pearson correlation.

    Query transformation remains paired with the served estimate across epochs.
    The previous epoch remains serviceable throughout scheduled recalibration.
    """
    def __init__(self, p, k, *, calibration_size=512, epoch_size=None,
                 marginal_lambda=3.0, parallel_lambda=3.0, residual_lambda=3.0,
                 scale_floor=1e-8, epsilon=1e-10):
        self._engine = _Dual(p, k, calibration_size=calibration_size, epoch_size=epoch_size,
            marginal_lambda=marginal_lambda, parallel_lambda=parallel_lambda,
            residual_lambda=residual_lambda, scale_floor=scale_floor, epsilon=epsilon)
        self._failure = None

    @property
    def p(self):
        return self._engine.p

    @property
    def k(self):
        return self._engine.k

    def reset(self):
        try:
            self._engine.reset()
        except BaseException as error:
            self._failure = f'{type(error).__name__}: {error}'
            raise
        self._failure = None
        return self

    def _ensure_healthy(self):
        if self._failure is not None:
            raise FailedStateError('Dual-Bound stream has partially failed; call reset() before continuing')

    def fit(self, X):
        values = matrix(X, self.p)
        self.reset()
        return self.update_batch(values)

    def update(self, x):
        self._ensure_healthy()
        values = row(x, self.p)
        try:
            self._engine._update_trusted(values)
        except BaseException as error:
            self._failure = f'{type(error).__name__}: {error}'
            raise
        return self

    def update_batch(self, X):
        self._ensure_healthy()
        values = matrix(X, self.p)
        try:
            self._engine._update_batch_trusted(values)
        except BaseException as error:
            self._failure = f'{type(error).__name__}: {error}'
            raise
        return self

    def get_normalized_scatter(self):
        self._ensure_healthy()
        return self._engine.get_correlation()

    finalize = get_normalized_scatter

    def get_scale_scatter(self):
        self._ensure_healthy()
        return self._engine.get_scale_scatter()

    def transform(self, x):
        self._ensure_healthy()
        return self._engine.transform(row(x, self.p))

    def get_subspace(self, rank=None):
        self._ensure_healthy()
        return self._engine.get_subspace(rank)

    def get_diagnostics(self):
        details = self._engine.get_diagnostics()
        sketch = self._engine._sketch
        details['sketch_rows_seen'] = 0 if sketch is None else sketch.rows_seen
        details['compressions'] = 0 if sketch is None else sketch.compressions
        details['sketch_next_row'] = 0 if sketch is None else sketch.next_row
        details['failure'] = self._failure
        details['lifecycle_status'] = 'failed' if self._failure else details['phase']
        details['observations_attempted'] = self._engine._attempted_total
        details['observations_committed'] = self._engine._committed_total
        return _diagnostics('Dual-Bound FORCE', 'normalized_calibration_centered_transformed_scatter',
            self._failure is None and details['ready'], self._engine._committed_total, details['state_bytes'],
            'retained numerical array bytes; excludes Python object overhead and epoch history', details)
