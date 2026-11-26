"""FORCE-family estimators with explicit, distinct statistical targets."""
from ._exceptions import NotFittedError, FailedStateError
from ._mad.diagnostics import DiagnosticDeliveryError
from .api import (
    ForceEstimator, StreamingForceEstimator, MADForceEstimator,
    StreamingMADForceEstimator, DualBoundForceEstimator,
)

__version__ = '0.2.1'
__all__ = ['ForceEstimator', 'StreamingForceEstimator', 'MADForceEstimator',
           'StreamingMADForceEstimator', 'DualBoundForceEstimator',
           'NotFittedError', 'FailedStateError', 'DiagnosticDeliveryError']
