class NotFittedError(RuntimeError):
    """No estimate is available yet (calibration alone is not an estimate)."""


class FailedStateError(RuntimeError):
    """Core processing partially mutated a stream; reset before continuing."""
