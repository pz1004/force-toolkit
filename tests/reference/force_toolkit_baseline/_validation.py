"""Public input validation, completed before any chunk is consumed."""
import numpy as np
from ._force._quantiles import _as_real_array


def matrix(values, p, *, min_rows=0):
    array = _as_real_array(values)
    if array.ndim != 2 or array.shape[1] != p or len(array) < min_rows:
        raise ValueError(f"Expected a matrix with {p} columns and at least {min_rows} rows")
    if not np.isfinite(array).all():
        raise ValueError("Observations must contain only finite values")
    return array


def row(values, p):
    array = _as_real_array(values)
    if array.shape != (p,) or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite real row with {p} features")
    return array
