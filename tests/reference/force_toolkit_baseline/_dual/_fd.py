"""Small deterministic Frequent Directions primitive."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


class FrequentDirections:
    """A ``2k x p`` Frequent Directions sketch of supplied rows."""

    def __init__(self, p: int, k: int, *, epsilon: float = 1e-10):
        for name, value in (("p", p), ("k", k)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if p <= 0 or k <= 0 or k > p:
            raise ValueError("require p > 0 and 0 < k <= p")
        if (
            isinstance(epsilon, (bool, np.bool_))
            or not isinstance(epsilon, Real)
            or not np.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("epsilon must be a finite positive real number")
        self.p = int(p)
        self.k = int(k)
        self.epsilon = float(epsilon)
        self.B = np.zeros((2 * self.k, self.p), dtype=float)
        self.next_row = 0
        self.rows_seen = 0
        self.compressions = 0

    def update(self, row: np.ndarray) -> None:
        value = np.asarray(row, dtype=float)
        if value.shape != (self.p,) or not np.isfinite(value).all():
            raise ValueError("FD row must be finite with shape (p,)")
        self.B[self.next_row] = value
        self.next_row += 1
        self.rows_seen += 1
        if self.next_row == 2 * self.k:
            self._compress()

    def _compress(self) -> None:
        _, singular_values, right_vectors = np.linalg.svd(self.B, full_matrices=False)
        cutoff = (
            float(singular_values[self.k]) if len(singular_values) > self.k else 0.0
        )
        leading = singular_values[: self.k]
        # Equation (12): sqrt(s_i**2 - s_{k+1}**2). The equivalent ratio
        # expression avoids squaring tiny or huge singular values. At k=p
        # the extended singular value is exactly zero, so retain s_i itself.
        retained = leading.copy()
        if cutoff > 0.0:
            ratio = cutoff / leading
            retained *= np.sqrt(np.maximum((1.0 - ratio) * (1.0 + ratio), 0.0))
        if not np.isfinite(retained).all():
            raise FloatingPointError("FD shrinkage produced non-finite singular values")
        self.B.fill(0.0)
        self.B[: len(retained)] = retained[:, None] * right_vectors[: len(retained)]
        self.next_row = min(self.k, len(retained))
        self.compressions += 1

    def gram(self, denominator: int) -> np.ndarray:
        if denominator <= 0:
            return np.zeros((self.p, self.p), dtype=float)
        return self.B.T @ self.B / int(denominator)

    def basis(self, rank: int | None = None) -> np.ndarray:
        requested = self.k if rank is None else int(rank)
        if requested <= 0:
            raise ValueError("rank must be positive")
        if not np.any(self.B):
            return np.zeros((self.p, 0), dtype=float)
        _, singular_values, right_vectors = np.linalg.svd(self.B, full_matrices=False)
        positive = int(np.sum(singular_values > self.epsilon))
        used = min(requested, positive)
        return right_vectors[:used].T.copy()

    @property
    def state_bytes(self) -> int:
        return int(self.B.nbytes)
