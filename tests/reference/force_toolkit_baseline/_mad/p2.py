"""
P² Quantile Estimator — streaming quantile estimation with O(1) memory and O(1) time per update.

Reference: Jain & Chlamtac, "The P² Algorithm for Dynamic Calculation of Quantiles
and Histograms Without Storing Observations", CACM 1985.

Provides both a pure-Python class and Numba @njit functions.
"""

import numpy as np
from numba import njit

# ─────────────────────────────────────────────────────────────
# Numba JIT functions (flat-array interface, no Python objects)
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def p2_init(phi):
    """Allocate and return P² state arrays for quantile *phi*.

    Returns
    -------
    q         : float64[5]  — marker heights
    n         : int64[5]    — marker positions
    n_prime   : float64[5]  — desired marker positions
    count     : int64       — total observations seen
    init_buf  : float64[5]  — buffer for the first 5 observations
    initialized : int64     — 1 once first 5 obs processed, else 0
    """
    if not np.isfinite(phi) or phi <= 0.0 or phi >= 1.0:
        raise ValueError("phi must be finite and strictly between 0 and 1")
    q = np.empty(5, dtype=np.float64)
    n = np.zeros(5, dtype=np.int64)
    n_prime = np.empty(5, dtype=np.float64)
    init_buf = np.empty(5, dtype=np.float64)
    return q, n, n_prime, np.int64(0), init_buf, np.int64(0)


@njit(cache=True)
def p2_update(q, n, n_prime, count, init_buf, initialized, phi, x):
    """Process one observation *x*.  Modifies arrays in-place.

    Returns updated (count, initialized).
    """
    if not np.isfinite(x):
        raise ValueError("P2 observations must be finite")
    count += 1

    # ── Initialization phase: collect first 5 obs ──────────────
    if initialized == 0:
        init_buf[count - 1] = x
        if count < 5:
            return count, initialized

        # Sort the 5-element buffer (insertion sort, tiny)
        for i in range(1, 5):
            key = init_buf[i]
            j = i - 1
            while j >= 0 and init_buf[j] > key:
                init_buf[j + 1] = init_buf[j]
                j -= 1
            init_buf[j + 1] = key

        for i in range(5):
            q[i] = init_buf[i]
            n[i] = i           # 0-indexed positions
        n_prime[0] = 0.0
        n_prime[1] = 2.0 * phi
        n_prime[2] = 4.0 * phi
        n_prime[3] = 2.0 + 2.0 * phi
        n_prime[4] = 4.0
        initialized = 1
        return count, initialized

    # ── Steady-state update ───────────────────────────────────
    # Step 1: find cell k
    if x < q[0]:
        q[0] = x
        k = 0
    elif x >= q[4]:
        q[4] = x
        k = 3   # logically cell 4, but we increment j >= k+1 = 4
    else:
        k = 0
        for i in range(1, 5):
            if x < q[i]:
                k = i - 1
                break

    # Step 2: increment positions for markers above insertion cell
    for j in range(k + 1, 5):
        n[j] += 1

    # Step 3: update desired positions
    total = float(count)
    n_prime[0] = 0.0
    n_prime[1] = (total - 1.0) * phi / 2.0
    n_prime[2] = (total - 1.0) * phi
    n_prime[3] = (total - 1.0) * (1.0 + phi) / 2.0
    n_prime[4] = total - 1.0

    # Step 4: adjust interior markers
    for i in range(1, 4):
        d_i = n_prime[i] - float(n[i])
        if (d_i >= 1.0 and n[i + 1] - n[i] > 1) or \
           (d_i <= -1.0 and n[i - 1] - n[i] < -1):
            d = 1 if d_i > 0 else -1
            d_f = float(d)

            # Parabolic (P²) adjustment
            ni1_ni = float(n[i + 1] - n[i])
            ni_ni_1 = float(n[i] - n[i - 1])
            denom = float(n[i + 1] - n[i - 1])

            q_new = q[i] + (d_f / denom) * (
                (ni_ni_1 + d_f) * (q[i + 1] - q[i]) / ni1_ni
                + (ni1_ni - d_f) * (q[i] - q[i - 1]) / ni_ni_1
            )

            # Monotonicity check
            if q[i - 1] < q_new < q[i + 1]:
                q[i] = q_new
            else:
                # Linear fallback
                idx = i + d
                q[i] = q[i] + d_f * (q[idx] - q[i]) / float(n[idx] - n[i])
            n[i] += d

    return count, initialized


@njit(cache=True)
def p2_estimate(q, initialized):
    """Return current quantile estimate (q[2]), or NaN if not yet initialized."""
    if initialized == 0:
        return np.nan
    return q[2]


# ─────────────────────────────────────────────────────────────
# Pure-Python class (wraps the Numba functions)
# ─────────────────────────────────────────────────────────────

class P2Estimator:
    """Streaming quantile estimator using the P² algorithm.

    Parameters
    ----------
    quantile : float in (0, 1)
        The target quantile to track.
    """

    def __init__(self, quantile: float):
        quantile = float(quantile)
        if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be finite and strictly between 0 and 1")
        self.phi = quantile
        self.q, self.n, self.n_prime, self._count, self._buf, self._init = p2_init(quantile)

    def update(self, x: float) -> None:
        """Process one observation."""
        x = float(x)
        if not np.isfinite(x):
            raise ValueError("P2 observations must be finite")
        self._count, self._init = p2_update(
            self.q, self.n, self.n_prime, self._count, self._buf, self._init, self.phi, x
        )

    def estimate(self) -> float:
        """Return the current quantile estimate."""
        return p2_estimate(self.q, self._init)

    @property
    def count(self) -> int:
        return int(self._count)

    @property
    def initialized(self) -> bool:
        return self._init == 1

    def reset(self, seed_values=None):
        """Reset to initial state.  Optionally seed with values."""
        self.q, self.n, self.n_prime, self._count, self._buf, self._init = p2_init(self.phi)
        if seed_values is not None:
            for v in seed_values:
                self.update(v)

    def provisional_estimate(self) -> float:
        """Return the exact bootstrap quantile before five observations.

        The public :meth:`estimate` method deliberately remains ``NaN`` until
        P² initializes. This internal bootstrap accessor supplies the
        previous-prefix median required by MAD-FORCE Equation (2).
        """
        if self.initialized:
            return self.estimate()
        if self.count == 0:
            return np.nan
        return float(np.quantile(self._buf[: self.count], self.phi))

    def validate_state(self) -> None:
        """Raise if a marker invariant is violated."""
        if self.count < 0:
            raise RuntimeError("P2 observation count is negative")
        if not self.initialized:
            if self.count >= 5:
                raise RuntimeError("P2 must be initialized after five observations")
            return
        if self.count < 5:
            raise RuntimeError("initialized P2 state has fewer than five observations")
        if np.any(np.diff(self.q) < 0.0):
            raise RuntimeError("P2 marker heights are not ordered")
        if np.any(np.diff(self.n) <= 0):
            raise RuntimeError("P2 marker positions are not strictly ordered")
        if np.any(np.diff(self.n_prime) < 0.0):
            raise RuntimeError("P2 desired positions are not ordered")
        if self.n[0] != 0 or self.n[-1] != self.count - 1:
            raise RuntimeError("P2 endpoint marker positions are inconsistent")
        expected = np.array(
            [
                0.0,
                (self.count - 1.0) * self.phi / 2.0,
                (self.count - 1.0) * self.phi,
                (self.count - 1.0) * (1.0 + self.phi) / 2.0,
                self.count - 1.0,
            ],
            dtype=np.float64,
        )
        if not np.allclose(self.n_prime, expected, rtol=0.0, atol=1e-12):
            raise RuntimeError("P2 desired positions do not match the standard recurrence")
