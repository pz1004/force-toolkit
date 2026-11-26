"""
Two-Stage Streaming MAD (Median Absolute Deviation).

Stage 1: P² tracker for the median of raw observations.
Stage 2: P² tracker for the median of |x − median|.

The one-step delay is critical: Stage 2 uses the median from *before* the
current observation updates Stage 1.
"""

import numpy as np
from numba import njit

from .p2 import P2Estimator, p2_init, p2_update, p2_estimate

MAD_SCALE = 1.4826  # Gaussian consistency factor


class StreamingMAD:
    """Streaming robust location and scale estimator via two-stage P² MAD."""

    def __init__(self):
        self.stage1 = P2Estimator(0.50)  # median of x
        self.stage2 = P2Estimator(0.50)  # median of |x − median|

    def update(self, x: float) -> None:
        """Process one observation with one-step-delayed median."""
        mu_prev = self.stage1.provisional_estimate()
        self.stage1.update(x)
        if not np.isnan(mu_prev):
            dev = abs(x - mu_prev)
            self.stage2.update(dev)

    def location(self) -> float:
        """Current median estimate."""
        return self.stage1.estimate()

    def scale(self) -> float:
        """Robust scale = 1.4826 × MAD."""
        mad = self.stage2.estimate()
        if np.isnan(mad):
            return np.nan
        return MAD_SCALE * mad

    def raw_mad(self) -> float:
        """Raw MAD without Gaussian scaling."""
        return self.stage2.estimate()

    @property
    def initialized(self) -> bool:
        """True once both stages are initialized."""
        return self.stage1.initialized and self.stage2.initialized

    @property
    def deviation_count(self) -> int:
        """Number of Equation (2) absolute deviations processed."""
        return self.stage2.count

    def reset(self, seed_values=None) -> None:
        """Reset both stages. Optionally replay seed values."""
        self.stage1.reset()
        self.stage2.reset()
        if seed_values is not None:
            for x in seed_values:
                self.update(float(x))


# ─────────────────────────────────────────────────────────────
# Numba-compatible functions
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def streaming_mad_init():
    """Allocate state for both P² stages (phi=0.50)."""
    s1_q, s1_n, s1_np, s1_cnt, s1_buf, s1_init = p2_init(0.5)
    s2_q, s2_n, s2_np, s2_cnt, s2_buf, s2_init = p2_init(0.5)
    return (s1_q, s1_n, s1_np, s1_cnt, s1_buf, s1_init,
            s2_q, s2_n, s2_np, s2_cnt, s2_buf, s2_init)


@njit(cache=True)
def streaming_mad_update(s1_q, s1_n, s1_np, s1_cnt, s1_buf, s1_init,
                          s2_q, s2_n, s2_np, s2_cnt, s2_buf, s2_init, x):
    """Update both stages.  Returns updated (s1_cnt, s1_init, s2_cnt, s2_init)."""
    # Read the stale median. During P² bootstrap, Equation (2) still begins at
    # i=2, so use the exact median of the available prefix.
    mu_prev = p2_estimate(s1_q, s1_init)
    if s1_init == 0 and s1_cnt > 0:
        tmp = np.empty(s1_cnt, dtype=np.float64)
        for i in range(s1_cnt):
            tmp[i] = s1_buf[i]
        tmp.sort()
        middle = s1_cnt // 2
        if s1_cnt % 2 == 1:
            mu_prev = tmp[middle]
        else:
            mu_prev = 0.5 * (tmp[middle - 1] + tmp[middle])
    # Update Stage 1
    s1_cnt, s1_init = p2_update(s1_q, s1_n, s1_np, s1_cnt, s1_buf, s1_init, 0.5, x)
    # Update Stage 2 whenever a previous-prefix median is available.
    if not np.isnan(mu_prev):
        dev = abs(x - mu_prev)
        s2_cnt, s2_init = p2_update(s2_q, s2_n, s2_np, s2_cnt, s2_buf, s2_init, 0.5, dev)
    return s1_cnt, s1_init, s2_cnt, s2_init
