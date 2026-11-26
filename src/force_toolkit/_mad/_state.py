"""Owned contiguous MAD tracker state and Python views over that state.

Adapters never pack or synchronize copies. Bootstrap deliberately calls NumPy's
production quantile interpolation, including its large-value rounding behavior.
"""
import numpy as np

from .p2 import P2Estimator
from .streaming_mad import StreamingMAD


class TrackerState:
    def __init__(self, p):
        self.q = np.zeros((p, 4, 5))
        self.n = np.zeros((p, 4, 5), dtype=np.int64)
        self.desired = np.zeros((p, 4, 5))
        self.buffer = np.zeros((p, 4, 5))
        self.counts = np.zeros((p, 4, 2), dtype=np.int64)
        self.mad, self.low, self.high = [], [], []
        for j in range(p):
            trackers = [ArrayP2(self, j, k, phi)
                        for k, phi in enumerate((.5, .5, .01, .99))]
            mad = StreamingMAD.__new__(StreamingMAD)
            mad.stage1, mad.stage2 = trackers[:2]
            self.mad.append(mad)
            self.low.append(trackers[2])
            self.high.append(trackers[3])

    @property
    def arrays(self):
        return self.q, self.n, self.desired, self.buffer, self.counts

    def bootstrap(self, j, value):
        self.mad[j].update(float(value))
        self.low[j].update(float(value))
        self.high[j].update(float(value))


class ArrayP2(P2Estimator):
    def __init__(self, state, j, k, phi):
        self.phi = phi
        self.q, self.n = state.q[j, k], state.n[j, k]
        self.n_prime, self._buf = state.desired[j, k], state.buffer[j, k]
        self._counters = state.counts[j, k]

    @property
    def _count(self):
        return int(self._counters[0])

    @_count.setter
    def _count(self, value):
        self._counters[0] = value

    @property
    def _init(self):
        return int(self._counters[1])

    @_init.setter
    def _init(self, value):
        self._counters[1] = value

    def reset(self, seed_values=None):
        self._counters.fill(0)
        self.n.fill(0)
        if seed_values is not None:
            for value in seed_values:
                self.update(value)


class RecentRing:
    def __init__(self, values, counters):
        self.values, self.counters = values, counters

    def append(self, value):
        self.values[self.counters[0]] = value
        self.counters[0] = (self.counters[0] + 1) % len(self.values)
        self.counters[1] = min(self.counters[1] + 1, len(self.values))

    def __len__(self):
        return int(self.counters[1])

    def __iter__(self):
        for offset in range(len(self)):
            yield float(self.values[(self.counters[0] - len(self) + offset) % len(self.values)])
