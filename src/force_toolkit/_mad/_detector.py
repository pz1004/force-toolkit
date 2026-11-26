"""ADWIN bucket rings shared by the native controller and Python diagnostics.

Level capacity grows before insertion; it is never a detector window cap.
Each level is a FIFO ring, traversed oldest first from highest to lowest level.
"""
from collections import deque
import math
import numpy as np
from numba import njit

from .adwin import ADWINDetector, _Bucket
from .common import explicit_state_bytes
from .diagnostics import queue_diagnostic


class DetectorState:
    def __init__(self, detectors, levels=4):
        self.p = len(detectors)
        self.capacity = detectors[0].max_buckets + 1
        self.buckets = np.zeros((self.p, levels, self.capacity, 2))
        self.rings = np.zeros((self.p, levels, 2), dtype=np.int64)
        self.meta = np.zeros((self.p, 5), dtype=np.int64)
        self.totals = np.zeros((self.p, 4))
        self.totals[:, 2:] = np.nan
        self.cut_counts = np.zeros((self.p, 4), dtype=np.int64)
        self.cut_values = np.zeros((self.p, 2))
        self.adapters = [ArrayDetector(self, j, detector) for j, detector in enumerate(detectors)]

    @property
    def arrays(self):
        return self.buckets, self.rings, self.meta, self.totals, self.cut_counts, self.cut_values

    def grow(self):
        levels = self.buckets.shape[1]
        buckets = np.zeros((self.p, 2 * levels, self.capacity, 2))
        rings = np.zeros((self.p, 2 * levels, 2), dtype=np.int64)
        buckets[:, :levels], rings[:, :levels] = self.buckets, self.rings
        self.buckets, self.rings = buckets, rings


@njit(cache=True)
def needs_growth(j, buckets, rings, meta):
    # One insertion can propagate one merge through every occupied level.
    for level in range(buckets.shape[1]):
        if rings[j, level, 1] < buckets.shape[2] - 1:
            return False
    return True


@njit(cache=True)
def insert(j, x, buckets, rings, meta, totals):
    capacity = buckets.shape[2]
    slot = (rings[j, 0, 0] + rings[j, 0, 1]) % capacity
    buckets[j, 0, slot, 0], buckets[j, 0, slot, 1] = x, x * x
    rings[j, 0, 1] += 1
    meta[j, 0] += 1
    totals[j, 0] += x
    totals[j, 1] += x * x
    meta[j, 4] = max(meta[j, 4], 1)
    level = 0
    while level < meta[j, 4]:
        if rings[j, level, 1] < capacity:
            level += 1
            continue
        head = rings[j, level, 0]
        second = (head + 1) % capacity
        slot = (rings[j, level + 1, 0] + rings[j, level + 1, 1]) % capacity
        for k in range(2):
            buckets[j, level + 1, slot, k] = buckets[j, level, head, k] + buckets[j, level, second, k]
        rings[j, level, 0] = (head + 2) % capacity
        rings[j, level, 1] -= 2
        rings[j, level + 1, 1] += 1
        meta[j, 4] = max(meta[j, 4], level + 2)


@njit(cache=True)
def drop(j, width, exact, buckets, rings, meta, totals):
    remaining = width
    for level in range(meta[j, 4] - 1, -1, -1):
        while rings[j, level, 1] and remaining > 0:
            size = 1 << level
            if exact and size > remaining:
                raise RuntimeError('Bucket boundary mismatch during ADWIN cut')
            head = rings[j, level, 0]
            remaining -= size
            meta[j, 0] -= size
            totals[j, 0] -= buckets[j, level, head, 0]
            totals[j, 1] -= buckets[j, level, head, 1]
            rings[j, level, 0] = (head + 1) % buckets.shape[2]
            rings[j, level, 1] -= 1
        if remaining <= 0:
            break
    if exact and remaining != 0:
        raise RuntimeError('Failed to remove the requested ADWIN prefix')
    while meta[j, 4] and rings[j, meta[j, 4] - 1, 1] == 0:
        meta[j, 4] -= 1
    return width - remaining


@njit(cache=True)
def trim(j, cap, buckets, rings, meta, totals):
    if cap > 0 and meta[j, 0] > cap:
        meta[j, 3] += drop(j, meta[j, 0] - cap, False, buckets, rings, meta, totals)


@njit(cache=True)
def scan(j, delta, min_subwindow, buckets, rings, meta, totals, cut_counts, cut_values):
    count = meta[j, 0]
    mean = totals[j, 0] / max(count, 1)
    var = max(totals[j, 1] / max(count, 1) - mean * mean, 0.)
    dd = math.log(2.) - math.log(delta) + math.log(math.log2(max(meta[j, 1], 2)) + 1.)
    prefix_count, prefix_total = 0, 0.
    for level in range(meta[j, 4] - 1, -1, -1):
        for offset in range(rings[j, level, 1]):
            slot = (rings[j, level, 0] + offset) % buckets.shape[2]
            prefix_count += 1 << level
            prefix_total += buckets[j, level, slot, 0]
            suffix_count = count - prefix_count
            if prefix_count < min_subwindow or suffix_count < min_subwindow:
                continue
            mu0, mu1 = prefix_total / prefix_count, (totals[j, 0] - prefix_total) / suffix_count
            difference = abs(mu0 - mu1)
            m = 1. / (1. / prefix_count + 1. / suffix_count)
            threshold = math.sqrt(2. * var * dd / m) + (2. * dd) / (3. * m)
            if difference > threshold:
                totals[j, 2], totals[j, 3] = threshold, difference
                meta[j, 2] += 1
                cut_counts[j, :] = meta[j, 1], count, prefix_count, suffix_count
                cut_values[j, :] = difference, threshold
                drop(j, prefix_count, True, buckets, rings, meta, totals)
                return True
    return False


def _integer_field(index):
    return property(lambda self: int(self._state.meta[self._j, index]),
                    lambda self, value: self._state.meta.__setitem__((self._j, index), value))


def _float_field(index, optional=False):
    def read(self):
        value = float(self._state.totals[self._j, index])
        return None if optional and math.isnan(value) else value
    def write(self, value):
        self._state.totals[self._j, index] = np.nan if value is None else value
    return property(read, write)


class ArrayDetector(ADWINDetector):
    _count, _total_time, _alarm_count, _expired_count = [_integer_field(i) for i in range(4)]
    _total, _total_sq = _float_field(0), _float_field(1)
    _last_cut_threshold, _last_cut_diff = _float_field(2, True), _float_field(3, True)

    def __init__(self, state, j, original):
        self._state, self._j = state, j
        for name, value in vars(original).items():
            if name not in {'_rows', '_count', '_total_time', '_alarm_count', '_expired_count',
                            '_total', '_total_sq', '_last_cut_threshold', '_last_cut_diff'}:
                setattr(self, name, value)

    @property
    def _rows(self):
        rows = []
        for level in range(self.n_rows):
            head, size = self._state.rings[self._j, level]
            row = deque()
            for offset in range(size):
                values = self._state.buckets[self._j, level, (head + offset) % self._state.capacity]
                row.append(_Bucket(1 << level, float(values[0]), float(values[1])))
            rows.append(row)
        return rows

    @property
    def n_rows(self):
        return int(self._state.meta[self._j, 4])

    @property
    def bucket_count(self):
        return int(self._state.rings[self._j, :, 1].sum())

    def _insert(self, x):
        state = self._state
        if needs_growth(self._j, state.buckets, state.rings, state.meta):
            state.grow()
        insert(self._j, x, state.buckets, state.rings, state.meta, state.totals)

    def _trim_to_window(self):
        state = self._state
        trim(self._j, self.max_window or 0, state.buckets, state.rings, state.meta, state.totals)

    def _detect_change(self):
        if scan(self._j, self.delta, self.min_subwindow, *self._state.arrays):
            self._record_cut()
            return True
        return False

    def _record_cut(self):
        integers = self._state.cut_counts[self._j]
        values = self._state.cut_values[self._j]
        cut = dict(zip(('time', 'count', 'prefix_count', 'suffix_count'), map(int, integers)))
        cut.update(zip(('diff', 'threshold'), map(float, values)))
        self._cut_history.append(cut)
        queue_diagnostic(self, {'kind': 'detector_cut', **cut})

    def reset(self):
        if self._diagnostic_active:
            raise RuntimeError('update/reset cannot be called from a diagnostic callback')
        for array in self._state.arrays:
            array[self._j].fill(0)
        self._state.totals[self._j, 2:] = np.nan
        self._cut_history.clear()
        self._cut_history.total_count = 0

    @property
    def state_bytes(self):
        return (sum(array[self._j].nbytes for array in self._state.arrays)
                + explicit_state_bytes([self._cut_history, self._diagnostic_pending]))
