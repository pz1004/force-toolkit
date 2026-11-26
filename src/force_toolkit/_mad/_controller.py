"""Sequential MAD controller with explicit, mutation-safe continuations.

No fastmath or parallel reductions. A continuation identifies the *next* phase;
Python must complete that phase exactly once before resuming. Attempted and
committed counters live in arrays so even an escaping native error exposes the
original partial-row boundary.
"""
import math
import numpy as np
from numba import njit

from .p2 import p2_update
from .ter import compute_asymmetric_ter
from . import _kernels, _detector

DONE, BOOTSTRAP, BOUNDS, DETECTOR, HISTORY, FAULT, CUT, GROW = range(8)


@njit(cache=True)
def update_trackers(j, value, q, n, desired, buffer, counts):
    previous = q[j, 0, 2]
    for k in range(4):
        x = abs(value - previous) if k == 1 else value
        phi = .01 if k == 2 else .99 if k == 3 else .5
        count, initialized = p2_update(q[j, k], n[j, k], desired[j, k],
                                      counts[j, k, 0], buffer[j, k],
                                      counts[j, k, 1], phi, x)
        counts[j, k, 0], counts[j, k, 1] = count, initialized


@njit(cache=True)
def tracker_pass(X, cursor, q, n, desired, buffer, counts):
    while cursor[0] < len(X):
        while cursor[1] < X.shape[1]:
            j = cursor[1]
            if counts[j, 0, 1] == 0:
                return BOOTSTRAP
            update_trackers(j, X[cursor[0], j], q, n, desired, buffer, counts)
            cursor[1] += 1
        cursor[0] += 1
        cursor[1] = 0
    return DONE


@njit(cache=True)
def ordinary_bounds(location, scale, lam, lower_mult, upper_mult):
    if not math.isfinite(location) or not math.isfinite(scale):
        raise ValueError('location and scale must be finite')
    if scale < 0.:
        raise ValueError('scale must be non-negative')
    if not (math.isfinite(lam) and math.isfinite(lower_mult) and math.isfinite(upper_mult)):
        raise ValueError('bound factors must be finite')
    if scale == 0. or lam == 0.:
        return location, location, False
    first_low, first_high = lam * lower_mult, lam * upper_mult
    width_low, width_high = first_low * scale, first_high * scale
    lower, upper = location - width_low, location + width_high
    tiny = np.finfo(np.float64).tiny
    exceptional = (not math.isfinite(first_low) or not math.isfinite(first_high)
                   or not math.isfinite(width_low) or not math.isfinite(width_high)
                   or not math.isfinite(lower) or not math.isfinite(upper)
                   or abs(first_low) < tiny or abs(first_high) < tiny
                   or abs(width_low) < tiny or abs(width_high) < tiny)
    return lower, upper, exceptional


@njit(cache=True)
def run(X, cursor, observations, trackers, recent, recent_counts, counters,
        pairs, masks, bound_inputs, options, fault, detectors, detector_options):
    q, n, desired, buffer, counts = trackers
    post_reset, acceptance_total, acceptance_sum, reset_counts = counters
    accepted, reset_step, ready = masks
    lam, ter_max, warm, min_reset, guarded, detection, history = options
    p = X.shape[1]
    while cursor[0] < len(X):
        i, j, phase = cursor
        # Internal test seam: stop at a native phase without bypassing the core.
        if observations[0] == fault[0] and phase == fault[1] and j == fault[2]:
            return FAULT
        if phase == 0:
            observations[0] += 1
            for k in range(p):
                recent[k, recent_counts[k, 0]] = X[i, k]
                recent_counts[k, 0] = (recent_counts[k, 0] + 1) % recent.shape[1]
                recent_counts[k, 1] = min(recent_counts[k, 1] + 1, recent.shape[1])
                accepted[k], reset_step[k] = True, False
            cursor[2] = 1
        elif phase == 1:
            if counts[j, 0, 1] == 0:
                return BOOTSTRAP
            update_trackers(j, X[i, j], q, n, desired, buffer, counts)
            cursor[2] = 2
        elif phase == 2:
            if counts[j, 0, 1] and counts[j, 1, 1]:
                location, scale = q[j, 0, 2], 1.4826 * q[j, 1, 2]
                lower_mult, upper_mult = 1., 1.
                if guarded and math.isfinite(q[j, 2, 2]) and math.isfinite(q[j, 3, 2]):
                    plus, minus = compute_asymmetric_ter(q[j, 2, 2], location, q[j, 3, 2], scale, ter_max)
                    dominant, opposite = max(plus, minus), min(plus, minus)
                    imbalance = dominant / opposite
                    if (dominant >= 1.10 and dominant <= 1.50 and opposite <= 1.05
                            and imbalance >= 1.15 and imbalance <= 1.75):
                        if plus >= minus:
                            upper_mult = plus
                        else:
                            lower_mult = minus
                lower, upper, exceptional = ordinary_bounds(location, scale, lam, lower_mult, upper_mult)
                if exceptional:
                    bound_inputs[:] = location, scale, lam, lower_mult, upper_mult
                    return BOUNDS
                accepted[j] = lower <= X[i, j] <= upper
            cursor[2] = 3
        elif phase == 3:
            acceptance_total[j] += 1
            acceptance_sum[j] += int(accepted[j])
            if reset_counts[j] > 0 and post_reset[j] < min_reset:
                post_reset[j] += 1
            cursor[2] = 4
        elif phase == 4:
            if detection:
                buckets, rings, meta, totals, cut_counts, cut_values = detectors
                delta, cap, min_window, min_subwindow = detector_options
                if _detector.needs_growth(j, buckets, rings, meta):
                    return GROW
                meta[j, 1] = observations[0]
                _detector.insert(j, 1.0 if accepted[j] else 0.0, buckets, rings, meta, totals)
                _detector.trim(j, cap, buckets, rings, meta, totals)
                if (observations[0] > warm and post_reset[j] >= min_reset
                        and meta[j, 0] >= min_window):
                    if _detector.scan(j, delta, min_subwindow, *detectors):
                        cursor[2] = 5
                        return CUT
            cursor[2] = 5
        elif phase == 5:
            cursor[1] += 1
            if cursor[1] == p:
                cursor[1], cursor[2] = 0, 6
            else:
                cursor[2] = 1
        elif phase == 6:
            if history:
                return HISTORY
            cursor[2] = 7
        elif phase == 7:
            if observations[0] > warm:
                for k in range(p):
                    ready[k] = (accepted[k] and not reset_step[k] and counts[k, 0, 1]
                                and counts[k, 1, 1] and post_reset[k] >= min_reset
                                and (not guarded or (counts[k, 2, 1] and counts[k, 3, 1])))
                _kernels.update(X[i], ready, *pairs)
            cursor[2] = 8
        else:
            observations[1] += 1
            cursor[0] += 1
            cursor[1], cursor[2] = 0, 0
    return DONE
