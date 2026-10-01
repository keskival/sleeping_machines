"""Compact C1 temporal integration using event-driven exponential moments.

Reference autograd implementation, not a fused scheduler. Event order and
membership are discrete; zero kernel and slope at either boundary make these
membership changes differentiable for a fixed finite input event set. Creating
or deleting nonzero upstream spikes is a separate credit problem.
"""
from collections import deque
import math

import torch


def temporal_window(times, values, queries, width, decay):
    """Return sum_i v_i K(query-t_i), K=(a/H)^2(1-a/H)^2 exp(-beta*a).

    Sorted one-dimensional times/queries; values [events, payload]. Positive
    scalar width H and nonnegative scalar decay beta, all same dtype/device.
    Each input enters once and expires once. Five moment vectors advance only
    at arrivals and queries; no silent-time ticks or event-by-query tensor.
    """
    if times.ndim != 1 or queries.ndim != 1 or values.ndim != 2 or len(values) != len(times):
        raise ValueError('Sorted scalar times/queries and event-by-payload values required')
    for tensor in (times, queries, width, decay):
        if tensor.dtype != values.dtype or tensor.device != values.device:
            raise ValueError('All inputs must share floating dtype and device')
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Finite inputs required')
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError('Finite floating values required')
    if width.ndim or decay.ndim or not bool(width > 0) or not bool(decay >= 0):
        raise ValueError('Scalar positive width and nonnegative decay required')
    if bool((times[1:] < times[:-1]).any()) or bool((queries[1:] < queries[:-1]).any()):
        raise ValueError('Input arrivals and queries must be nondecreasing')
    # Anchor keeps mathematically zero derivatives connected for empty windows.
    zero = values.sum(0)*0 + (times.sum()+queries.sum()+width+decay)*0
    moments = [zero for _ in range(5)]
    live = deque()
    last = None
    cursor = 0
    outputs = []

    def advance(now):
        nonlocal moments, last
        if last is not None:
            gap = now-last
            factor = torch.exp(-decay*gap)
            moments = [factor*sum(math.comb(k,j)*gap.pow(k-j)*moments[j]
                for j in range(k+1)) for k in range(5)]
        last = now
        # Lazy expiry at the next addressed read/arrival; no autonomous scan.
        while live and bool((now-times[live[0]]) >= width):
            index = live.popleft()
            age = now-times[index]
            contribution = torch.exp(-decay*age)*values[index]
            moments = [m-contribution*age.pow(k) for k,m in enumerate(moments)]

    for query in queries:
        while cursor < len(times) and bool(times[cursor] <= query):
            advance(times[cursor])
            moments[0] = moments[0]+values[cursor]
            live.append(cursor)
            cursor += 1
        advance(query)
        outputs.append(moments[2]/width.square()-2*moments[3]/width.pow(3)+moments[4]/width.pow(4))
    return torch.stack(outputs) if outputs else zero.expand(0,values.shape[1])
