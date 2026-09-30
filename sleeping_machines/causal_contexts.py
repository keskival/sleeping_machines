"""Contexts for predicting x[t], measurable from x[:t] only.

The old E63 partial-word context reset on x[t] == space. That is target
leakage. Keep the historical implementation available for audit; corrected
experiments must use this function and rebuild their training tables.
"""
import numpy as np


def word_codes_before(x, alphabet=27, max_length=10):
    x = np.asarray(x, dtype=np.int64)
    n = len(x)
    # Shift the boundary detector: the current target never decides a reset.
    last = np.r_[-1, np.maximum.accumulate(np.where(x[:-1] == 0,
                                                   np.arange(max(n-1, 0)), -1))] if n else np.empty(0, int)
    length = np.minimum(np.arange(n)-last-1, max_length)
    code = np.zeros(n, dtype=np.int64)
    for j in range(1, max_length+1):
        if j < n:
            code[j:] += x[:-j]*(alphabet**(j-1))*(length[j:] >= j)
    return code+(alphabet**max_length)*(length+1)
