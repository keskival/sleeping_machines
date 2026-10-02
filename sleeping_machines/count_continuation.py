"""Kneser-Ney continuation statistics for the lower cascade orders (THEORY §393 result).

The stratified audit shows frozen Kneser-Ney beating the composition only where the top context is unseen.  KN's
lower orders estimate the law of novel continuations with continuation counts N1+(. h y): the number of distinct
symbols that preceded the pair (h, y), not how often (h, y) occurred.  Here orders 1..K-1 carry those statistics
and order K keeps occurrence counts, as in interpolated KN.  Streams have the count_carrying_language layout
(K, L, A): row p holds the statistics of the context ending at p for predicting p+1.

  fitting stream   leave-one-out: removing position p's own transition decrements N1+(. h y) only when the
                   extended triple (x_{p-k}, h, y) occurs once in the fitting text;
  evaluation       prequential: fit statistics plus evaluation transitions strictly before p; N1+(. h y)
                   increases only when a new extended triple appears.

Order-k statistics need the symbol before the order-k context; at the first positions of a stream that symbol is
absent and the triple is not counted, as in the reference continuation tables.
"""
from collections import defaultdict

import numpy as np

from .count_carrying_language import A, context_codes, fit_stream_counts, eval_stream_counts


def _triples(x, k):
    """per position p (predicting x[p+1]): code of (x[p-k], context of length k ending at p), or -1."""
    x = np.asarray(x, np.int64)
    out = np.full(len(x), -1, np.int64)
    ctx = context_codes(x, k)
    for p in range(k, len(x)):
        out[p] = x[p - k] * A ** k + ctx[p]
    return out


def fit_stream_continuation_counts(fit, K):
    x = np.asarray(fit, np.int64)
    out = fit_stream_counts(x, K)          # order K keeps leave-one-out occurrence counts
    for k in range(1, K):
        ctx, tri = context_codes(x, k), _triples(x, k)
        seen = defaultdict(int)            # (triple, y) -> occurrences
        for p in range(len(x) - 1):
            if tri[p] >= 0:
                seen[(int(tri[p]), int(x[p + 1]))] += 1
        cont = defaultdict(lambda: np.zeros(A, np.float64))
        for (t, y) in seen:
            cont[t % A ** k][y] += 1
        stats = np.zeros((len(x), A), np.float32)
        for p in range(len(x) - 1):
            if ctx[p] < 0:
                continue
            v = cont.get(int(ctx[p]))
            if v is None:
                continue
            v = v.copy()
            if tri[p] >= 0 and seen[(int(tri[p]), int(x[p + 1]))] == 1:
                v[x[p + 1]] -= 1           # leave-one-out: this position was the triple's only occurrence
            stats[p] = v
        out[k - 1] = stats
    return out


def eval_stream_continuation_counts(fit, stream, K):
    f, x = np.asarray(fit, np.int64), np.asarray(stream, np.int64)
    out = eval_stream_counts(f, x, K)      # order K keeps prequential occurrence counts
    for k in range(1, K):
        seen, cont = set(), defaultdict(lambda: np.zeros(A, np.float64))
        ftri = _triples(f, k)
        for p in range(len(f) - 1):
            if ftri[p] >= 0 and (int(ftri[p]), int(f[p + 1])) not in seen:
                seen.add((int(ftri[p]), int(f[p + 1]))); cont[int(ftri[p]) % A ** k][f[p + 1]] += 1
        ctx, tri = context_codes(x, k), _triples(x, k)
        stats = np.zeros((len(x), A), np.float32)
        for p in range(len(x) - 1):
            if ctx[p] >= 0 and int(ctx[p]) in cont:
                stats[p] = cont[int(ctx[p])]
            if tri[p] >= 0 and (int(tri[p]), int(x[p + 1])) not in seen:  # observe only after scoring p
                seen.add((int(tri[p]), int(x[p + 1]))); cont[int(ctx[p])][x[p + 1]] += 1
        out[k - 1] = stats
    return out
