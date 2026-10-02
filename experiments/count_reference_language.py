"""Count-model information references for the native character-language protocol.

Purpose: calibrate what a development bpc on the shared protocol means.  The
integrated/native language fits use text8[0:N] for fitting and score every
position of text8[90,000,000:90,008,192] from an empty state.  This script scores
classical smoothed context-count predictors on exactly those windows (same
27-symbol alphabet, same 8,191 scored targets, same development hash).

These are reference predictors, not proposed architecture and not dense
Transformer/LSTM controls.  They answer one narrow question: how much of the
predictable structure that is visible from N fitting characters do our fitted
models currently capture, compared with closed-form counting?

Variants (all causal):
  frozen   counts come only from the fitting window; the development prefix is
           used only as conditioning context.
  adaptive counts additionally include the development prefix seen so far
           (prequential, like a compressor).  Our models also carry persistent
           state through the development stream, but do not update weights, so
           neither variant is an exact information match; both are reported.

Smoothers: Witten-Bell interpolation (PPM-C-like), interpolated absolute
discounting, and interpolated Kneser-Ney (continuation counts for lower
orders; frozen only).  Order 0 interpolates with the uniform distribution.
Counting work is reported as a raw count of table updates and lookups; it is
not converted to FLOPs here.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np



def text_slice(start, n):
    """Same mapping as e120_shared_tasks.text_slice (space -> 0, a..z -> 1..26)."""
    with (Path(__file__).resolve().parents[1] / 'data/text8/text8').open('rb') as f:
        f.seek(start)
        raw = np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)
    if len(raw) != n or not np.all((raw == 32) | ((raw >= 97) & (raw <= 122))):
        raise ValueError('Missing or invalid text8 data')
    return np.where(raw == 32, 0, raw - 96)

A = 27
DEV_START, DEV_LEN = 90_000_000, 8192
DEV_SHA = '65efabd8ceea'  # prefix of development_data_sha256 in the native/episodic results


class Counts:
    """Context tables for orders 0..K: ctx code -> length-A count vector."""

    def __init__(self, K):
        self.K = K
        self.t = [defaultdict(lambda: np.zeros(A, np.float64)) for _ in range(K + 1)]

    def add_sequence(self, x):
        x = np.asarray(x, np.int64)
        for k in range(self.K + 1):
            if len(x) <= k:
                continue
            code = np.zeros(len(x) - k, np.int64)
            for j in range(k):
                code = code * A + x[j:len(x) - k + j]
            pair = code * A + x[k:]
            u, c = np.unique(pair, return_counts=True)
            tab = self.t[k]
            for p, n in zip(u.tolist(), c.tolist()):
                tab[p // A][p % A] += n

    def add(self, hist, y):
        for k in range(self.K + 1):
            if len(hist) >= k:
                self.t[k][ctx_code(hist, k)][y] += 1


def ctx_code(hist, k):
    c = 0
    for s in hist[len(hist) - k:] if k else ():
        c = c * A + int(s)
    return c


def continuation_tables(fit, K):
    """Kneser-Ney lower-order tables: N1+(. h c) for orders 0..K-1."""
    x = np.asarray(fit, np.int64)
    out = []
    for k in range(K):
        n = k + 2  # (prefix symbol, h of length k, c)
        if len(x) < n:
            out.append({})
            continue
        code = np.zeros(len(x) - n + 1, np.int64)
        for j in range(n):
            code = code * A + x[j:len(x) - n + 1 + j]
        u = np.unique(code)
        hc = u % (A ** (k + 1))  # drop the prefix symbol
        v, c = np.unique(hc, return_counts=True)
        tab = defaultdict(lambda: np.zeros(A, np.float64))
        for p, m in zip(v.tolist(), c.tolist()):
            tab[p // A][p % A] += m
        out.append(tab)
    return out


def predict(tables, hist, K, method, D=0.75, kn=None):
    p = np.full(A, 1.0 / A)
    for k in range(min(K, len(hist)) + 1):
        h = ctx_code(hist, k)
        if kn is not None and k < K and k < len(hist) and kn[k]:
            c = kn[k].get(h)  # continuation counts for every non-highest order
        else:
            c = tables.t[k].get(h)
        if c is None:
            continue
        n = c.sum()
        if n <= 0:
            continue
        T = float((c > 0).sum())
        if method == 'wb':
            p = (c + T * p) / (n + T)
        else:  # interpolated absolute discounting / Kneser-Ney
            p = np.maximum(c - D, 0.0) / n + (D * T / n) * p
    return p


def score(fit, dev, K, method, adaptive, D=0.75):
    tables = Counts(K)
    tables.add_sequence(fit)
    kn = continuation_tables(fit, K) if method == 'kn' else None
    total, lookups = 0.0, 0
    for i in range(1, len(dev)):
        hist = dev[max(0, i - K):i]
        p = predict(tables, hist, K, method, D, kn)
        total -= math.log2(p[dev[i]])
        lookups += min(K, len(hist)) + 1
        if adaptive:
            tables.add(hist, dev[i])
    return total / (len(dev) - 1), lookups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fits', default='2048,8192,32768,131072,1048576')
    ap.add_argument('--orders', default='0,1,2,3,4,5,6,8')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    dev = text_slice(DEV_START, DEV_LEN).tolist()
    sha = hashlib.sha256(np.asarray(dev, np.uint8).tobytes()).hexdigest()
    if not sha.startswith(DEV_SHA):
        raise SystemExit(f'development hash mismatch {sha}')
    t0 = time.time()
    rows = []
    for N in [int(v) for v in a.fits.split(',')]:
        fit = text_slice(0, N).tolist()
        for K in [int(v) for v in a.orders.split(',')]:
            for method, adaptive in (('wb', False), ('ad', False), ('kn', False), ('wb', True), ('ad', True)):
                bpc, lookups = score(fit, dev, K, method, adaptive)
                rows.append(dict(fit=N, order=K, method=method, adaptive=adaptive, bpc=bpc,
                                 table_updates=N * (K + 1) + (DEV_LEN - 1) * (K + 1) * adaptive,
                                 lookups=lookups))
                print(json.dumps(rows[-1]), flush=True)
    result = dict(
        status='completed', kind='count_reference_language',
        protocol=dict(fitting_prefixes=[0, 'N'], development=[DEV_START, DEV_START + DEV_LEN],
                      scored_targets=DEV_LEN - 1, tokenizer='27-character text8',
                      state='development scored from an empty context, as native_language_helpers.evaluate'),
        development_data_sha256=sha, hyperparameters=dict(discount=0.75, base='uniform over 27'),
        scope=('Closed-form smoothed context counts; reference predictors, not proposed architecture and not '
               'neural controls. Order is a hyperparameter: a minimum over orders is dev-selected.'),
        rows=rows, wall_seconds=time.time() - t0,
        hardware=dict(device='cpu', threads=1, platform=platform.platform(), python=sys.version.split()[0]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
