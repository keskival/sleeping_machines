"""Large-data count references and escape statistics (THEORY §§376-381).

Vectorized interpolated Kneser-Ney (fixed D=.75 and modified KN with count-of-count
discounts, Chen & Goodman 1999) over sorted pair tables, counted in chunks so that
90M fitting characters fit under the host guard.  Scores:
  * the E64 control test segment text8[95M:96M] (999,999 targets, stateful from the
    segment start, as e64_lm_baselines.score for the LSTM), and
  * the shared native/integrated development window text8[90M:90M+8192].
Also records the top-level escape mass e_h and responsibility r_y of the next
coarser predictive (THEORY §379) on both segments.

Reference predictors only: no neural model, no tuning on the test segment (modified-KN
discounts come from fitting counts; order is reported per value, not selected).
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

A = 27
ROOT = Path(__file__).resolve().parents[1]


def load():
    raw = np.fromfile(ROOT / 'data/text8/text8', dtype=np.uint8)
    if len(raw) != 100_000_000:
        raise SystemExit('text8 missing or truncated')
    return np.where(raw == 32, 0, raw - 96).astype(np.uint8)


def pair_codes(x, k):
    """code of (k previous characters, next character); previous char is the lowest context digit."""
    n = len(x) - k
    ctx = np.zeros(n, np.int64)
    for j in range(1, k + 1):
        ctx += x[k - j:k - j + n].astype(np.int64) * (A ** (j - 1))
    return ctx * A + x[k:k + n]


def merge(u1, c1, u2, c2):
    u = np.concatenate([u1, u2])
    c = np.concatenate([c1, c2])
    o = np.argsort(u, kind='stable')
    u, c = u[o], c[o]
    first = np.r_[True, u[1:] != u[:-1]]
    idx = np.flatnonzero(first)
    return u[idx], np.add.reduceat(c, idx).astype(np.int64)


def count_pairs(x, k, chunk=8_000_000):
    u = np.zeros(0, np.int64)
    c = np.zeros(0, np.int64)
    for s in range(0, len(x) - k, chunk):
        part = pair_codes(x[s:min(len(x), s + chunk + k)], k)
        pu, pc = np.unique(part, return_counts=True)
        u, c = merge(u, c, pu, pc.astype(np.int64))
    return u, c


def discounts(cnt):
    n = [int((cnt == i).sum()) for i in (1, 2, 3, 4)]
    if min(n) == 0:
        return np.array([0.0, .75, .75, .75])
    Y = n[0] / (n[0] + 2 * n[1])
    D = np.array([0.0, 1 - 2 * Y * n[1] / n[0], 2 - 3 * Y * n[2] / n[1], 3 - 4 * Y * n[3] / n[2]])
    return np.clip(D, 0.0, np.array([0.0, 1.0, 2.0, 3.0]))  # valid range for count-of-count estimates


class Level:
    """Sorted (context, symbol) table with per-context totals and discount mass."""

    def __init__(self, pairs, cnt, D):
        self.pairs, self.cnt = pairs, cnt
        ctx = pairs // A
        self.ctx, first = np.unique(ctx, return_index=True)
        self.n = np.add.reduceat(cnt, first).astype(np.float64)
        self.D = D  # array indexed by min(count,3); D[0]=0
        d = D[np.minimum(cnt, 3)]
        self.mass = np.add.reduceat(d, first)

    def lookup(self, ctx_codes):
        """counts (L, A), totals and discount mass for context codes; -1 or unseen -> zeros."""
        L = len(ctx_codes)
        C = np.zeros((L, A))
        n = np.zeros(L)
        m = np.zeros(L)
        ok = ctx_codes >= 0
        j = np.clip(np.searchsorted(self.ctx, ctx_codes), 0, len(self.ctx) - 1)
        hit = ok & (self.ctx[j] == ctx_codes)
        n[hit] = self.n[j[hit]]
        m[hit] = self.mass[j[hit]]
        for a in range(A):
            q = ctx_codes * A + a
            i = np.clip(np.searchsorted(self.pairs, q), 0, len(self.pairs) - 1)
            h = hit & (self.pairs[i] == q)
            C[h, a] = self.cnt[i[h]]
        return C, n, m


def build(x, K, modified):
    """raw[k]: ordinary counts (used at the top order); cont[k]: KN continuation counts (lower orders)."""
    pairs = [count_pairs(x, k) for k in range(K + 1)]
    fixed = np.array([0.0, .75, .75, .75])
    raw, cont = [], []
    for k in range(K + 1):
        p, c = pairs[k]
        raw.append(Level(p, c, discounts(c) if modified else fixed))
        if k < K:  # distinct order-(k+1) pairs reduced modulo A^(k+1)
            p2, c2 = np.unique(pairs[k + 1][0] % (A ** (k + 1)), return_counts=True)
            c2 = c2.astype(np.int64)
            cont.append(Level(p2, c2, discounts(c2) if modified else fixed))
    return raw, cont


def levels_for(tables, K):
    raw, cont = tables
    return cont[:K] + [raw[K]]


def ctx_codes(stream, k, start, stop):
    """context codes of the k previous characters within `stream` only (cold start: -1)."""
    pos = np.arange(start, stop)
    code = np.zeros(len(pos), np.int64)
    for j in range(1, k + 1):
        code += np.where(pos - j >= 0, stream[np.maximum(pos - j, 0)].astype(np.int64), 0) * (A ** (j - 1))
    code[pos < k] = -1
    return code


def score(levels, K, stream, block=65536):
    """bits/char of stream[1:] given stream[:t]; returns bpc, mean escape e, mean responsibility r, fractions."""
    y = stream[1:].astype(np.int64)
    tot, stats = 0.0, dict(e=[], r=[])
    for s in range(1, len(stream), block):
        t = min(len(stream), s + block)
        p = np.full((t - s, A), 1.0 / A)
        e_top = r_top = None
        for k in range(K + 1):
            C, n, m = levels[k].lookup(ctx_codes(stream, k, s, t))
            seen = n > 0
            Dk = levels[k].D[np.minimum(C, 3).astype(np.int64)]
            disc = np.maximum(C - Dk, 0) / np.maximum(n, 1)[:, None]
            esc = np.where(seen, m / np.maximum(n, 1), 1.0)
            newp = np.where(seen[:, None], disc + esc[:, None] * p, p)
            if k == K:
                yy = y[s - 1:t - 1]
                base = p[np.arange(len(yy)), yy]
                py = newp[np.arange(len(yy)), yy]
                e_top, r_top = esc, esc * base / py
            p = newp
        tot -= np.log2(p[np.arange(t - s), y[s - 1:t - 1]]).sum()
        stats['e'].append(e_top)
        stats['r'].append(r_top)
    e = np.concatenate(stats['e'])
    r = np.concatenate(stats['r'])
    return dict(bpc=tot / (len(stream) - 1), mean_escape=float(e.mean()), mean_responsibility=float(r.mean()),
                frac_escape_lt_05=float((e < .05).mean()), frac_escape_lt_10=float((e < .1).mean()),
                frac_resp_lt_05=float((r < .05).mean()), frac_resp_lt_20=float((r < .2).mean()),
                frac_unseen_top=float((e == 1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fits', default='10000000,90000000')
    ap.add_argument('--K', type=int, default=7)
    ap.add_argument('--orders', default='4,5,6,7')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    x = load()
    test = x[95_000_000:96_000_000]
    dev = x[90_000_000:90_008_192]
    hashes = dict(test=hashlib.sha256(test.tobytes()).hexdigest(), development=hashlib.sha256(dev.tobytes()).hexdigest())
    if not hashes['development'].startswith('65efabd8ceea'):
        raise SystemExit('development hash mismatch')
    rows, t0 = [], time.time()
    for N in [int(v) for v in a.fits.split(',')]:
        for modified in (False, True):
            tb = time.time()
            tables = build(x[:N], a.K, modified)
            sizes = [int(len(L.pairs)) for L in tables[0]]
            for K in [int(v) for v in a.orders.split(',')]:
                for name, seg in (('e64_test_95M_1M', test), ('shared_dev_90M_8192', dev)):
                    r = score(levels_for(tables, K), K, seg)
                    rows.append(dict(fit=N, order=K, method='mkn' if modified else 'kn75', segment=name,
                                     targets=len(seg) - 1, table_pairs_by_order=sizes[:K + 1],
                                     build_seconds=time.time() - tb, **r))
                    print(json.dumps(rows[-1]), flush=True)
            del tables
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        status='completed', kind='count_reference_scale', segment_sha256=hashes, rows=rows,
        scope='Closed-form interpolated KN references; frozen counts of text8[0:N]; segments scored from a cold '
              'context. Responsibility uses the next-coarser KN predictive as the base (THEORY §379 proxy).',
        wall_seconds=time.time() - t0,
        hardware=dict(device='cpu', threads=1, platform=platform.platform(), python=sys.version.split()[0])), indent=1))


if __name__ == '__main__':
    main()
