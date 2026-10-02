"""Stream-adaptive interpolated Kneser-Ney reference on the shared language protocol (THEORY §393 result).

The count cascade of count_carrying_language with KN continuation statistics in orders 1..K-1, occurrence counts
in order K, fixed absolute discount D in every order, theta -> 0 (interpolated absolute discounting) and a
uniform base: interpolated Kneser-Ney whose statistics update prequentially on the development stream (fit
statistics plus development transitions strictly before each target).  No learned parameters.  Order K and
discount D are selected on the development window, so the selected row is a development-selected reference,
the same convention as the earlier count references.  Per-position bits are stratified by prequential order-4
context evidence for the §393 audit.
"""
import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from e120_shared_tasks import text_slice  # noqa: E402
from sleeping_machines.count_carrying_language import compose, eval_stream_counts  # noqa: E402
from sleeping_machines.count_continuation import eval_stream_continuation_counts  # noqa: E402

STRATA = (('unseen', 0, 0), ('1-2', 1, 2), ('3-9', 3, 9), ('>=10', 10, 10 ** 9))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True)
    p.add_argument('--fits', type=int, nargs='+', default=[2048, 8192, 32768, 131072])
    p.add_argument('--orders', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8])
    p.add_argument('--discounts', type=float, nargs='+', default=[.6, .75, .85, .9, .95])
    a = p.parse_args()
    out = ROOT / 'experiments/results/count_reference' / f'{a.tag}.json'
    if out.exists():
        raise ValueError('preserve prior result')
    started = time.perf_counter(); torch.set_num_threads(1)
    dev = np.array(text_slice(90_000_000, 8192)); y = torch.tensor(dev[1:])
    rows, best = [], {}
    for N in a.fits:
        fit = np.array(text_slice(0, N))
        evidence = eval_stream_counts(fit, dev, 4)[3].sum(-1)[:len(dev) - 1]
        for K in a.orders:
            stats = torch.tensor(eval_stream_continuation_counts(fit, dev, K), dtype=torch.float64)
            log_q = torch.full((len(dev), 27), -math.log(27.), dtype=torch.float64)
            for D in a.discounts:
                lp = compose(log_q, stats, torch.full((K,), D, dtype=torch.float64), torch.full((K,), 1e-9, dtype=torch.float64))
                bits = (-lp[:-1].gather(1, y[:, None])[:, 0] / math.log(2)).numpy()
                row = dict(fit=N, order=K, discount=D, bpc=float(bits.mean()), method='kn_interpolated', adaptive=True,
                           strata={s: float(bits[(evidence >= lo) & (evidence <= hi)].mean()) for s, lo, hi in STRATA})
                rows.append(row)
                if N not in best or row['bpc'] < best[N]['bpc']:
                    best[N] = row
        print(json.dumps(dict(fit=N, selected=best[N])), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(status='completed', kind='stream-adaptive interpolated Kneser-Ney (continuation lower orders)',
        development=[90_000_000, 90_008_192], rows=rows, selected={str(k): v for k, v in best.items()},
        wall_s=time.perf_counter() - started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Development-selected count reference (order and discount chosen on the scored window); no learned '
              'parameters; prequential development statistics; same targets as every language row.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
