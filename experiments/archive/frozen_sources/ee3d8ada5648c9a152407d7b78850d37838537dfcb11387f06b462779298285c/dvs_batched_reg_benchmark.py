"""Episode-batched DVS training with regularization for the subject-disjoint generalization gap (THEORY §406).

Every native DVS fit shows a 20-25 point fit/development gap (fit-subset 81-84% versus dev 57-66%).  The strong
controls are regularized (RBF C, kernel controls); the native fits had only epoch selection.  Options on the batched
driver (exact batched training, §405):
  --weight-decay w   decoupled weight decay (AdamW) on all model parameters;
  --input-noise s    training-only Gaussian noise (std s) on the normalized packet contents (query flag untouched),
                     drawn from a dedicated generator seeded by (seed, epoch, window) so race noise is unchanged;
  --bins 4           coarse 250 ms packets through the AWS coarse loader (aws_coarse_native.load).
Evaluation is unchanged (no noise).
"""
from contextlib import contextmanager
import functools
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import aws_coarse_native as K  # noqa: E402
import dvs_batched_le_benchmark as BL  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402


def parser():
    p = BL.parser(); p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--input-noise', type=float, default=0.); p.add_argument('--bins', type=int, choices=(4, 20), default=20)
    return p


def noisy(rows, std, seed):
    if std <= 0:
        return rows
    g = np.random.default_rng(seed); out = []
    for r in rows:
        events = []
        for t, c in r['events']:
            c = np.asarray(c, dtype=np.float32).copy()
            if c[-1] == 0:                                   # packet content only; the query event is unchanged
                c[:-1] += g.normal(0, std, len(c) - 1).astype(np.float32)
            events.append((t, c))
        out.append(dict(r, events=events))
    return out


def train_window(model, optimizer, rows, a, epoch, trace=False):
    window_seed = (a.seed * 1_000_003 + epoch * 10_007 + rows[0]['index']) % 2 ** 32
    return BL.train_window(model, optimizer, noisy(rows, a.input_noise, window_seed), a, epoch, trace)


def load(a):
    return K.load(a) if a.bins != 20 else N_BASE_LOAD(a)


N_BASE_LOAD = N.load


def sources():
    return {**BL.sources(), **K.sources(), 'experiments/dvs_batched_reg_benchmark.py': N.sha(ROOT / 'experiments/dvs_batched_reg_benchmark.py')}


@contextmanager
def activate(a):
    old = N.make_model, N.parser, N.sources, N.train_window, N.evaluate, N.load, torch.optim.Adam
    N.make_model, N.parser, N.sources, N.train_window, N.evaluate = BL.make_model, parser, sources, train_window, BL.evaluate
    N.load = load
    if a.weight_decay > 0:   # N.run builds torch.optim.Adam(model.parameters(), lr=...); decoupled decay instead
        torch.optim.Adam = functools.partial(torch.optim.AdamW, weight_decay=a.weight_decay)
    try:
        yield
    finally:
        N.make_model, N.parser, N.sources, N.train_window, N.evaluate, N.load, torch.optim.Adam = old


if __name__ == '__main__':
    import json
    args = parser().parse_args()
    with activate(args):
        N.run(args)
    out = ROOT / 'experiments/results/dvs_native' / (args.tag + '.json')
    result = json.loads(out.read_text())
    if result.get('status') == 'completed':
        result['regularization'] = dict(weight_decay=args.weight_decay, input_noise=args.input_noise, bins=args.bins,
                                        optimizer='AdamW (decoupled)' if args.weight_decay > 0 else 'Adam')
        out.write_text(json.dumps(result, indent=2) + '\n')
