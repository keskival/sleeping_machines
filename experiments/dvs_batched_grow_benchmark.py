"""Episode-batched training of a depth-grown native model (§399 growth by nesting × §405 batching × §402 credit).

The model is the clock-calibrated native core at --depth, grown from a completed shallower --parent (near-identity
appended layers, dvs_grow_depth_benchmark.grow), trained with the batched driver, optionally with all-race replay credit.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_batched_le_benchmark as BL  # noqa: E402
import dvs_grow_depth_benchmark as G  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402


def parser():
    p = BL.parser(); p.add_argument('--parent', required=True)
    return p


def make_model(a, fast=True):
    return G.make_model(a, True)


def sources():
    return {**BL.sources(), **G.sources(), 'experiments/dvs_batched_grow_benchmark.py': N.sha(ROOT / 'experiments/dvs_batched_grow_benchmark.py')}


@contextmanager
def activate():
    old = N.make_model, N.parser, N.sources, N.train_window, N.evaluate
    N.make_model, N.parser, N.sources, N.train_window, N.evaluate = make_model, parser, sources, BL.train_window, BL.evaluate
    try:
        yield
    finally:
        N.make_model, N.parser, N.sources, N.train_window, N.evaluate = old


if __name__ == '__main__':
    with activate():
        N.run(parser().parse_args())
