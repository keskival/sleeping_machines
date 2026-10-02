"""DVS native fit with the diagnostic pathwise race credit (THEORY §399, depth plan step 2).

Wraps the clock-calibrated driver and switches AddressedEventHeads to its declared credit='pathwise' mode:
identical forward clocks and winner content, but only the realized winner's interior clock/value derivatives and
no losing-route teacher.  At depth 4 it tests whether the counterfactual surrogate is what limits deep training.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402


def make_model(a, fast=True):
    model = C.make_model(a, fast); model.credit = 'pathwise'
    return model


def sources():
    return {**C.sources(), 'experiments/dvs_pathwise_benchmark.py': N.sha(ROOT / 'experiments/dvs_pathwise_benchmark.py')}


@contextmanager
def activate():
    old = N.make_model, N.parser, N.sources
    N.make_model, N.parser, N.sources = make_model, C.parser, sources
    try:
        yield
    finally:
        N.make_model, N.parser, N.sources = old


if __name__ == '__main__':
    with activate():
        N.run(C.parser().parse_args())
