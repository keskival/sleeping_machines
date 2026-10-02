"""DVS native fit with exact-pi linearized race credit (THEORY §400).

Wraps the clock-calibrated driver; every race's score credit is pi_i g.(v_i - sum_j pi_j v_j) with exact pi from the
scores instead of the one-sample rate_i * T estimate (same expectation, no estimator noise, same cost).  Forward
races, noise, winners, values and delays are unchanged.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402
from sleeping_machines.exact_pi_race import use_exact_pi_credit  # noqa: E402


def make_model(a, fast=True):
    return use_exact_pi_credit(C.make_model(a, fast))


def sources():
    return {**C.sources(), 'experiments/dvs_exact_pi_benchmark.py': N.sha(ROOT / 'experiments/dvs_exact_pi_benchmark.py'),
            'sleeping_machines/exact_pi_race.py': N.sha(ROOT / 'sleeping_machines/exact_pi_race.py')}


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
