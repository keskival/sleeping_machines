"""Coarse 250 ms packets + corrected local-expectation route credit (THEORY §402 with note 92 correction).

Combines the AWS coarse loader (aws_coarse_native.load: four causal 250 ms count packets plus the 1 s query) with
the local-expectation window step (first-time-preserving counterfactual replays for choice credit, factorized race
for payload and common first-time clock credit).  --route-samples 0 runs the same loader with the base
counterfactual teacher, as the matched coarse baseline under identical settings.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import aws_coarse_native as K  # noqa: E402
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_local_expectation_benchmark as LE  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402


def parser():
    p = C.parser(); p.add_argument('--bins', type=int, choices=(4, 20), default=4)
    p.add_argument('--route-samples', type=int, default=4); p.set_defaults(clock_step=.25)
    return p


def sources():
    return {**K.sources(), **LE.sources(),
            'experiments/dvs_coarse_le_benchmark.py': N.sha(ROOT / 'experiments/dvs_coarse_le_benchmark.py')}


@contextmanager
def activate(a):
    old = N.load, N.make_model, N.train_window, N.sources, N.parser
    base_window = N.train_window
    N.load, N.sources, N.parser = K.load, sources, parser
    if a.route_samples > 0:
        N.make_model, N.train_window = LE.make_model, LE.train_window
    else:
        N.make_model = C.make_model
    try:
        yield
    finally:
        N.load, N.make_model, N.train_window, N.sources, N.parser = old


if __name__ == '__main__':
    args = parser().parse_args()
    with activate(args):
        N.run(args)
