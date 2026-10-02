"""DVS native fit with receiver maps tied across each pool (THEORY §398 capacity–exposure test).

Wraps the clock-calibrated driver (same data, controls, protocol and clocks).  Within every (depth, head,
source) pool, units share their input, output, gate, control and key-read maps; keys, clock biases and
timescales (raw rate, frequency) stay private, as does every unit's persistent memory.  Available receivers and
selected updates per event are unchanged.  Parametric dilution (§398.1) is removed: every routed event trains
the shared maps.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402

SHARED = ('input', 'output', 'gate', 'control', 'key_read')


def tie_pools(model):
    for depth in model.units:
        for head in depth:
            for pool in head:
                for unit in list(pool)[1:]:
                    for name in SHARED:
                        setattr(unit, name, getattr(pool[0], name))
    return model


def make_model(a, fast=True):
    return tie_pools(C.make_model(a, fast))


def sources():
    return {**C.sources(), 'experiments/dvs_tied_pool_benchmark.py': N.sha(ROOT / 'experiments/dvs_tied_pool_benchmark.py')}


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
