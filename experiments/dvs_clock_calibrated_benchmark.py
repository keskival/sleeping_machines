"""Matched native gesture fit with packet-scale temporal initialization only."""
from contextlib import contextmanager
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_native_benchmark as N

BASE_FACTORY = N.make_model
BASE_PARSER = N.parser
BASE_SOURCES = N.sources


def parser():
    p = BASE_PARSER()
    p.add_argument('--clock-step', type=float, default=.05)
    return p


def scale_initial_clocks(model, step):
    if not math.isfinite(step) or not .01 <= step <= 1:
        raise ValueError('Bounded observed clock step required')
    scale = 1. / step
    with torch.no_grad():
        rates = [model.transport_rate]
        frequencies = [model.transport_frequency]
        for depth in model.units:
            for head in depth:
                for source in head:
                    for unit in source:
                        rates.append(unit.raw_rate); frequencies.append(unit.frequency)
        for parameter in rates:
            # Include the existing1e-6 floor: actual new rate=scale*old rate.
            desired = (F.softplus(parameter) + 1e-6) * scale - 1e-6
            parameter.copy_(torch.log(torch.expm1(desired)))
        for parameter in frequencies:
            parameter.mul_(scale)
    return model


def make_model(a, fast=True):
    return scale_initial_clocks(BASE_FACTORY(a, fast), a.clock_step)


def sources():
    names = ['experiments/dvs_clock_calibrated_benchmark.py',
        'experiments/dvs_clock_calibrated_contracts.py',
        'experiments/theory/76_packet_scale_temporal_initialization.md']
    return {**BASE_SOURCES(), **{n: N.sha(ROOT / n) for n in names}}


@contextmanager
def activate():
    old = N.make_model, N.parser, N.sources
    N.make_model, N.parser, N.sources = make_model, parser, sources
    try:
        yield
    finally:
        N.make_model, N.parser, N.sources = old


if __name__ == '__main__':
    with activate():
        N.run(parser().parse_args())
