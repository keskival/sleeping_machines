"""Grow a trained DVS native model in depth by near-identity layers (THEORY §399, depth plan step 3).

Wraps the clock-calibrated driver.  The model is built at --depth D (> parent depth) with the clock-calibrated
initialization, then every parameter of the parent's layers, embeddings, content, source gate and head is copied
from the parent's selected weights, and the parent's per-layer unit gain is restored.  Appended layers keep identity
channel mixes and closed unit gates (gate bias -20).  Intermediate new layers transport with zero frequency and
negligible decay; the new top layer inherits the parent's top-layer transport, because that transport also aligns
the context carried to the next event.  All of it stays learnable, so each new layer passes its input almost
unchanged:
the grown model starts close to the parent (contract-tested on development gestures) and can use the extra depth
only if credit moves the new layers.  Remaining differences: race noise for the parent's layers is drawn
from a shifted RNG stream (more races per event), as with a different race seed.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402

CLOSED_GATE = -20.


def parser():
    p = C.parser(); p.add_argument('--parent', required=True, help='completed shallower DVS result (.json with .progress.pt)')
    return p


def grow(model, parent_state, parent_depth):
    own = model.state_dict()
    for name, value in parent_state.items():
        parts = name.split('.')
        if parts[0] in ('channel_mix', 'queries', 'units') and int(parts[1]) >= parent_depth:
            continue
        if parts[0] in ('transport_rate', 'transport_frequency'):
            own[name][:parent_depth] = value
        elif name in own and own[name].shape == value.shape:
            own[name] = value
    model.load_state_dict(own)
    gain = .5 / parent_depth ** .5
    with torch.no_grad():
        model.transport_rate[parent_depth:-1] = CLOSED_GATE    # softplus(-20) ~ 2e-9: no decay across intermediate new layers
        model.transport_frequency[parent_depth:-1] = 0.         # no rotation across intermediate new layers
        # the top layer's transport also aligns the carried context for the next event (align(..., depth - 1)),
        # so the new top layer inherits the parent's top-layer transport
        model.transport_rate[-1] = parent_state['transport_rate'][parent_depth - 1]
        model.transport_frequency[-1] = parent_state['transport_frequency'][parent_depth - 1]
        for depth, layer in enumerate(model.units):
            for head in layer:
                for pool in head:
                    for unit in pool:
                        if depth < parent_depth:
                            unit.gain = gain
                        else:
                            unit.gate.bias.fill_(CLOSED_GATE)
    return model


def make_model(a, fast=True):
    parent = json.loads((ROOT / a.parent).read_text())
    pa = parent['args']
    for key in ('payload', 'heads', 'pool', 'clock_step'):
        if pa[key] != getattr(a, key):
            raise ValueError(f'parent {key} differs')
    if a.depth <= pa['depth']:
        raise ValueError('growth requires a deeper model')
    ck = torch.load(ROOT / a.parent.replace('.json', '.progress.pt'), weights_only=False)
    return grow(C.make_model(a, fast), ck['best_state'], pa['depth'])


def sources():
    return {**C.sources(), 'experiments/dvs_grow_depth_benchmark.py': N.sha(ROOT / 'experiments/dvs_grow_depth_benchmark.py')}


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
