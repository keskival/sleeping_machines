"""Bounded-score integrated sibling using unchanged full-write local expectation."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_batched_le_benchmark as B
import dvs_clock_calibrated_benchmark as C
import dvs_native_benchmark as N
from sleeping_machines.bridge_addressed_events import BridgeAddressedEvents
from sleeping_machines.bridge_batched_episodes import batched_logits


def parser():
    p = B.parser(); p.add_argument('--score-bridge', type=float, default=.1)
    return p


def make_model(a, fast=True):
    torch.manual_seed(a.seed)
    model = BridgeAddressedEvents(sources=1, content_dim=33, classes=11,
        payload=a.payload, depth=a.depth, heads=a.heads, pool=a.pool, score_bridge=a.score_bridge)
    return C.scale_initial_clocks(model, a.clock_step)


def sources():
    names = ['experiments/dvs_bridge_batched_benchmark.py',
        'sleeping_machines/bridge_addressed_events.py', 'sleeping_machines/bridge_batched_episodes.py',
        'sleeping_machines/bounded_score_sensitivity.py', 'sleeping_machines/factorized_race.py',
        'experiments/theory/106_integrated_bounded_score_learning_contract.md']
    return {**B.sources(), **{n: N.sha(ROOT / n) for n in names}}


@contextmanager
def activate():
    old = B.batched_logits, N.make_model, N.parser, N.sources
    B.batched_logits = batched_logits
    try:
        with B.activate():
            N.make_model, N.parser, N.sources = make_model, parser, sources
            yield
    finally:
        B.batched_logits = old[0]
        N.make_model, N.parser, N.sources = old[1:]


if __name__ == '__main__':
    a = parser().parse_args()
    with activate():
        N.run(a)
    out = ROOT / 'experiments/results/dvs_native' / (a.tag + '.json')
    result = json.loads(out.read_text())
    if result.get('status') == 'completed':
        result['route_credit'] = dict(enabled=a.route_credit, estimator='first-time-preserving local expectation',
            replay_lanes_this_process=B.REPLAYS[0], replay_work='included in traced forward stages')
        result['score_bridge'] = dict(alpha=a.score_bridge, bound=12., extra_key_scoring=0,
            scope='Restricted learning/accounting smoke, not benchmark advantage')
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
