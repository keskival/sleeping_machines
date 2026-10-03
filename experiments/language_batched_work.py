"""Inference work per character of a segment-batched native language configuration (THEORY §§409-412).

Traces the evaluation forward pass (no gradients) of experiments/language_batched_benchmark.window_scores on E64
windows, on a fresh model of the configuration: operator work depends on shapes, not weights.  The batched path
evaluates every unit's proposal and then selects one per race, so this is the emulator's work, an upper bound on a
sparse implementation that would score every key but compute only the selected units' proposals.  Same unit/special
convention as the fitting work (race_language_screen.capture).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import language_batched_benchmark as L  # noqa: E402
from race_language_screen import capture  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402


def inference_work(payload, depth, heads, pool, segment=128, lanes=8, chars=4096):
    torch.set_num_threads(1); torch.manual_seed(0)
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=27, classes=27, payload=payload, depth=depth,
                                            heads=heads, pool=pool)
    text = L.load_text(90_000_000, chars)
    box = {}
    rec = capture(lambda: box.update(score=L.window_scores(model, text, segment, 314159, lanes)))
    scored = box['score'][1]
    events = len(range(0, len(text) - segment - 1, segment // 2)) * segment     # every window position is evaluated
    return dict(payload=payload, depth=depth, heads=heads, pool=pool, segment=segment, scored_targets=scored,
                evaluated_positions=events,
                unit_special_flops_per_evaluated_position=(rec['arithmetic_flops'] + rec['special_function_evaluations']) / events,
                unit_special_flops_per_scored_target=(rec['arithmetic_flops'] + rec['special_function_evaluations']) / scored,
                key_scores_per_position=depth * heads * pool, selected_writes_per_position=depth * heads,
                scope='batched emulator evaluation: all proposals computed; E64 windows overlap, so each scored '
                      'target costs about two evaluated positions')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', default='16,8,2,2;32,4,2,2;32,8,2,2;32,8,2,4')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = ROOT / a.out
    if out.exists():
        raise ValueError('Preserve existing result')
    rows = [inference_work(*[int(v) for v in c.split(',')]) for c in a.configs.split(';')]
    out.write_text(json.dumps(dict(rows=rows), indent=2) + '\n'); print(json.dumps(rows, indent=1))
