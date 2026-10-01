"""Frozen query interventions; observed query flags, never labels, activate them.

Instrument receiver proposals and key reads without changing model parameters.
These interventions measure learned reliance, not refitted capability or work.
"""
from contextlib import contextmanager

import torch


CHANNELS = {
    'full': (True, True, True),
    'no_context': (True, True, False),
    'keys_only': (True, False, False),
    'values_only': (False, True, False),
    'context_only': (False, False, True),
    'no_history': (False, False, False),
}


@contextmanager
def query_interventions(model, condition):
    if condition not in CHANNELS:
        raise ValueError('A declared frozen information intervention is required')
    keys, values, context = CHANNELS[condition]
    switch = dict(query=False)
    hooks, proposals = [], []
    try:
        for layer in model.units:
            for head in layer:
                for source in head:
                    for unit in source:
                        original = unit.propose
                        had_override = 'propose' in unit.__dict__
                        def propose(x, memory, previous, arrival, original=original):
                            if switch['query'] and not values:
                                memory, previous = torch.zeros_like(memory), None
                            return original(x, memory, previous, arrival)
                        proposals.append((unit, original, had_override))
                        unit.propose = propose
                        def key_hook(module, inputs, output):
                            return torch.zeros_like(output) if switch['query'] and not keys else output
                        hooks.append(unit.key_read.register_forward_hook(key_hook))
        yield switch, context
    finally:
        for hook in hooks:
            hook.remove()
        for unit, original, had_override in proposals:
            if had_override:
                unit.propose = original
            else:
                del unit.__dict__['propose']


@torch.no_grad()
def predict_channels(model, row, condition):
    if model.training:
        raise ValueError('Information probes require frozen inference mode')
    state = model.new_state()
    outputs, targets = [], []
    with query_interventions(model, condition) as (switch, keep_context):
        for event in row:
            # Query presence is observed input. Target values are read only
            # after the prediction, and never enter a model or hook decision.
            switch['query'] = bool(event.mark[1])
            if switch['query'] and not keep_context:
                state.contexts.pop(event.source, None)
            logits, _ = model.consume_event(event.source, event.time, event.mark, state)
            if switch['query']:
                if event.target is None:
                    raise ValueError('Scored query must have an external target')
                outputs.append(logits)
                targets.append(event.target)
    return torch.stack(outputs), torch.tensor(targets), state
