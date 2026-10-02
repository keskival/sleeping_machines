"""Forked counterfactual replays: resume at the forced race's event instead of replaying from the start (§404).

The factual episode records, at every event boundary, a detached copy of the persistent state, the RNG state and the
race counter.  A replay that forces race r restores the snapshot of r's event and runs only the suffix, with the same
random numbers and the first-time-preserving conditional law.  The result equals the full replay exactly
(contract-tested); cost falls from the whole episode to the suffix.
"""
import copy

import torch
from torch.nn import functional as F

import dvs_local_expectation_benchmark as LE
from sleeping_machines.factorized_race import force_at_first_time


def _snapshot(state):
    s = copy.copy(state)
    s.memories = {k: v.detach().clone() for k, v in state.memories.items()}
    s.arrivals = {k: v.detach().clone() for k, v in state.arrivals.items()}
    s.contexts = {k: (x.detach().clone(), t.detach().clone()) for k, (x, t) in state.contexts.items()}
    s.visited_units = set(state.visited_units)
    return s


def realized(model, row, seed, record):
    """factual episode with the factorized race; returns loss, race count, state and per-event snapshots."""
    counter = [0]; snaps = []

    def race(scores, values=None):
        counter[0] += 1; record.append(scores)
        return LE.PATHWISE(scores, values)
    model.race = race
    try:
        state = model.new_state()
        if model.training and hasattr(model, '_fast_layers'):
            model._fast_layers = {}
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for timestamp, content in row['events']:
                snaps.append((_snapshot(state), torch.get_rng_state(), counter[0]))
                logits, _ = model.consume_event(0, timestamp, content, state)
    finally:
        del model.race
        if hasattr(model, '_fast_layers'):
            model._fast_layers = None
    return F.cross_entropy(logits[None], torch.tensor([row['target']])), counter[0], state, snaps


@torch.no_grad()
def forked(model, row, snaps, force):
    """replay from the event containing race force[0], forcing alternative force[1]; returns (loss, events replayed)."""
    e = max(i for i, (_, _, start) in enumerate(snaps) if start <= force[0])
    state0, rng, start = snaps[e]
    counter = [start]

    def race(scores, values=None):
        r = counter[0]; counter[0] += 1
        if r == force[0]:
            return force_at_first_time(scores, values, force[1])
        return LE.PATHWISE(scores, values)
    model.race = race
    try:
        state = _snapshot(state0)
        if model.training and hasattr(model, '_fast_layers'):
            model._fast_layers = {}
        with torch.random.fork_rng():
            torch.set_rng_state(rng)
            for timestamp, content in row['events'][e:]:
                logits, _ = model.consume_event(0, timestamp, content, state)
    finally:
        del model.race
        if hasattr(model, '_fast_layers'):
            model._fast_layers = None
    return F.cross_entropy(logits[None], torch.tensor([row['target']])), len(row['events']) - e
