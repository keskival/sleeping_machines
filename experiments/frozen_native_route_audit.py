"""Guarded checkpoint-only route/state replay; no optimizer or fitted benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from native_event_contracts import predict
from native_event_tasks import episodes
from sleeping_machines.addressed_event_heads import AddressedEventHeads


def weight_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def replay(model, row, node, seed, forced=None, gradients=False):
    original = model.race; trace = {}; count = 0
    def race(scores, values=None):
        nonlocal count
        index = count; count += 1
        value, delay, winner = original(scores, values)
        if index == node:
            trace.update(scores=scores.detach().clone(), values=values.detach().clone(),
                         winner=int(winner), delay=float(delay.detach()))
            if gradients:
                value.retain_grad(); trace['live_value'] = value
            if forced is not None:
                # Preserve the sampled delay and RNG consumption. Return the
                # forced winner so consume_event commits THAT candidate memory.
                value = values[forced]
                winner = torch.tensor(forced, device=scores.device)
        return value, delay, winner
    model.race = race
    try:
        model.zero_grad(set_to_none=True); model.train()
        torch.manual_seed(seed)
        with torch.set_grad_enabled(gradients):
            logits, target, state, _ = predict(model, row)
            loss = F.cross_entropy(logits, target, reduction='sum')
            if gradients:
                loss.backward()
                trace['error_value'] = trace['live_value'].grad.detach().clone()
        trace.pop('live_value', None)
        trace.update(loss=float(loss.detach()), races=count,
                     commits=state.selected_updates, rng=torch.get_rng_state().clone())
        return trace
    finally:
        model.race = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True); parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    output = ROOT / 'experiments/results/diagnostics' / (args.tag + '.json')
    if output.exists():
        raise ValueError('Preserve completed audit; use a new tag')
    checkpoint = ROOT / args.checkpoint
    saved = torch.load(checkpoint, weights_only=False)
    source_result = saved['result']; a = source_result['args']
    if source_result['status'] != 'completed':
        raise ValueError('Completed checkpoint required')
    for name, sha in source_result['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen model lineage changed: ' + name)
    torch.set_num_threads(1); started = time.monotonic()
    with torch.random.fork_rng():
        model = AddressedEventHeads(sources=a['sources'], payload=a['payload'], depth=a['depth'],
            heads=a['heads'], pool=a['pool'], classes=4 if a['task'] == 'order' else 2, credit=a['credit'])
        model.load_state_dict(saved['best_state']); before = weight_hash(model)
        row = episodes(a['task'], a['sources'], a['sources'], 2201)[0]
        # Predeclared one population, all4 events of address0, shallow/middle/deep
        # blocks, head0, common continuation seed. No best-probe selection.
        events = [index for index, event in enumerate(row) if event.source == 0]
        nodes = [(event, depth, (event*a['depth']+depth)*a['heads'])
                 for event in events for depth in (0, a['depth']//2, a['depth']-1)]
        probes = []
        for event, depth, node in nodes:
            base = replay(model, row, node, 619, gradients=True)
            branches = [replay(model, row, node, 619, forced=i) for i in range(a['pool'])]
            torch.testing.assert_close(torch.tensor(branches[base['winner']]['loss']), torch.tensor(base['loss']), rtol=0, atol=0)
            for branch in branches:
                assert branch['races'] == base['races']
                assert branch['commits'] == base['commits']
                assert torch.equal(branch['rng'], base['rng'])
                assert branch['delay'] == base['delay']
            rates = base['scores'].double().exp()
            delay_fraction = (base['delay']-.001)/.010
            minimum_time = delay_fraction/(1-delay_fraction)
            values = base['values']; error = base['error_value']
            direction = ((values-values.mean(0)) @ error).double()
            teacher = rates*minimum_time*direction
            teacher[base['winner']] -= teacher.sum()
            losses = torch.tensor([b['loss'] for b in branches], dtype=torch.float64)
            # Joint-race boundary reference: actual scalar branch losses replace
            # endpoint value linearization. Winner clock interior is excluded
            # from BOTH sides. Full expectation/suffix-time credit is unaudited.
            reference = rates*minimum_time*(losses-losses.mean())
            reference[base['winner']] -= reference.sum()
            denom = float(teacher.norm()*reference.norm())
            cosine = float(teacher@reference)/denom if denom > 1e-14 else None
            probes.append(dict(event_index=event, depth=depth, head=0, node=node,
                winner=base['winner'], loss=base['loss'], branch_losses=losses.tolist(),
                local_teacher=teacher.tolist(), conditional_boundary_reference=reference.tolist(),
                cosine=cosine, minimum_time=minimum_time, delay=base['delay'],
                includes_candidate_memory_commit=True, common_rng_and_delay_verified=True,
                races_per_replay=base['races']))
        assert before == weight_hash(model)
    available = [p['cosine'] for p in probes if p['cosine'] is not None]
    result = dict(status='completed', args=vars(args), source_sha256={
        **source_result['source_sha256'],
        'experiments/frozen_native_route_audit.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_checkpoint_tag=a['tag'], parameter_weights_preserved=True,
        torch_rng_preserved=True, probes=probes,
        summary=dict(probes=len(probes), nonzero_comparisons=len(available),
            opposed=sum(c < 0 for c in available), mean_cosine=sum(available)/len(available) if available else None),
        accounting=dict(forward_replays=len(probes)*(a['pool']+1), backward_replays=len(probes),
            total_races=len(probes)*(a['pool']+1)*probes[0]['races_per_replay'],
            scope='Actual replay/route counts and whole audit wall/RSS; arithmetic FLOPs uninstrumented and not claimed free'),
        wall_s=time.monotonic()-started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen one-development-population, one-continuation-seed state-aware conditional boundary diagnosis; not expected sequence/time gradient or refitted quality',
        limitations=['Winner clock interior excluded from both comparisons', 'Downstream hard-route/time expectation not integrated',
                     'One address and one population; no population/seed inference'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
