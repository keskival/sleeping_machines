"""Bounded, streaming forward diagnostics of original saved native language weights.

Reports actual wins separately from expected probability mass; seedwise NLL
and its mixture Jensen gap; and two distinct argmax clock policies. All
native contracts and inference must run through the guarded host queue.
--help and standard-library summary checks do not load the model runtime.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments')]
import legacy_batched_driver_binding as B
from routing_measurement_summary import RoutingSummary, mixture_summary


@contextmanager
def race_policy(torch, race_class, policy, heads, depth, summary=None):
    """Change the identity rule at each current history; retain RNG draw order."""
    descriptor = race_class.__dict__['forward']
    original = race_class.forward
    calls = [0]

    def forward(ctx, scores, proposals, noise, forced_alt):
        if bool((forced_alt >= 0).any()):
            raise ValueError('Routing measurements do not combine external forced sites')
        actual_noise, actual_force = noise, forced_alt
        if policy == 'argmax_preserve_first':
            actual_force = scores.argmax(-1)
        elif policy == 'argmax_unit_clocks':
            actual_noise = torch.ones_like(noise)
        elif policy != 'sampled':
            raise ValueError('Unknown routing policy')
        value, delay, winner = original(ctx, scores, proposals, actual_noise, actual_force)
        index = calls[0]
        calls[0] += 1
        if summary is not None:
            rates = scores.double().exp()
            pi = rates/rates.sum(-1, keepdim=True)
            maximum = pi.max(-1).values
            entropy = -(pi*pi.clamp_min(1e-300).log2()).sum(-1)
            first = (actual_noise[None, :]/rates).min(-1).values
            sensitivity = .010*first/(1+first).square()
            summary.add((index//heads) % depth, index % heads,
                        torch.bincount(winner, minlength=summary.pool).tolist(),
                        pi.sum(0).tolist(), float(maximum.sum()), float(entropy.sum()),
                        int((maximum > .9).sum()), len(scores),
                        boundary_keys=int((scores.abs() >= 12).sum()),
                        clock_sensitivity_sum=float(sensitivity.sum()),
                        weak_clock_races=int((sensitivity < .000025).sum()))
        return value, delay, winner

    race_class.forward = staticmethod(forward)
    try:
        yield calls
    finally:
        race_class.forward = descriptor


def native_contracts(torch, np, batch, language, factory):
    checks = []
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(141)
        model = factory(sources=1, content_dim=27, classes=27, payload=4,
                        depth=2, heads=2, pool=2).double().eval()
        text = np.arange(15, dtype=np.uint8) % 27
        rows = language.rows_of(text, [0, 5], 4)
        baseline = batch.batched_logits(model, rows, 314159, all_logits=True)
        stats = RoutingSummary(2, 2, 2)
        descriptor = batch.LaneRace.__dict__['forward']
        with race_policy(torch, batch.LaneRace, 'sampled', 2, 2, stats) as calls:
            observed = batch.batched_logits(model, rows, 314159, all_logits=True)
        assert torch.equal(baseline, observed) and calls[0] == 16
        assert sum(r['races'] for r in stats.report()) == 32
        assert batch.LaneRace.__dict__['forward'] is descriptor
        checks.append('Every synthetic sampled logit unchanged by actual-win collection; exact callback recovery/counts')

        # The sampled winner is deliberately different from the score argmax.
        scores = torch.tensor([[math.log(3.), 0.]], dtype=torch.float64)
        proposals = torch.tensor([[[1., 2.], [7., 8.]]], dtype=torch.float64)
        noise = torch.tensor([3., .5], dtype=torch.float64)
        forced = torch.tensor([-1])
        sampled = batch.LaneRace.apply(scores, proposals, noise, forced)
        with race_policy(torch, batch.LaneRace, 'argmax_preserve_first', 1, 1):
            identity = batch.LaneRace.apply(scores, proposals, noise, forced)
        with race_policy(torch, batch.LaneRace, 'argmax_unit_clocks', 1, 1):
            unit = batch.LaneRace.apply(scores, proposals, noise, forced)
        assert sampled[2].item() == 1 and identity[2].item() == unit[2].item() == 0
        assert torch.equal(sampled[1], identity[1]) and not torch.equal(sampled[1], unit[1])
        assert torch.equal(identity[0], proposals[:, 0])
        checks.append('Constructed race changes only identity at its sampled first time; unit-clocks policy changes delay')

        collected = []
        for row in rows:
            collected.append(batch.batched_logits(model, [row], 314159, all_logits=True))
        assert torch.allclose(baseline, torch.cat(collected), rtol=1e-10, atol=1e-10)
        checks.append('Tiny double full-window outputs agree across together/separate lanes; no large float32 guarantee')
    return checks


def run(args):
    begin = time.perf_counter()
    parent_path, output = ROOT/args.result, ROOT/args.out
    if output.exists():
        raise ValueError('Preserve the existing diagnostic output')
    parent_hash = B.sha(parent_path)
    parent = json.loads(parent_path.read_text())
    if parent.get('status') != 'completed' or 'final_weights' not in parent:
        raise ValueError('A completed result with saved final weights is required')
    if parent['source_sha256'].get(B.BATCH_NAME) != B.BATCH_DIGEST:
        raise ValueError('This protocol binds the original pre-linear-credit numerical producer')
    config = parent['args']
    segment = config['segment']
    if args.samples < 1 or args.lanes < 1 or segment < 2 or segment % 2 or args.chars <= segment+1:
        raise ValueError('Positive samples/lanes and a sufficient span with even segment length required')
    weights_path = ROOT/parent['final_weights']
    if not weights_path.is_file():
        raise ValueError('Saved weights are unavailable on this host; use the producer checkpoint, never reconstruct from scores')
    weights_hash = B.sha(weights_path)
    if B.sha(ROOT/B.BATCH_DEPENDENCY_NAME) != B.BATCH_DEPENDENCY_DIGEST:
        raise ValueError('Changed temporal transport dependency of the archived numerical producer')
    factory_sources = ('sleeping_machines/addressed_event_heads.py', 'sleeping_machines/fast_native_core.py')
    for name in factory_sources:
        if B.sha(ROOT/name) != parent['source_sha256'][name]:
            raise ValueError('Changed saved-model factory: '+name)
    source_names = factory_sources + ('experiments/language_routing_measurements.py',
                    'experiments/routing_measurement_summary.py',
                    'experiments/legacy_batched_driver_binding.py',
                    'experiments/language_batched_benchmark.py',
                    'experiments/e120_shared_tasks.py',
                    B.BATCH_DEPENDENCY_NAME, B.BATCH_ARCHIVE)
    sources = {name: B.sha(ROOT/name) for name in source_names}

    # Lazy imports keep a source/help inspection from starting a tensor runtime.
    import numpy as np
    import torch
    import language_batched_benchmark as L
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.fast_native_core import fast_class

    torch.set_num_threads(1)
    caller_rng = torch.get_rng_state().clone()
    batch = B.load_batched()
    race_descriptor = batch.LaneRace.__dict__['forward']
    with torch.random.fork_rng(), torch.no_grad():
        checks = native_contracts(torch, np, batch, L, fast_class(AddressedEventHeads))
        model = fast_class(AddressedEventHeads)(sources=1, content_dim=27, classes=27,
                    payload=config['payload'], depth=config['depth'], heads=config['heads'], pool=config['pool'])
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location='cpu'))
        model.eval()
        saved = {name: value.clone() for name, value in model.state_dict().items()}
        gains = {name: float(module.gain) for name, module in model.named_modules() if hasattr(module, 'gain')}
        text = L.load_text(90_000_000, args.chars)
        half = segment//2
        starts = list(range(0, len(text)-segment-1, half))
        targets = segment+(len(starts)-1)*half
        seeds = [314159+7919*k for k in range(args.samples)]
        seed_bits, mixture_bits = [0.]*args.samples, 0.
        policies = {'argmax_preserve_first': 0., 'argmax_unit_clocks': 0.}
        summary = RoutingSummary(model.depth, model.heads, model.pool)
        calls_total = 0

        def bits(log_target, chunk):
            return math.fsum(-float((log_target[i] if s == 0 else log_target[i, half:]).sum())/math.log(2)
                             for i, s in enumerate(chunk))

        for offset in range(0, len(starts), args.lanes):
            chunk = starts[offset:offset+args.lanes]
            rows = L.rows_of(text, chunk, segment)
            y = torch.tensor(np.stack([text[s+1:s+segment+1] for s in chunk]), dtype=torch.long)
            mix = None
            for k, seed in enumerate(seeds):
                with race_policy(torch, batch.LaneRace, 'sampled', model.heads, model.depth,
                                 summary if k == 0 else None) as calls:
                    z = batch.batched_logits(model, rows, seed, all_logits=True)
                assert calls[0] == segment*model.depth*model.heads
                calls_total += calls[0]
                log_target = torch.log_softmax(z.double(), -1).gather(-1, y[..., None]).squeeze(-1)
                seed_bits[k] += bits(log_target, chunk)
                mix = log_target.clone() if mix is None else torch.logaddexp(mix, log_target)
            mixture_bits += bits(mix-math.log(args.samples), chunk)
            for policy in policies:
                with race_policy(torch, batch.LaneRace, policy, model.heads, model.depth) as calls:
                    z = batch.batched_logits(model, rows, seeds[0], all_logits=True)
                assert calls[0] == segment*model.depth*model.heads
                calls_total += calls[0]
                log_target = torch.log_softmax(z.double(), -1).gather(-1, y[..., None]).squeeze(-1)
                policies[policy] += bits(log_target, chunk)
            print(json.dumps(dict(windows_processed=min(offset+args.lanes, len(starts)), windows=len(starts))), flush=True)
        for name, value in model.state_dict().items():
            assert torch.equal(value, saved[name])
        assert gains == {name: float(module.gain) for name, module in model.named_modules() if hasattr(module, 'gain')}
        assert all(p.grad is None for p in model.parameters())
        checks.append('Every saved weight/Python residual gain unchanged; no backward or optimizer update')
        mixture = mixture_summary(seed_bits, mixture_bits, targets)
        actual = summary.report()
        expected_races = len(starts)*segment*model.depth*model.heads
        assert sum(row['races'] for row in actual) == expected_races
        checks.append('Actual sampled winners and expected probability mass have the same recorded race denominator; valid mixture Jensen bound')
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert batch.LaneRace.__dict__['forward'] is race_descriptor
    for name, digest in sources.items():
        assert B.sha(ROOT/name) == digest, name
    assert B.sha(parent_path) == parent_hash and B.sha(weights_path) == weights_hash
    checks.append('Source/result/weights/caller RNG/race callback retained')
    forward_passes = args.samples+len(policies)
    events = len(starts)*segment*forward_passes
    result = dict(status='completed', args=vars(args), contracts_passed=len(checks), contracts=checks,
        parent=args.result, parent_sha256=parent_hash, weights=parent['final_weights'], weights_sha256=weights_hash,
        producer_source_sha256=parent['source_sha256'], source_sha256=sources, python_gains=gains,
        protocol=dict(data_span=[90_000_000, 90_000_000+args.chars], data_sha256=hashlib.sha256(text.tobytes()).hexdigest(),
                      segment=segment, lanes=args.lanes, scored_targets_per_policy=targets, seeds=seeds,
                      windows=len(starts), state='reset per window', precision='float32 model; double log-space scoring',
                      routing_counts='First factual seed, all warming/scored input positions, separate depth/head; shared row noise is not IID'),
        clock_metric_scope='Local derivative of delay to a common shift of transmitted post-clamp scores; upstream clamp and downstream loss sensitivity are separate. Boundary hits do not distinguish equality from clipping outside the bound',
        mixture=mixture, routing=actual, policy_bpc={k:v/targets for k,v in policies.items()},
        policy_scope=dict(argmax_preserve_first='Argmax identity with the same sampled first time at each own current history; future state/rates can change',
                          argmax_unit_clocks='Argmax identity and first time1/max(rate); both identity and clocks change'),
        execution=dict(forward_passes=forward_passes, window_evaluations=len(starts)*forward_passes,
                       forward_input_events=events, selected_writes=events*model.depth*model.heads,
                       scored_keys=events*model.depth*model.heads*model.pool,
                       computed_candidate_values=events*model.depth*model.heads*model.pool,
                       scored_target_predictions=targets*forward_passes, race_callback_calls=calls_total,
                       scope='Main production passes only; contracts/collection/log scoring additional. Campaign FLOPs/traffic/energy unknown, not zero'),
        hardware=dict(device='cpu', threads=torch.get_num_threads(), torch=torch.__version__),
        wall_s=time.perf_counter()-begin, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Small fixed DEV-span forward intervention; no fitting/model selection, global context proof or benchmark advantage')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed', contracts=len(checks), mixture=mixture,
                         policy_bpc=result['policy_bpc'], wall_s=result['wall_s'])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--chars', type=int, default=2048)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--lanes', type=int, default=8)
    run(p.parse_args())


if __name__ == '__main__':
    main()
