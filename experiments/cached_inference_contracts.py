"""Bounded saved-weight winner/cache contracts; run only in a guarded queue.

Observe real min decisions and final local state without changing either
producer. Float64 fixtures are distinct from trained float32 validation.
Imports/help and source admission use only the standard library.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments')]
from sparse_inference_accounting import cache_accounting, race_certificate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def observe(torch, function, lanes, pool):
    """Readonly, process-local observation; restore min/profile on all exits."""
    original = torch.Tensor.min
    owned_min = 'min' in torch.Tensor.__dict__
    original_local_min = torch.Tensor.__dict__.get('min')
    previous_profile = sys.getprofile()
    if previous_profile is not None:
        raise ValueError('Do not nest the diagnostic inside another Python profiler')
    code = getattr(function, '__wrapped__', function).__code__
    tape = dict(races=[], state=None)

    def minimum(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        dim = args[0] if args else kwargs.get('dim')
        if dim == -1 and value.dtype == torch.float64 and tuple(value.shape) == (lanes, pool):
            tape['races'].append((value.detach().clone(), result.indices.detach().clone()))
        return result

    def profile(frame, event, arg):
        if frame.f_code is code and event == 'return' and arg is not None:
            if tape['state'] is not None:
                raise ValueError('Unexpected recursive/multiple producer returns')
            local = frame.f_locals
            names = ['mem', 'arr', 'seen', 'ctx_vals', 'ctx_arr', 'has_ctx']
            if 'reads' in local:
                names.append('reads')
            state = {}
            for name in names:
                value = local[name]
                if isinstance(value, list):
                    value = torch.stack(value, 1)
                state[name] = value.detach().clone()
            tape['state'] = state

    torch.Tensor.min = minimum
    sys.setprofile(profile)
    try:
        yield tape
    finally:
        sys.setprofile(previous_profile)
        if owned_min:
            torch.Tensor.min = original_local_min
        else:
            delattr(torch.Tensor, 'min')


def pair(torch, reference, sparse, model, rows, seed, label):
    dtype = model.embedding.weight.dtype
    tolerance = 1e-10 if dtype == torch.float64 else 1e-4
    before = {name: value.clone() for name, value in model.state_dict().items()}
    gains = {name: float(m.gain) for name, m in model.named_modules() if hasattr(m, 'gain')}
    caller_rng = torch.get_rng_state().clone()
    min_method, profile = torch.Tensor.min, sys.getprofile()
    collected, tapes = [], []
    for function in (reference, sparse):
        with observe(torch, function, len(rows), model.pool) as tape:
            logits = function(model, rows, seed, all_logits=True)
        collected.append(logits)
        tapes.append(tape)
        assert torch.equal(caller_rng, torch.get_rng_state())
        assert torch.Tensor.min is min_method and sys.getprofile() is profile
    length = max(len(row['events']) for row in rows)
    expected_calls = length*model.depth*model.heads
    assert all(len(tape['races']) == expected_calls for tape in tapes)
    assert all(tape['state'] is not None for tape in tapes)
    active_races, agreements, certified = 0, 0, 0
    disagreements = []
    smallest_margin, smallest_residual, maximum_log_error = None, None, 0.
    for index, ((first, winner), (other, selected)) in enumerate(zip(tapes[0]['races'], tapes[1]['races'])):
        event = index//(model.depth*model.heads)
        for lane, row in enumerate(rows):
            if event >= len(row['events']):
                continue
            result = race_certificate(first[lane].tolist(), other[lane].tolist(),
                                      int(winner[lane]), int(selected[lane]))
            active_races += 1
            agreements += int(result['winner_agrees'])
            certified += int(result['certified'])
            if not result['winner_agrees']:
                disagreements.append(dict(race=index, event=event, lane=lane,
                                          reference_winner=int(winner[lane]), candidate_winner=int(selected[lane]),
                                          reference_times=first[lane].tolist(), candidate_times=other[lane].tolist(),
                                          certificate=result))
            maximum_log_error = max(maximum_log_error, result['max_abs_log_time_difference'])
            for name, key in (('margin', 'reference_log_margin'), ('residual', 'conservative_residual_margin')):
                value = result[key]
                if value is not None:
                    if name == 'margin':
                        smallest_margin = value if smallest_margin is None else min(smallest_margin, value)
                    else:
                        smallest_residual = value if smallest_residual is None else min(smallest_residual, value)
    assert active_races == sum(len(row['events']) for row in rows)*model.depth*model.heads
    errors, checks = {}, {}
    for name in ['logits', 'mem', 'arr', 'seen', 'ctx_vals', 'ctx_arr', 'has_ctx']:
        a, b = collected if name == 'logits' else (tapes[0]['state'][name], tapes[1]['state'][name])
        if a.dtype == torch.bool:
            checks[name] = torch.equal(a, b)
            errors[name] = int((a != b).sum())
        else:
            assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            allowed = 1e-10 if name in ('arr', 'ctx_arr') else tolerance
            checks[name] = torch.allclose(a, b, atol=allowed, rtol=allowed)
            errors[name] = float((a-b).abs().max())
    layers = model._stacked(0)
    expected_reads = torch.stack([layer['key'].view(model.heads, model.pool, model.payload)+
                                 torch.einsum('hupq,nhuq->nhup',
                                              layer['key_read'].view(model.heads, model.pool, model.payload, model.payload),
                                              tapes[1]['state']['mem'][:, depth])
                                 for depth, layer in enumerate(layers)], 1)
    cached = tapes[1]['state']['reads']
    assert bool(torch.isfinite(cached).all()) and bool(torch.isfinite(expected_reads).all())
    checks['cache_invariant'] = torch.allclose(cached, expected_reads, atol=tolerance, rtol=tolerance)
    errors['cache_invariant'] = float((cached-expected_reads).abs().max())
    # Observer nesting itself must leave numerical output unchanged.
    for function, recorded in zip((reference, sparse), collected):
        assert torch.equal(function(model, rows, seed, all_logits=True), recorded)
    assert torch.equal(caller_rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert gains == {name: float(m.gain) for name, m in model.named_modules() if hasattr(m, 'gain')}
    assert all(p.grad is None for p in model.parameters())
    return dict(label=label, dtype=str(dtype), seed=seed, lengths=[len(row['events']) for row in rows],
                depth=model.depth, heads=model.heads, pool=model.pool, payload=model.payload,
                passed=all(checks.values()) and agreements == active_races,
                checks=checks, maximum_absolute_differences=errors,
                tolerance=dict(values=tolerance, arrival_stamps=1e-10, winners='every active race must agree'),
                active_races=active_races, winner_agreements=agreements, margin_certified_races=certified,
                winner_disagreements=disagreements,
                minimum_reference_log_margin=smallest_margin, minimum_conservative_residual_margin=smallest_residual,
                maximum_log_time_difference=maximum_log_error, batched_race_calls_per_pass=expected_calls,
                production_calls_observed=2, observer_nesting_calls=2,
                scope='Observed finite trajectories only; final state/cache checked, every intermediate payload not exposed')


def run(args):
    begin = time.perf_counter()
    output, parent_path = ROOT/args.out, ROOT/args.result
    if output.exists():
        raise ValueError('Preserve existing output')
    parent_hash = sha(parent_path)
    parent = json.loads(parent_path.read_text())
    if parent.get('status') != 'completed' or not parent.get('final_weights'):
        raise ValueError('Completed fit with its actual final checkpoint required')
    weights = ROOT/parent['final_weights']
    if not weights.is_file():
        raise ValueError('Producer checkpoint unavailable here; do not reconstruct weights from scores')
    weights_hash = sha(weights)
    config = parent['args']
    if config.get('skip_init_from', 0) != 0:
        raise ValueError('This admission covers the current nongrown/skip0 language producer')
    producer_names = ['sleeping_machines/batched_episodes.py', 'sleeping_machines/fast_native_core.py',
                      'sleeping_machines/addressed_event_heads.py']
    for name in producer_names:
        if sha(ROOT/name) != parent['source_sha256'][name]:
            raise ValueError('Changed numerical producer/factory: '+name)
    names = producer_names+['sleeping_machines/sparse_inference.py', 'sleeping_machines/parallel_stream_language.py',
                           'experiments/cached_inference_contracts.py', 'experiments/sparse_inference_accounting.py',
                           'experiments/language_batched_benchmark.py', 'experiments/e120_shared_tasks.py']
    sources = {name: sha(ROOT/name) for name in names}

    import numpy as np
    import torch
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.fast_native_core import fast_class
    from sleeping_machines.batched_episodes import batched_logits
    from sleeping_machines.sparse_inference import sparse_logits
    import language_batched_benchmark as L

    torch.set_num_threads(1)
    caller_rng = torch.get_rng_state().clone()
    cases = []
    factory = fast_class(AddressedEventHeads)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for pool in (1, 3, 4):
            for dtype in (torch.float32, torch.float64):
                torch.manual_seed(143+pool)
                model = factory(sources=1, content_dim=5, classes=5, payload=8, depth=2, heads=2, pool=pool).to(dtype).eval()
                for parameter in model.parameters():
                    parameter.add_(torch.randn_like(parameter)*.1)
                generator = np.random.default_rng(143+pool)
                rows = [dict(events=[(float(t)+float(generator.random()), generator.normal(size=5))
                                     for t in range(length)]) for length in (9, 3, 7)]
                for seed in (311, 733):
                    cases.append(pair(torch, batched_logits, sparse_logits, model, rows, seed, 'synthetic'))
        admitted = all(case['passed'] for case in cases)
        data_hash = None
        if admitted:
            symbols = L.load_text(0, 128)
            data_hash = hashlib.sha256(symbols.tobytes()).hexdigest()
            rows = [L.rows_of(symbols, [start], length)[0] for start, length in ((0, 32), (39, 17), (78, 24))]
            saved = torch.load(weights, weights_only=True, map_location='cpu')
            for dtype in (torch.float32, torch.float64):
                model = factory(sources=1, content_dim=27, classes=27, payload=config['payload'],
                                depth=config['depth'], heads=config['heads'], pool=config['pool']).to(dtype).eval()
                model.load_state_dict(saved)
                for seed in (314159, 322078):
                    cases.append(pair(torch, batched_logits, sparse_logits, model, rows, seed, 'actual_trained_FIT'))
        admitted = admitted and all(case['passed'] for case in cases)
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert sha(parent_path) == parent_hash and sha(weights) == weights_hash
    for name, digest in sources.items():
        assert sha(ROOT/name) == digest, name
    result = dict(status='completed' if admitted else 'failed_contract', contracts_admitted=admitted,
                  args=vars(args), parent_sha256=parent_hash, weights=parent['final_weights'], weights_sha256=weights_hash,
                  producer_source_sha256=parent['source_sha256'], source_sha256=sources, cases=cases,
                  protocol=dict(data='text8 FIT[0:128]; fixed spans0/39/78, lengths32/17/24, no target loss',
                                data_sha256=data_hash, precisions='original float32 weights and same represented weights promoted to float64',
                                synthetic_gate_first=True, profile='readonly min observer and return-frame state snapshot, exactly nested outputs'),
                  accounting=cache_accounting(config['payload'], config['depth'], config['heads'], config['pool'], 3),
                  hardware=dict(device='cpu', threads=torch.get_num_threads(), torch=torch.__version__),
                  wall_s=time.perf_counter()-begin, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Paid numerical/cache/RNG/state contracts only; no fitting, full DEV/test rescore, FLOP or wall-speed comparison')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], cases=len(cases), admitted=admitted, wall_s=result['wall_s'])))
    if not admitted:
        raise RuntimeError('Inference contracts failed; retained diagnostics do not admit the backend')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
