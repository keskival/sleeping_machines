"""Bounded trained-FIT delivery/write factorial audit; guarded queue only.

No fitting or optimizer. Help/source admission use only the standard library.
Shadow branches keep the selected race's first time and future RNG draws.
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
from route_fidelity_geometry import addition_fidelity, factorial, fidelity, pooled_addition_fidelity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def intervention(torch, batch, site, delivery=None, commit=None, probe=False):
    """One-lane eager diagnostic; restore inherited apply and write helper.

    Probe mode removes ALL auxiliary write-score credit and adds a zero
    independent leaf at the chosen write to measure its state cotangent.
    It must nest EVERY parameter gradient of message-only linear credit.
    Hybrids are forward-only diagnostic interventions, not new legal routes.
    """
    race = batch.LaneRace
    original_apply, original_write = race.apply, batch.linear_write_credit
    owned = 'apply' in race.__dict__
    descriptor = race.__dict__.get('apply')
    trace = dict(calls=0, noise_tape=[])

    def apply(scores, proposals, noise, forced_alt):
        index = trace['calls']
        trace['calls'] += 1
        trace['noise_tape'].append(noise.detach().clone())
        assert len(scores) == 1 and not bool((forced_alt >= 0).any())
        value, delay, winner = original_apply(scores, proposals, noise, forced_alt)
        if index == site:
            assert 'scores' not in trace
            trace.update(scores=scores.detach().clone(), proposals=proposals.detach().clone(),
                         winner=int(winner[0]), delay=delay.detach().clone(), noise=noise.detach().clone())
            if probe:
                value.retain_grad()
                trace['live_value'] = value
            if delivery is not None:
                value = proposals[:, delivery]
            if commit is not None:
                winner = winner.new_full(winner.shape, commit)
            trace.update(delivered=trace['winner'] if delivery is None else delivery,
                         committed=int(winner[0]))
        return value, delay, winner

    def write(scores, old, new, active):
        assert probe
        at_site = trace['calls'] == site+1
        zero = torch.zeros_like(new, requires_grad=at_site)
        if at_site:
            assert 'live_write' not in trace
            caller = sys._getframe(1)
            assert caller.f_code is batch.batched_logits.__code__
            local = caller.f_locals
            head = local['head']
            # These locals are admitted against the exact saved producer hash.
            assert torch.equal(old, local['m'][:, head]) and torch.equal(new, local['m_new'][:, head])
            trace.update(live_write=zero, before=old.detach().clone(), after=new.detach().clone(),
                         written=local['written'][:, head].detach().clone(),
                         ages=local['age'][:, head].detach().clone(),
                         forget=local['forget'][:, head].detach().clone(),
                         seen_before=local['seen'][:, local['depth'], head].detach().clone(),
                         arrival=local['arrival'].detach().clone())
        return zero

    race.apply = staticmethod(apply)
    if probe:
        batch.linear_write_credit = write
    try:
        yield trace
    finally:
        batch.linear_write_credit = original_write
        if owned:
            race.apply = descriptor
        else:
            delattr(race, 'apply')


def gradients(model):
    return {name: None if p.grad is None else p.grad.detach().clone() for name, p in model.named_parameters()}


def equal_gradients(torch, first, second):
    assert first.keys() == second.keys()
    for name in first:
        a, b = first[name], second[name]
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.equal(a, b), name


def loss_of(torch, model, z, targets):
    # Preserve float32 per-target CE but accumulate in double to avoid losing
    # a small suffix difference when subtracting two rounded sequence sums.
    terms = torch.nn.functional.cross_entropy(z.reshape(-1, model.classes), targets.reshape(-1), reduction='none')
    return terms.double().sum()


def ordinary(torch, batch, model, rows, targets, seed):
    model.zero_grad(set_to_none=True)
    z = batch.batched_logits(model, rows, seed, all_logits=True, route_credit='linear')
    loss = loss_of(torch, model, z, targets)
    loss.backward()
    return z.detach().clone(), float(loss.detach()), gradients(model)


def replay(torch, batch, model, rows, targets, seed, site, probe=False, delivery=None, commit=None):
    model.zero_grad(set_to_none=True)
    caller_rng = torch.get_rng_state().clone()
    old_apply = batch.LaneRace.__dict__.get('apply')
    old_write = batch.linear_write_credit
    with torch.set_grad_enabled(probe), intervention(torch, batch, site, delivery, commit, probe) as trace:
        z = batch.batched_logits(model, rows, seed, all_logits=True,
                                 route_credit='linear_rw' if probe else 'linear')
        loss = loss_of(torch, model, z, targets)
        if probe:
            loss.backward()
            value = trace.pop('live_value')
            write = trace.pop('live_write')
            assert value.grad is not None
            trace['value_cotangent'] = value.grad.detach().clone()
            trace['state_cotangent'] = torch.zeros_like(write) if write.grad is None else write.grad.detach().clone()
            trace['parameter_gradients'] = gradients(model)
    assert batch.LaneRace.__dict__.get('apply') is old_apply and batch.linear_write_credit is old_write
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert trace['calls'] == len(rows[0]['events'])*model.depth*model.heads
    trace.update(logits=z.detach().clone(), loss=float(loss.detach()), caller_rng=caller_rng)
    return trace


def paired_case(torch, batch, model, rows, targets, seed, site, baseline):
    z, loss, grad = baseline
    actual = replay(torch, batch, model, rows, targets, seed, site, probe=True)
    assert torch.equal(actual['logits'], z) and actual['loss'] == loss
    equal_gradients(torch, actual.pop('parameter_gradients'), grad)
    winner = actual['winner']
    assert model.pool == 2
    other = 1-winner
    shadows = [replay(torch, batch, model, rows, targets, seed, site, delivery=d, commit=c)
               for d, c in ((other, winner), (winner, other), (other, other))]
    event = site//(model.depth*model.heads)
    for shadow in shadows:
        for name in ('scores', 'proposals', 'delay', 'noise'):
            assert torch.equal(actual[name], shadow[name]), name
        assert shadow['winner'] == winner and torch.equal(shadow['caller_rng'], actual['caller_rng'])
        assert len(shadow['noise_tape']) == len(actual['noise_tape'])
        assert all(torch.equal(a, b) for a, b in zip(actual['noise_tape'], shadow['noise_tape']))
        assert torch.equal(shadow['logits'][:, :event], z[:, :event])
    # A pure private write cannot affect the current event's delivered value.
    assert torch.equal(shadows[1]['logits'][:, :event+1], z[:, :event+1])
    if event == len(rows[0]['events'])-1:
        assert shadows[1]['loss'] == loss and torch.equal(shadows[0]['logits'], shadows[2]['logits'])
    pi = actual['scores'][0].double().softmax(-1)
    value_coefficient = (actual['proposals'][0].double()*actual['value_cotangent'][0].double()).sum(-1)
    state_error = actual['state_cotangent'][0].double()
    stored = (state_error*(actual['after'][0]-actual['before'][0]).double()).sum(-1)
    written = (state_error*actual['written'][0].double()).sum(-1)
    utilities = [None, None]
    utilities[winner], utilities[other] = loss, shadows[2]['loss']
    linear = float(value_coefficient[other]-value_coefficient[winner])
    coefficients = dict(value=value_coefficient, value_plus_stored=value_coefficient+stored,
                        value_plus_written=value_coefficient+written)
    before_norm = actual['before'][0].double().norm(dim=-1)
    return dict(seed=seed, site=site, event=event, depth=(site//model.heads) % model.depth,
                head=site % model.heads, winner=winner, alternative=other,
                probabilities=pi.tolist(), scores=actual['scores'][0].tolist(),
                delay=float(actual['delay'][0]), legal_branch_losses_nats_sum=utilities,
                noise_tape_sha256=hashlib.sha256(torch.stack(actual['noise_tape']).numpy().tobytes()).hexdigest(),
                factorial_losses_nats_sum=dict(factual=loss, value_only=shadows[0]['loss'],
                                                commit_only=shadows[1]['loss'], both=shadows[2]['loss']),
                factorial=factorial(loss, *(s['loss'] for s in shadows), linear),
                coefficients_nats=dict(value=value_coefficient.tolist(), stored_state=stored.tolist(),
                                       written_state=written.tolist(), homogeneous_state=(stored-written).tolist()),
                teachers={name: fidelity(pi.tolist(), utilities, a.tolist()) for name, a in coefficients.items()},
                write_addition_geometry={name: addition_fidelity(pi.tolist(), utilities, value_coefficient.tolist(), a.tolist())
                                         for name, a in (('stored', stored), ('written', written))},
                state=dict(memory_norms=before_norm.tolist(), gaps=actual['ages'][0].tolist(),
                           forget=actual['forget'][0].tolist(), seen=actual['seen_before'][0].tolist(),
                           state_cotangent_norms=state_error.norm(dim=-1).tolist(),
                           written_norms=actual['written'][0].double().norm(dim=-1).tolist()),
                hard_score_boundary_count=int((actual['scores'][0].abs() >= 12).sum()),
                contracts=['Exact factual logits/loss/EVERY gradient nesting', 'Same selected first time and future draws',
                           'Unchanged earlier outputs; commit-only leaves current output unchanged',
                           'Restored callbacks/caller RNG'],
                scope='One factual cotangent under message-only learning; write coefficients are diagnostic additions, not full linear_rw/rwn update gradients')


def native_contracts(torch, np, batch, factory, language):
    torch.manual_seed(144)
    model = factory(sources=1, content_dim=27, classes=27, payload=4, depth=2, heads=2, pool=2).double().eval()
    text = np.arange(6, dtype=np.uint8)
    rows = language.rows_of(text, [0], 5)
    targets = torch.tensor(text[1:6].astype('int64'))[None]
    baseline = ordinary(torch, batch, model, rows, targets, 144)
    cases = [paired_case(torch, batch, model, rows, targets, 144, site, baseline) for site in (4, 19)]
    model.zero_grad(set_to_none=True)
    return dict(status='passed', cases=cases,
                checks=['Every-parameter probe nesting', 'Inherited apply/helper recovery',
                        'Fixed-time factorial causal-prefix preservation', 'Last-event write has no future loss effect'])


def run(args):
    begin = time.perf_counter()
    parent_path, out = ROOT/args.result, ROOT/args.out
    if out.exists():
        raise ValueError('Preserve prior output')
    parent_hash = sha(parent_path)
    parent = json.loads(parent_path.read_text())
    if parent.get('status') != 'completed' or not parent.get('final_weights'):
        raise ValueError('Completed actual-weight language producer required')
    weights = ROOT/parent['final_weights']
    if not weights.is_file():
        raise ValueError('Actual producer checkpoint unavailable; no reconstruction from scores')
    weights_hash, settings = sha(weights), parent['args']
    if settings['pool'] != 2 or settings.get('skip_init_from', 0) or settings['route_credit'] != 'linear':
        raise ValueError('Current scope: nongrown pool2 message-linear trained producer')
    producer_names = ['sleeping_machines/batched_episodes.py', 'sleeping_machines/fast_native_core.py',
                      'sleeping_machines/addressed_event_heads.py']
    for name in producer_names:
        if sha(ROOT/name) != parent['source_sha256'][name]:
            raise ValueError('Changed producer: '+name)
    names = producer_names+['experiments/language_route_fidelity.py', 'experiments/route_fidelity_geometry.py',
                           'experiments/language_batched_benchmark.py', 'experiments/e120_shared_tasks.py',
                           'sleeping_machines/parallel_stream_language.py']
    sources = {name: sha(ROOT/name) for name in names}
    import numpy as np
    import torch
    import language_batched_benchmark as L
    import sleeping_machines.batched_episodes as B
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.fast_native_core import fast_class
    torch.set_num_threads(1)
    rng = torch.get_rng_state().clone()
    factory = fast_class(AddressedEventHeads)
    with torch.random.fork_rng(devices=[]):
        contracts = native_contracts(torch, np, B, factory, L)
        model = factory(sources=1, content_dim=27, classes=27, payload=settings['payload'],
                        depth=settings['depth'], heads=settings['heads'], pool=2).eval()
        model.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
        saved = {name: value.clone() for name, value in model.state_dict().items()}
        gains = {name: float(m.gain) for name, m in model.named_modules() if hasattr(m, 'gain')}
        text = L.load_text(0, 33)
        rows = L.rows_of(text, [0], 32)
        targets = torch.tensor(text[1:33].astype('int64'))[None]
        probes = []
        for seed in (314159, 322078):
            baseline = ordinary(torch, B, model, rows, targets, seed)
            for event in (7, 23):
                for depth in (0, settings['depth']-1):
                    site = (event*settings['depth']+depth)*settings['heads']
                    probes.append(paired_case(torch, B, model, rows, targets, seed, site, baseline))
        for name, value in model.state_dict().items():
            assert torch.equal(value, saved[name]), name
        assert gains == {name: float(m.gain) for name, m in model.named_modules() if hasattr(m, 'gain')}
        model.zero_grad(set_to_none=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert sha(parent_path) == parent_hash and sha(weights) == weights_hash
    for name, digest in sources.items():
        assert sha(ROOT/name) == digest, name
    forwards, backwards = 2+4*len(probes), 2+len(probes)
    calibration = {name: pooled_addition_fidelity([
        (p['probabilities'], p['legal_branch_losses_nats_sum'], p['coefficients_nats']['value'],
         p['coefficients_nats'][field]) for p in probes])
        for name, field in (('stored', 'stored_state'), ('written', 'written_state'))}
    result = dict(status='completed', args=vars(args), native_contracts=contracts, probes=probes,
                  shared_alpha_diagnostics=calibration,
                  parent_sha256=parent_hash, weights=parent['final_weights'], weights_sha256=weights_hash,
                  source_sha256=sources, producer_source_sha256=parent['source_sha256'],
                  data=dict(source='text8 FIT[0:33]', sha256=hashlib.sha256(text.tobytes()).hexdigest(),
                            length=32, targets=32, state_reset=True, selection='fixed sites/seeds; no DEV/test'),
                  numerics=dict(loss='Per-target model-precision CE, float64 sum; summed nats, not native mean/clip/update',
                                coefficients='Float64 dot products of captured represented proposals/cotangents; ideal local coefficient comparison'),
                  accounting=dict(actual_trained_forward_replays=forwards, actual_trained_backward_replays=backwards,
                                  actual_trained_forward_input_positions=forwards*32,
                                  actual_trained_forward_races=forwards*32*settings['depth']*settings['heads'],
                                  synthetic_forward_replays=9, synthetic_backward_replays=3,
                                  optimizer_updates=0, scope='All replay counts charged; native FLOPs/traffic/energy unmeasured'),
                  hardware=dict(device='cpu', threads=torch.get_num_threads(), torch=torch.__version__),
                  wall_s=time.perf_counter()-begin, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Local conditional two-route factorial fidelity at one FIT span/two seeds/eight sites; '
                        'finite suffixes include future route switches. No fitted improvement, complete expected gradient, '
                        'divergence attribution, warm-Adam alignment or benchmark claim')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed', probes=len(probes), wall_s=result['wall_s'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
