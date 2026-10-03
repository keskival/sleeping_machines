"""Integrated bounded-score training, recovery and full-work contracts (note106)."""
import argparse
import copy
import io
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_bridge_batched_benchmark as D
import dvs_batched_le_benchmark as B
import dvs_native_benchmark as N
import dvs_local_expectation_benchmark as LE
from sleeping_machines.batched_episodes import batched_logits as old_batch
from sleeping_machines.factorized_race import factorized_race
from sleeping_machines.bounded_score_sensitivity import bounded_score_bridge
from race_language_screen import capture


def close(a, b, exact=False):
    torch.testing.assert_close(a, b, rtol=0 if exact else 2e-7, atol=0 if exact else 2e-9)


def gradients(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def compare_grads(a, b, exact=False):
    if exact: assert a.keys() == b.keys()
    for name in a.keys() | b.keys():
        left = a.get(name, torch.zeros_like(b[name]) if name not in a else None)
        right = b.get(name, torch.zeros_like(a[name]) if name not in b else None)
        try: close(left, right, exact)
        except AssertionError as err: raise AssertionError(name) from err


def seq_objective(model, rows, seed):
    model.train(); objective = model.embedding.weight.new_zeros(())
    for row in rows:
        scores = []
        loss, races, _ = LE.run(model, row, seed, record=scores)
        with torch.no_grad():
            losses = [torch.stack([LE.run(model, row, seed, force=(r, i))[0]
                                  for i in range(model.pool)]) for r in range(races)]
        objective = objective + loss + sum((s.softmax(0) * u).sum() for s, u in zip(scores, losses))
    return objective


def seq_step(model, optimizer, rows, a, epoch):
    optimizer.zero_grad(set_to_none=True)
    seq_objective(model, rows, 100000 + a.seed + 10000 * epoch).backward()
    for p in model.parameters():
        if p.grad is not None: p.grad.div_(len(rows))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a0 = p.parse_args()
    out = ROOT / 'experiments/results/diagnostics' / (a0.tag + '.json')
    assert not out.exists() and Path(a0.tag).name == a0.tag
    torch.set_num_threads(1); started = time.perf_counter(); LE.PATHWISE = factorized_race
    a = SimpleNamespace(payload=8, depth=4, heads=2, pool=2, clock_step=.05, seed=7,
                        score_bridge=0., route_credit=True, route_races=0)
    rng = np.random.default_rng(106)
    rows = [dict(index=j, target=j % 11, events=[((k + 1) * .05,
            torch.tensor(rng.standard_normal(33), dtype=torch.float64, requires_grad=True))
            for k in range(2 + j)]) for j in range(2)]
    passed = []; counts = {}; coverage = {}
    model = D.make_model(a).double()
    with torch.no_grad():
        for par in model.parameters(): par.add_(torch.randn_like(par) * .05)
    state = copy.deepcopy(model.state_dict())
    # Exact alpha0 nesting against the immutable old episode-batched computation.
    model._fast_layers = {}; z0 = old_batch(model, rows, 20261003)
    F.cross_entropy(z0, torch.tensor([r['target'] for r in rows])).backward(); g0 = gradients(model)
    input0 = [v.grad.clone() for r in rows for _, v in r['events']]
    model.zero_grad(); model._fast_layers = {}
    for r in rows:
        for _, v in r['events']: v.grad = None
    z = D.batched_logits(model, rows, 20261003)
    F.cross_entropy(z, torch.tensor([r['target'] for r in rows])).backward()
    close(z0, z, True); compare_grads(g0, gradients(model), True)
    for old, new in zip(input0, [v.grad for r in rows for _, v in r['events']]): close(old, new, True)
    passed.append('alpha0 logits, all parameter and observed-content gradients exactly nest old batched path')
    # Positive native sequential branch and complete first-time alternatives.
    a.score_bridge = .1; model.score_bridge = .1
    model.zero_grad(); model._fast_layers = {}; record = []
    z = D.batched_logits(model, rows, 3007, record=record)
    bat_ce = F.cross_entropy(z, torch.tensor([r['target'] for r in rows]), reduction='sum')
    for r in rows:
        for _, v in r['events']: v.grad = None
    bat_ce.backward(); gb = gradients(model)
    content_b = [v.grad.clone() for r in rows for _, v in r['events']]
    model.zero_grad()
    for r in rows:
        for _, v in r['events']: v.grad = None
    seq_losses = [LE.run(model, r, 3007)[0] for r in rows]
    close(bat_ce, torch.stack(seq_losses).sum()); torch.stack(seq_losses).sum().backward()
    compare_grads(gb, gradients(model))
    for old, new in zip(content_b, [v.grad for r in rows for _, v in r['events']]): close(old, new)
    passed.append('positive bridge factual logits/all parameter/content gradients match independent sequential reference')
    # Evaluate every alternative at a selected interior race; state writes are real.
    forces = [(4, i) for i in range(model.pool)]
    with torch.no_grad():
        model.train(); model._fast_layers = {}
        forced = D.batched_logits(model, [rows[0]] * model.pool, 3007, forces)
        sf = [LE.run(model, rows[0], 3007, force=f) for f in forces]
        for i, (loss, races, receiver_state) in enumerate(sf):
            close(F.cross_entropy(forced[i:i+1], torch.tensor([rows[0]['target']])), loss)
            assert receiver_state.selected_updates == len(rows[0]['events']) * a.depth * a.heads
            assert (2, 0, 0, i) in receiver_state.visited_units
        assert not torch.equal(forced[0], forced[1])
    passed.append('forced alternatives match sequential loss and commit different addressed receiver writes')
    # Full local-expectation surrogate, optimizer normalization and clipping.
    for alpha in (0., .1):
        a.score_bridge = alpha
        bat = D.make_model(a).double(); bat.load_state_dict(state)
        seq = D.make_model(a).double(); seq.load_state_dict(state)
        bo = torch.optim.Adam(bat.parameters(), lr=.003); so = torch.optim.Adam(seq.parameters(), lr=.003)
        with D.activate(): result = B.train_window(bat, bo, rows, a, 1, trace=True)
        seq_step(seq, so, rows, a, 1)
        compare_grads(gradients(bat), gradients(seq))
        for n, par in bat.named_parameters(): close(par, dict(seq.named_parameters())[n])
        assert result['route_replays'] == sum(len(r['events']) for r in rows) * a.depth * a.heads * a.pool
        assert result['selected_updates'] * a.pool == result['key_scores']
        for name, tr in result['stages'].items():
            assert tr['formula_coverage_complete'], (name, tr)
        coverage[str(alpha)] = result['stages']; counts[str(alpha)] = {k:v for k,v in result.items() if k != 'stages'}
        passed.append(f'alpha{alpha} full-route all-gradient/Adam normalization/clipping equivalence and complete stage accounting')
    # Partial-window exact optimizer/RNG continuation including existing Adam state.
    saved_rng = torch.get_rng_state().clone(); buf = io.BytesIO()
    torch.save(dict(model=bat.state_dict(), optimizer=bo.state_dict(), rng=saved_rng, cursor={'epoch':2,'window':0}), buf)
    with D.activate(): partial = B.train_window(bat, bo, rows[:1], a, 2, trace=True)
    after_rng = torch.get_rng_state().clone()
    buf.seek(0); saved = torch.load(buf, weights_only=False)
    recovered = D.make_model(a).double(); recovered.load_state_dict(saved['model'])
    ro = torch.optim.Adam(recovered.parameters(), lr=.003); ro.load_state_dict(saved['optimizer']); torch.set_rng_state(saved['rng'])
    assert saved['cursor'] == {'epoch':2,'window':0}
    with D.activate(): resumed = B.train_window(recovered, ro, rows[:1], a, 2)
    for n, par in bat.named_parameters(): close(par, dict(recovered.named_parameters())[n], True)
    compare_grads(gradients(bat), gradients(recovered), True); close(after_rng, torch.get_rng_state(), True)
    for k, val in bo.state_dict()['state'].items():
        for name, v in val.items(): close(v, ro.state_dict()['state'][k][name], True)
    for name, tr in partial['stages'].items(): assert tr['formula_coverage_complete'], name
    coverage['partial'] = partial['stages']; passed.append('partial-window model/Adam/RNG/cursor exact recovery and complete work coverage')
    raw = torch.tensor([13., -13.], dtype=torch.float64, requires_grad=True)
    derivative, = torch.autograd.grad(bounded_score_bridge(raw, .1).sum(), raw)
    assert bool((derivative > 0).all())
    passed.append('actual bounded-score derivative outside the old cap is positive')
    # Native evaluation uses only selected proposals, agrees with batched evaluation.
    with torch.no_grad():
        bat.eval(); bat._fast_layers = None
        z = D.batched_logits(bat, rows, 314159)
        native_states = []
        for j, row in enumerate(rows):
            logits, ss = N.predict(bat, row, 314159, False); close(z[j], logits)
            native_states.append(ss.storage())
            assert ss.counterfactual_values == 0
        tr = capture(lambda: N.predict(bat, rows[0], 314159, False))
        assert tr['formula_coverage_complete']
    coverage['selected_native_inference'] = tr
    passed.append('selected-value native inference equals batched output, real state retained and arithmetic covered')
    result = dict(status='completed', args=vars(a0), contracts_passed=len(passed), contracts=passed,
        source_sha256=D.sources(), workload=dict(payload=8, depth=4, heads=2, pool=2, variable_event_counts=[2,3]),
        activity=counts, coverage=coverage, native_state=native_states,
        wall_s=time.perf_counter()-started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Integrated numerical training/recovery/accounting contracts; no DEV/test quality or advantage evidence')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n'); print(json.dumps({k:result[k] for k in ('status','contracts_passed','wall_s','max_rss_kb')}))


if __name__ == '__main__': main()
