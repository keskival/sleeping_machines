"""Numerical and artifact contracts for gain-preserving DVS depth growth.

Only tiny forward/backward correctness cases and temporary synthetic parent
artifacts are used.  No encoder is fitted and no accuracy gain is claimed.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import resource
import sys
import tempfile
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_clock_calibrated_benchmark as C
import dvs_grow_depth_benchmark as G
import dvs_grow_depth_lineage_benchmark as D
import dvs_native_benchmark as N
from sleeping_machines.batched_episodes import batched_logits


def args(depth, parent=None):
    value = argparse.Namespace(payload=4, depth=depth, heads=2, pool=2,
        clock_step=.05, seed=7, tag='synthetic_parent', epochs=1,
        update_targets=2, lr=.003, data='synthetic_contract_only',
        controls='synthetic_contract_only', contracts=None, fit=2, dev=2,
        resume=False, stop_after_updates=None)
    if parent is not None:
        value.parent = str(parent)
    return value


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert type(a) == type(b) and a == b


def rejects(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError, KeyError) as error:
        return str(error)
    raise AssertionError('Invalid lineage accepted')


def write_parent(folder, name, model, a, kind='ordinary', info=None):
    source = C.sources()
    if kind == 'legacy':
        source = {**source, D.LEGACY: N.sha(ROOT / D.LEGACY)}
    elif kind == 'new':
        source = D.sources()
    data = dict(scope='Temporary synthetic lineage contract; no DVS quality evidence')
    result = dict(status='completed', args=vars(a), source_sha256=source,
        data=data, selected_epoch=1, final=dict(nll=2.),
        parameters=sum(p.numel() for p in model.parameters()))
    if info is not None:
        result['growth_lineage'] = copy.deepcopy(info)
    state = copy.deepcopy(model.state_dict())
    settings = {k: v for k, v in vars(a).items() if k not in ('tag', 'resume', 'stop_after_updates')}
    ck = dict(source_sha256=source, data=data, settings=settings,
        best_state=state, online_model=copy.deepcopy(state), best=2.,
        result=copy.deepcopy(result), cursor=dict(epoch=2, window=0, updates=0))
    if info is not None:
        ck['growth_lineage'] = copy.deepcopy(info)
    path = folder / (name + '.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    torch.save(ck, path.with_suffix('.progress.pt'))
    return path


def factual(model, rows):
    model.zero_grad(set_to_none=True)
    model.train()
    before = torch.get_rng_state().clone()
    logits = batched_logits(model, rows, 713)
    F.cross_entropy(logits, torch.tensor([r['target'] for r in rows])).backward()
    gradient = {n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()}
    assert torch.equal(before, torch.get_rng_state())
    return logits.detach(), gradient


def sequential(model, rows):
    model.eval(); outputs = []; states = []
    with torch.no_grad(), torch.random.fork_rng():
        for row in rows:
            torch.manual_seed(713); state = model.new_state()
            for t, mark in row['events']:
                z, _ = model.consume_event(0, t, mark, state)
            outputs.append(z); states.append(copy.deepcopy(vars(state)))
    return torch.stack(outputs), states


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args()
    if Path(a.tag).name != a.tag:
        raise ValueError('Unused plain tag required')
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if out.exists():
        raise ValueError('Preserve completed contracts')
    started = time.perf_counter(); torch.set_num_threads(1)
    checks = []; rows = []
    generator = torch.Generator().manual_seed(921)
    cases = [dict(index=j, target=j + 1,
        events=[((k + 1) * .05, torch.randn(33, generator=generator)) for k in range(3 + j)])
        for j in range(2)]
    parent = C.make_model(args(2), True).double()
    parent_state = copy.deepcopy(parent.state_dict())
    legacy = G.grow(C.make_model(args(4), True).double(), parent_state, 2)
    fixed = D.grow(C.make_model(args(4), True).double(), parent_state, D.layer_gains(parent))
    exact(legacy.state_dict(), fixed.state_dict())
    exact(D.layer_gains(legacy), D.layer_gains(fixed))
    before = torch.get_rng_state().clone()
    exact(factual(legacy, cases), factual(fixed, cases))
    exact(sequential(legacy, cases), sequential(fixed, cases))
    assert torch.equal(before, torch.get_rng_state())
    exact(parent_state, parent.state_dict())
    checks.append('Direct D2->D4 exactly nests legacy weights, gains, every factual logit/gradient/state and caller RNG')
    correct6 = D.grow(C.make_model(args(6), True).double(), fixed.state_dict(), D.layer_gains(fixed))
    incorrect6 = G.grow(C.make_model(args(6), True).double(), fixed.state_dict(), 4)
    expected = [.5 / math.sqrt(2)] * 2 + [.5 / math.sqrt(4)] * 2 + [.5 / math.sqrt(6)] * 2
    assert D.layer_gains(correct6) == expected
    assert D.layer_gains(incorrect6)[:2] != expected[:2]
    for name, value in fixed.state_dict().items():
        if name in ('transport_rate', 'transport_frequency'):
            exact(correct6.state_dict()[name][:4], value)
        else:
            exact(correct6.state_dict()[name], value)
    checks.append('D2->D4->D6 preserves every old tensor and all heterogeneous gains; legacy reset is explicitly witnessed')
    assert all(float(u.gate.bias.mean()) == G.CLOSED_GATE
        for layer in correct6.units[4:] for head in layer for pool in head for u in pool)
    assert torch.equal(correct6.transport_frequency[4], torch.zeros_like(correct6.transport_frequency[4]))
    exact(correct6.transport_frequency[-1], fixed.transport_frequency[-1])
    exact(correct6.transport_rate[-1], fixed.transport_rate[-1])
    checks.append('New layers retain legacy closed gates, no intermediate rotation and inherited top temporal transport')
    source_names = list(D.sources())
    source_before = {name: N.sha(ROOT / name) for name in source_names}
    with tempfile.TemporaryDirectory(prefix='dvs-growth-lineage-') as temporary:
        folder = Path(temporary)
        p2 = C.make_model(args(2), True)
        path2 = write_parent(folder, 'ordinary2', p2, args(2))
        p4 = G.grow(C.make_model(args(4), True), p2.state_dict(), 2)
        path4 = write_parent(folder, 'legacy4', p4, args(4, path2), 'legacy')
        bytes_before = {path: path.read_bytes() for path in folder.iterdir()}
        resolved = D.resolve_parent(path4)
        assert resolved['layer_gains'] == D.layer_gains(p4)
        grow_args = args(6, path4); grow_args.seed = 9
        model6 = D.make_model(grow_args)
        assert D.layer_gains(model6) == expected
        info6 = model6._depth_growth_lineage
        path6 = write_parent(folder, 'new6', model6, grow_args, 'new', info6)
        resolved6 = D.resolve_parent(path6)
        assert resolved6['layer_gains'] == expected
        model8 = D.make_model(args(8, path6))
        assert D.layer_gains(model8)[:6] == expected
        checks.append('Actual ordinary/direct legacy parents reconstruct gains; new metadata survives D6 checkpoint reload and D6->D8')
        # Historical progressive parents are reconstructed as they actually ran.
        old6 = G.grow(C.make_model(args(6), True), p4.state_dict(), 4)
        oldpath6 = write_parent(folder, 'legacy_reset6', old6, args(6, path4), 'legacy')
        assert D.resolve_parent(oldpath6)['layer_gains'] == D.layer_gains(old6)
        checks.append('An existing legacy progressive parent retains its historical reset rather than rewriting evidence')
        with D.activate(grow_args):
            active_model = N.make_model(grow_args)
            assert D.layer_gains(active_model) == expected
            actual_sources = N.sources()
            assert all(actual_sources[path] == digest for path, digest in resolved['bindings'].items())
        checks.append('Activated actual batched factory preserves heterogeneous gains and binds every ancestor result/checkpoint')
        captured = []; original_save = torch.save
        def capture_save(payload, destination, *positional, **keywords):
            captured.append(copy.deepcopy(payload))
        torch.save = capture_save
        try:
            with D.activate(grow_args):
                payload = dict(source_sha256=actual_sources, settings=vars(grow_args),
                    cursor=dict(updates=0), result=dict(status='running'))
                destination = ROOT / 'experiments/results/dvs_native' / (grow_args.tag + '.progress.tmp')
                torch.save(payload, destination)
                assert captured[-1]['growth_lineage'] == info6
                assert captured[-1]['result']['growth_lineage'] == info6
                unrelated = dict(source_sha256=actual_sources, settings={}, cursor={}, result={})
                torch.save(unrelated, folder / 'unrelated.pt')
                assert 'growth_lineage' not in captured[-1]
        finally:
            torch.save = original_save
        checks.append('Every actual recovery-snapshot destination receives gain metadata; unrelated serialization remains untouched')
        badsource = copy.deepcopy(json.loads(path2.read_text()))
        badsource['source_sha256']['experiments/dvs_unknown_representational_benchmark.py'] = '0' * 64
        unknownpath = folder / 'unknown_constructor.json'; unknownpath.write_text(json.dumps(badsource))
        rejects(lambda: D.resolve_parent(unknownpath))
        badsource = copy.deepcopy(json.loads(path2.read_text()))
        badsource['source_sha256']['sleeping_machines/sparse_race_language.py'] = '0' * 64
        unknownpath.write_text(json.dumps(badsource)); rejects(lambda: D.resolve_parent(unknownpath))
        checks.append('Unknown parent construction and changed frozen forward-source bindings are refused')
        rejects(lambda: D.grow(C.make_model(args(6), True), p4.state_dict(), [float('nan')] * 4))
        bad = args(6, path4); bad.pool = 3
        rejects(lambda: D.make_model(bad))
        missing = copy.deepcopy(json.loads(path4.read_text())); missing['args'].pop('parent')
        badpath = folder / 'missing.json'; badpath.write_text(json.dumps(missing))
        rejects(lambda: D.resolve_parent(badpath))
        cyclic = copy.deepcopy(json.loads(path4.read_text())); cyclic['args']['parent'] = str(folder / 'cyclic.json')
        (folder / 'cyclic.json').write_text(json.dumps(cyclic))
        rejects(lambda: D.resolve_parent(folder / 'cyclic.json'))
        checks.append('Nonfinite gains, dimension mismatch, missing lineage and ancestry cycles are refused')
        badresult = copy.deepcopy(json.loads(path6.read_text()))
        badresult['growth_lineage']['layer_gains'][0] = .25
        badpath = folder / 'metadata_mismatch.json'; badpath.write_text(json.dumps(badresult))
        rejects(lambda: D.resolve_parent(badpath))
        cp = path6.with_suffix('.progress.pt'); original_cp = cp.read_bytes()
        broken = torch.load(cp, weights_only=False)
        broken['growth_lineage']['layer_gains'][0] = .25
        torch.save(broken, cp); rejects(lambda: D.resolve_parent(path6)); cp.write_bytes(original_cp)
        broken = torch.load(cp, weights_only=False)
        broken['best_state']['head.weight'] = broken['best_state']['head.weight'][:1]
        torch.save(broken, cp); rejects(lambda: D.resolve_parent(path6)); cp.write_bytes(original_cp)
        broken = torch.load(cp, weights_only=False); broken['source_sha256'] = {}
        torch.save(broken, cp); rejects(lambda: D.resolve_parent(path6)); cp.write_bytes(original_cp)
        checks.append('Persisted gain/result/checkpoint disagreement, state-shape mismatch and source mismatch are refused')
        for path, blob in bytes_before.items():
            assert path.read_bytes() == blob
        assert cp.read_bytes() == original_cp
        exact(parent_state, parent.state_dict())
        checks.append('All original parent result/checkpoint bytes and in-memory weights remain unchanged')
        rows.append(dict(parent_depths=[2, 4], grown_depth=6, layer_gains=expected,
            legacy_d6_layer_gains=D.layer_gains(incorrect6), checkpoint_reloaded_depth=8,
            source_and_checkpoint_bindings=resolved['bindings'],
            scope='Temporary synthetic numerical/artifact contracts; no fitted quality, optimizer update or benchmark work comparison'))
    assert source_before == {name: N.sha(ROOT / name) for name in source_names}
    checks.append('Every existing/new contracted source remains unchanged during admission')
    result = dict(status='completed', args=vars(a), contracts_passed=len(checks), contracts=checks,
        rows=rows, source_sha256=D.sources(), wall_s=time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen-safe lineage sibling correctness only. Closed-gate plasticity, function-preserving clock/noise extension, '
              'restored Adam continuity and deeper-model quality remain separate open checks.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'contracts_passed', 'wall_s', 'max_rss_kb')}))


if __name__ == '__main__':
    main()
