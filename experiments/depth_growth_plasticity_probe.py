"""Bounded FIT-only gate/plasticity audit of legacy direct D2->D4 growth.

Four fixed gate-bias interventions (-20, -8, -4, 0), one fresh normalized clip1
Adam step each. The larger-gate arms are diagnostics, not trained near-identity
models. No DEV/test evaluation, tuning or benchmark quality claim.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
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
import dvs_clock_calibrated_benchmark as C
import dvs_grow_depth_benchmark as G
import dvs_native_benchmark as N
import sleeping_machines.batched_episodes as K

PARENT = 'experiments/results/dvs_native/local_dvs_clock_full_20261002T153000Z.json'
SELF = 'experiments/depth_growth_plasticity_probe.py'
NOTE = 'experiments/theory/123_depth_growth_plasticity_probe.md'
SEED = 171323


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    else:
        assert a == b


def fit_inputs(parent):
    """Reproduce the parent's fine-packet FIT transform; never read DEV arrays."""
    a = argparse.Namespace(**parent['args'])
    data_path, controls_path = ROOT / a.data, ROOT / a.controls
    data = json.loads(data_path.read_text()); controls = json.loads(controls_path.read_text())
    if data['status'] != 'completed' or controls['status'] != 'completed':
        raise ValueError('Completed calibrated data required')
    if controls['data_result_sha256'] != N.sha(data_path):
        raise ValueError('Changed calibration binding')
    for artifact in (data, controls):
        for name, digest in artifact['source_sha256'].items():
            if N.sha(ROOT / name) != digest:
                raise ValueError('Changed calibration source: ' + name)
    count_path = ROOT / data['data_artifact']
    if N.sha(count_path) != data['data_sha256']:
        raise ValueError('Changed observed count artifact')
    with np.load(count_path, allow_pickle=False) as z:
        counts, labels, identities = z['fit_counts'], z['fit_labels'], z['fit_ids']
        raw = counts.reshape(len(labels), -1)
        center = np.log1p(raw).mean(0); scale = np.maximum(np.log1p(raw).std(0), .5)
        transform_sha = hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest()
        rows = [dict(index=i, identity=str(identities[i]), target=int(labels[i]),
            events=N.encode(counts[i], center, scale)) for i in range(16)]
    for name, digest in dict(data_result_sha256=N.sha(data_path), controls_sha256=N.sha(controls_path),
            data_artifact_sha256=N.sha(count_path), transform_sha256=transform_sha).items():
        if parent['data'][name] != digest:
            raise ValueError('Parent FIT input protocol mismatch: ' + name)
    return rows, dict(data_result=a.data, data_result_sha256=N.sha(data_path),
        controls=a.controls, controls_sha256=N.sha(controls_path),
        count_artifact=data['data_artifact'], count_artifact_sha256=N.sha(count_path),
        transform_sha256=transform_sha, fit_indices=list(range(16)),
        scope='Original fine-packet FIT-only transform; DEV/test arrays are not read')


def stats(values):
    x = torch.cat([v.detach().double().flatten() for v in values])
    assert len(x) and bool(torch.isfinite(x).all())
    return dict(elements=x.numel(), minimum=float(x.min()), mean=float(x.mean()),
        median=float(x.median()), maximum=float(x.max()), l2=float(x.norm()),
        nonzero_fraction=float((x != 0).double().mean()))


@contextmanager
def telemetry(model, rows):
    sigmoid, race = torch.sigmoid, K.LaneRace
    lengths = torch.tensor([len(r['events']) for r in rows])
    records = {d: dict(gate=[], gate_derivative=[], raw_gate=[], candidate_delta=[], selected_delta=[])
        for d in range(model.depth)}
    holder = dict(gate_calls=0, race_calls=0, incoming=None)
    hooks = []
    def incoming_hook(module, inputs, output):
        holder['incoming'] = output.detach().view(len(rows), model.heads, model.payload)
    for mix in model.channel_mix:
        hooks.append(mix.register_forward_hook(incoming_hook))
    def instrumented_sigmoid(value, *args, **kwargs):
        result = sigmoid(value, *args, **kwargs)
        if value.ndim == 4 and tuple(value.shape) == (len(rows), model.heads, model.pool, model.payload):
            index = holder['gate_calls']; depth = index % model.depth; event = index // model.depth
            active = lengths > event; holder['gate_calls'] += 1
            records[depth]['gate'].append(result.detach()[active].clone())
            records[depth]['gate_derivative'].append((result.detach() * (1 - result.detach()))[active].clone())
            records[depth]['raw_gate'].append(value.detach()[active].clone())
        return result
    def instrumented_race(scores, proposals, noise, forced):
        index = holder['race_calls']; head = index % model.heads
        depth = (index // model.heads) % model.depth
        event = index // (model.heads * model.depth); active = lengths > event
        holder['race_calls'] += 1
        value, delay, winner = race.apply(scores, proposals, noise, forced)
        incoming = holder['incoming'][:, head, None, :].expand_as(proposals)
        delta = proposals.detach() - incoming
        records[depth]['candidate_delta'].append(delta[active].clone())
        lane = torch.arange(len(rows))
        records[depth]['selected_delta'].append(delta[lane, winner][active].clone())
        return value, delay, winner
    torch.sigmoid = instrumented_sigmoid; K.LaneRace = SimpleNamespace(apply=instrumented_race)
    try:
        yield records, holder
    finally:
        torch.sigmoid, K.LaneRace = sigmoid, race
        for hook in hooks:
            hook.remove()


def summarize(records, holder, rows, model):
    T = max(len(r['events']) for r in rows)
    assert holder['gate_calls'] == T * model.depth
    assert holder['race_calls'] == T * model.depth * model.heads
    return dict(layers=[dict(depth=d, added=d >= 2,
        **{name: stats(values) for name, values in records[d].items()}) for d in range(model.depth)],
        gate_calls=holder['gate_calls'], race_calls=holder['race_calls'],
        scope='Only active episode lanes. Proposal delta is the actual FP32 subtraction, including rounded-away branches')


def forward(model, rows, instrument=False):
    model.train(); expected_rng = torch.get_rng_state().clone()
    if instrument:
        with telemetry(model, rows) as (records, holder):
            logits = K.batched_logits(model, rows, SEED)
        observed = summarize(records, holder, rows, model)
    else:
        logits = K.batched_logits(model, rows, SEED); observed = None
    assert torch.equal(expected_rng, torch.get_rng_state())
    loss_sum = F.cross_entropy(logits, torch.tensor([r['target'] for r in rows]), reduction='sum')
    return logits, loss_sum, observed


def gradients(model, rows, instrument=False):
    model.zero_grad(set_to_none=True)
    logits, loss, observed = forward(model, rows, instrument)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(len(rows))
    gradient = {name: torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()}
    return logits.detach(), float(loss.detach()) / len(rows), gradient, observed


def slices(name, tensor):
    if name.startswith('units.'):
        parts = name.split('.'); return [(f'layer{parts[1]}/units/{parts[5]}', tensor)]
    if name.startswith('queries.') or name.startswith('channel_mix.'):
        parts = name.split('.'); return [(f'layer{parts[1]}/{parts[0]}', tensor)]
    if name in ('transport_rate', 'transport_frequency'):
        return [(f'layer{d}/{name}', tensor[d]) for d in range(tensor.shape[0])]
    return [(name.split('.')[0], tensor)]


def step(model, rows, arrays, arm):
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    # Independent plain backward demonstrates instrumentation's exact nesting.
    plain_z, plain_loss, plain_g, _ = gradients(model, rows, False)
    z, initial_loss, raw, measured_before = gradients(model, rows, True)
    exact(z, plain_z); exact(raw, plain_g); assert initial_loss == plain_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    assert not optimizer.state
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    clipped = {name: torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()}
    optimizer.step()
    actual = {name: p.detach() - before[name] for name, p in model.named_parameters()}
    eps = optimizer.param_groups[0]['eps']; lr = optimizer.param_groups[0]['lr']
    formula = {name: -lr * g / (g.abs() + eps) for name, g in clipped.items()}
    for name, p in model.named_parameters():
        error = (p.detach() - (before[name] + formula[name])).abs()
        bound = 4 * torch.finfo(p.dtype).eps * before[name].abs().clamp_min(1)
        assert bool((error <= bound).all()), name
    with torch.no_grad():
        post_z, post_sum, measured_after = forward(model, rows, True)
    group = {}
    for name in before:
        for block, old in slices(name, before[name]):
            record = group.setdefault(block, dict(raw=[], clipped=[], actual=[], formula=[], before=[]))
            for field, mapping in (('raw', raw), ('clipped', clipped), ('actual', actual),
                    ('formula', formula), ('before', before)):
                record[field].append(dict(slices(name, mapping[name]))[block].double().flatten())
    blocks = []
    for block, record in sorted(group.items()):
        vectors = {name: torch.cat(values) for name, values in record.items()}
        clipped_g = vectors['clipped']; active = clipped_g != 0
        cancellation = clipped_g.abs() / (clipped_g.abs() + eps)
        blocks.append(dict(block=block, parameter_elements=clipped_g.numel(),
            raw_mean_gradient_l2=float(vectors['raw'].norm()), clipped_gradient_l2=float(clipped_g.norm()),
            actual_FP32_displacement_l2=float(vectors['actual'].norm()),
            fresh_Adam_formula_displacement_l2=float(vectors['formula'].norm()),
            actual_changed_coordinate_fraction=float((vectors['actual'] != 0).double().mean()),
            nonzero_gradient_fraction=float(active.double().mean()),
            clipped_gradient_below_eps_fraction=float((clipped_g.abs() < eps).double().mean()),
            active_gradient_below_eps_fraction=None if not bool(active.any()) else
                float((clipped_g[active].abs() < eps).double().mean()),
            active_Adam_sign_step_fraction_mean=None if not bool(active.any()) else float(cancellation[active].mean()),
            actual_formula_displacement_difference_l2=float((vectors['actual'] - vectors['formula']).norm())))
    names = list(before)
    for field, values in (('raw_gradient', raw), ('clipped_gradient', clipped),
            ('actual_displacement', actual), ('formula_displacement', formula)):
        arrays[f'{arm}_{field}'] = torch.cat([values[n].double().flatten() for n in names]).numpy()
    arrays[f'{arm}_parameter_names'] = np.asarray(names)
    arrays[f'{arm}_parameter_elements'] = np.asarray([before[n].numel() for n in names], dtype=np.int64)
    gates = []
    for depth in range(model.depth):
        nms = [n for n in before if n.startswith(f'units.{depth}.') and n.endswith('.gate.bias')]
        gates.append(dict(depth=depth, added=depth >= 2,
            before=stats([before[n] for n in nms]),
            after=stats([dict(model.named_parameters())[n].detach() for n in nms])))
    return dict(arm=arm, initial_gate_bias={'closed20': -20., 'open8': -8., 'skip4': -4., 'open0': 0.}[arm],
        other_host_section410_gate_only_counterpart=arm == 'skip4',
        fit_targets=len(rows), optimizer_updates=1, optimizer='Fresh Adam .003, default betas/eps',
        epsilon=eps, normalized_raw_gradient_norm=norm, clip_cap=1.,
        observed_clip_scale=1. if norm == 0 else min(1., 1. / (norm + 1e-6)),
        initial_same_FIT_nll=initial_loss, post_same_FIT_nll=float(post_sum) / len(rows),
        finite_same_FIT_nll_change=float(post_sum) / len(rows) - initial_loss,
        initial_to_post_mean_KL=float((z.softmax(-1) * (z.log_softmax(-1) - post_z.log_softmax(-1))).sum(-1).mean()),
        gate_biases=gates, before=measured_before, after=measured_after,
        parameter_blocks=blocks, instrumentation_all_logits_and_gradients_bitwise=True,
        fresh_Adam_formula_with_coordinate_storage_bound_passed=True,
        scope='One diagnostic update on the SAME fixed16 FIT gestures. No heldout performance or learning-curve inference')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True); a = parser.parse_args()
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    artifact = out.with_suffix('.vectors.npz')
    if Path(a.tag).name != a.tag or out.exists() or artifact.exists():
        raise ValueError('Unused plain tag required; preserve completed evidence')
    started = time.perf_counter(); torch.set_num_threads(1)
    parent_path = ROOT / PARENT; checkpoint_path = parent_path.with_suffix('.progress.pt')
    hashes = {str(parent_path): N.sha(parent_path), str(checkpoint_path): N.sha(checkpoint_path)}
    parent = json.loads(parent_path.read_text())
    if parent['status'] != 'completed' or parent['args']['depth'] != 2:
        raise ValueError('Completed fixed D2 parent required')
    for name, digest in parent['source_sha256'].items():
        if N.sha(ROOT / name) != digest:
            raise ValueError('Changed parent model source: ' + name)
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if ck['source_sha256'] != parent['source_sha256'] or ck['data'] != parent['data']:
        raise ValueError('Parent checkpoint/data mismatch')
    parent_weights = copy.deepcopy(ck['best_state'])
    inputs, data = fit_inputs(parent)
    source = {**G.sources(), **{name: N.sha(ROOT / name) for name in
        (SELF, NOTE, 'sleeping_machines/batched_episodes.py', 'experiments/dvs_batched_le_benchmark.py')}}
    caller_rng = torch.get_rng_state().clone(); original_sigmoid, original_race = torch.sigmoid, K.LaneRace
    arrays = {}; result_rows = []
    with torch.random.fork_rng():
        settings = argparse.Namespace(**parent['args']); settings.depth = 4
        grown = G.grow(C.make_model(settings, True), parent_weights, 2)
        unchanged = copy.deepcopy(grown.state_dict())
        for arm, bias in (('closed20', -20.), ('open8', -8.), ('skip4', -4.), ('open0', 0.)):
            model = copy.deepcopy(grown)
            with torch.no_grad():
                for layer in model.units[2:]:
                    for head in layer:
                        for pool in head:
                            for unit in pool:
                                unit.gate.bias.fill_(bias)
            result_rows.append(step(model, inputs, arrays, arm))
            exact(grown.state_dict(), unchanged)
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert torch.sigmoid is original_sigmoid and K.LaneRace is original_race
    exact(ck['best_state'], parent_weights)
    assert all(N.sha(path) == digest for path, digest in hashes.items())
    assert all(N.sha(ROOT / path) == digest for path, digest in source.items())
    np.savez_compressed(artifact, **arrays)
    result = dict(status='completed', args=vars(a), rows=result_rows, source_sha256=source,
        parent=PARENT, parent_result_sha256=hashes[str(parent_path)],
        parent_checkpoint_sha256=hashes[str(checkpoint_path)], parent_selected_epoch=parent['selected_epoch'],
        parent_weight_scope='Selected parent best_state; fresh Adam is intentional, saved terminal moments are not paired',
        data=data, race_noise_seed=SEED, payload=16, depth=4, heads=2, pool=2,
        actual_fit_targets=64, actual_optimizer_updates=4,
        artifact=str(artifact.relative_to(ROOT)), artifact_sha256=N.sha(artifact),
        instrumentation_contracts_passed=4, fresh_step_formula_contracts_passed=4,
        parent_and_base_immutability_passed=True, caller_RNG_and_kernel_functions_preserved=True,
        wall_s=time.perf_counter() - started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fitting_FLOPs=None, traffic_bytes=None, energy_joules=None,
        scope='FIT-only bounded plasticity mechanism diagnostic. Interventions -8/-4/0 are deliberately different initial functions, '
              'not demonstrated near-identity growth repairs. The -4 arm is the gate-only counterpart to other-host section410; '
              'it retains the trained shallow parent and does not reproduce that full scratch skip-init protocol. '
              'Four independent one-step fresh-Adam forks; '
              'full instrumentation checks/plain backwards and before/after forwards paid in campaign wall. '
              'No DEV/test, tuning, trained quality claim or inference-work advantage; total audit FLOPs/traffic/energy unknown, not zero.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'wall_s', 'max_rss_kb', 'actual_optimizer_updates')}))


if __name__ == '__main__':
    main()
