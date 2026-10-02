"""Frozen native recurrent-state attribution with fixed causal count evidence.

This is a small development diagnostic, not retraining, a semantic probe, or a
full-development quality comparison. The learned gate is allowed to respond to
the changed base; the count table and its absolute stream cursor stay fixed.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from e120_shared_tasks import text_slice
from language_learning_audit import fingerprint, layer_gradients, piece
from sleeping_machines.count_carrying_language import compose, eval_stream_counts, fit_stream_counts
from sleeping_machines.count_escape_gate import GatedCountCarryingNativeModel


def erase(state, kind, layer=None):
    """Erase stored content, preserving absolute time and count-table position."""
    if kind in ('receiver_content', 'all_content'):
        state.memories = {k: torch.zeros_like(v) if layer is None or k[0] == layer else v
                          for k, v in state.memories.items()}
    if kind in ('source_context', 'all_content'):
        state.contexts = {k: (torch.zeros_like(v), t) for k, (v, t) in state.contexts.items()}
    return state


def statistics(logp, targets, reference=None):
    row = dict(bpc=float(F.nll_loss(logp, targets)) / math.log(2),
               accuracy=float((logp.argmax(-1) == targets).float().mean()))
    if reference is not None:
        row.update(mean_kl_from_intact=float((reference.exp() * (reference - logp)).sum(-1).mean()),
                   max_log_probability_change=float((reference - logp).abs().max()))
    return row


def audit(stem):
    path = ROOT / 'experiments/results/count_carrying_language' / (stem + '.json')
    row = json.loads(path.read_text()); assert row['status'] == 'completed'
    for source, sha in row['source_sha256'].items():
        if source.startswith('sleeping_machines/'):
            assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == sha, source
    checkpoint = path.with_suffix('.progress.pt')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['result']['status'] == 'completed' and saved['result']['final'] == row['final']
    a = row['args']; assert a['escape_gate'] and not a['count_message']
    model = GatedCountCarryingNativeModel(a['payload'], a['depth'], a['pool'], heads=a['heads'],
        orders=a['orders'], escape_gate=True, count_message=False)
    model.load_state_dict(saved['model']); before = fingerprint(model)
    fit = torch.tensor(text_slice(0, a['fit'])); dev = torch.tensor(text_slice(90_000_000, a['dev']))
    for key, data in (('fitting_data_sha256', fit), ('development_data_sha256', dev)):
        assert hashlib.sha256(data.numpy().astype('uint8').tobytes()).hexdigest() == row[key]
    model.register_stream('fit', fit_stream_counts(fit.numpy(), a['orders']))
    model.register_stream('dev', eval_stream_counts(fit.numpy(), dev.numpy(), a['orders']))
    begin, end = 128, 160
    model.eval(); model.use_stream('dev'); raw = []
    handle = model.head.register_forward_hook(lambda _m, _i, z: raw.append(z))
    arms = [('intact', None, None), ('erase_source_context', 'source_context', None),
            ('erase_receiver_content', 'receiver_content', None), ('erase_all_content', 'all_content', None)]
    arms += [(f'erase_layer_{d+1}_receiver_content', 'receiver_content', d) for d in range(model.depth)]
    interventions = []; reference = None; raw_reference = None
    with torch.no_grad(), torch.random.fork_rng():
        _, warm = piece(model, dev, 0, begin, True); warm.detach()
        for name, kind, layer in arms:
            state = copy.deepcopy(warm); cursor = state.events
            if kind is not None: erase(state, kind, layer)
            assert state.events == cursor == begin
            raw.clear(); logp, state = piece(model, dev, begin, end, True, state)
            logq = torch.stack(raw).log_softmax(-1); targets = dev[begin+1:end+1]
            if reference is None: reference, raw_reference = logp, logq
            interventions.append(dict(intervention=name,
                composed=statistics(logp, targets, reference), base=statistics(logq, targets, raw_reference),
                scored_targets=end-begin, count_cursor=cursor))
        counts = model.streams['dev'][:, begin:end]
        D, th = model.escape_gate(raw_reference, counts, model.raw_discount, model.raw_theta)
        uniform = torch.full_like(raw_reference, -math.log(27))
        uniform_score = statistics(compose(uniform, counts, D, th), targets)
    # A genuine fitting-data window and actual trained model, without optimizer steps.
    # Compare output-logit credit, isolating the direct residual path by freezing
    # D/theta at their identical-forward values; retain the actual dynamic gate too.
    model.use_stream('fit'); model.train()
    with torch.no_grad(), torch.random.fork_rng():
        raw.clear(); _, state = piece(model, fit, 0, 64, True); state.detach()
    raw.clear(); model.zero_grad(set_to_none=True)
    with torch.random.fork_rng():
        actual, _ = piece(model, fit, 64, 128, True, state)
        z = torch.stack(raw); logq = z.log_softmax(-1); targets = fit[65:129]
        counts = model.streams['fit'][:, 64:128]
        D, th = model.escape_gate(logq, counts, model.raw_discount, model.raw_theta)
        dynamic = compose(logq, counts, D, th)
        torch.testing.assert_close(dynamic, actual, atol=2e-6, rtol=2e-6)
        fixed = compose(logq, counts, D.detach(), th.detach())
        losses = [F.nll_loss(dynamic, targets), F.nll_loss(fixed, targets), F.nll_loss(logq, targets)]
        grads = [torch.autograd.grad(loss, z, retain_graph=True)[0].detach() for loss in losses]
        q = logq.detach().exp(); escape = torch.ones(len(targets))
        for k in range(a['orders']):
            c = counts[k]; n = c.sum(-1); types = (c > 0).sum(-1)
            factor = (th[k].detach().squeeze(-1) + D[k].detach().squeeze(-1)*types) / (n + th[k].detach().squeeze(-1))
            escape *= torch.where(n > 0, factor, torch.ones_like(factor))
        responsibility = escape*q.gather(1, targets[:, None]).squeeze(1) / fixed.detach().exp().gather(1, targets[:, None]).squeeze(1)
        expected = responsibility[:, None] * (q - F.one_hot(targets, 27)) / len(targets)
        torch.testing.assert_close(grads[1], expected, atol=2e-7, rtol=2e-5)
        losses[0].backward(); layers = layer_gradients(model, native=True)
    handle.remove(); assert fingerprint(model) == before
    credit = dict(fitting_positions=[64, 128], targets=64, credit_events=64,
        direct_residual_responsibility=dict(mean=float(responsibility.mean()), min=float(responsibility.min()),
            max=float(responsibility.max()), quantiles=torch.quantile(responsibility, torch.tensor([.25,.5,.75])).tolist()),
        responsibility_gradient_max_error=float((grads[1]-expected).abs().max()),
        output_logit_gradient_norms=dict(dynamic_gate=float(grads[0].norm()), fixed_gate=float(grads[1].norm()),
                                        standalone_base=float(grads[2].norm())),
        gate_path_gradient_norm=float((grads[0]-grads[1]).norm()),
        dynamic_to_standalone_gradient_ratio=float(grads[0].norm()/grads[2].norm()),
        fixed_to_standalone_gradient_ratio=float(grads[1].norm()/grads[2].norm()),
        fitting_bpc=dict(composed=float(losses[0].detach())/math.log(2), base=float(losses[2].detach())/math.log(2)),
        actual_dynamic_gate_layer_gradients=layers,
        all_layers_receive_gradient=all(r['gradient_norm'] > 0 for r in layers))
    return dict(result=str(path.relative_to(ROOT)), result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), model_fingerprint=before,
        development_positions=[begin, end], interventions=interventions,
        uniform_base_with_intact_gate=uniform_score, credit=credit, weights_preserved=True, optimizer_steps=0,
        forward_input_tokens=begin+len(arms)*(end-begin)+128,
        output_logit_gradient_evaluations=3, full_model_backwards=1,
        scope='One 32-target development slice; memory erased once at its start, then allowed to rebuild. '
              'Arrival timestamps, absolute count cursor and causal count vectors preserved; learned gate can respond. '
              'Uniform-base diagnostic freezes intact gate. Raw base is a trained residual, not a standalone model. '
              'Three output-logit gradient contrasts and one actual surrogate model backward on 64 fit targets; '
              'standalone-base gradient is a diagnostic, not an alternative training recommendation.')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--tag', required=True); a = p.parse_args()
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists(): raise ValueError('Unique unused plain tag required')
    started = time.perf_counter(); torch.set_num_threads(1)
    stems = ['local_count_credit64_full_D2048_20261002T060000Z',
             'local_count_credit64_minimal_D2048_20261002T060000Z']
    rows = []
    for stem in stems:
        rows.append(audit(stem)); print(json.dumps(dict(completed=stem)), flush=True)
    names = ['experiments/deep_core_attribution_audit.py', 'experiments/language_learning_audit.py',
             'experiments/e120_shared_tasks.py']
    sources = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    for row in rows:
        sources.update(json.loads((ROOT/row['result']).read_text())['source_sha256'])
    result = dict(status='completed', models=rows, optimizer_steps=0, weights_preserved=True,
        official_test_read=False, arithmetic_flops_instrumented=False,
        source_sha256=sources, hardware=dict(host=__import__('os').uname().nodename, device='cpu',
            torch=torch.__version__, threads=1), wall_s=time.perf_counter()-started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(completed=a.tag, wall_s=result['wall_s'], max_rss_kb=result['max_rss_kb'])), flush=True)


if __name__ == '__main__': main()
