"""Guarded frozen language/history and actual count-composition gradient audit.

No optimizer, parameter update, checkpoint substitution or official-test access.
Equal cross entropy to an n-gram is a calibration, not a feature measurement.
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
from sleeping_machines.native_stream_language import NativeStreamLanguageModel
from sleeping_machines.selective_stream_language import SelectiveEventLanguageModel
from sleeping_machines.count_composed_stream import CountComposedModel
from sleeping_machines.count_carrying_language import fit_stream_counts


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def layer_gradients(model, native=False):
    rows = []
    for depth in range(model.depth):
        prefix = f'units.{depth}.' if native else f'layers.{depth}.'
        groups = [prefix, f'queries.{depth}.', f'channel_mix.{depth}.'] if native else [prefix]
        values = {name: None if p.grad is None else float(p.grad.norm())
                  for name, p in model.named_parameters() if any(name.startswith(g) for g in groups)}
        assert all(value is None or math.isfinite(value) for value in values.values())
        rows.append(dict(depth=depth + 1, parameter_gradient_norms=values,
                         gradient_norm=math.sqrt(sum(v*v for v in values.values() if v is not None))))
    return rows


def piece(model, tokens, begin, end, native, state=None):
    fresh = state is None
    state = model.new_state() if fresh else state
    if native:
        if fresh: state.events = begin
        outputs = []
        for position in range(begin, end):
            # Coupled per-position noise: identical retained suffixes receive
            # identical clock draws across history interventions.
            torch.manual_seed(314159 + position)
            out, state = model.forward_chunk(tokens[position:position+1], state)
            outputs.append(out)
        return torch.cat(outputs), state
    if fresh: state.position = begin
    return model.forward_chunk(tokens[begin:end], state)


def frozen_audit(name, native, tokens):
    path = ROOT / name
    row = json.loads(path.read_text()); assert row['status'] == 'completed'
    for source, sha in row['source_sha256'].items():
        if source.startswith('sleeping_machines/'):
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == sha, source
    checkpoint = path.with_suffix('.progress.pt')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['result']['status'] == 'completed' and saved['result']['final'] == row['final']
    args = row['args']
    model = (NativeStreamLanguageModel(payload=args['payload'], depth=args['depth'],
        pool=args['pool'], heads=args['heads']) if native else
        SelectiveEventLanguageModel(width=args['width'], modes=args['modes'], depth=args['depth']))
    model.load_state_dict(saved['model']); before = fingerprint(model)
    model.eval(); positions = list(range(64, 96)); histories = (64, 16, 8, 4, 2, 1)
    references = []; evidence = {}; forward_tokens = 0
    with torch.no_grad(), torch.random.fork_rng():
        for history in histories:
            logits = []
            for position in positions:
                z, _ = piece(model, tokens, position-history+1, position+1, native)
                logits.append(z[-1]); forward_tokens += history
            logits = torch.stack(logits); log_q = logits.log_softmax(-1)
            if history == 64:
                references = log_q
            kl = (references.exp()*(references-log_q)).sum(-1)
            evidence[str(history)] = dict(n=len(positions),
                bpc=float(F.cross_entropy(logits, tokens[torch.tensor(positions)+1]))/math.log(2),
                mean_kl_from_64_character_history=float(kl.mean()),
                max_log_probability_change=float((references-log_q).abs().max()))
    # Actual declared native 16-character credit window; the carrier audit
    # also uses 16 here. Native hard-route teachers are not finite-difference
    # derivatives of the discontinuous realized router.
    model.eval()
    with torch.no_grad(), torch.random.fork_rng():
        _, warm_state = piece(model, tokens, 0, 64, native)
    warm_state.detach()
    model.train(); model.zero_grad(set_to_none=True)
    with torch.random.fork_rng():
        z, _ = piece(model, tokens, 64, 80, native, warm_state)
        loss = F.cross_entropy(z, tokens[65:81]); loss.backward()
    gradients = layer_gradients(model, native)
    assert fingerprint(model) == before
    return dict(result=name, result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        history_interventions=evidence, layer_gradients=gradients,
        weights_preserved=True, forwards=len(histories)*len(positions)+2,
        forward_input_tokens=forward_tokens+80, backwards=1, optimizer_steps=0,
        scope='32 fixed development positions and coupled per-position noise, not the saved full-development score. '
              'History dependence and gradient reach do not establish useful hierarchical features; no retraining.')


def composition_gradient_audit():
    torch.manual_seed(6)
    base = SelectiveEventLanguageModel(width=32, modes=16, depth=6)
    composed = CountComposedModel(copy.deepcopy(base), 5)
    fit = torch.tensor(text_slice(0, 2048)); inputs, targets = fit[:64], fit[1:65]
    counts = torch.tensor(fit_stream_counts(fit.numpy(), 5)[:, :64])
    composed.register_stream('fit', counts); composed.use_stream('fit'); composed.train()
    captured = {}
    def capture_logits(module, args, output):
        output.retain_grad(); captured['logits'] = output
    handle = composed.base.head.register_forward_hook(capture_logits)
    before = fingerprint(composed)
    lp, _ = composed.forward_chunk(inputs); F.cross_entropy(lp, targets).backward(); handle.remove()
    z = captured['logits']; q = z.detach().softmax(-1)
    # The effective escape coefficient is the product over the cascade.
    discount, theta = composed.escape_parameters(); escape = torch.ones(len(inputs))
    for k, c in enumerate(counts):
        n = c.sum(-1); kinds = (c > 0).sum(-1)
        e = (theta[k].detach()+discount[k].detach()*kinds)/(n+theta[k].detach())
        escape = escape * torch.where(n > 0, e, torch.ones_like(e))
    responsibility = escape*q.gather(1, targets[:, None]).squeeze(1)/lp.detach().exp().gather(1, targets[:, None]).squeeze(1)
    expected = responsibility[:, None]*(q-F.one_hot(targets, 27))/len(inputs)
    error = float((z.grad-expected).abs().max())
    torch.testing.assert_close(z.grad, expected, atol=2e-7, rtol=2e-5)
    gradients = layer_gradients(composed.base)
    assert all(r['gradient_norm'] > 0 for r in gradients)
    assert fingerprint(composed) == before
    return dict(base_logit_gradient_matches_responsibility_identity=True,
        maximum_absolute_gradient_error=error, mean_responsibility=float(responsibility.mean()),
        layer_gradients=gradients, all_six_layers_receive_gradients=True,
        forward_input_tokens=64, forwards=1, backwards=1, optimizer_steps=0,
        weights_preserved=True, scope='Actual current carrier/count modules at initialization, 64 fitting targets '
            'with 2K leave-one-out count state. Not a replay of missing composed fitted checkpoints.')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--tag', required=True)
    args = parser.parse_args(); output = ROOT / 'experiments/results/diagnostics' / (args.tag+'.json')
    if Path(args.tag).name != args.tag or output.exists():
        raise ValueError('Unused plain tag required')
    torch.set_num_threads(1); started = time.perf_counter(); tokens = torch.tensor(text_slice(90_000_000, 129))
    models = [('experiments/results/parallel_language/local_selective_w128_D131072_selective_20260930T161050Z.json', False),
              ('experiments/results/native_language/local_native_language_D8192_H2_d16_depth8_s6_20261001T174000Z.json', True)]
    rows = [frozen_audit(name, native, tokens) for name, native in models]
    composition = composition_gradient_audit()
    sources = [Path(__file__), ROOT/'experiments/e120_shared_tasks.py',
        *[p for p in (ROOT/'sleeping_machines').glob('*.py') if p.name in
          ('native_stream_language.py','addressed_event_heads.py','selective_stream_language.py',
           'parallel_stream_language.py','stream_language.py','event_memory.py','event_state.py',
           'count_composed_stream.py','count_carrying_language.py','parallel_head_race_language.py')]]
    result = dict(status='completed', args=vars(args), frozen_models=rows, composition_gradient=composition,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        wall_s=time.perf_counter()-started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        protocol=dict(official_test_read=False, optimizer_steps=0, development_start=90_000_000,
            floating_arithmetic_instrumented=False, work='Every replay token/forward/backward counted; wall and whole-process RSS measured. No FLOP or energy claim.'),
        scope='Frozen local diagnostic of credit reach and context dependence, not a benchmark, semantic-feature proof or superiority claim.')
    output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(dict(completed=args.tag, wall_s=result['wall_s'])), flush=True)


if __name__ == '__main__':
    main()
