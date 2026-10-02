"""Joint text + irregular-event benchmark for the native core on one persistent address (THEORY §394).

Model: unchanged AddressedEventHeads with one source address, 32-wide content (characters, marks, query flag),
two classes, observed physical timestamps.  The batched training path (fast_native_core, contract-equal to the
reference) is used by default.  Reported bars, all on the same development episodes:
  ours (observed time) | rank-time control (timestamps replaced by event index) | cleared-text control (character
  events removed: an information ablation) | table bar (best time-blind lookup fitted on the fit episodes).
Learned dense controls with the same events (Δt GRU, time-encoded Transformer) are an AWS job.
Fitting work (forward, backward, losing proposals, normalization, clipping, Adam) is fully traced for the first
--trace-windows optimizer windows and extrapolated per fitting event (a labelled estimate, the repository's
representative-window convention); per-step tracing made a full fit about 50x slower.
"""
import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import joint_event_language_tasks as J  # noqa: E402
from parallel_head_accumulated_language import merge  # noqa: E402
from race_language_screen import capture  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402


def sources():
    names = ('experiments/joint_event_language_benchmark.py', 'experiments/joint_event_language_tasks.py',
             'experiments/native_event_tasks.py', 'sleeping_machines/addressed_event_heads.py',
             'sleeping_machines/fast_native_core.py', 'sleeping_machines/parallel_head_race_language.py',
             'sleeping_machines/sparse_race_language.py', 'sleeping_machines/parallel_stream_language.py',
             'experiments/race_language_screen.py', 'experiments/parallel_head_accumulated_language.py')
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}


def predict(model, row, mode='observed', drop_text=False):
    if model.training and hasattr(model, '_fast_layers'):
        model._fast_layers = {}
    state = model.new_state(); outputs, targets = [], []
    try:
        index = 0
        for event in row:
            if drop_text and 1. in event.mark[:27]:
                continue
            timestamp = event.time if mode == 'observed' else float(index)
            logits, _ = model.consume_event(0, timestamp, event.mark, state); index += 1
            if event.target is not None:
                outputs.append(logits); targets.append(event.target)
    finally:
        if hasattr(model, '_fast_layers'):
            model._fast_layers = None
    return torch.stack(outputs), torch.tensor(targets), state


@torch.no_grad()
def evaluate(model, rows, mode='observed', drop_text=False):
    with torch.random.fork_rng():
        torch.manual_seed(314159); model.eval(); loss = correct = n = events = 0
        for row in rows:
            z, y, state = predict(model, row, mode, drop_text)
            loss += float(F.cross_entropy(z, y, reduction='sum')); correct += int((z.argmax(-1) == y).sum())
            n += len(y); events += state.events
    return dict(n=n, nll=loss / n, accuracy=correct / n, events=events)


def table_bar(fit, dev):
    """best time-blind lookup: (word, last mark, events after the named mark) -> majority label from fit episodes."""
    table = defaultdict(Counter)
    for row in fit:
        table[J.blind_features(row)[:3]][row[-1].target] += 1
    hits = [((c[1] > c[0]) if (c := table.get(J.blind_features(row)[:3])) else True) == bool(row[-1].target) for row in dev]
    return dict(accuracy=float(np.mean(hits)), features='named word, last mark, events after the named mark',
                scope='Fitted on the fit episodes; time-blind by construction')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--fit', type=int, default=512)
    p.add_argument('--dev', type=int, default=256); p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--seed', type=int, default=6); p.add_argument('--payload', type=int, default=16)
    p.add_argument('--depth', type=int, default=8); p.add_argument('--heads', type=int, default=2)
    p.add_argument('--pool', type=int, default=2); p.add_argument('--update-episodes', type=int, default=16)
    p.add_argument('--lr', type=float, default=.003); p.add_argument('--time-input', choices=('observed', 'rank'), default='observed')
    p.add_argument('--reference-core', action='store_true', help='unbatched reference training path')
    p.add_argument('--background', type=int, nargs=2, default=[2, 6], help='[low, high) phase-A events: history length')
    p.add_argument('--trace-windows', type=int, default=2, help='optimizer windows of pass 1 traced for the work estimate')
    a = p.parse_args()
    out = ROOT / 'experiments/results/joint_event_language' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required; prior results are preserved')
    if a.fit % a.update_episodes or not 1 <= a.epochs <= 32:
        raise ValueError('Whole optimizer windows and a bounded pass budget required')
    torch.set_num_threads(1); torch.manual_seed(a.seed); started = time.perf_counter()
    fit, dev = J.episodes(a.fit, 1301, tuple(a.background)), J.episodes(a.dev, 2301, tuple(a.background))
    cls = AddressedEventHeads if a.reference_core else fast_class(AddressedEventHeads)
    model = cls(sources=1, content_dim=J.WIDTH, classes=2, payload=a.payload, depth=a.depth, pool=a.pool, heads=a.heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed + 10)
    ledger = {k: [] for k in ('forward_and_loss', 'backward', 'gradient_normalization', 'gradient_clipping', 'optimizer')}
    activity = dict(events=0, key_scores=0, selected_updates=0, counterfactual_proposals=0)
    result = dict(status='running', args=vars(a), source_sha256=sources(), parameters=sum(q.numel() for q in model.parameters()),
                  data_sha256=dict(fit=J.data_hash(fit), dev=J.data_hash(dev)), table_bar=table_bar(fit, dev), curve=[],
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, device='cpu', threads=1),
                  protocol=dict(fit_seed=1301, dev_seed=2301, delta=J.DELTA, address='one source address for text and events',
                                input='character / mark / query-flag one-hot and physical timestamp; targets never input',
                                selection='lowest frozen development NLL over fixed passes', chance=.5,
                                scope='Synthetic joint-modality capability pilot (Theory §394); one seed; dense controls on AWS'))
    best, best_state = float('inf'), None
    traced_events = traced_windows = 0
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(len(fit)).tolist(); model.train(); total = 0.
        for begin in range(0, len(order), a.update_episodes):
            optimizer.zero_grad(set_to_none=True); pending = 0
            traced = epoch == 1 and begin // a.update_episodes < a.trace_windows
            run = capture if traced else (lambda f: f())
            for index in order[begin:begin + a.update_episodes]:
                box = {}
                def forward():
                    z, y, state = predict(model, fit[index], a.time_input)
                    box.update(loss=F.cross_entropy(z, y, reduction='sum'), state=state, n=len(y))
                record = run(forward)
                if traced:
                    ledger['forward_and_loss'].append(record)
                if not torch.isfinite(box['loss']):
                    raise FloatingPointError('Nonfinite loss')
                record = run(lambda: box['loss'].backward())
                if traced:
                    ledger['backward'].append(record); traced_events += box['state'].events
                pending += box['n']; total += float(box['loss'].detach()); s = box['state']
                activity['events'] += s.events; activity['key_scores'] += s.candidate_scores
                activity['selected_updates'] += s.selected_updates; activity['counterfactual_proposals'] += s.counterfactual_values
                s.detach(); del box, s
            def normalize():
                for q in model.parameters():
                    if q.grad is not None:
                        q.grad.div_(pending)
            for key, step in (('gradient_normalization', normalize),
                              ('gradient_clipping', lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)),
                              ('optimizer', optimizer.step)):
                record = run(step)
                if traced:
                    ledger[key].append(record)
            if traced:
                traced_windows += 1
                for key in ledger:
                    ledger[key] = [merge(ledger[key])]
        score = evaluate(model, dev, a.time_input)
        result['curve'].append(dict(epoch=epoch, dev=score, training_nll=total / a.fit, wall_s=time.perf_counter() - started))
        if score['nll'] < best:
            best, best_state = score['nll'], copy.deepcopy(model.state_dict()); result['selected_epoch'] = epoch
        print(json.dumps(dict(epoch=epoch, dev_accuracy=score['accuracy'], dev_nll=score['nll'])), flush=True)
    model.load_state_dict(best_state)
    result['final'] = dict(dev=evaluate(model, dev, a.time_input), rank_time=evaluate(model, dev, 'rank'),
                           cleared_text=evaluate(model, dev, a.time_input, drop_text=True),
                           confirmation=evaluate(model, J.episodes(1024, 3301, tuple(a.background)), a.time_input))
    model.eval(); box = {}
    with torch.no_grad():
        def inference():
            box['z'], box['y'], box['state'] = predict(model, dev[0], a.time_input)
        inference_trace = capture(inference)
    train = {k: merge(v) for k, v in ledger.items()}
    traced_arith = sum(t['arithmetic_flops'] for t in train.values()); traced_special = sum(t['special_function_evaluations'] for t in train.values())
    scale = activity['events'] / traced_events  # whole fit = traced work per fitting event x all fitting events
    arithmetic, special = traced_arith * scale, traced_special * scale
    result['work'] = dict(fitting_episodes=a.fit * a.epochs, fitting_events=activity['events'],
                          traced_windows=traced_windows, traced_events=traced_events, estimate='traced work per fitting event x fitting events',
                          total_training_arithmetic_flops=arithmetic, training_special_function_evaluations=special,
                          total_training_unit_special_flops=arithmetic + special, traced_training_stages=train,
                          fit_mflops_per_query=(arithmetic + special) / (a.fit * a.epochs) / 1e6,
                          inference_unit_special_flops_per_query=inference_trace['arithmetic_flops'] + inference_trace['special_function_evaluations'],
                          inference_events_per_query=len(dev[0]), exact_fitting_activity=activity,
                          available_receivers=a.depth * a.heads * a.pool, selected_updates_per_event=a.depth * a.heads,
                          key_scores_per_event=a.depth * a.heads * a.pool, inference_storage=box['state'].storage(),
                          scope='Representative-window estimate (first windows of pass 1 fully traced) incl. losing proposals, '
                                'backward, normalization, clipping, Adam; '
                                '2 FLOPs/MAC, specials separate plus unit weight; dev/RNG/traffic/energy separate')
    result.update(status='completed', wall_s=time.perf_counter() - started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed=a.tag, accuracy=result['final']['dev']['accuracy'],
                          table=result['table_bar']['accuracy'])), flush=True)


if __name__ == '__main__':
    main()
