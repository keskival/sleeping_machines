"""Native sparse temporal gesture fit on the calibrated observed packet data."""
import argparse
import copy
import hashlib
import json
import math
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
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.fast_native_core import fast_class
from race_language_screen import capture
from parallel_head_accumulated_language import merge


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    names = ['experiments/dvs_native_benchmark.py', 'sleeping_machines/addressed_event_heads.py',
        'sleeping_machines/fast_native_core.py', 'sleeping_machines/parallel_head_race_language.py',
        'sleeping_machines/native_stream_language.py', 'sleeping_machines/sparse_race_language.py',
        'sleeping_machines/parallel_stream_language.py', 'experiments/race_language_screen.py',
        'sleeping_machines/operation_audit.py', 'experiments/theory/73_real_gesture_integrated_admission.md']
    return {name: sha(ROOT / name) for name in names}


def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--tag', required=True)
    p.add_argument('--data', required=True); p.add_argument('--controls', required=True)
    p.add_argument('--payload', type=int, default=16); p.add_argument('--depth', type=int, default=2)
    p.add_argument('--heads', type=int, default=2); p.add_argument('--pool', type=int, default=2)
    p.add_argument('--epochs', type=int, default=8); p.add_argument('--update-targets', type=int, default=16)
    p.add_argument('--lr', type=float, default=.003); p.add_argument('--seed', type=int, default=6)
    p.add_argument('--fit', type=int); p.add_argument('--dev', type=int); p.add_argument('--contracts')
    p.add_argument('--resume', action='store_true'); p.add_argument('--stop-after-updates', type=int)
    return p


def make_model(a, fast=True):
    torch.manual_seed(a.seed)
    cls = fast_class(AddressedEventHeads) if fast else AddressedEventHeads
    return cls(sources=1, content_dim=33, classes=11, payload=a.payload, depth=a.depth, heads=a.heads, pool=a.pool)


def encode(counts, center, scale):
    normalized = ((np.log1p(counts.reshape(-1)) - center) / scale).reshape(20, 32)
    rows = []
    for k in range(20):
        if counts[k].sum() > 0:
            rows.append(((k + 1) * .05, np.r_[normalized[k], 0.].astype(np.float32)))
    rows.append((1., np.r_[np.zeros(32), 1.].astype(np.float32)))
    return rows


def predict(model, row, seed, training):
    model.train(training); state = model.new_state()
    if training and hasattr(model, '_fast_layers'): model._fast_layers = {}
    try:
        with torch.random.fork_rng():
            # Common uniforms across clips: no identity/index/label becomes
            # an unintended randomness channel. Pass and fit seed only.
            torch.manual_seed(seed)
            for timestamp, content in row['events']:
                logits, _ = model.consume_event(0, timestamp, content, state)
    finally:
        if hasattr(model, '_fast_layers'): model._fast_layers = None
    return logits, state


@torch.no_grad()
def evaluate(model, rows):
    losses = []; probabilities = []; correct = 0; events = keys = commits = max_bytes = 0
    for row in rows:
        z, state = predict(model, row, 314159, False)
        losses.append(float(F.cross_entropy(z[None], torch.tensor([row['target']]))))
        probabilities.append(z.softmax(-1).tolist()); correct += int(z.argmax()) == row['target']
        events += state.events; keys += state.candidate_scores; commits += state.selected_updates
        max_bytes = max(max_bytes, state.storage()['persistent_tensor_bytes'])
    return dict(targets=len(rows), nll=float(np.mean(losses)), accuracy=correct / len(rows),
        per_target_nll=losses, probabilities=probabilities, events=events, key_scores=keys,
        selected_updates=commits, max_state_tensor_bytes=max_bytes)


def train_window(model, optimizer, rows, a, epoch, trace=False):
    optimizer.zero_grad(set_to_none=True); stages = {}; loss_sum = 0.; events = keys = commits = teachers = 0
    def traced(name, fn):
        if trace: stages.setdefault(name, []).append(capture(fn))
        else: fn()
    for row in rows:
        box = {}
        def forward():
            logits, state = predict(model, row, 100000 + a.seed + 10000 * epoch, True)
            box.update(loss=F.cross_entropy(logits[None], torch.tensor([row['target']])), state=state)
        traced('forward_and_loss', forward)
        if not torch.isfinite(box['loss']): raise FloatingPointError('Nonfinite query loss')
        traced('backward', lambda: box['loss'].backward()); loss_sum += float(box['loss'].detach())
        state = box['state']; events += state.events; keys += state.candidate_scores
        commits += state.selected_updates; teachers += state.counterfactual_values
    def normalize():
        for parameter in model.parameters():
            if parameter.grad is not None: parameter.grad.div_(len(rows))
    traced('gradient_normalization', normalize)
    traced('gradient_clipping', lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    traced('optimizer', optimizer.step)
    return dict(targets=len(rows), loss_sum=loss_sum, events=events, key_scores=keys,
                selected_updates=commits, counterfactual_values=teachers,
                stages={name: merge(values) for name, values in stages.items()})


def load(a):
    parent = json.loads((ROOT / a.data).read_text()); control = json.loads((ROOT / a.controls).read_text())
    if parent['status'] != control['status'] or parent['status'] != 'completed': raise ValueError('Completed calibration required')
    if control['data_result_sha256'] != sha(ROOT / a.data): raise ValueError('Changed control data')
    if not control['development_region_admission_passed']: raise ValueError('No practical region admitted')
    artifact = ROOT / parent['data_artifact']
    if sha(artifact) != parent['data_sha256']: raise ValueError('Changed data artifact')
    for result in (parent, control):
        for name, digest in result['source_sha256'].items():
            if sha(ROOT / name) != digest: raise ValueError('Changed calibration source')
    z = np.load(artifact, allow_pickle=False); raw = z['fit_counts'].reshape(len(z['fit_labels']), -1)
    center = np.log1p(raw).mean(0); scale = np.maximum(np.log1p(raw).std(0), .5)
    sets = []
    for split, limit in [('fit', a.fit), ('dev', a.dev)]:
        n = len(z[split + '_labels']) if limit is None else limit
        if not 1 <= n <= len(z[split + '_labels']): raise ValueError('Within saved split required')
        sets.append([dict(index=i, identity=str(z[split + '_ids'][i]), target=int(z[split + '_labels'][i]),
                          events=encode(z[split + '_counts'][i], center, scale)) for i in range(n)])
    return *sets, dict(data_result=a.data, data_result_sha256=sha(ROOT / a.data), controls=a.controls,
        controls_sha256=sha(ROOT / a.controls), data_artifact_sha256=parent['data_sha256'], protocol=parent['protocol'],
        feature_transform='Same fit-only 640-coordinate log-count mean/std-clamp.5; native float32 content',
        race_noise='Common per-pass/fitting-seed noise and fixed dev seed314159; no identity/index/label seed input',
        transform_sha256=hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest())


def run(a, directory=None):
    directory = Path(directory) if directory else ROOT / 'experiments/results/dvs_native'
    directory.mkdir(parents=True, exist_ok=True); out = directory / (a.tag + '.json')
    checkpoint = out.with_suffix('.progress.pt'); started = time.perf_counter(); torch.set_num_threads(1)
    if out.exists() and json.loads(out.read_text()).get('status') == 'completed': raise ValueError('Preserve completed run')
    if Path(a.tag).name != a.tag or not 1 <= a.epochs <= 16: raise ValueError('Unique bounded run required')
    fit, dev, data_info = load(a); settings = {k: v for k, v in vars(a).items() if k not in ('tag', 'resume', 'stop_after_updates')}
    model = make_model(a); optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    if a.contracts:
        contract = json.loads((ROOT / a.contracts).read_text())
        if contract['status'] != 'completed': raise ValueError('Completed numerical contracts required')
        for name, digest in contract['source_sha256'].items():
            if sha(ROOT / name) != digest: raise ValueError('Changed contracted source')
    if a.resume:
        saved = torch.load(checkpoint, weights_only=False)
        if saved['settings'] != settings or saved['source_sha256'] != sources() or saved['data'] != data_info:
            raise ValueError('Changed recovery protocol')
        model.load_state_dict(saved['online_model']); optimizer.load_state_dict(saved['optimizer'])
        best_state = saved['best_state']; best = saved['best']; cursor = saved['cursor']; result = saved['result']
        torch.set_rng_state(saved['torch_rng']); prior_wall = saved['wall_s']
    else:
        if checkpoint.exists(): raise ValueError('Explicit recovery required')
        best_state = None; best = math.inf; cursor = dict(epoch=1, window=0, updates=0)
        result = dict(status='running', args=vars(a), source_sha256=sources(), data=data_info,
            initial_dev=evaluate(model, dev), initial_fit=evaluate(model, fit[:min(32, len(fit))]), curve=[],
            activity=dict(targets=0, events=0, key_scores=0, selected_updates=0, counterfactual_values=0),
            work_samples=[], window_size_counts={}, parameters=sum(p.numel() for p in model.parameters()))
        prior_wall = 0.
    def snapshot():
        payload = dict(settings=settings, source_sha256=result['source_sha256'], data=data_info,
            online_model=model.state_dict(), optimizer=optimizer.state_dict(), best_state=best_state, best=best,
            cursor=cursor, result=result, torch_rng=torch.get_rng_state(), wall_s=prior_wall + time.perf_counter() - started)
        temporary = checkpoint.with_suffix('.tmp'); torch.save(payload, temporary); temporary.replace(checkpoint)
    while cursor['epoch'] <= a.epochs:
        epoch = cursor['epoch']; order = np.random.default_rng(a.seed + 10 + epoch).permutation(len(fit)).tolist()
        windows = [order[i:i + a.update_targets] for i in range(0, len(order), a.update_targets)]
        first = {}; last = {}
        for k, indices in enumerate(windows): first.setdefault(len(indices), k); last[len(indices)] = k
        while cursor['window'] < len(windows):
            k = cursor['window']; indices = windows[k]; n = len(indices)
            traced = (epoch == 1 and k == first[n]) or (epoch == a.epochs and k == last[n])
            update = train_window(model, optimizer, [fit[i] for i in indices], a, epoch, traced)
            for name in result['activity']: result['activity'][name] += update[name]
            key = str(n); result['window_size_counts'][key] = result['window_size_counts'].get(key, 0) + 1
            if traced: result['work_samples'].append(dict(epoch=epoch, window=k, targets=n, stages=update['stages']))
            cursor['window'] += 1; cursor['updates'] += 1; snapshot()
            if a.stop_after_updates is not None and cursor['updates'] >= a.stop_after_updates:
                return result
        score = evaluate(model, dev); result['curve'].append(dict(epoch=epoch, dev=score, updates=cursor['updates']))
        if score['nll'] < best: best = score['nll']; best_state = copy.deepcopy(model.state_dict()); result['selected_epoch'] = epoch
        print(json.dumps(dict(epoch=epoch, dev_accuracy=score['accuracy'], dev_nll=score['nll'])), flush=True)
        cursor.update(epoch=epoch + 1, window=0); snapshot()
    model.load_state_dict(best_state); result['final'] = evaluate(model, dev)
    result['final_fit_diagnostic'] = evaluate(model, fit[:min(32, len(fit))])
    traces = {}
    with torch.no_grad():
        for row in dev[:min(11, len(dev))]:
            def forward(): predict(model, row, 314159, False)
            tr = capture(forward)
            for name, value in tr.items():
                if name in ('arithmetic_flops', 'special_function_evaluations'):
                    traces[name] = traces.get(name, 0) + value
            if not tr['formula_coverage_complete']: raise ValueError('Incomplete inference accounting')
    estimates = {}
    for size, count in result['window_size_counts'].items():
        samples = [s for s in result['work_samples'] if s['targets'] == int(size)]
        for stage in samples[0]['stages']:
            value = float(np.mean([s['stages'][stage]['arithmetic_flops'] + s['stages'][stage]['special_function_evaluations'] for s in samples])) * count
            estimates[stage] = estimates.get(stage, 0) + value
    total = sum(estimates.values()); targets = result['activity']['targets']; sample_n = min(11, len(dev))
    result['work'] = dict(fitting_targets=targets, optimizer_updates=cursor['updates'], whole_fit_unit_special_flops_estimate=total,
        fit_unit_special_flops_per_target_estimate=total / targets, stage_estimates=estimates,
        inference_unit_special_flops_per_target_estimate=sum(traces.values()) / sample_n,
        native_available_receivers=a.depth * a.heads * a.pool,
        scope='All prefix/query forward/backward, candidate values, clipping/normalization/Adam; first/last window samples stratified by actual target count. '
              'Inference mean first11 dev prefixes;2FLOPs/MAC+unit specials; common raw preprocessing/traffic/RNG/energy separate.')
    result.update(status='completed', wall_s=prior_wall + time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        small_fit_learning_passed=result['final_fit_diagnostic']['nll'] < result['initial_fit']['nll'] - .01,
        scope='One fitted native seed, same calibrated causal packets; dev selection over fixed passes. No official test, dense-control duplication, '
              'raw-spike execution, whole-route exact gradient or supremacy claim.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n'); return result


if __name__ == '__main__': run(parser().parse_args())
