"""Guarded frozen-memory diagnostic from completed native event checkpoints.

No optimizer, tuning or new test-set selection. Exact original forward and
all-history-cleared contracts precede fixed query-channel interventions.
"""
import argparse
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
sys.path.insert(0, str(ROOT))
from native_event_tasks import episodes, data_hash
from native_event_contracts import predict
from native_information_channels import CHANNELS, predict_channels
from sleeping_machines.addressed_event_heads import AddressedEventHeads


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights_digest(model):
    hashed = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        hashed.update(name.encode())
        hashed.update(tensor.detach().cpu().numpy().tobytes())
    return hashed.hexdigest()


def evaluate(model, rows, condition):
    with torch.random.fork_rng():
        torch.manual_seed(314159)
        started = time.perf_counter()
        loss, hits, events = 0., [], 0
        for row in rows:
            logits, targets, state = predict_channels(model, row, condition)
            loss += float(F.cross_entropy(logits, targets, reduction='sum'))
            hits.append((logits.argmax(-1) == targets).tolist())
            events += state.events
        array = np.asarray(hits, dtype=np.float64)
        return dict(accuracy=float(array.mean()), nll=loss/array.size,
                    n=int(array.size), events=events, population_hits=hits,
                    wall_s=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--targets', type=int, default=256)
    args = parser.parse_args()
    directory = ROOT/'experiments/results/native_information'
    directory.mkdir(exist_ok=True)
    output = directory/(args.tag+'.json')
    parent = (ROOT/args.parent).resolve()
    if Path(args.tag).name != args.tag or output.exists():
        raise ValueError('Unused local tag required')
    if not parent.is_relative_to(ROOT/'experiments/results/native_event'):
        raise ValueError('A completed native-event result is required')
    result = json.loads(parent.read_text())
    if result['status'] != 'completed' or 'final' not in result or result['args']['epochs'] < 8:
        raise ValueError('A completed fixed-budget pilot is required')
    for name, sha in result['source_sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Parent source changed: '+name)
    a = result['args']
    if a['time_input'] != 'observed' or args.targets != a['dev_targets']:
        raise ValueError('Match the completed observed-time development protocol')
    torch.set_num_threads(1)
    started = time.perf_counter()
    checkpoint = parent.with_suffix('.progress.pt')
    checkpoint_sha = digest(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['result'] != result:
        raise ValueError('Checkpoint must be the completed matching snapshot')
    model = AddressedEventHeads(sources=a['sources'], classes=4 if a['task']=='order' else 2,
                               payload=a['payload'], depth=a['depth'], pool=a['pool'],
                               heads=a['heads'], credit=a['credit']).eval()
    model.load_state_dict(saved['best_state'])
    before = weights_digest(model)
    dev = episodes(a['task'], a['sources'], args.targets, result['protocol']['development_data_seed'])
    if data_hash(dev) != result['data_sha256']['dev']:
        raise ValueError('Development data mismatch')
    with torch.random.fork_rng():
        torch.manual_seed(314159)
        noise = torch.get_rng_state()
        with torch.no_grad():
            expected, _, _, _ = predict(model, dev[0])
            torch.set_rng_state(noise)
            actual, _, _ = predict_channels(model, dev[0], 'full')
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.set_rng_state(noise)
            expected, _, _, _ = predict(model, dev[0], clear=True)
            torch.set_rng_state(noise)
            actual, _, _ = predict_channels(model, dev[0], 'no_history')
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    scores = {}
    for stretch in ((1., 8., 64.) if a['task']=='order' else (1.,)):
        rows = episodes(a['task'], a['sources'], args.targets,
                        result['protocol']['development_data_seed'], stretch)
        key = str(stretch)
        scores[key] = dict(data_sha256=data_hash(rows), conditions={})
        for condition in CHANNELS:
            metrics = evaluate(model, rows, condition)
            scores[key]['conditions'][condition] = metrics
            print(json.dumps(dict(stretch=stretch, condition=condition,
                                  accuracy=metrics['accuracy'])), flush=True)
        full = np.asarray(scores[key]['conditions']['full']['population_hits'], dtype=float)
        for condition, metrics in scores[key]['conditions'].items():
            # Pair at the independent population boundary; no flattened-event CI.
            differences = (full-np.asarray(metrics['population_hits'], dtype=float)).mean(1)
            rng = np.random.default_rng(431)
            resamples = rng.choice(differences, (1000, len(differences)), replace=True).mean(1)
            metrics.update(drop_from_full=float(differences.mean()),
                           paired_population_bootstrap_drop_interval=np.quantile(resamples,(.025,.975)).tolist())
    base = scores['1.0']['conditions']['full']
    if base['accuracy'] != result['final']['dev']['accuracy'] or abs(base['nll']-result['final']['dev']['nll']) > 1e-7:
        raise ValueError('Original completed development score did not reproduce')
    if weights_digest(model) != before or digest(checkpoint) != checkpoint_sha:
        raise ValueError('Frozen weights/checkpoint were modified')
    names = ('experiments/native_information_probe.py', 'experiments/native_information_channels.py')
    record = dict(status='completed', args=vars(args), parent=args.parent,
                  parent_result_sha256=digest(parent), checkpoint_sha256=checkpoint_sha,
                  source_sha256={**result['source_sha256'], **{n:digest(ROOT/n) for n in names}},
                  parent_args=a, parent_selected_epoch=result['selected_epoch'],
                  numerical_contracts=dict(original_forward_exact=True, all_clear_exact=True,
                                          completed_score_reproduced=True, parameters_unchanged=True),
                  conditions={k:dict(keys=v[0], values=v[1], context=v[2]) for k,v in CHANNELS.items()},
                  scores=scores, wall_s=time.perf_counter()-started,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  hardware=dict(platform=platform.platform(),torch=torch.__version__,device='cpu',threads=1),
                  protocol=dict(optimizer_updates=0, official_test_read=False, development_only=True,
                                observation='Explicit observed query flag activates intervention; external labels only score predictions',
                                scope='Frozen learned reliance; not refitted capability, a causal training effect or a resource comparison; population intervals do not correct development checkpoint selection'))
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(output)


if __name__ == '__main__':
    main()
