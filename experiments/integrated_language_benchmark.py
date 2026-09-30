"""Resumable integrated language fits; use only through queue/run_safe.sh.

The old development driver and its active source contracts remain unchanged.
Official scores require the fixed 10M / four-pass / 200K-validation protocol.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from e120_shared_tasks import text_slice
from integrated_language_protocol import learn_chunk, numerical_contracts, sources
from sparse_language_contracts import hashes
from sparse_language_screen import evaluate
from sleeping_machines.sparse_race_language import SparseRaceLanguageModel
from race_language_screen import capture
import parallel_event_language as accounting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True); parser.add_argument('--contracts', required=True)
    parser.add_argument('--fit', type=int, default=32768)
    parser.add_argument('--dev', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--payload', type=int, default=16)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--pool', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--checkpoint-every', type=int, default=512)
    parser.add_argument('--official-test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    if Path(a.tag).name != a.tag or min(a.fit, a.dev, a.epochs, a.chunk, a.checkpoint_every) < 1:
        raise ValueError('Positive settings and a plain unique tag required')
    if a.fit < 2 * a.chunk + 1 or a.fit > 90_000_000 or not 2 <= a.dev <= 200_000:
        raise ValueError('Invalid disjoint fitting/development budgets')
    if a.official_test and (a.fit, a.dev, a.epochs) != (10_000_000, 200_000, 4):
        raise ValueError('Official comparison requires exactly 10M fit / 200K validation / four passes')
    contract = json.loads((ROOT / a.contracts).read_text())
    if contract['status'] != 'completed' or contract['source_sha256'] != hashes():
        raise ValueError('Source-matching core numerical contracts required')
    out = ROOT / 'experiments/results/parallel_language' / f'{a.tag}.json'
    running, checkpoint = out.with_suffix('.running.json'), out.with_suffix('.progress.pt')
    if out.exists() or (running.exists() and not (a.resume and checkpoint.exists())):
        raise ValueError('Preserve results; unfinished runs require --resume and their checkpoint')
    if a.resume and not checkpoint.exists():
        raise ValueError('--resume requires the original progress checkpoint')
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    started = time.perf_counter(); previous_wall = 0.
    source_hashes = sources(__file__)
    checks = numerical_contracts(a.payload, a.depth, a.pool)
    train = torch.tensor(text_slice(0, a.fit)); dev = torch.tensor(text_slice(90_000_000, a.dev))
    model = SparseRaceLanguageModel(a.payload, a.depth, a.pool)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    initial = copy.deepcopy(model.state_dict())
    result = dict(status='running', args=vars(a), parameters=sum(p.numel() for p in model.parameters()),
        capacity_units=model.capacity_units, source_sha256=source_hashes,
        contract_result=a.contracts, numerical_contracts=checks, curve=[],
        fitting_data_sha256=hashlib.sha256(train.numpy().astype('uint8').tobytes()).hexdigest(),
        development_data_sha256=hashlib.sha256(dev.numpy().astype('uint8').tobytes()).hexdigest(),
        hardware=dict(device='cpu', threads=1, platform=platform.platform(), torch=torch.__version__),
        protocol=dict(fitting=[0, a.fit], development=[90_000_000, 90_000_000 + a.dev],
            test=[95_000_000, 96_000_000] if a.official_test else None,
            official_test_read=False, cold_context=True, excluded_first_target=True,
            statistical_experts=[], credit_truncation=a.chunk,
            selection='minimum frozen full-development bpc over fixed fitting passes',
            weights_frozen_on_test=True, tokenizer='fixed 27-character text8 alphabet',
            index='observed-character pools; learned contextual state keys and race clocks',
            scope='Integrated sparse/timed model; fixed topology, representative arithmetic, no measured energy'))
    epoch, next_start, total, steps = 1, 0, 0., 0
    best, best_state, state = float('inf'), None, model.new_state()
    if a.resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        old_args, new_args = dict(saved['result']['args']), dict(vars(a))
        old_args.pop('resume', None); new_args.pop('resume', None)
        if old_args != new_args or saved['result']['source_sha256'] != source_hashes:
            raise ValueError('Cannot resume changed settings or sources')
        if saved['result']['fitting_data_sha256'] != result['fitting_data_sha256']:
            raise ValueError('Fitting data changed')
        if saved['result']['development_data_sha256'] != result['development_data_sha256']:
            raise ValueError('Development data changed')
        result = saved['result']; previous_wall = result['wall_s']
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        initial = saved['initial']
        epoch, next_start, total, steps, best, best_state, state = (
            saved[k] for k in ('epoch', 'next_start', 'total', 'steps', 'best', 'best_state', 'stream_state'))
        torch.set_rng_state(saved['torch_rng'])
    else:
        result['initial_dev'] = evaluate(model, dev, a.chunk)
    def persist():
        result.update(wall_s=previous_wall + time.perf_counter() - started,
            max_rss_kb=max(result.get('max_rss_kb', 0), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        result['progress'] = dict(epoch=epoch, targets=next_start, steps=steps,
            online_bpc=total / next_start / math.log(2) if next_start else None)
        temporary = checkpoint.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), result=result,
            initial=initial, epoch=epoch, next_start=next_start, total=total, steps=steps,
            best=best, best_state=best_state, stream_state=state.detach(), torch_rng=torch.get_rng_state()), temporary)
        temporary.replace(checkpoint)
        target = out if result['status'] == 'completed' else running
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n'); temporary.replace(target)
    persist(); print(json.dumps(dict(started=a.tag, contracts=checks, resumed=a.resume)), flush=True)
    while epoch <= a.epochs:
        for start in range(next_start, len(train) - 1, a.chunk):
            end = min(start + a.chunk, len(train) - 1)
            loss, state, _ = learn_chunk(model, optimizer, train[start:end], train[start+1:end+1], state)
            total += loss; steps += 1; next_start = end
            if steps % a.checkpoint_every == 0:
                persist(); print(json.dumps(result['progress']), flush=True)
        score = evaluate(model, dev, a.chunk)
        result['curve'].append(dict(epoch=epoch, dev=score, online_bpc=total/(len(train)-1)/math.log(2),
            deliveries=state.deliveries, candidate_scores=state.candidate_scores,
            counterfactual_values=state.counterfactual_values, selected_updates=state.selected_updates,
            optimizer_steps=math.ceil((len(train)-1)/a.chunk)))
        if score['bpc'] < best:
            best, best_state = score['bpc'], copy.deepcopy(model.state_dict())
            result['selected_epoch'] = epoch
        print(json.dumps(result['curve'][-1]), flush=True)
        epoch, next_start, total, state = epoch + 1, 0, 0., model.new_state()
        persist()
    model.load_state_dict(best_state)
    result['final'] = dict(dev=evaluate(model, dev, a.chunk))
    if a.official_test:
        test = torch.tensor(text_slice(95_000_000, 1_000_000))
        result['final']['official_test'] = evaluate(model, test, a.chunk)
        result['protocol']['official_test_read'] = True
        result['test_data_sha256'] = hashlib.sha256(test.numpy().astype('uint8').tobytes()).hexdigest()
    accounting.capture = capture
    result['work'] = accounting.fitting_work(model, optimizer, train, a)
    result['work']['scope'] += ' Sparse address/optimizer activity varies; representative extrapolation, not exact whole-run work. RNG, indexing and physical traffic remain additional.'
    evaluation_targets = (a.epochs + 2) * (a.dev - 1) + (999_999 if a.official_test else 0)
    result['evaluation_work'] = dict(targets=evaluation_targets,
        representative_arithmetic_flops=evaluation_targets * result['work']['inference_arithmetic_flops_per_character'],
        scope='Additional initial/epoch/final development and frozen test scoring; representative inference extrapolation')
    result['parameter_change_norms_by_category'] = {}
    for name, p in model.named_parameters():
        category = 'unit' if name.startswith('units') else name.split('.')[0]
        result['parameter_change_norms_by_category'][category] = result['parameter_change_norms_by_category'].get(category, 0.) + float((p.detach() - initial[name]).norm())
    result['training_random_draws'] = sum(r['candidate_scores'] for r in result['curve'])
    result['status'] = 'completed'; persist(); running.unlink(missing_ok=True)
    print(json.dumps(dict(completed=a.tag, final=result['final'], fitting_flops=result['work']['total_training_arithmetic_flops'])), flush=True)


if __name__ == '__main__':
    main()
