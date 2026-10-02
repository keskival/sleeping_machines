"""Confirm fixed joint/local race learning with two new fit seeds and reserved suffixes."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import joint_outcome_benchmark as J
from balanced_joint_protocol import joint_examples, data_hash


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--plan', required=True)
    a = p.parse_args(); out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unused plain tag required')
    started = time.perf_counter(); torch.set_num_threads(1)
    plan = json.loads((ROOT / a.plan).read_text()); rows = []; initial = {}; common = None
    fresh = joint_examples(8, 64, 75001)
    for job in plan['jobs']:
        if job['stage'] != 'pilot': continue
        r = json.loads((ROOT / job['result']).read_text()); args = r['args']
        if r['status'] != 'completed' or args['tag'] != job['tag']:
            raise ValueError('Completed matching fit required')
        for name, sha in r['source_sha256'].items():
            if digest(name) != sha: raise ValueError('Changed fitting source')
        settings = {key: args[key] for key in ('payload', 'depth', 'heads', 'pool', 'gap', 'key_width',
                    'value_width', 'update_targets', 'lr', 'fit_groups', 'dev_groups', 'epochs',
                    'fit_data_seed', 'dev_data_seed')}
        settings.update(fit_hash=r['fitting_data_sha256'], dev_hash=r['development_data_sha256'])
        if common is None: common = settings
        elif settings != common: raise ValueError('Unmatched protocol')
        if args['seed'] not in (7, 8): raise ValueError('Confirmation seeds 7 and 8 required')
        if not all(abs(x['query_bits'] - 1) < 1e-12 for x in r['counts']['rows']):
            raise ValueError('Query-count control failed')
        model = J.make_model(SimpleNamespace(**args))
        checkpoint = Path(job['result']).with_suffix('.progress.pt')
        saved = torch.load(ROOT / checkpoint, weights_only=False); model.load_state_dict(saved['best_state'])
        before = {name: value.clone() for name, value in model.state_dict().items()}
        score = J.evaluate(model, fresh)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        # Prediction-only intervention: remove delivered value content while
        # preserving query/key races, native state and contextual decoder.
        with torch.no_grad(): model.values.weight.zero_()
        zero_value_score = J.evaluate(model, fresh)
        model.load_state_dict(before)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        initial[(args['seed'], args['read_credit'])] = r['initial_dev']
        work = r['work']
        rows.append(dict(seed=args['seed'], read_credit=args['read_credit'], result=job['result'],
            result_sha256=digest(job['result']), checkpoint_sha256=digest(checkpoint),
            selected_epoch=r['selected_epoch'], development=r['final'], fresh=score,
            zero_delivered_value=zero_value_score, intervention_restored=True,
            fresh_dependency_gate_passed=score['accuracy'] >= .75 and score['query_bits'] <= .8,
            fitting_targets=work['fitting_targets'], optimizer_updates=work['optimizer_updates'],
            whole_fit_gflops_estimate=work['whole_fit_unit_special_flops_estimate'] / 1e9,
            fit_mflops_per_target_estimate=work['fit_unit_special_flops_per_target_estimate'] / 1e6,
            inference_mflops_per_target=work['inference_unit_special_flops_per_target'] / 1e6,
            parameters=r['parameters'], activity=r['activity'], wall_s=r['wall_s'], max_rss_kb=r['max_rss_kb']))
        print(json.dumps(dict(seed=args['seed'], credit=args['read_credit'], fresh_bits=score['query_bits'],
                              accuracy=score['accuracy'])), flush=True)
    if set(initial) != {(seed, credit) for seed in (7, 8) for credit in ('joint', 'local')}:
        raise ValueError('All four declared confirmation fits required')
    comparisons = []
    for seed in (7, 8):
        for key in ('per_target_bits', 'class1_probabilities', 'terminal_chosen_addresses'):
            if initial[(seed, 'joint')][key] != initial[(seed, 'local')][key]:
                raise ValueError('Initial model/race mismatch')
        pair = {r['read_credit']: r for r in rows if r['seed'] == seed}
        gain = pair['local']['fresh']['query_bits'] - pair['joint']['fresh']['query_bits']
        differences = (np.array(pair['local']['fresh']['per_target_bits']) -
                       np.array(pair['joint']['fresh']['per_target_bits'])).reshape(-1, 4).mean(1)
        rng = np.random.default_rng(418 + seed)
        boots = differences[rng.integers(0, 64, size=(2000, 64))].mean(1)
        comparisons.append(dict(seed=seed, joint_vs_local_fresh_bits=gain,
            joint_credit_gate_passed=pair['joint']['fresh_dependency_gate_passed'] and gain >= .05,
            paired_suffix_group_95_interval=np.quantile(boots, [.025, .975]).tolist(),
            joint_local_fitting_work_ratio_estimate=pair['joint']['whole_fit_gflops_estimate'] /
                                                   pair['local']['whole_fit_gflops_estimate']))
    result = dict(status='completed', args=vars(a), matched_protocol=common, common_unit_ledger=rows,
        comparisons=comparisons, confirmation_gate_passed=all(c['joint_credit_gate_passed'] for c in comparisons),
        fresh_seed=75001, fresh_groups=64, fresh_targets=256, fresh_data_sha256=data_hash(fresh),
        source_sha256={**J.sources(), 'experiments/joint_outcome_replication.py': digest('experiments/joint_outcome_replication.py')},
        wall_s=time.perf_counter() - started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Seeds7/8 fixed configuration; common new suffixes unused for fitting or selection; both seeds and arms retained. '
              'Matched inference, conditional terminal objective/credit comparison. Query-suffix/count bound only; '
              'no whole-core exact gradient, depth, natural-language, energy or iso-quality resource claim.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__': main()
