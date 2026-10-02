"""Completed-only joint/local terminal credit and same-width depth comparison."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import joint_outcome_benchmark as J
from balanced_joint_protocol import joint_examples,data_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Fresh plain tag required')
    started=time.perf_counter();torch.set_num_threads(1);plan=json.loads((ROOT/a.plan).read_text())
    rows=[];loaded={};common=None;fresh_hash=None
    for job in plan['jobs']:
        if job['stage']!='pilot':continue
        path=ROOT/job['result'];r=json.loads(path.read_text());ckpath=path.with_suffix('.progress.pt')
        if r['status']!='completed' or r['args']['tag']!=job['tag']:raise ValueError('Completed matching pilot required')
        for n,h in r['source_sha256'].items():
            if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise ValueError('Changed fitting source')
        args=r['args'];settings={k:args[k] for k in ['gap','fit_groups','dev_groups','epochs','payload','heads','pool',
            'seed','lr','update_targets','fit_data_seed','dev_data_seed','key_width','value_width']}
        settings.update(fit_hash=r['fitting_data_sha256'],dev_hash=r['development_data_sha256'])
        if common is None:common=settings
        elif settings!=common:raise ValueError('Unmatched learning/data/width protocol')
        if not all(abs(c['query_bits']-1)<1e-12 for c in r['counts']['rows']):raise ValueError('Restricted count bound/control failed')
        model=J.make_model(SimpleNamespace(**args));ck=torch.load(ckpath,weights_only=False);model.load_state_dict(ck['best_state'])
        before={n:t.clone() for n,t in model.state_dict().items()}
        fresh=joint_examples(args['gap'],32,74001);fresh_hash=data_hash(fresh);fresh_score=J.evaluate(model,fresh)
        for n,t in model.state_dict().items():torch.testing.assert_close(t,before[n],rtol=0,atol=0)
        fitted=joint_examples(args['gap'],args['fit_groups'],args['fit_data_seed']);C=np.array([len(set(x['inputs'][:-1])) for x in fitted])
        work=r['work'];targets=work['fitting_targets'];native_keys=args['depth']*args['heads']*args['pool']*(args['gap']+7)
        if r['activity']['key_scores']!=int((native_keys+2*C).sum())*args['epochs']:raise ValueError('Uncharged/mismatched discovery scores')
        loaded[job['arm']]=r
        row=dict(arm=job['arm'],result=job['result'],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),read_credit=args['read_credit'],depth=args['depth'],
            payload=args['payload'],parameters=r['parameters'],selected_epoch=r['selected_epoch'],development=r['final'],
            fresh=fresh_score,fresh_data_sha256=fresh_hash,fresh_seed=74001,fresh_groups=32,encoder_and_head_preserved=True,
            fresh_dependency_gate_passed=fresh_score['accuracy']>=.75 and fresh_score['query_bits']<=.8,
            whole_fit_gflops_estimate=work['whole_fit_unit_special_flops_estimate']/1e9,
            fit_mflops_per_target_estimate=work['fit_unit_special_flops_per_target_estimate']/1e6,
            inference_mflops_per_target=work['inference_unit_special_flops_per_target']/1e6,
            fitting_targets=targets,optimizer_updates=work['optimizer_updates'],
            native_available_receivers=args['depth']*args['heads']*args['pool'],outcome_address_capacity=27,
            mean_occupied_outcome_addresses=float(C.mean()),native_selected_updates_per_target=args['depth']*args['heads']*(args['gap']+7),
            raw_outcome_writes_per_target=args['gap']+6,terminal_keys_per_target=float(2*C.mean()),
            terminal_value_deliveries_per_target=2,terminal_unique_candidate_values_per_training_target=float(C.mean()),
            terminal_counterfactual_loss_pairs_per_training_target=float((C*C).mean()) if args['read_credit']=='joint' else 0,
            raw_outcome_packed_integer_bytes_max=r['final']['raw_outcome_packed_integer_bytes_max'],
            native_state_tensor_bytes=r['final']['max_state_tensor_bytes'],activity=r['activity'],
            wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb'],
            curve=[dict(epoch=x['epoch'],fitting_query_bits=x['fitting_query_bits'],
                query_bits=x['dev']['query_bits'],accuracy=x['dev']['accuracy']) for x in r['curve']])
        rows.append(row);print(json.dumps(dict(arm=job['arm'],fresh_bits=fresh_score['query_bits'],fresh_accuracy=fresh_score['accuracy'])),flush=True)
    if set(loaded)!=set(['joint_full','local_full','joint_shallow']):raise ValueError('All matched arms required')
    for key in ('per_target_bits','class1_probabilities','terminal_chosen_addresses'):
        if loaded['joint_full']['initial_dev'][key]!=loaded['local_full']['initial_dev'][key]:raise ValueError('Initial forward/policy mismatch')
    by={r['arm']:r for r in rows};full=by['joint_full'];local=by['local_full'];shallow=by['joint_shallow']
    gain=local['fresh']['query_bits']-full['fresh']['query_bits'];depth_gain=shallow['fresh']['query_bits']-full['fresh']['query_bits']
    paired=np.asarray(local['fresh']['per_target_bits'])-np.asarray(full['fresh']['per_target_bits']);groups=paired.reshape(-1,4).mean(1)
    rng=np.random.default_rng(418);boot=groups[rng.integers(0,len(groups),size=(2000,len(groups)))].mean(1)
    result=dict(status='completed',args=vars(a),matched_protocol=common,common_unit_ledger=rows,
        joint_vs_local_fresh_bits=gain,full_vs_shallow_fresh_bits=depth_gain,
        joint_credit_gate_passed=full['fresh_dependency_gate_passed'] and gain>=.05,
        any_fresh_dependency_gate_passed=any(r['fresh_dependency_gate_passed'] for r in rows),
        joint_local_fitting_work_ratio_estimate=full['whole_fit_gflops_estimate']/local['whole_fit_gflops_estimate'],
        paired_query_groups_95_percentile_interval=np.quantile(boot,[.025,.975]).tolist(),
        restricted_query_count_lower_bound_bits=1.,fresh_data_sha256=fresh_hash,
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={**J.sources(),'experiments/joint_outcome_analysis.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='Fixed selected models on reserved seed74001 suffixes; all three declared arms, one fitted seed and synthetic task. Matched joint/local inference, terminal credit differs. Whole-fit sampled estimates include all native prefix/decoder/candidate/loser/optimizer work; metadata/traffic/RNG/energy separate. Capacity, candidate scores, state commits, raw writes and value deliveries distinguishable. No all-counting, natural-language, dense-control, exact whole-core gradient or supremacy claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
