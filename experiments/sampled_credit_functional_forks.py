"""Actual finite FIT-only Adam forks of frozen depth-sampling parameter banks."""
import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import dvs_clock_calibrated_benchmark as C
import depth_growth_plasticity_probe as P
import depth_route_sampling_variance_probe as V
from sleeping_machines.batched_episodes import batched_logits

PARENT='experiments/results/diagnostics/local_depth_route_sampling_variance_20261003T033000Z.json'


def predict(model,rows,seed):
    rng=torch.get_rng_state().clone()
    with torch.no_grad():
        z=batched_logits(model,rows,seed)
        losses=F.cross_entropy(z,torch.tensor([r['target'] for r in rows]),reduction='none')
    assert torch.equal(rng,torch.get_rng_state())
    return dict(logits=z.numpy(),losses=losses.numpy(),log_probabilities=z.log_softmax(-1).numpy())


def metric(pred,base,full,offset):
    sl=slice(*offset);lp=pred['log_probabilities'][sl]
    def kl(reference):
        q=reference['log_probabilities'][sl]
        return float((np.exp(q)*(q-lp)).sum(-1).mean())
    return dict(targets=offset[1]-offset[0],mean_nll=float(pred['losses'][sl].mean()),
        mean_nll_change_from_initial=float((pred['losses'][sl]-base['losses'][sl]).mean()),
        mean_nll_difference_from_full=float((pred['losses'][sl]-full['losses'][sl]).mean()),
        initial_to_fork_prediction_KL=kl(base),full_to_fork_prediction_KL=kl(full),
        logit_difference_from_full_L2=float(np.linalg.norm(pred['logits'][sl]-full['logits'][sl])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');artifact=out.with_suffix('.predictions.npz')
    assert Path(a.tag).name==a.tag and not out.exists() and not artifact.exists()
    torch.set_num_threads(1);begin=time.perf_counter();caller=torch.get_rng_state().clone()
    parent=json.loads((ROOT/PARENT).read_text());parent_hash=N.sha(ROOT/PARENT)
    if parent['status']!='completed' or parent['contracts_passed']!=5:raise ValueError('Completed contracted parent required')
    for name,digest in parent['source_sha256'].items():
        if N.sha(ROOT/name)!=digest:raise ValueError('Changed parent source '+name)
    vp=ROOT/parent['vectors']['path']
    if N.sha(vp)!=parent['vectors']['sha256']:raise ValueError('Changed parameter bank')
    data_parent=json.loads((ROOT/P.PARENT).read_text());rows,data=P.fit_inputs(data_parent)
    if data!=parent['data']:raise ValueError('Changed exact FIT inputs')
    seeds=(171323,171324);arrays={};cases=[];checks=[];forks=0;targets=0
    with np.load(vp,allow_pickle=False) as vectors,torch.random.fork_rng():
        for c in parent['cases']:
            depth=c['depth'];settings=SimpleNamespace(seed=7,payload=16,depth=depth,heads=2,pool=2,clock_step=.05)
            model=C.make_model(settings).double();weights=copy.deepcopy(model.state_dict())
            theta=torch.cat([p.detach().flatten() for p in model.parameters()]).numpy().copy()
            assert np.array_equal(theta,vectors[f'd{depth}_weights'])
            factual=vectors[f'd{depth}_factual'];route=vectors[f'd{depth}_route']
            banks=[vectors[f'd{depth}_episode{j}_parameter_route_vectors'] for j in range(2)]
            np.testing.assert_allclose(sum(v.sum(0) for v in banks),route,rtol=3e-7,atol=3e-9)
            arms=[('factorized',factual,None),('full',factual+route,None)]
            reference,_=V.adam_delta(factual+route,theta)
            for dist in c['distributions']:
                for i,indices in enumerate(dist['sample_indices']):
                    sampled=sum(len(v)/len(s)*v[s].sum(0) for v,s in zip(banks,indices));g=factual+sampled
                    delta,_=V.adam_delta(g,theta)
                    measured=float(np.linalg.norm(delta-reference)/np.linalg.norm(reference))
                    assert abs(measured-dist['samples'][i]['Adam_delta_relative_error'])<1e-12
                    if i<8:arms.append((f"k{dist['k']}_draw{i}",g,dict(k=dist['k'],draw=i,indices=indices)))
            initial={seed:predict(model,rows,seed) for seed in seeds};targets+=len(rows)*len(seeds)
            assert abs(float(initial[171323]['losses'][:2].mean())-c['initial_same_FIT_nll'])<1e-12
            for seed,pred in initial.items():
                arrays[f'd{depth}_initial_seed{seed}_logits']=pred['logits']
                arrays[f'd{depth}_initial_seed{seed}_losses']=pred['losses']
            outcomes=[];full=None
            # Full is evaluated first to define a common reference, without selecting a winner.
            for label,g,selection in sorted(arms,key=lambda arm:arm[0]!='full'):
                fork=copy.deepcopy(model);error=V.actual_adam_contract(fork,g);forks+=1
                predicted={seed:predict(fork,rows,seed) for seed in seeds};targets+=len(rows)*len(seeds)
                if label=='full':full=predicted
                assert full is not None
                displacement=torch.cat([p.detach().flatten() for p in fork.parameters()]).numpy()-theta
                outcomes.append(dict(arm=label,selection=selection,actual_update_norm=float(np.linalg.norm(displacement)),
                    actual_fresh_Adam_formula_error_norm=error,
                    evaluations=[dict(noise_seed=seed,same_FIT=metric(predicted[seed],initial[seed],full[seed],(0,2)),
                        anchor_FIT=metric(predicted[seed],initial[seed],full[seed],(2,16))) for seed in seeds]))
                for seed,pred in predicted.items():
                    arrays[f'd{depth}_{label}_seed{seed}_logits']=pred['logits']
                    arrays[f'd{depth}_{label}_seed{seed}_losses']=pred['losses']
            assert all(torch.equal(weights[n],p) for n,p in model.state_dict().items())
            cases.append(dict(depth=depth,parameters=len(theta),same_FIT_indices=[0,1],anchor_FIT_indices=list(range(2,16)),
                baseline=[dict(noise_seed=seed,same_FIT_nll=float(pred['losses'][:2].mean()),
                    anchor_FIT_nll=float(pred['losses'][2:].mean())) for seed,pred in initial.items()],outcomes=outcomes))
            checks.append(f'D{depth}: exact saved initialization/vector/sample metrics, actual Adam formulas, original weights and RNG preserved')
            print(json.dumps(dict(depth=depth,status='functional_case_completed',wall_s=time.perf_counter()-begin)),flush=True)
    assert torch.equal(caller,torch.get_rng_state());assert N.sha(ROOT/PARENT)==parent_hash
    assert forks==54 and targets==1824
    checks.append('Caller RNG, parent bytes and declared optimizer/prediction counts preserved')
    np.savez_compressed(artifact,**arrays)
    sources={**parent['source_sha256'],**{n:N.sha(ROOT/n) for n in
        ('experiments/sampled_credit_functional_forks.py','experiments/theory/125_sampled_credit_functional_forks.md')}}
    result=dict(status='completed',args=vars(a),contracts_passed=len(checks),contracts=checks,cases=cases,
        parent=PARENT,parent_sha256=parent_hash,data=data,source_sha256=sources,
        artifact=str(artifact.relative_to(ROOT)),artifact_sha256=N.sha(artifact),
        work=dict(executed_fresh_Adam_forks=forks,prediction_target_evaluations=targets,
            new_gradient_or_counterfactual_replay_evaluations=0,total_FLOPs=None,traffic=None,energy=None,
            scope='Reuses paid parent vector bank;54 actual optimizer steps and1824 forward target evaluations, unknown total work not zero'),
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='First-step FIT-only functional fork audit; fixed first8 stored draws, no DEV/test, heldout quality, trained moments, convergence or advantage claim')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',contracts=len(checks),wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'])))


if __name__=='__main__':main()
