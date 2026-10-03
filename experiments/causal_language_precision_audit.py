"""All-gradient and all-shadow branch precision audit of winner reuse."""
import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import causal_language_replay_contracts as C
import causal_language_replay_resource_admission as P
import causal_language_replay_helpers_rng as Old
import causal_language_replay_winner_reuse as New
import causal_language_winner_reuse_contracts as W
from sleeping_machines.causal_language_shadow_winner_reuse import batched_chunks as kernel


def promote(st):
    other=copy.deepcopy(st)
    other.memories={k:v.double() for k,v in other.memories.items()}
    other.arrivals={k:v.double() for k,v in other.arrivals.items()}
    other.contexts={k:tuple(v.double() for v in pair) for k,pair in other.contexts.items()}
    return other


def evaluate(helper,model,tokens,targets,state,seed):
    traces=[];old=helper.batched_chunks
    def tapped(*args,**kwargs):
        winners=kwargs.get('winners')
        if winners is None:winners=[]
        kwargs['winners']=winners;begin=len(winners)
        value=kernel(*args,**kwargs)
        forces=args[4] if len(args)>4 else kwargs.get('forces')
        traces.append(dict(winners=torch.stack(winners[begin:],dim=1),forces=forces))
        return value
    helper.batched_chunks=tapped;model.zero_grad(set_to_none=True)
    try:
        objective,logits,following,activity=helper.batched_objective(model,tokens,targets,state,seed)
        objective.backward()
    finally:helper.batched_chunks=old
    return dict(gradients=C.grads(model),objective=float(objective.detach()),logits=logits.detach(),
        state=following.detach(),activity=activity,traces=traces)


def errors(candidate,reference):
    rows=[];error=0.;norm=0.;maximum=0.
    for n in reference.keys()|candidate.keys():
        x=candidate[n].double() if n in candidate else torch.zeros_like(reference[n])
        y=reference[n].double() if n in reference else torch.zeros_like(x)
        delta=x-y;e=float(delta.square().sum());v=float(y.square().sum());error+=e;norm+=v
        mx=float(delta.abs().max());maximum=max(maximum,mx)
        fail=not bool(torch.all(delta.abs()<=3e-6+3e-4*y.abs()))
        rows.append(dict(parameter=n,relative_l2=(e/max(v,1e-300))**.5,maximum_absolute=mx,
                         reference_maximum_absolute=float(y.abs().max()),misses_116_coordinate_tolerance=fail))
    return dict(relative_l2=(error/max(norm,1e-300))**.5,maximum_absolute=maximum,
                failed_parameter_tensors=sum(r['misses_116_coordinate_tolerance'] for r in rows),parameters=sorted(rows,key=lambda r:r['parameter']))


def route_mismatches(left,right):
    assert len(left)==len(right)
    rows=[]
    for a,b in zip(left,right):
        assert a['forces']==b['forces']
        rows.append(dict(lanes=a['winners'].shape[0],races=a['winners'].shape[1],
                         mismatches=int((a['winners']!=b['winners']).sum())))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');arrays=out.with_suffix('.vectors.npz')
    assert not out.exists() and not arrays.exists();torch.set_num_threads(1);begin=time.perf_counter()
    prior=json.loads((ROOT/'experiments/results/diagnostics/local_language_winner_reuse_resource_audit_20261003T014800Z.json').read_text())
    for name,digest in prior['source_sha256'].items():assert N.sha(ROOT/name)==digest,name
    observed=torch.tensor([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3,2]);inputs=observed[:16];targets=observed[1:17]
    checks=[];families=[];saved={}
    for family in ('private','depth'):
        model=P.make(family);state=P.initial(model);double=copy.deepcopy(model).double();ds=promote(state)
        torch.manual_seed(116329);seed=torch.get_rng_state().clone();caller=seed.clone();arms={}
        for precision,base,entering in [('float32',model,state),('float64',double,ds)]:
            for kind,helper in [('original',Old),('reuse',New)]:
                name=precision+'_'+kind;arms[name]=evaluate(helper,copy.deepcopy(base),inputs,targets,copy.deepcopy(entering),seed)
                C.close(torch.get_rng_state(),caller,True);C.close(arms[name]['activity']['factual_end_rng'],arms['float32_original']['activity']['factual_end_rng'],True)
                names=sorted(arms[name]['gradients']);saved[family+'_'+name+'_parameter_names']=np.asarray(names)
                saved[family+'_'+name+'_parameter_sizes']=np.asarray([arms[name]['gradients'][n].numel() for n in names])
                saved[family+'_'+name+'_gradients']=np.concatenate([arms[name]['gradients'][n].double().numpy().reshape(-1) for n in names])
                for i,trace in enumerate(arms[name]['traces']):saved[family+'_'+name+'_winners_'+str(i)]=trace['winners'].numpy()
        C.grads_close(arms['float64_original']['gradients'],arms['float64_reuse']['gradients'])
        C.close(arms['float64_original']['logits'],arms['float64_reuse']['logits'],True)
        C.state_close(arms['float64_original']['state'],arms['float64_reuse']['state'],True)
        checks.append(f'{family}: EVERY production double gradient agrees for original/full versus winner reuse; factual logits/private state exact')
        same32=route_mismatches(arms['float32_original']['traces'][:1],arms['float32_reuse']['traces'][:1])
        same64=route_mismatches(arms['float64_original']['traces'][:1],arms['float64_reuse']['traces'][:1])
        assert sum(r['mismatches'] for r in same32+same64)==0
        checks.append(f'{family}: original/reuse factual race histories and all caller/end RNG match; no optimizer/parameter updates')
        rows={}
        for kind in ('original','reuse'):
            f32=arms['float32_'+kind];f64=arms['float64_'+kind]
            torch.testing.assert_close(f32['logits'].double(),f64['logits'],rtol=3e-5,atol=1e-5)
            rows[kind]=dict(gradient_error_against_double=errors(f32['gradients'],f64['gradients']),
                all_factual_and_shadow_route_precision_mismatches=route_mismatches(f32['traces'],f64['traces']))
        dg=errors(arms['float64_reuse']['gradients'],arms['float64_original']['gradients'])
        fg=errors(arms['float32_reuse']['gradients'],arms['float32_original']['gradients'])
        families.append(dict(family=family,parameters=sum(p.numel() for p in model.parameters()),depth=8,heads=2,pool=2,payload=16,targets=16,
            precision_comparisons=rows,double_original_reuse_error=dg,float32_original_reuse_error=fg,
            no_factual_or_shadow_precision_route_changes=all(not r['mismatches'] for v in rows.values() for r in v['all_factual_and_shadow_route_precision_mismatches'])))
    np.savez_compressed(arrays,**saved)
    names=['experiments/causal_language_precision_audit.py','experiments/theory/117_production_counterfactual_precision_audit.md']
    result=dict(status='completed',args=vars(a),contracts_passed=len(checks),contracts=checks,families=families,
        source_sha256={**prior['source_sha256'],**W.sources(),**{n:N.sha(ROOT/n) for n in names}},
        parent_result='experiments/results/diagnostics/local_language_winner_reuse_resource_audit_20261003T014800Z.json',
        parent_result_sha256=N.sha(ROOT/'experiments/results/diagnostics/local_language_winner_reuse_resource_audit_20261003T014800Z.json'),
        vectors=dict(path=str(arrays.relative_to(ROOT)),sha256=N.sha(arrays),bytes=arrays.stat().st_size),
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Same represented weights/state promoted to production double; exact race-history/gradient precision audit only, '
            'no optimizer or trained text8 quality. Total diagnostic fitting-work/traffic/energy unknown, not zero.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',contracts_passed=len(checks),wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'],
        families=[dict(family=f['family'],double_gradient_relative_error=f['double_original_reuse_error']['relative_l2'],
            no_precision_route_changes=f['no_factual_or_shadow_precision_route_changes'],
            float32_old_error=f['precision_comparisons']['original']['gradient_error_against_double']['relative_l2'],
            float32_reuse_error=f['precision_comparisons']['reuse']['gradient_error_against_double']['relative_l2']) for f in families])))


if __name__=='__main__':main()
