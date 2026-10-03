"""Hutchinson conditional race-credit covariance at the actual16-episode batch."""
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


def projector(model,scores,q,batch):
    assert model.pool==2 and not q.requires_grad
    parameters=list(model.parameters());pi=torch.stack(scores,1).detach().softmax(-1)
    coefficient=pi[:,:,0]*pi[:,:,1]*(q[:,:,0]-q[:,:,1])/batch
    difference=torch.stack(scores,1)[:,:,0]-torch.stack(scores,1)[:,:,1]
    dual=torch.zeros_like(difference,requires_grad=True)
    h=torch.autograd.grad((difference*dual).sum(),parameters,retain_graph=True,create_graph=True,allow_unused=True)
    def apply(z):
        pieces=[];cursor=0
        for p,g in zip(parameters,h):
            value=torch.from_numpy(z[cursor:cursor+p.numel()]).reshape_as(p);cursor+=p.numel()
            if g is not None and g.requires_grad:pieces.append((g*value).sum())
        assert cursor==len(z) and pieces
        jz=torch.autograd.grad(sum(pieces),dual,retain_graph=True)[0]
        return (coefficient*jz).detach().numpy()
    return apply


def estimate(draws,denominator):
    x=np.asarray(draws);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
    assert np.isfinite(x).all() and (x>=0).all()
    return dict(directions=len(x),parameter_trace_covariance_estimate=mean,
        parameter_trace_covariance_empirical_SE=se,empirical_relative_SE=se/max(mean,1e-300),
        combined_MSE_ratio_estimate=mean/denominator,combined_MSE_ratio_empirical_SE=se/denominator,
        worst_case_relative_RMS_bound=float(np.sqrt(2/len(x))),all_trace_probes=x.tolist(),
        scope='Unbiased conditional trace estimate; empirical SE descriptive, no guaranteed confidence interval or Adam variance')


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert Path(a.tag).name==a.tag and not out.exists()
    torch.set_num_threads(1);begin=time.perf_counter();caller=torch.get_rng_state().clone()
    parent=json.loads((ROOT/PARENT).read_text());bindings={PARENT:N.sha(ROOT/PARENT)}
    assert parent['status']=='completed' and parent['contracts_passed']==5
    bindings.update(parent['source_sha256']);vp=parent['vectors']['path'];bindings[vp]=parent['vectors']['sha256']
    for name,digest in bindings.items():
        if N.sha(ROOT/name)!=digest:raise ValueError('Changed parent binding '+name)
    rows,data=P.fit_inputs(json.loads((ROOT/P.PARENT).read_text()));assert data==parent['data']
    sources={**parent['source_sha256'],**{n:N.sha(ROOT/n) for n in
        ('experiments/projected_batch_credit_variance.py','experiments/theory/127_projected_actual_batch_credit_variance.md')}}
    contracts=[];cases=[]
    with torch.random.fork_rng():
        with np.load(ROOT/vp,allow_pickle=False) as vectors:
            for c in parent['cases']:
                depth=c['depth'];model=C.make_model(SimpleNamespace(seed=7,payload=16,depth=depth,heads=2,pool=2,clock_step=.05)).double()
                theta=torch.cat([p.detach().flatten() for p in model.parameters()]).numpy()
                assert np.array_equal(theta,vectors[f'd{depth}_weights']);weights=copy.deepcopy(model.state_dict());scores=[]
                z=batched_logits(model,rows[:2],V.SEED,record=scores)
                loss=F.cross_entropy(z,torch.tensor([r['target'] for r in rows[:2]]))
                assert abs(float(loss.detach())-c['initial_same_FIT_nll'])<1e-12
                q=torch.from_numpy(vectors[f'd{depth}_returns']);apply=projector(model,scores,q,2)
                banks=[vectors[f'd{depth}_episode{j}_parameter_route_vectors'] for j in range(2)]
                rng=np.random.default_rng(127000+depth);errors=[]
                for _ in range(3):
                    direction=(2*rng.integers(0,2,len(theta))-1).astype(np.float64);actual=apply(direction)
                    expected=np.stack([v@direction for v in banks])
                    np.testing.assert_allclose(actual,expected,rtol=3e-7,atol=3e-9)
                    np.testing.assert_allclose(actual.sum(),vectors[f'd{depth}_route']@direction,rtol=3e-7,atol=3e-9)
                    errors.append(float(np.max(np.abs(actual-expected))))
                assert all(torch.equal(weights[n],v) for n,v in model.state_dict().items())
                contracts.append(dict(depth=depth,projected_directions=3,per_race_entries_per_direction=int(expected.size),
                    maximum_entry_errors=errors,scope='Every projected route contribution equals saved exact parameter bank dot direction'))
                print(json.dumps(dict(depth=depth,status='projection_contract_completed',wall_s=time.perf_counter()-begin)),flush=True)
                del model,scores,apply,banks,z,loss
        for depth in (4,6):
            model=C.make_model(SimpleNamespace(seed=7,payload=16,depth=depth,heads=2,pool=2,clock_step=.05)).double()
            weights=copy.deepcopy(model.state_dict());d=V.prepare(model,rows);apply=projector(model,d['scores'],d['q'],len(rows))
            parameters=sum(p.numel() for p in model.parameters());rng=np.random.default_rng(127160+depth)
            traces={k:[] for k in (8,32)}
            for index in range(32):
                direction=(2*rng.integers(0,2,parameters)-1).astype(np.float64);projection=apply(direction)
                for k in traces:traces[k].append(sum(V.covariance(projection[j,:R,None],k) for j,R in enumerate(d['lengths'])))
                if index%8==7:print(json.dumps(dict(depth=depth,directions=index+1,status='projection_progress',wall_s=time.perf_counter()-begin)),flush=True)
            combined=d['factual']+d['full'];denominator=float(combined.square().sum())
            assert denominator>0 and all(torch.equal(weights[n],v) for n,v in model.state_dict().items())
            cases.append(dict(depth=depth,batch=len(rows),fit_indices=list(range(16)),parameters=parameters,
                precision='FP32 represented initialization promoted to double',noise_seed=V.SEED,
                races_per_episode=d['lengths'],available_receivers=depth*4,key_scores=sum(d['lengths'])*2,
                selected_updates=sum(d['lengths']),full_return_shadow_lanes=d['lanes'],full_return_shadow_events=d['events'],
                initial_same_FIT_nll=float(d['loss'].detach()),factual_gradient_norm=float(d['factual'].norm()),
                full_route_gradient_norm=float(d['full'].norm()),combined_gradient_norm=float(combined.norm()),
                estimates=[dict(k=k,**estimate(x,denominator)) for k,x in traces.items()]))
            del model,d,apply
    assert torch.equal(caller,torch.get_rng_state())
    for name,digest in {**bindings,**sources}.items():
        if N.sha(ROOT/name)!=digest:raise ValueError('Concurrent source/artifact change '+name)
    result=dict(status='completed',args=vars(a),contracts_passed=4,contracts=contracts,
        additional_contract='Caller RNG, original weights, every frozen source/data/parent artifact byte preserved',
        parent=PARENT,parent_sha256=bindings[PARENT],source_sha256=sources,data=data,cases=cases,
        work=dict(exact_bank_contract_projections=9,main_projected_pullbacks=64,executed_optimizer_steps=0,
            main_full_return_shadow_lanes=sum(c['full_return_shadow_lanes'] for c in cases),
            main_full_return_shadow_events=sum(c['full_return_shadow_events'] for c in cases),
            total_FLOPs=None,traffic=None,energy=None),
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Actual B16 initialization-only declared-choice-surrogate covariance ESTIMATE, not exact trace, Adam variance, trained causality, fit quality or advantage')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',contracts=4,wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'])))


if __name__=='__main__':main()
