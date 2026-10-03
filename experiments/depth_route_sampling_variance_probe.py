"""Conditional parameter variance of uniform race subsampling across depth."""
import argparse
import copy
import itertools
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
import dvs_batched_le_benchmark as BL
import depth_growth_plasticity_probe as P
from sleeping_machines.batched_episodes import batched_logits

SEED=171323


def vector(gradient,parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p,g in zip(parameters,gradient)])


def grad(objective,parameters):
    return vector(torch.autograd.grad(objective,parameters,retain_graph=True,allow_unused=True),parameters)


def all_returns(model,rows):
    lengths=[len(r['events'])*model.depth*model.heads for r in rows]
    lanes=[];forces=[];indices=[]
    for j,row in enumerate(rows):
        for race in range(lengths[j]):
            for alt in range(model.pool):
                lanes.append(row);forces.append((race,alt));indices.append((j,race,alt))
    with torch.no_grad():
        z=batched_logits(model,lanes,SEED,forces)
        losses=F.cross_entropy(z,torch.tensor([r['target'] for r in lanes]),reduction='none')
    q=z.new_zeros(len(rows),max(lengths),model.pool)
    for (j,r,a),value in zip(indices,losses):q[j,r,a]=value
    return q,lengths,len(lanes),sum(len(r['events']) for r in lanes)


def prepare(model,rows):
    parameters=list(model.parameters());weights=copy.deepcopy(model.state_dict())
    rng=torch.get_rng_state().clone();scores=[]
    z=batched_logits(model,rows,SEED,record=scores)
    loss=F.cross_entropy(z,torch.tensor([r['target'] for r in rows]))
    q,lengths,lanes,events=all_returns(model,rows)
    pi=torch.stack(scores,dim=1).softmax(-1)
    terms=(pi*q).sum(-1)/len(rows)
    factual=grad(loss,parameters);full=grad(terms.sum(),parameters)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(weights[n],v) for n,v in model.state_dict().items())
    return dict(parameters=parameters,z=z,loss=loss,q=q,lengths=lengths,
        terms=terms,scores=scores,factual=factual,full=full,lanes=lanes,events=events)


def covariance(v,k):
    R=len(v);k=min(k,R)
    if k==R:return 0.
    centered=v-v.mean(0)
    return R*(R-k)/(k*(R-1))*float(np.square(centered).sum())


def tiny_contract():
    a=SimpleNamespace(seed=7,payload=4,depth=2,heads=2,pool=2,clock_step=.05)
    model=C.make_model(a).double()
    rows=[dict(index=0,target=1,events=[(.05,np.linspace(-1,1,33))])]
    d=prepare(model,rows);v=torch.stack([grad(t,d['parameters']) for t in d['terms'][0]])
    torch.testing.assert_close(v.sum(0),d['full'],rtol=3e-7,atol=3e-9)
    estimates=torch.stack([2*v[list(s)].sum(0) for s in itertools.combinations(range(4),2)])
    torch.testing.assert_close(estimates.mean(0),d['full'],rtol=3e-7,atol=3e-9)
    expected=covariance(v.numpy(),2)
    actual=float((estimates-d['full']).square().sum(-1).mean())
    assert abs(actual-expected)<=1e-10*max(1.,expected)
    original,_=BL.route_term(model,rows,SEED,d['scores'])
    torch.testing.assert_close(grad(original,d['parameters']),d['full'],rtol=3e-7,atol=3e-9)
    return dict(R=4,k=2,subsets=6,exact_trace_covariance=expected,
        enumerated_trace_covariance=actual,scope='Actual FP64 native parameter vectors, mean/covariance and original driver equivalence')


def adam_delta(g,weights,lr=.003):
    k=min(1.,1./(float(np.linalg.norm(g))+1e-6))
    h=k*g
    return (weights-lr*h/(np.abs(h)+1e-8))-weights,k


def distribution(vectors,factual,weights,k,seed,draws=64):
    route=sum(v.sum(0) for v in vectors);full=factual+route
    reference,reference_clip=adam_delta(full,weights)
    rng=np.random.default_rng(seed);samples=[];indices=[]
    for _ in range(draws):
        route_hat=np.zeros_like(route);chosen=[]
        for v in vectors:
            R=len(v);n=min(k,R);s=np.sort(rng.choice(R,n,replace=False));chosen.append(s.tolist())
            route_hat+=(R/n)*v[s].sum(0)
        g=factual+route_hat;delta,clip=adam_delta(g,weights)
        def cosine(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-300))
        samples.append(dict(route_squared_error=float(np.square(route_hat-route).sum()),
            combined_relative_error=float(np.linalg.norm(g-full)/max(np.linalg.norm(full),1e-300)),
            combined_cosine=cosine(g,full),clip_factor=clip,
            Adam_delta_relative_error=float(np.linalg.norm(delta-reference)/max(np.linalg.norm(reference),1e-300)),
            Adam_delta_cosine=cosine(delta,reference),
            Adam_sign_disagreement_fraction=float(np.mean(np.sign(delta)!=np.sign(reference)))))
        indices.append(chosen)
    metrics={name:dict(mean=float(np.mean([s[name] for s in samples])),
        minimum=float(np.min([s[name] for s in samples])),maximum=float(np.max([s[name] for s in samples])))
        for name in samples[0]}
    trace=sum(covariance(v,k) for v in vectors)
    return dict(k=k,draws=draws,seed=seed,exact_parameter_trace_covariance=trace,
        exact_route_MSE_over_squared_full_route_mean=trace/max(float(route@route),1e-300),
        exact_combined_MSE_over_squared_combined_mean=trace/max(float(full@full),1e-300),
        all_race_clip_factor=reference_clip,metrics=metrics,samples=samples,sample_indices=indices,
        scope='Exact conditional covariance; cached draws/descriptive fresh-Adam transforms, not executed sampling fits')


def actual_adam_contract(model,g):
    before=torch.cat([p.detach().flatten() for p in model.parameters()]);cursor=0
    for p in model.parameters():
        p.grad=torch.from_numpy(g[cursor:cursor+p.numel()]).reshape_as(p).clone();cursor+=p.numel()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    torch.optim.Adam(model.parameters(),lr=.003).step()
    delta=torch.cat([p.detach().flatten() for p in model.parameters()])-before
    expected,_=adam_delta(g,before.numpy())
    torch.testing.assert_close(delta,torch.from_numpy(expected),rtol=3e-7,atol=3e-9)
    return float(np.linalg.norm(delta.numpy()-expected))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');artifact=out.with_suffix('.vectors.npz')
    assert Path(a.tag).name==a.tag and not out.exists() and not artifact.exists()
    torch.set_num_threads(1);started=time.perf_counter();caller_rng=torch.get_rng_state().clone()
    parent_path=ROOT/P.PARENT;parent=json.loads(parent_path.read_text());parent_hash=N.sha(parent_path)
    rows,data=P.fit_inputs(parent);rows=rows[:2]
    arrays={};cases=[];checks=[]
    with torch.random.fork_rng():
        tiny=tiny_contract();checks.append('Tiny actual parameter-vector unbiased mean/exact covariance/original-driver contract')
        for depth in (2,4,6):
            settings=SimpleNamespace(seed=7,payload=16,depth=depth,heads=2,pool=2,clock_step=.05)
            model=C.make_model(settings).double();weights=copy.deepcopy(model.state_dict());d=prepare(model,rows)
            theta=torch.cat([p.detach().flatten() for p in model.parameters()]).numpy().copy()
            banks=[]
            for j,R in enumerate(d['lengths']):
                bank=np.empty((R,len(theta)),dtype=np.float64)
                for r in range(R):bank[r]=grad(d['terms'][j,r],d['parameters']).numpy()
                banks.append(bank)
            full=sum(v.sum(0) for v in banks)
            torch.testing.assert_close(torch.from_numpy(full),d['full'],rtol=3e-7,atol=3e-9)
            replay,_=BL.route_term(model,rows,SEED,d['scores'])
            torch.testing.assert_close(grad(replay/len(rows),d['parameters']),d['full'],rtol=3e-7,atol=3e-9)
            rng=np.random.default_rng(124900+depth);subset=d['terms'].new_zeros(())
            cached=np.zeros_like(full)
            for j,v in enumerate(banks):
                R=len(v);n=min(8,R);choice=rng.choice(R,n,replace=False)
                subset+=(R/n)*d['terms'][j,choice].sum();cached+=(R/n)*v[choice].sum(0)
            torch.testing.assert_close(torch.from_numpy(cached),grad(subset,d['parameters']),rtol=3e-7,atol=3e-9)
            factual=d['factual'].numpy().copy();combined=factual+full
            distributions=[distribution(banks,factual,theta,k,124323+depth+k) for k in (8,32)]
            errors=[]
            for g in (combined,factual+cached):errors.append(actual_adam_contract(copy.deepcopy(model),g))
            assert all(torch.equal(weights[n],p) for n,p in model.state_dict().items())
            arrays[f'd{depth}_factual']=factual;arrays[f'd{depth}_route']=full;arrays[f'd{depth}_weights']=theta
            arrays[f'd{depth}_returns']=d['q'].detach().numpy()
            for j,v in enumerate(banks):arrays[f'd{depth}_episode{j}_parameter_route_vectors']=v
            cases.append(dict(depth=depth,initialization_seed=7,precision='float32-initialized represented weights promoted to FP64',
                parameters=len(theta),fit_indices=[0,1],targets=2,events_per_episode=[len(r['events']) for r in rows],
                races_per_episode=d['lengths'],available_receivers=depth*4,
                selected_updates_per_episode=[R for R in d['lengths']],key_scores_per_episode=[2*R for R in d['lengths']],
                shadow_lanes=d['lanes'],shadow_events=d['events'],individual_route_VJPs=sum(d['lengths']),
                initial_same_FIT_nll=float(d['loss'].detach()),factual_gradient_norm=float(np.linalg.norm(factual)),
                full_route_gradient_norm=float(np.linalg.norm(full)),combined_gradient_norm=float(np.linalg.norm(combined)),
                per_episode_route_cancellation=[float(np.linalg.norm(v.sum(0))/max(sum(np.linalg.norm(row) for row in v),1e-300)) for v in banks],
                distributions=distributions,actual_fresh_Adam_formula_error_norms=errors,
                scope='One fixed two-episode initialization history; not trained weights, whole-risk gradient, isolated depth perturbation or benchmark'))
            checks.append(f'D{depth}: all parameter route VJPs sum to both full objectives and fixed sampled8 backward; two actual fresh Adam forks; weights unchanged')
            print(json.dumps(dict(depth=depth,status='diagnostic_case_completed',elapsed_s=time.perf_counter()-started)),flush=True)
            del d,model,banks,replay,subset
    assert torch.equal(caller_rng,torch.get_rng_state());assert N.sha(parent_path)==parent_hash
    checks.append('Caller RNG and original source-data parent preserved')
    sources={**BL.sources(),**{n:N.sha(ROOT/n) for n in ('experiments/depth_route_sampling_variance_probe.py',
        'experiments/theory/124_depth_route_sampling_variance.md','experiments/depth_growth_plasticity_probe.py')}}
    np.savez_compressed(artifact,**arrays)
    result=dict(status='completed',args=vars(a),contracts_passed=len(checks),contracts=checks,tiny_contract=tiny,
        cases=cases,data=data,parent=P.PARENT,parent_sha256=parent_hash,source_sha256=sources,
        vectors=dict(path=str(artifact.relative_to(ROOT)),sha256=N.sha(artifact)),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Conditional INITIALIZATION-only FP64 choice-credit sampling diagnostic; unknown total FLOPs/traffic/energy, not zero. '
              'All alternative forward losses paid once; per-race VJPs and actual contract forks paid. No DEV/test/quality fit or training recommendation.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',contracts=len(checks),wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'])))


if __name__=='__main__':main()
