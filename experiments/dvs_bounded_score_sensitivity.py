"""Primitive/native bounded bridge contracts and conditioned Jacobian comparison."""
import argparse
import copy
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from dvs_native_window_intervention import equal_state
from dvs_key_score_decomposition import replay
from dvs_conditional_clock_choice_geometry import flat_grad,geometry,block
from sleeping_machines.bounded_score_sensitivity import bounded_score_bridge,smooth_slope
from sleeping_machines.native_bounded_score_diagnostic import NativeBoundedScoreDiagnostic


def primitive():
    raw=torch.tensor([-40.,-20.,-13.,-10.,-.2,0.,.2,10.,13.,20.,40.],dtype=torch.float64,requires_grad=True)
    for alpha in (0.,.1,1.):
        result=bounded_score_bridge(raw,alpha)
        assert torch.autograd.gradcheck(lambda r:bounded_score_bridge(r,alpha),(raw,),eps=1e-6,atol=1e-7,rtol=1e-6)
        derivative=torch.autograd.grad(result.sum(),raw)[0]
        expected=(1-alpha)*(raw.detach().abs()<12).double()+alpha*smooth_slope(raw.detach())
        torch.testing.assert_close(derivative,expected,rtol=1e-12,atol=1e-12)
        if alpha:assert bool((derivative>0).all())
        else:assert torch.equal(result,raw.clamp(-12,12))
        grid=torch.linspace(-1000,1000,100001,dtype=torch.float64);values=bounded_score_bridge(grid,alpha)
        assert bool((values.abs()<=12+1e-12).all()) and bool((values[1:]>=values[:-1]).all())
        torch.testing.assert_close(values,-bounded_score_bridge(-grid,alpha),rtol=0,atol=0)
    witness=[]
    for alpha in (0.,.1):
        point=torch.tensor([20.,0.],dtype=torch.float64,requires_grad=True)
        scores=bounded_score_bridge(point,alpha);jac=torch.autograd.functional.jacobian(lambda r:bounded_score_bridge(r,alpha),point)
        fisher=jac.T@torch.diag(scores.softmax(0))@jac;rank=int(torch.linalg.matrix_rank(fisher))
        assert rank==(1 if alpha==0. else 2)
        witness.append(dict(alpha=alpha,scores=scores.detach().tolist(),score_jacobian=jac.tolist(),
            fisher_eigenvalues=torch.linalg.eigvalsh(fisher).detach().tolist(),rank=rank))
    return dict(contracts_passed=5,rank_witness=witness)


def make(config,weights):
    torch.manual_seed(config.seed)
    model=NativeBoundedScoreDiagnostic(sources=1,content_dim=33,classes=11,payload=config.payload,
        depth=config.depth,heads=config.heads,pool=config.pool)
    model.load_state_dict(weights);model.eval();return model


def original_gradient_contract(model,reference,row,seed):
    model.bridge=0.;model.clear_trace()
    for m in (model,reference):m.zero_grad(set_to_none=True)
    z,_=N.predict(model,row,seed,True);zr,_=N.predict(reference,row,seed,True);assert torch.equal(z,zr)
    F.cross_entropy(z[None],torch.tensor([row['target']])).backward()
    F.cross_entropy(zr[None],torch.tensor([row['target']])).backward();count=0
    for (name,p),(other,q) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other and (p.grad is None)==(q.grad is None)
        if p.grad is not None:assert torch.equal(p.grad,q.grad),name;count+=1
    model.clear_trace();model.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--geometry',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    base=json.loads((ROOT/a.geometry).read_text());assert base['status']=='completed'
    for name,digest in base['source_sha256'].items():assert N.sha(ROOT/name)==digest
    oldpath=ROOT/base['artifact'];assert N.sha(oldpath)==base['artifact_sha256'];old=np.load(oldpath)
    torch.set_num_threads(1);begin=time.perf_counter();law=primitive();rows=[];checks=[];arrays={};sources=dict(base['source_sha256'])
    seeds=base['noise_seed'];indices=base['indices'];sites=base['sites'];alpha=.1
    for lineage in base['parents']:
        name=lineage['result'];parent=json.loads((ROOT/name).read_text());assert N.sha(ROOT/name)==lineage['result_sha256']
        config=argparse.Namespace(**parent['args']);expanded=copy.copy(config);expanded.fit=984
        fit,_,data=N.load(expanded);assert data==parent['data'];cp=(ROOT/name).with_suffix('.progress.pt')
        assert N.sha(cp)==lineage['checkpoint_sha256'];saved=torch.load(cp,weights_only=False)
        initial=N.make_model(config,fast=False).state_dict()
        for encoder,weights in [('initial',initial),('fixed_pass4',saved['online_model'])]:
            model=make(config,weights);reference=N.make_model(config,fast=False);reference.load_state_dict(weights)
            gradients=original_gradient_contract(model,reference,fit[indices[0]],seeds)
            params=tuple(model.parameters());groups={b:[] for b in ('clock_bias','key_query','message_state_transport','classifier')}
            offset=0;names=[n for n,_ in model.named_parameters()];assert names==base['parameter_names']
            for name,param in model.named_parameters():groups[block(name)].extend(range(offset,offset+param.numel()));offset+=param.numel()
            before={n:v.detach().clone() for n,v in model.state_dict().items()}
            for fit_index in indices:
                row=fit[fit_index];model.bridge=0.;model.clear_trace()
                zr,sr,rngr=replay(reference,row,seeds);z,st,rng=replay(model,row,seeds)
                assert torch.equal(z,zr) and torch.equal(rng,rngr);equal_state(st,sr)
                factual=[dict(delay=r['delay'],winner=r['winner']) for r in model.records]
                model.bridge=alpha;model.clear_trace();model.observed_trace=factual;model.condition_previous_clocks=True
                N.predict(model,row,seeds,False);trace=model.records
                for event,depth,head in sites:
                    index=event*4+depth*2+head;r=trace[index];score=r['scores'];pi=score.detach().double().softmax(0)
                    u=flat_grad(score[0]-score[1],params);v=flat_grad(score.double().logsumexp(0),params)
                    key=f's{config.seed}_{encoder}_i{fit_index}_e{event}d{depth}h{head}'
                    uh=torch.from_numpy(old[key+'_choice']);vh=torch.from_numpy(old[key+'_clock'])
                    original=next(x for x in base['rows'] if x['seed']==config.seed and x['encoder']==encoder
                        and x['fit_index']==fit_index and x['site']==[event,depth,head])
                    hard=geometry(uh,vh,torch.tensor(original['probabilities'],dtype=torch.float64))
                    local=dict(seed=config.seed,encoder=encoder,fit_index=fit_index,site=[event,depth,head],
                        raw_scores=r['raw'].detach().tolist(),bridge_scores=score.detach().tolist(),
                        raw_slope=((1-alpha)*(r['raw'].detach().abs()<12).double()+alpha*smooth_slope(r['raw'].detach().double())).tolist(),
                        original_hard=hard,bridge=geometry(u,v,pi),original_clamped=any(abs(s)>=12 for s in original['scores']),
                        bridge_blocks={b:geometry(u[ids],v[ids],pi) for b,ids in groups.items()})
                    arrays[key+'_choice']=u.numpy();arrays[key+'_clock']=v.numpy();rows.append(local)
                # Full unconditioned positive native primal retains actual coupled histories.
                model.clear_trace();model.bridge=alpha;zb,sb,rngb=replay(model,row,seeds)
                assert torch.equal(rngb,rngr) and (sb.events,sb.candidate_scores,sb.selected_updates,sb.counterfactual_values)==(21,168,84,0)
                assert float(sb.contexts[0][1].max())>=1.
                model.clear_trace();zs,ss,rngs=replay(model,dict(row,target=(row['target']+1)%11),seeds)
                assert torch.equal(zb,zs) and torch.equal(rngb,rngs);equal_state(sb,ss)
                assert all(torch.equal(before[n],value.detach()) for n,value in model.state_dict().items())
                checks.append(dict(seed=config.seed,encoder=encoder,fit_index=fit_index,alpha0_exact_primal_state_rng=True,
                    positive_full_native_exact_rng_target_invariance=True,real_sparse_updates=sb.selected_updates,
                    positive_final_ready_time=float(sb.contexts[0][1].max()),positive_live_state_bytes=sb.storage()['persistent_tensor_bytes'],
                    alpha0_exact_parameter_gradients=gradients,extra_preclamp_dot_products_per_prefix=168))
                print(json.dumps(dict(seed=config.seed,encoder=encoder,fit_index=fit_index,
                    rows=[{k:r[k] for k in ('site','raw_scores','raw_slope','original_clamped','bridge')} for r in rows[-4:]])),flush=True)
            model.clear_trace()
            try:N.predict(model,fit[indices[0]],seeds,True)
            except ValueError as e:assert 'training not installed' in str(e)
            else:raise AssertionError('Positive bridge training must be refused')
    artifact=out.with_suffix('.directions.npz');np.savez_compressed(artifact,**arrays)
    own=['sleeping_machines/bounded_score_sensitivity.py','sleeping_machines/native_bounded_score_diagnostic.py',
        'experiments/dvs_bounded_score_sensitivity.py','experiments/theory/104_hard_score_clamps_and_bounded_sensitivity.md']
    sources.update({name:N.sha(ROOT/name) for name in own})
    result=dict(status='completed',args=vars(a),base_geometry_sha256=N.sha(ROOT/a.geometry),parents=base['parents'],
        primitive=law,rows=rows,native_contracts=checks,bridge_alpha=alpha,parameter_vjps=64,
        artifact=str(artifact.relative_to(ROOT)),artifact_sha256=N.sha(artifact),source_sha256=sources,
        optimizer_steps=0,development_evaluations=0,whole_audit_flops=None,
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(host=platform.node(),device='CPU',threads=1),
        scope='Bounded native score-map prototype: exact alpha0 forward/state/RNG/all-gradient nesting, '
            'positive full native causal sparse forward/target invariance with actual changed clocks and routes. '
            'Hard-vs-bridge parameter geometry conditions on the SAME old winner/time histories, not a new quality evaluation. '
            'Hooks/recomputed168 extra dot products are paid diagnostic work. Positive training refuses; '
            'no optimizer, counterfactual loss/utility, DEV/test or inference/quality/work advantage claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
