"""Frozen native conditional route/common-clock parameter geometry."""
import argparse
import copy
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from dvs_native_window_intervention import equal_state
from dvs_key_score_decomposition import replay
from sleeping_machines.addressed_event_heads import AddressedEventHeads


class ConditionedHeads(AddressedEventHeads):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.records=[];self.conditioned_trace=None

    def race(self,scores,values=None):
        if self.training or values is not None:raise ValueError('Conditioned geometry is inference-only')
        _,delay,winner=super().race(scores)
        index=len(self.records)
        if self.conditioned_trace is not None:
            record=self.conditioned_trace[index];delay=record['delay'];winner=record['winner']
        self.records.append(dict(scores=scores,delay=delay.detach().clone(),winner=winner.detach().clone()))
        return None,delay.detach(),winner.detach()


def make(config,weights):
    torch.manual_seed(config.seed)
    model=ConditionedHeads(sources=1,content_dim=33,classes=11,payload=config.payload,
        depth=config.depth,heads=config.heads,pool=config.pool)
    model.load_state_dict(weights);model.eval();return model


def block(name):
    if name.endswith('clock_bias'):return 'clock_bias'
    if name.startswith('queries.') or name.endswith('.key') or '.key_read.' in name:return 'key_query'
    if name.startswith('head.'):return 'classifier'
    return 'message_state_transport'


def flat_grad(value,params):
    gradients=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
    return torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten().double()
        for p,g in zip(params,gradients)])


def geometry(u,v,probability):
    u2=float(u@u);v2=float(v@v);uv=float(u@v)
    cosine=uv/math.sqrt(u2*v2) if u2>0 and v2>0 else None
    choice=float(probability.prod());gram=torch.tensor([[choice*u2,math.sqrt(choice)*uv],
        [math.sqrt(choice)*uv,v2]],dtype=torch.float64)
    eigen=torch.linalg.eigvalsh(gram);minimum=max(0.,float(eigen[0]));maximum=float(eigen[-1])
    return dict(choice_sensitivity_norm=math.sqrt(u2),clock_sensitivity_norm=math.sqrt(v2),cosine=cosine,
        independent_direction_fraction=None if cosine is None else max(0.,1-cosine*cosine),
        categorical_information_trace=choice*u2,clock_information_trace=v2,
        joint_nonzero_eigenvalues=[minimum,maximum],information_eigenvalue_ratio=minimum/maximum if maximum else None)


def conditioned_scores(model,row,seed,trace):
    model.records=[];model.conditioned_trace=trace
    with torch.no_grad():replay(model,row,seed)
    return torch.stack([r['scores'].detach().double() for r in model.records])


def finite_difference(model,row,seed,trace,params,u,v,index,before):
    direction=u/u.norm();parts=[];offset=0
    for p in params:parts.append(direction[offset:offset+p.numel()].reshape(p.shape).to(p));offset+=p.numel()
    expected=[float(u@direction),float(v@direction)];checks=[]
    try:
        for eps in (.002,.001):
            observed=[]
            for sign in (1.,-1.):
                with torch.no_grad():
                    for (name,p),d in zip(model.named_parameters(),parts):p.copy_(before[name]+sign*eps*d)
                scores=conditioned_scores(model,row,seed,trace)[index]
                observed.append(torch.stack([scores[0]-scores[1],scores.logsumexp(0)]))
            fd=(observed[0]-observed[1])/(2*eps);error=(fd-torch.tensor(expected,dtype=torch.float64)).abs()
            torch.testing.assert_close(fd,torch.tensor(expected,dtype=torch.float64),rtol=.02,atol=.005)
            checks.append(dict(epsilon=eps,expected=expected,finite_difference=fd.tolist(),absolute_error=error.tolist(),
                relative_tolerance=.02,absolute_tolerance=.005))
    finally:
        with torch.no_grad():
            for name,p in model.named_parameters():p.copy_(before[name])
        model.records=[];model.conditioned_trace=None
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    torch.set_num_threads(1);begin=time.perf_counter();records=[];parents=[];sources={};arrays={};checks=[]
    excluded=set(np.linspace(256,983,16,dtype=int))
    available=[i for i in range(256,984) if i not in excluded]
    excluded.update(available[i] for i in np.linspace(0,len(available)-1,16,dtype=int))
    available=[i for i in range(256,984) if i not in excluded];indices=[available[0],available[-1]]
    assert not set(indices)&excluded
    sites=[(1,1,0),(19,0,0),(20,0,0),(20,1,0)];seed=510173;parameter_names=None
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        config=argparse.Namespace(**parent['args']);expanded=copy.copy(config);expanded.fit=984
        assert (config.fit,config.dev,config.epochs,config.depth,config.heads,config.pool)==(256,192,4,2,2,2)
        fit,_,data=N.load(expanded);assert data==parent['data']
        for f,digest in parent['source_sha256'].items():assert N.sha(ROOT/f)==digest
        sources.update(parent['source_sha256']);cp=(ROOT/name).with_suffix('.progress.pt')
        saved=torch.load(cp,weights_only=False);assert saved['cursor']['epoch']==5
        assert saved['source_sha256']==parent['source_sha256']
        initial=N.make_model(config,fast=False).state_dict()
        parents.append(dict(result=name,result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(cp),
            original_core_fit_gflops=parent['work']['whole_fit_unit_special_flops_estimate']/1e9))
        for encoder,weights in [('initial',initial),('fixed_pass4',saved['online_model'])]:
            model=make(config,weights);reference=N.make_model(config,fast=False);reference.load_state_dict(weights)
            params=tuple(model.parameters());names=[n for n,_ in model.named_parameters()]
            parameter_names=names;groups={b:[] for b in ('clock_bias','key_query','message_state_transport','classifier')}
            offset=0;slices={}
            for n,param in model.named_parameters():
                ids=list(range(offset,offset+param.numel()));groups[block(n)].extend(ids);slices[n]=(offset,offset+param.numel());offset+=param.numel()
            before={n:v.detach().clone() for n,v in model.state_dict().items()}
            for fit_index in indices:
                row=fit[fit_index];assert len(row['events'])==21
                z,st,rng=replay(reference,row,seed);model.records=[];zc,sc,rngc=replay(model,row,seed)
                assert torch.equal(z,zc) and torch.equal(rng,rngc);equal_state(st,sc)
                model.records=[];zs,ss,rngs=replay(model,dict(row,target=(row['target']+1)%11),seed)
                assert torch.equal(zc,zs) and torch.equal(rngc,rngs);equal_state(sc,ss)
                model.records=[];N.predict(model,row,seed,False);trace=model.records
                assert len(trace)==84
                for event,depth,head in sites:
                    index=event*4+depth*2+head;scores=trace[index]['scores'];prob=scores.detach().double().softmax(0)
                    u=flat_grad(scores[0]-scores[1],params);v=flat_grad(scores.double().logsumexp(0),params)
                    local=dict(seed=config.seed,encoder=encoder,fit_index=fit_index,site=[event,depth,head],
                        scores=scores.detach().tolist(),probabilities=prob.tolist(),full=geometry(u,v,prob),
                        blocks={b:geometry(u[ids],v[ids],prob) for b,ids in groups.items()})
                    analytic=[]
                    if bool((scores.abs()<12).all()):
                        for candidate in (0,1):
                            n=f'units.{depth}.{head}.0.{candidate}.clock_bias';start,end=slices[n]
                            torch.testing.assert_close(u[start:end],torch.tensor([1. if candidate==0 else -1.],dtype=torch.float64),rtol=0,atol=1e-6)
                            torch.testing.assert_close(v[start:end],prob[candidate:candidate+1],rtol=1e-6,atol=1e-7)
                            analytic.append(candidate)
                    local['analytic_clock_bias_coordinates_passed']=analytic
                    key=f's{config.seed}_{encoder}_i{fit_index}_e{event}d{depth}h{head}'
                    arrays[key+'_choice']=u.numpy();arrays[key+'_clock']=v.numpy();records.append(local)
                    if (event,depth,head)==sites[-1]:
                        # Save detached physical observations before rebuilding finite-difference primals.
                        fixed=[dict(delay=r['delay'],winner=r['winner']) for r in trace]
                        local['finite_difference_checks']=finite_difference(model,row,seed,fixed,params,u,v,index,before)
                assert all(torch.equal(before[n],value.detach()) for n,value in model.state_dict().items())
                checks.append(dict(seed=config.seed,encoder=encoder,fit_index=fit_index,
                    exact_primal_state_rng=True,target_invariance=True,parameters_bitwise_restored=True))
                print(json.dumps(dict(seed=config.seed,encoder=encoder,fit_index=fit_index,
                    geometry=[r['full'] for r in records[-4:]])),flush=True)
            try:N.predict(model,fit[indices[0]],seed,True)
            except ValueError as e:assert 'inference-only' in str(e)
            else:raise AssertionError('Conditioned geometry training must be refused')
    artifact=out.with_suffix('.directions.npz');np.savez_compressed(artifact,**arrays)
    own=['experiments/dvs_conditional_clock_choice_geometry.py','experiments/theory/103_native_conditional_clock_choice_geometry.md',
        'experiments/dvs_key_score_decomposition.py','experiments/dvs_native_window_intervention.py']
    sources.update({f:N.sha(ROOT/f) for f in own})
    result=dict(status='completed',args=vars(a),parents=parents,rows=records,primal_contracts=checks,indices=indices,
        sites=sites,noise_seed=seed,score_parameter_vjps=64,parameter_names=parameter_names,
        artifact=str(artifact.relative_to(ROOT)),artifact_sha256=N.sha(artifact),source_sha256=sources,
        optimizer_steps=0,development_evaluations=0,whole_audit_flops=None,
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(host=platform.node(),device='CPU',threads=1),
        scope='Exact native primal with conditioned previous winner/time observations for parameter score sensitivities. '
            'Actual causal sparse contents/state writes retained. Two fixed-pass producers and initial reservoirs; '
            'two disjoint unused FIT inputs/four fixed sites. Conditional local information geometry, not ordinary '
            'sampled-noise time gradients, target loss/utility, expected-risk gradient, optimizer or architecture replacement. '
            'Prior fits and all VJPs/audit costs paid; full arithmetic/traffic/energy unknown, not zero.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
