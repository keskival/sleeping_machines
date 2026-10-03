"""Native every-gradient contracts and paid Adam/recovery admission for129."""
import argparse,copy,io,json,resource,sys,time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import dvs_native_benchmark as N
import dvs_clock_calibrated_benchmark as C
import dvs_batched_le_benchmark as BL
import sleeping_machines.batched_episodes as K
from sleeping_machines.conditional_branch_content_credit import conditional_branch_objective as objective
from race_language_screen import capture
from parallel_head_accumulated_language import merge
SEED=171323
def sources():return {**BL.sources(),**{n:N.sha(ROOT/n) for n in (
    'experiments/conditional_branch_content_contracts.py','experiments/theory/129_conditional_branch_content_credit.md',
    'sleeping_machines/conditional_branch_content_credit.py')}}
def model():return C.make_model(SimpleNamespace(seed=7,payload=4,depth=2,heads=2,pool=2,clock_step=.05)).double()
def rows(lengths):return [dict(index=j,target=j+1,events=[(.05*(i+1),np.sin(np.arange(33)*.17+i*.4+j)) for i in range(T)]) for j,T in enumerate(lengths)]
def gradients(loss,parameters):return [torch.zeros_like(p) if g is None else g for p,g in zip(parameters,torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True))]
def close(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=3e-7,atol=3e-9)
@contextmanager
def clock_reference(history=None):
    original=K.LaneRace;records=[];cursor=0
    def apply(scores,proposals,noise,forced):
        nonlocal cursor
        rates=scores.double().exp()
        if history is None:
            result=original.apply(scores,proposals,noise,forced);first=(noise[None,:]/rates).min(-1).values
            records.append(dict(winner=result[2].detach().clone(),latent=(first*rates.sum(-1)).detach(),first=first.detach(),forced=forced.detach().clone()))
            return result
        saved=history[cursor];cursor+=1
        assert torch.equal(forced,saved['forced']) and len(scores)==len(saved['winner'])
        first=saved['latent']/rates.sum(-1)
        return proposals[torch.arange(len(scores)),saved['winner']],.001+.010*first/(1+first),saved['winner']
    K.LaneRace=SimpleNamespace(apply=apply)
    try:yield records
    finally:
        K.LaneRace=original
        if history is not None:assert cursor==len(history)
def explicit(m,data,selected):
    scores=[];factual=K.batched_logits(m,data,SEED,record=scores);pi=torch.stack([scores[r][j] for j,r in enumerate(selected)]).softmax(-1)
    losses=torch.stack([torch.stack([F.cross_entropy(K.batched_logits(m,[row],SEED,[(selected[j],i)]),torch.tensor([row['target']])) for i in range(2)]) for j,row in enumerate(data)])
    R=pi.new_tensor([len(row['events'])*4 for row in data]);joint=(pi.detach()*losses).sum(-1).mean()+(R*(pi*losses.detach()).sum(-1)).mean()
    return gradients(joint,list(m.parameters())),factual,losses
def update(m,opt,data,selected,trace=False):
    opt.zero_grad(set_to_none=True);box={};stages={}
    def run(name,fn):
        if trace:stages[name]=capture(fn)
        else:fn()
    run('forward_and_normalized_loss',lambda:box.update(objective(m,data,SEED,selected,True)))
    run('backward',lambda:box['objective'].backward())
    run('gradient_clipping',lambda:torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True))
    run('optimizer',opt.step);return box,stages
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    assert Path(a.tag).name==a.tag and not out.exists();torch.set_num_threads(1);begin=time.perf_counter();caller=torch.get_rng_state().clone();bindings=sources()
    checks=[];cases=[];original=K.LaneRace;replays=BL.REPLAYS[0]
    with torch.random.fork_rng():
        for lengths,selected in [([1],[0]),([2],[7]),([2,1],[0,3])]:
            m=model();ps=list(m.parameters());weights=copy.deepcopy(m.state_dict());data=rows(lengths)
            with clock_reference() as history:d=objective(m,data,SEED,selected,True)
            g=gradients(d['objective'],ps);gb=gradients(d['branch_objective'],ps);gc=gradients(d['choice_objective'],ps)
            close(g,[x+y for x,y in zip(gb,gc)]);ge,ze,qe=explicit(m,data,selected);close(g,ge)
            torch.testing.assert_close(d['factual_logits'],ze,rtol=3e-7,atol=3e-9);torch.testing.assert_close(d['branch_losses'],qe,rtol=3e-7,atol=3e-9)
            with clock_reference(history):ref=objective(m,data,SEED,selected,True)
            close(g,gradients(ref['objective'],ps));torch.testing.assert_close(d['factual_logits'],ref['factual_logits'],rtol=3e-7,atol=3e-9)
            for j,r in enumerate(selected):
                torch.testing.assert_close(history[r]['first'][j].expand(2),history[max(lengths)*4+r]['first'][j*2:j*2+2],rtol=3e-7,atol=3e-9)
                winner=int(history[r]['winner'][j]);z=K.batched_logits(m,[data[j]],SEED,[(r,winner)]);fact=K.batched_logits(m,[data[j]],SEED)
                torch.testing.assert_close(z,fact,rtol=3e-7,atol=3e-9);target=torch.tensor([data[j]['target']])
                close(gradients(F.cross_entropy(z,target),ps),gradients(F.cross_entropy(fact,target),ps))
            matched=objective(m,data,SEED,selected,False);scores=[];factual=K.batched_logits(m,data,SEED,record=scores)
            class Fixed:
                def __init__(self):self.r=iter(selected)
                def choice(self,R,size,replace):assert size==1 and replace is False;return np.array([next(self.r)])
            route,_=BL.route_term(m,data,SEED,scores,1,Fixed());old=F.cross_entropy(factual,torch.tensor([r['target'] for r in data]))+route/len(data)
            close(gradients(matched['objective'],ps),gradients(old,ps))
            changed=objective(m,[dict(r,target=(r['target']+3)%11) for r in data],SEED,selected,True)
            assert torch.equal(d['factual_logits'],changed['factual_logits']) and all(torch.equal(weights[n],v) for n,v in m.state_dict().items())
            cases.append(dict(lengths=lengths,selected_races=selected,shadow_lanes=d['shadow_lanes'],shadow_events=d['shadow_events'],every_parameter_contracts=True))
            checks.append(f'Lengths{lengths}: explicit/decomposition/clock reference/forced winner/BLk1/causal logits/weights')
        m=model();ps=list(m.parameters())
        with clock_reference() as history:d=objective(m,rows([1]),SEED,[0],True)
        pi=d['probabilities'][0].detach();gj=gradients(d['objective'],ps);gb=gradients(d['branch_objective'],ps)
        branches=[gradients(loss,ps) for loss in d['branch_losses'][0]];close(gb,[sum(pi[i]*branches[i][j] for i in range(2)) for j in range(len(ps))])
        variance=sum(float(pi[i])*sum(float((branches[i][j]-gb[j]).square().sum()) for j in range(len(ps))) for i in range(2));assert variance>0
        gf=gradients(d['factual_losses'].mean(),ps);winner=int(history[0]['winner'][0]);loser=1-winner;unit=m.units[0][0][0][loser]
        index=next(i for i,p in enumerate(ps) if p is unit.output.weight)
        assert float(gf[index].norm())==0. and float(gj[index].norm())>0.;close([gj[index]],[pi[loser]*branches[loser][index]])
        exposure=dict(winner=winner,loser=loser,pi=pi.tolist(),factual_loser_output_gradient_norm=0.,joint_loser_output_gradient_norm=float(gj[index].norm()),conditional_current_winner_parameter_variance=variance,scope='One episode/site, not total shared-noise batch variance')
        checks.append('Conditional branch mean and pi-weighted direct losing output-map credit')
        m=model();opt=torch.optim.Adam(m.parameters(),lr=.003);data=rows([2,1]);weights=copy.deepcopy(m.state_dict());_,stages=update(m,opt,data,[0,3],True)
        assert any(not torch.equal(weights[n],v) for n,v in m.state_dict().items());buffer=io.BytesIO();torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),buffer);buffer.seek(0);saved=torch.load(buffer,weights_only=False)
        restored=model();restored.load_state_dict(saved['model']);other=torch.optim.Adam(restored.parameters(),lr=.003);other.load_state_dict(saved['optimizer'])
        update(m,opt,data,[7,0]);update(restored,other,data,[7,0])
        for x,y in zip(m.parameters(),restored.parameters()):assert torch.equal(x,y)
        for key,state in opt.state_dict()['state'].items():
            for name,value in state.items():assert torch.equal(value,other.state_dict()['state'][key][name])
        checks.append('Actual normalized clip1 Adam movement and serialized next-update recovery')
        inference=capture(lambda:K.batched_logits(restored,data,SEED));total=merge(list(stages.values()));fit_ops=total['arithmetic_flops']+total['special_function_evaluations'];infer_ops=inference['arithmetic_flops']+inference['special_function_evaluations']
        checks.append('Complete live-shadow/backward/normalization/clip/Adam and native inference coverage')
        for selected in ([True],[4],[-1],[]):
            try:objective(model(),rows([1]),SEED,selected,True)
            except ValueError:pass
            else:raise AssertionError('Invalid race admitted')
        checks.append('Invalid race rejection')
    BL.REPLAYS[0]=replays;assert K.LaneRace is original and torch.equal(caller,torch.get_rng_state());assert bindings==sources();checks.append('Caller RNG/kernel/source bytes and replay counter preserved')
    result=dict(status='completed',args=vars(a),contracts_passed=len(checks),contracts=checks,cases=cases,exposure=exposure,source_sha256=bindings,
        work=dict(targets=2,whole_fit_unit_special_flops=fit_ops,fit_unit_special_flops_per_target=fit_ops/2,inference_unit_special_flops_per_target=infer_ops/2,stages=stages,inference=inference,scope='One tiny traced joint update; total contract campaign work unknown, not zero'),
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Conditional native estimator contracts and tiny learning/resource admission; no quality or batch variance claim')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='completed',contracts=len(checks),wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'])))
if __name__=='__main__':main()
