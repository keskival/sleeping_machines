"""Fully counted real-FIT nativeD4 learning admission of live branch credit."""
import argparse,copy,json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import dvs_native_benchmark as N
import dvs_clock_calibrated_benchmark as C
import dvs_batched_le_benchmark as BL
import depth_growth_plasticity_probe as P
import conditional_branch_content_contracts as Q
from sleeping_machines.conditional_branch_content_credit import conditional_branch_objective as objective
from sleeping_machines.batched_episodes import batched_logits
from race_language_screen import capture
from parallel_head_accumulated_language import merge
CONTRACT='experiments/results/diagnostics/local_conditional_branch_content_contracts_20261003T041000Z.json'

def make():return C.make_model(SimpleNamespace(seed=7,payload=4,depth=4,heads=2,pool=2,clock_step=.05))
def inputs():
    parent=json.loads((ROOT/P.PARENT).read_text());_,data=P.fit_inputs(parent)
    with np.load(ROOT/data['count_artifact'],allow_pickle=False) as z:
        counts,labels,ids=z['fit_counts'],z['fit_labels'],z['fit_ids'];raw=np.log1p(counts.reshape(len(labels),-1));center=raw.mean(0);scale=np.maximum(raw.std(0),.5)
        rows=[dict(index=i,target=int(labels[i]),identity=str(ids[i]),events=N.encode(counts[i],center,scale)) for i in range(32)]
    return rows,data
@torch.no_grad()
def evaluate(m,rows):
    z=batched_logits(m,rows,314159);loss=F.cross_entropy(z,torch.tensor([r['target'] for r in rows]),reduction='none')
    return dict(targets=len(rows),nll=float(loss.mean()),accuracy=float((z.argmax(-1)==torch.tensor([r['target'] for r in rows])).float().mean()),per_target_nll=loss.tolist(),logits=z.tolist())
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');progress=out.with_suffix('.running.json');assert Path(a.tag).name==a.tag and not out.exists() and not progress.exists()
    torch.set_num_threads(1);begin=time.perf_counter();caller=torch.get_rng_state().clone();admission=json.loads((ROOT/CONTRACT).read_text())
    assert admission['status']=='completed' and admission['contracts_passed']==8
    for name,digest in admission['source_sha256'].items():assert N.sha(ROOT/name)==digest,name
    sources={**admission['source_sha256'],**{n:N.sha(ROOT/n) for n in ('experiments/conditional_content_integrated_smoke.py','experiments/theory/130_conditional_content_integrated_smoke.md','experiments/depth_growth_plasticity_probe.py')}}
    rows,data=inputs();fit,anchor=rows[:24],rows[24:];rng=np.random.default_rng(17007);schedule=[]
    for epoch in range(2):
        order=rng.permutation(24)
        for start in range(0,24,4):
            batch=[fit[int(i)] for i in order[start:start+4]];selected=[int(rng.integers(len(r['events'])*8)) for r in batch]
            schedule.append(dict(epoch=epoch,indices=[r['index'] for r in batch],selected=selected,seed=171323+epoch*1000+start))
    checks=[];arms=[];old_replays=BL.REPLAYS[0]
    with torch.random.fork_rng():
        m=make().double();batch=[fit[i] for i in schedule[0]['indices']];selected=schedule[0]['selected'];seed=schedule[0]['seed'];ps=list(m.parameters())
        joint=objective(m,batch,seed,selected,True);Q.close(Q.gradients(joint['objective'],ps),[x+y for x,y in zip(Q.gradients(joint['branch_objective'],ps),Q.gradients(joint['choice_objective'],ps))])
        choice=objective(m,batch,seed,selected,False);scores=[];z=batched_logits(m,batch,seed,record=scores)
        class Fixed:
            def __init__(self):self.r=iter(selected)
            def choice(self,R,size,replace):assert size==1 and replace is False;return np.array([next(self.r)])
        route,_=BL.route_term(m,batch,seed,scores,1,Fixed());old=F.cross_entropy(z,torch.tensor([r['target'] for r in batch]))+route/4
        Q.close(Q.gradients(choice['objective'],ps),Q.gradients(old,ps));checks.append('Real depth4 full-prefix every-gradient joint decomposition and matched existing BLk1')
        initial=None
        for content in (False,True):
            m=make();weights=copy.deepcopy(m.state_dict())
            if initial is None:initial=copy.deepcopy(weights)
            else:assert all(torch.equal(initial[n],v) for n,v in weights.items())
            opt=torch.optim.Adam(m.parameters(),lr=.003);before=dict(fit=evaluate(m,fit),anchor=evaluate(m,anchor));steps=[];ledger=[];arm_begin=time.perf_counter()
            for window,s in enumerate(schedule):
                batch=[fit[i] for i in s['indices']];opt.zero_grad(set_to_none=True);box={};stages={}
                def traced(name,fn):stages[name]=capture(fn)
                traced('forward_and_normalized_loss',lambda:box.update(objective(m,batch,s['seed'],s['selected'],content)))
                assert bool(torch.isfinite(box['objective']))
                factual_ce=float(box['factual_losses'].mean().detach());conditional_ce=float((box['probabilities'].detach()*box['branch_losses'].detach()).sum(-1).mean())
                traced('backward',lambda:box['objective'].backward())
                traced('gradient_clipping',lambda:torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True))
                traced('optimizer',opt.step);ledger.extend(stages.values())
                steps.append(dict(window=window,epoch=s['epoch'],fit_indices=s['indices'],selected_races=s['selected'],noise_seed=s['seed'],factual_pre_update_ce=factual_ce,conditional_site_ce=conditional_ce,stages=stages))
                progress.write_text(json.dumps(dict(status='running',completed_arms=arms,current_arm='joint_content_choice' if content else 'choice_only',completed_updates=len(steps),successful_target_presentations=len(steps)*4,source_sha256=sources,scope='Partial execution accounting only; not paired benchmark or quality evidence'))+'\n')
            after=dict(fit=evaluate(m,fit),anchor=evaluate(m,anchor));summary=merge(ledger);ops=summary['arithmetic_flops']+summary['special_function_evaluations']
            def inference():
                with torch.no_grad():batched_logits(m,rows,314159)
            infer=capture(inference);infer_ops=infer['arithmetic_flops']+infer['special_function_evaluations'];changes={}
            for name,p in m.named_parameters():
                group='.'.join(name.split('.')[:2]) if name.startswith(('units.','queries.','channel_mix.')) else name.split('.')[0]
                changes[group]=changes.get(group,0.)+float((p.detach()-weights[name]).double().square().sum())
            assert all(torch.isfinite(p).all() for p in m.parameters()) and sum(changes.values())>0
            arms.append(dict(arm='joint_content_choice' if content else 'choice_only',parameters=sum(p.numel() for p in m.parameters()),available_receivers=16,
                before=before,after=after,learning_gate=after['fit']['nll']<=before['fit']['nll']-.01,
                parameter_change_L2_by_group={k:v**.5 for k,v in changes.items()},steps=steps,
                work=dict(fitting_targets=48,optimizer_updates=12,whole_fit_unit_special_flops=ops,fit_unit_special_flops_per_target=ops/48,
                    inference_unit_special_flops_per_target=infer_ops/32,inference_targets=32,summary=summary,inference=infer,
                    factual_events=sum(len(fit[i]['events']) for s in schedule for i in s['indices']),
                    full_shadow_lanes=96,full_shadow_events=2*sum(len(fit[i]['events']) for s in schedule for i in s['indices']),
                    normalization='Inside objective mean, fully charged in forward; no second normalization',scope='EVERY fitting update traced; evaluation and admission checks separate, traffic/energy unknown'),
                fitting_and_final_eval_wall_s=time.perf_counter()-arm_begin))
            print(json.dumps(dict(arm=arms[-1]['arm'],fit_nll_before=before['fit']['nll'],fit_nll_after=after['fit']['nll'],learning_gate=arms[-1]['learning_gate'])),flush=True)
        checks.append('Identical represented initialization/data/order/sites/noise and all normalized actual fitting updates traced')
        checks.append('Finite actual Adam parameter movement and common-boundary native inference for both arms')
    BL.REPLAYS[0]=old_replays;assert torch.equal(caller,torch.get_rng_state())
    for name,digest in sources.items():assert N.sha(ROOT/name)==digest,name
    checks.append('Frozen contracts/sources/caller RNG and existing driver replay counter preserved')
    result=dict(status='completed',args=vars(a),contracts_passed=len(checks),contracts=checks,source_sha256=sources,
        contract=CONTRACT,contract_sha256=N.sha(ROOT/CONTRACT),data=data,settings=dict(seed=7,payload=4,depth=4,heads=2,pool=2,clock_step=.05,batch=4,passes=2,lr=.003,clip=1),
        fit_indices=list(range(24)),anchor_FIT_indices=list(range(24,32)),schedule=schedule,arms=arms,
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Real native depth4 FIT-only learning/resource admission,48 presentations each; no DEV/test, model selection, batch variance, useful feature depth or superiority claim')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='completed',wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'])))
    progress.unlink()
if __name__=='__main__':main()
