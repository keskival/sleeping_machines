"""Matched trained nativeD4 gradient and actual historical-Adam variance."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import conditional_content_integrated_smoke as S
import conditional_branch_content_contracts as Q
import dvs_native_benchmark as N
from sleeping_machines.conditional_branch_content_credit import conditional_branch_objective as objective
from sleeping_machines.batched_episodes import batched_logits
from race_language_screen import capture
from parallel_head_accumulated_language import merge
PARENT='experiments/results/diagnostics/local_conditional_content_integrated_smoke_retry_20261003T042500Z.json'

def equal(a,b):
    if torch.is_tensor(a):assert torch.is_tensor(b) and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)
def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).detach().double().flatten() for g,p in zip(gs,ps)])
def add(a,b):return [y if x is None else x if y is None else x+y for x,y in zip(a,b)]
def grad(loss,ps):return list(torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True))
def close(a,b,ps,double=False):
    assert [x is None for x in a]==[x is None for x in b]
    x,y=flat(a,ps),flat(b,ps);relative=float((x-y).norm()/x.norm().clamp_min(1e-30))
    assert relative< (2e-12 if double else 4e-6),relative
    if double:Q.close([torch.zeros_like(p) if g is None else g for g,p in zip(a,ps)],[torch.zeros_like(p) if g is None else g for g,p in zip(b,ps)])
    return relative
def restored(weights,moments):
    m=S.make();m.load_state_dict(copy.deepcopy(weights));o=torch.optim.Adam(m.parameters(),lr=.003);o.load_state_dict(copy.deepcopy(moments));return m,o
def update(m,o,gs):
    o.zero_grad(set_to_none=True)
    for p,g in zip(m.parameters(),gs):p.grad=None if g is None else g.detach().clone()
    norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True));clipped=flat([p.grad for p in m.parameters()],list(m.parameters()));o.step()
    return norm,clipped
def predict(m,rows):
    with torch.no_grad():
        z=batched_logits(m,rows,314159);loss=F.cross_entropy(z,torch.tensor([r['target'] for r in rows]),reduction='none')
    return dict(fit_nll=float(loss[:4].mean()),anchor_nll=float(loss[4:].mean()),per_target_nll=loss.tolist(),logits=z.tolist())
def ensemble(x):
    x=np.asarray(x,dtype=np.float64);mean=x.mean(0);trace=float(np.square(x-mean).sum()/(len(x)-1))
    return dict(draws=len(x),sample_covariance_trace=trace,empirical_mean_squared_norm=float(mean@mean),noise_to_mean_squared_ratio=trace/max(float(mean@mean),1e-300))
def paired(x,y):
    a,b=ensemble(x),ensemble(y);ratios=[]
    for k in range(len(x)):
        take=np.arange(len(x))!=k;ratios.append(ensemble(y[take])['sample_covariance_trace']/max(ensemble(x[take])['sample_covariance_trace'],1e-300))
    return dict(factual=a,joint=b,joint_to_factual_variance_ratio=b['sample_covariance_trace']/max(a['sample_covariance_trace'],1e-300),leave_one_pair_out_ratio_range=[min(ratios),max(ratios)],empirical_mean_difference_L2=float(np.linalg.norm(np.mean(y-x,axis=0))),scope='16 paired draws; empirical ratio/range, no guaranteed interval or convergence claim')
def decomposition(c,a,total):
    c=c-c.mean(0);a=a-a.mean(0);cross=float((c*a).sum()/(len(c)-1));expected=ensemble(c)['sample_covariance_trace']+ensemble(a)['sample_covariance_trace']+2*cross;actual=ensemble(total)['sample_covariance_trace']
    err=abs(expected-actual)/max(actual,1e-30);assert err<2e-5,err
    return dict(content_trace=ensemble(c)['sample_covariance_trace'],choice_trace=ensemble(a)['sample_covariance_trace'],twice_content_choice_cross_trace=2*cross,total_trace=actual,composition_rounding_relative_error=err)
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args();out=ROOT/'experiments/results/diagnostics'/(args.tag+'.json');assert Path(args.tag).name==args.tag and not out.exists();torch.set_num_threads(1);begin=time.perf_counter();caller=torch.get_rng_state().clone()
    parent=json.loads((ROOT/PARENT).read_text());assert parent['status']=='completed' and parent['contracts_passed']==4
    for n,h in parent['source_sha256'].items():assert N.sha(ROOT/n)==h,n
    extra=['experiments/trained_content_batch_variance.py','experiments/theory/132_trained_content_batch_variance.md','experiments/race_language_screen.py','experiments/parallel_head_accumulated_language.py']
    sources={**parent['source_sha256'],**{n:N.sha(ROOT/n) for n in extra}};checks=[];draws=[];banks={};maxerr=0.
    with torch.random.fork_rng():
        rows,data=S.inputs();assert data==parent['data'];fit,anchor=rows[:24],rows[24:];m=S.make();initial=copy.deepcopy(m.state_dict());o=torch.optim.Adam(m.parameters(),lr=.003);reconstruction=[]
        for s,old in zip(parent['schedule'],parent['arms'][0]['steps']):
            batch=[fit[i] for i in s['indices']];o.zero_grad(set_to_none=True);d=objective(m,batch,s['seed'],s['selected'],False)
            actual=[float(d['factual_losses'].mean().detach()),float((d['probabilities'].detach()*d['branch_losses'].detach()).sum(-1).mean())];assert actual==[old['factual_pre_update_ce'],old['conditional_site_ce']],actual
            d['objective'].backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();reconstruction.append(actual)
        for label,r in [('fit',fit),('anchor',anchor)]:equal(S.evaluate(m,r),parent['arms'][0]['after'][label])
        movements={}
        for n,p in m.named_parameters():
            g='.'.join(n.split('.')[:2]) if n.startswith(('units.','queries.','channel_mix.')) else n.split('.')[0];movements[g]=movements.get(g,0.)+float((p.detach()-initial[n]).double().square().sum())
        equal({k:v**.5 for k,v in movements.items()},parent['arms'][0]['parameter_change_L2_by_group']);weights=copy.deepcopy(m.state_dict());moments=copy.deepcopy(o.state_dict());assert len(moments['state'])>0
        snapshot=out.with_suffix('.state.pt');assert not snapshot.exists();torch.save(dict(model=weights,optimizer=moments,parent=PARENT,parent_sha256=N.sha(ROOT/PARENT),source_sha256=sources,available_receivers=16,scope='Exactly reconstructed12-step choice-only smoke; no new benchmark'),snapshot);saved=torch.load(snapshot,weights_only=False,map_location='cpu');equal(weights,saved['model']);equal(moments,saved['optimizer']);checks.append('Exact12-step reconstruction: every saved loss/final prediction/parameter movement and serialized real moments')
        batch=rows[:4];evaluation=batch+anchor;rng=np.random.default_rng(13217007);schedule=[dict(noise_seed=1320000+int(rng.integers(100000000)),selected=[int(rng.integers(len(r['events'])*8)) for r in batch]) for _ in range(16)]
        md,_=restored(weights,moments);md.double();ps=list(md.parameters());d=objective(md,batch,schedule[0]['noise_seed'],schedule[0]['selected'],True)
        ca=grad(d['factual_losses'].mean(),ps);cb=grad(d['branch_objective'],ps);a=grad(d['choice_objective'],ps)
        close(add(cb,a),grad(d['objective'],ps),ps,True);control=objective(md,batch,schedule[0]['noise_seed'],schedule[0]['selected'],False);close(add(ca,a),grad(control['objective'],ps),ps,True);checks.append('Trained-state EVERY-parameter joint/control decomposition and unchanged choice gradients in double')
        del md,ps,d,ca,cb,a,control
        names=list(dict(m.named_parameters()));ps=list(m.parameters());baseline=predict(m,evaluation);wflat=flat(list(m.parameters()),ps);work=[]
        for number,s in enumerate(schedule):
            d=objective(m,batch,s['noise_seed'],s['selected'],True);ca=grad(d['factual_losses'].mean(),ps);cb=grad(d['branch_objective'],ps);a=grad(d['choice_objective'],ps);gf,gj=add(ca,a),add(cb,a)
            if number==0:maxerr=close(gj,grad(d['objective'],ps),ps)
            for label,g in [('factual_content',ca),('conditional_content',cb),('choice',a),('raw_factual',gf),('raw_joint',gj)]:banks.setdefault(label,[]).append(flat(g,ps).numpy())
            entry=dict(draw=number,**s,actual_factual_ce=float(d['factual_losses'].mean().detach()),conditional_site_ce=float((d['probabilities'].detach()*d['branch_losses'].detach()).sum(-1).mean()),gradient_masks={})
            for content,label,g in [(False,'factual',gf),(True,'joint',gj)]:
                fork,opt=restored(weights,moments);norm,clipped=update(fork,opt,g);delta=flat(list(fork.parameters()),list(fork.parameters()))-wflat;assert bool(torch.isfinite(delta).all());banks.setdefault('clipped_'+label,[]).append(clipped.numpy());banks.setdefault('update_'+label,[]).append(delta.numpy());entry[label]=dict(raw_norm=norm,clip_factor=min(1.,1./(norm+1e-6)),update_L2=float(delta.norm()),predictions=predict(fork,evaluation));entry['gradient_masks'][label]=[x is not None for x in g]
                if number==0:
                    real,ro=restored(saved['model'],saved['optimizer']);stages={};box={}
                    stages['forward_normalized_loss']=capture(lambda:box.update(objective(real,batch,s['noise_seed'],s['selected'],content)))
                    stages['backward']=capture(lambda:box['objective'].backward());maxerr=max(maxerr,close(g,[p.grad for p in real.parameters()],ps))
                    stages['clip']=capture(lambda:torch.nn.utils.clip_grad_norm_(real.parameters(),1.,error_if_nonfinite=True));stages['optimizer']=capture(ro.step);rd=flat(list(real.parameters()),list(real.parameters()))-wflat
                    relative=float((rd-delta).norm()/delta.norm().clamp_min(1e-30));assert relative<2e-4,relative
                    total=merge(list(stages.values()));ops=total['arithmetic_flops']+total['special_function_evaluations']
                    def infer():
                        with torch.no_grad():batched_logits(real,evaluation,314159)
                    inference=capture(infer);iops=inference['arithmetic_flops']+inference['special_function_evaluations'];work.append(dict(arm=label,targets=4,actual_step_unit_special_flops=ops,fit_unit_special_flops_per_target=ops/4,inference_unit_special_flops_per_target=iops/12,inference_targets=12,stages=stages,inference=inference,actual_update_global_relative_difference=relative,scope='ONE isolated actual fitting window per arm; not total diagnostic campaign work'))
                    resumed,resopt=restored(saved['model'],saved['optimizer']);update(resumed,resopt,g);equal(fork.state_dict(),resumed.state_dict());equal(opt.state_dict(),resopt.state_dict())
                    del real,ro,box,resumed,resopt
                del fork,opt
            draws.append(entry);equal(m.state_dict(),weights);equal(o.state_dict(),moments);del d,ca,cb,a,gf,gj
            print(json.dumps(dict(draw=number,raw_norm_factual=entry['factual']['raw_norm'],raw_norm_joint=entry['joint']['raw_norm'])),flush=True)
        checks.append('Every float32 gradient mask and actual warm clip1 Adam movement retained; isolated full-program gradients/updates and serialized recovery admitted')
        banks={k:np.asarray(v) for k,v in banks.items()};estimates={kind:paired(banks[kind+'_factual'],banks[kind+'_joint']) for kind in ('raw','clipped','update')};dec={label:decomposition(banks[c],banks['choice'],banks['raw_'+label]) for label,c in [('factual','factual_content'),('joint','conditional_content')]}
        blocks={};start=0
        for n,p in m.named_parameters():
            group='.'.join(n.split('.')[:2]) if n.startswith(('units.','queries.','channel_mix.')) else n.split('.')[0];blocks.setdefault(group,[]).extend(range(start,start+p.numel()));start+=p.numel()
        block_stats={k:{kind:paired(banks[kind+'_factual'][:,v],banks[kind+'_joint'][:,v]) for kind in ('raw','update')} for k,v in blocks.items()}
        artifact=out.with_suffix('.vectors.npz');assert not artifact.exists();np.savez_compressed(artifact,**banks)
        checks.append('Complete formula-covered matched isolated fitting/inference work and covariance decomposition; all ensemble vectors retained')
    assert torch.equal(caller,torch.get_rng_state())
    for n,h in sources.items():assert N.sha(ROOT/n)==h,n
    checks.append('Recovered model/moments, original sources and caller RNG unchanged')
    result=dict(status='completed',args=vars(args),contracts_passed=len(checks),contracts=checks,parent=PARENT,parent_sha256=N.sha(ROOT/PARENT),source_sha256=sources,data=data,snapshot=str(snapshot.relative_to(ROOT)),snapshot_sha256=N.sha(snapshot),vectors=str(artifact.relative_to(ROOT)),vectors_sha256=N.sha(artifact),settings=dict(batch=4,payload=4,depth=4,heads=2,pool=2,clock_step=.05,draws=16,clip=1,optimizer='Actual12-step historical Adam.003'),fit_indices=list(range(4)),anchor_FIT_indices=list(range(24,32)),baseline=baseline,schedule=schedule,draws=draws,estimates=estimates,decomposition=dec,block_stats=block_stats,maximum_float32_global_gradient_error=maxerr,work=work,reconstruction=dict(target_presentations=48,optimizer_updates=12,per_step_loss_pairs=reconstruction,original_whole_fit_unit_special_flops=parent['arms'][0]['work']['whole_fit_unit_special_flops'],scope='Original source-bound fit reproduced without tracing, identical all recorded outcomes; not new benchmark'),execution=dict(ensemble_factual_targets=64,ensemble_full_shadow_lanes=128,ensemble_full_shadow_events=2688,ensemble_component_reverse_calls=48,discarded_warm_Adam_updates=32,isolated_paid_updates=2,isolated_shadow_lanes=16,isolated_shadow_events=336,fork_prediction_targets=384,baseline_prediction_targets=12,scope='Plus reconstruction and admission: total campaign FLOPs/traffic/energy unknown, not zero'),wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One restricted trained nativeD4 state,16 paired draws and32 discarded real-moment forks; no heldout nomination, exact expected-risk reference, depth-general cause or supremacy')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='completed',contracts=len(checks),wall_s=result['wall_s'],estimates=estimates)))
if __name__=='__main__':main()
