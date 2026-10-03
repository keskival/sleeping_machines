"""Actual historical-Adam coverage ensemble using exact archived driver bytes."""
import argparse,copy,json,resource,sys,time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import trained_content_batch_variance as W
import legacy_batched_driver_binding as B
PARENT='experiments/results/diagnostics/local_trained_route_coverage_work_20261003T054700Z.json'
SEEDS='experiments/results/diagnostics/local_trained_content_batch_variance_20261003T052200Z.json'
@contextmanager
def observe_clip(box):
    original=torch.nn.utils.clip_grad_norm_
    def intercept(parameters,max_norm,**kw):
        ps=list(parameters);assert not box
        box['raw']=[None if p.grad is None else p.grad.detach().clone() for p in ps]
        norm=original(ps,max_norm,**kw);box['raw_norm']=float(norm);box['clipped']=[None if p.grad is None else p.grad.detach().clone() for p in ps];return norm
    torch.nn.utils.clip_grad_norm_=intercept
    try:yield
    finally:torch.nn.utils.clip_grad_norm_=original

def pair(x,y):
    d=W.paired(x,y)
    return dict(k1=d['factual'],k8=d['joint'],k8_to_k1_variance_ratio=d['joint_to_factual_variance_ratio'],leave_one_pair_out_ratio_range=d['leave_one_pair_out_ratio_range'],empirical_mean_difference_L2=d['empirical_mean_difference_L2'],scope=d['scope'])
def error(x,reference):
    e=x-reference;squared=np.square(e).sum(1);mean=e.mean(0);centered=W.ensemble(e)['sample_covariance_trace'];mse=float(squared.mean());bias=float(mean@mean);assert abs(mse-bias-(len(e)-1)/len(e)*centered)<1e-10*max(1.,mse)
    return dict(mean_squared_difference=mse,empirical_SE=float(squared.std(ddof=1)/np.sqrt(len(e))),empirical_mean_error_squared_norm=bias,centered_error_sample_trace=centered,relative_MSE_to_mean_reference_norm_squared=mse/max(float(np.square(reference).sum(1).mean()),1e-300),all_squared_errors=squared.tolist(),scope='Same-noise full-site actual teacher; nonlinear clipped/update differences include mean shifts, no unbiased-update claim')
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args();out=ROOT/'experiments/results/diagnostics'/(args.tag+'.json');assert Path(args.tag).name==args.tag and not out.exists();torch.set_num_threads(1);begin=time.perf_counter();caller=torch.get_rng_state().clone();parent_digest=B.sha(ROOT/PARENT);seed_digest=B.sha(ROOT/SEEDS);parent=json.loads((ROOT/PARENT).read_text());seed_record=json.loads((ROOT/SEEDS).read_text());assert parent['status']=='completed' and parent['contracts_passed']==6 and seed_record['status']=='completed'
    assert parent['snapshot']==seed_record['snapshot'] and parent['snapshot_sha256']==seed_record['snapshot_sha256']
    sources=B.resolved_sources(parent['source_sha256']);sources.update({n:B.sha(ROOT/n) for n in ('experiments/warm_adam_route_coverage.py','experiments/legacy_batched_driver_binding.py','experiments/theory/138_warm_adam_route_coverage.md')})
    for n,h in sources.items():assert B.sha(ROOT/n)==h,n
    assert B.sha(ROOT/parent['snapshot'])==parent['snapshot_sha256'];saved=torch.load(ROOT/parent['snapshot'],weights_only=False,map_location='cpu');driver=B.load_driver();replays=driver.REPLAYS[0];callback=torch.nn.utils.clip_grad_norm_;batch_callback=W.batched_logits;arrays={};draws=[];checks=[]
    try:
        W.batched_logits=driver.batched_logits
        with torch.random.fork_rng():
            rows,data=W.S.inputs();assert data==parent['data']==seed_record['data'];batch=rows[:4];evaluation=batch+rows[24:];initial,_=W.restored(saved['model'],saved['optimizer']);W.equal(initial.state_dict(),saved['model']);base=W.flat(list(initial.parameters()),list(initial.parameters()));baseline=W.predict(initial,evaluation);schedule=seed_record['schedule'];assert len(schedule)==16
            for index,s in enumerate(schedule):
                entry=dict(draw=index,noise_seed=s['noise_seed'],arms={})
                for k in (1,8,168):
                    a=SimpleNamespace(seed=s['noise_seed']-100000,route_credit=True,route_races=0 if k==168 else k);m,o=W.restored(saved['model'],saved['optimizer']);box={}
                    with observe_clip(box):stats=driver.train_window(m,o,batch,a,0,False)
                    assert stats['targets']==4 and stats['route_replays']==8*k and stats['loss_sum']/4==seed_record['draws'][index]['actual_factual_ce']
                    vectors=dict(raw=W.flat(box['raw'],list(m.parameters())),clipped=W.flat(box['clipped'],list(m.parameters())),update=W.flat(list(m.parameters()),list(m.parameters()))-base)
                    for kind,v in vectors.items():assert bool(torch.isfinite(v).all());arrays.setdefault(f'{kind}_k{k}',[]).append(v.numpy())
                    masks=[g is not None for g in box['raw']];entry['arms'][str(k)]=dict(raw_norm=box['raw_norm'],clip_factor=min(1.,1./(box['raw_norm']+1e-6)),actual_update_L2=float(vectors['update'].norm()),factual_ce=stats['loss_sum']/4,gradient_masks=masks,shadow_lanes=stats['route_replays'],predictions=W.predict(m,evaluation))
                    if index==0:
                        other,oo=W.restored(saved['model'],saved['optimizer']);plain=driver.train_window(other,oo,batch,a,0,False);W.equal(m.state_dict(),other.state_dict());W.equal(o.state_dict(),oo.state_dict());W.equal(stats,plain);checks.append(f'k{k}: EVERY original-driver versus intercepted-gradient next weight/moment and statistics agree')
                    del m,o,box,vectors
                draws.append(entry);W.equal(initial.state_dict(),saved['model']);print(json.dumps(dict(draw=index,status='completed',norms={k:v['raw_norm'] for k,v in entry['arms'].items()})),flush=True)
            arrays={k:np.asarray(v) for k,v in arrays.items()};paired={kind:pair(arrays[f'{kind}_k1'],arrays[f'{kind}_k8']) for kind in ('raw','clipped','update')};by_k={str(k):{kind:W.ensemble(arrays[f'{kind}_k{k}']) for kind in ('raw','clipped','update')} for k in (1,8,168)};errors={str(k):{kind:error(arrays[f'{kind}_k{k}'],arrays[f'{kind}_k168']) for kind in ('raw','clipped','update')} for k in (1,8)}
            costs={x['k']:x for x in parent['arms'] if x['k'] in (1,8,168)};ratio=costs[8]['whole_step_unit_special_flops']/costs[1]['whole_step_unit_special_flops'];heuristics={kind:dict(raw_or_update_variance_ratio=paired[kind]['k8_to_k1_variance_ratio'],saved_actual_step_work_ratio=ratio,variance_times_work_ratio=paired[kind]['k8_to_k1_variance_ratio']*ratio,full_reference_MSE_ratio=errors['8'][kind]['mean_squared_difference']/max(errors['1'][kind]['mean_squared_difference'],1e-300),full_reference_MSE_times_work_ratio=ratio*errors['8'][kind]['mean_squared_difference']/max(errors['1'][kind]['mean_squared_difference'],1e-300),scope='Actual16-draw float32 quantities times verified saved one-window cost; heuristic, not sustained learning/convergence/quality') for kind in paired}
            blocks={};start=0
            for n,param in initial.named_parameters():
                g='.'.join(n.split('.')[:2]) if n.startswith(('units.','queries.','channel_mix.')) else n.split('.')[0];blocks.setdefault(g,[]).extend(range(start,start+param.numel()));start+=param.numel()
            block_stats={g:{kind:pair(arrays[f'{kind}_k1'][:,indices],arrays[f'{kind}_k8'][:,indices]) for kind in ('raw','update')} for g,indices in blocks.items()};artifact=out.with_suffix('.vectors.npz');assert not artifact.exists();np.savez_compressed(artifact,**arrays);checks.append('All48 finite actual normalized gradients/clipped gradients/historical updates and causal fixed predictions retained; exact per-seed factual CE')
    finally:driver.REPLAYS[0]=replays;W.batched_logits=batch_callback
    assert torch.equal(caller,torch.get_rng_state()) and torch.nn.utils.clip_grad_norm_ is callback and driver.REPLAYS[0]==replays and W.batched_logits is batch_callback
    for n,h in sources.items():assert B.sha(ROOT/n)==h,n
    assert B.sha(ROOT/parent['snapshot'])==parent['snapshot_sha256'] and B.sha(ROOT/PARENT)==parent_digest and B.sha(ROOT/SEEDS)==seed_digest;checks.append('Original source archive/snapshot/parent/seed record/caller RNG/clip callback/replay counter retained; saved actual cost boundary verified')
    result=dict(status='completed',args=vars(args),contracts_passed=len(checks),contracts=checks,parent=PARENT,parent_sha256=B.sha(ROOT/PARENT),seeds=SEEDS,seeds_sha256=B.sha(ROOT/SEEDS),snapshot=parent['snapshot'],snapshot_sha256=parent['snapshot_sha256'],source_sha256=sources,legacy_binding=dict(original_name=B.NAME,original_sha256=B.DIGEST,archive=B.ARCHIVE,archive_sha256=B.sha(ROOT/B.ARCHIVE),scope='Only verified provenance paths changed in new caller; saved evidence and active compiled driver unchanged'),data=data,settings=dict(batch=4,payload=4,depth=4,heads=2,pool=2,draws=16,clip=1,optimizer='Recovered12-step historical Adam.003',k=[1,8,168]),baseline=baseline,schedule=schedule,draws=draws,paired=paired,by_k=by_k,full_reference_errors=errors,heuristics=heuristics,block_stats=block_stats,saved_work=list(costs.values()),vectors=str(artifact.relative_to(ROOT)),vectors_sha256=B.sha(artifact),execution=dict(main_actual_updates=48,exact_recovery_updates=3,main_presentation_targets=192,main_full_shadow_lanes=22656,main_full_shadow_events=475776,recovery_full_shadow_lanes=1416,recovery_full_shadow_events=29736,fork_prediction_targets=576,baseline_prediction_targets=12,scope='Saved-cost tracing is prior campaign; total current FLOPs/traffic/energy unknown, not zero'),wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One tiny trained nativeD4 state, original driver16 paired noise draws and48 discarded actual warm-Adam updates; no IID/DEV/test selection, exact risk gradient or sustained benchmark advantage')
    result['legacy_batch_binding']=dict(original_name=B.BATCH_NAME,original_sha256=B.BATCH_DIGEST,archive=B.BATCH_ARCHIVE,archive_sha256=B.sha(ROOT/B.BATCH_ARCHIVE),scope='Legacy fitting and fixed inference callbacks both use archived computation; current optional-credit core remains active')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='completed',contracts=len(checks),wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'],heuristics=heuristics)))
if __name__=='__main__':main()
