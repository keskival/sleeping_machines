"""Actual integrated warm-Adam work of existing conditional coverage driver."""
import argparse,copy,json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import trained_content_batch_variance as W
import dvs_batched_le_benchmark as BL
from race_language_screen import capture
from sleeping_machines.batched_episodes import batched_logits
PARENT='experiments/results/diagnostics/local_trained_route_site_noise_20261003T054100Z.json'
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args();out=ROOT/'experiments/results/diagnostics'/(args.tag+'.json');assert Path(args.tag).name==args.tag and not out.exists();torch.set_num_threads(1);begin=time.perf_counter();caller=torch.get_rng_state().clone();parent=json.loads((ROOT/PARENT).read_text());assert parent['status']=='completed' and parent['contracts_passed']==5
    for n,h in parent['source_sha256'].items():assert W.N.sha(ROOT/n)==h,n
    assert W.N.sha(ROOT/parent['snapshot'])==parent['snapshot_sha256'];saved=torch.load(ROOT/parent['snapshot'],weights_only=False,map_location='cpu');sources={**parent['source_sha256'],**{n:W.N.sha(ROOT/n) for n in ('experiments/trained_route_coverage_work.py','experiments/theory/136_trained_route_coverage_work.md')}};arms=[];checks=[];replays=BL.REPLAYS[0]
    try:
        with torch.random.fork_rng():
            rows,data=W.S.inputs();assert data==parent['data'];batch=rows[:4];evaluation=batch+rows[24:];seed=parent['cases'][0]['noise_seed'];initial,_=W.restored(saved['model'],saved['optimizer']);baseline=W.predict(initial,evaluation)
            for k in (0,1,8,32,168):
                a=SimpleNamespace(seed=seed-100000,route_credit=k>0,route_races=0 if k==168 else k);m,o=W.restored(saved['model'],saved['optimizer']);stats=BL.train_window(m,o,batch,a,0,True)
                assert stats['targets']==4 and stats['route_replays']==8*k
                other,oo=W.restored(saved['model'],saved['optimizer']);plain=BL.train_window(other,oo,batch,a,0,False);W.equal(m.state_dict(),other.state_dict());W.equal(o.state_dict(),oo.state_dict());W.equal({n:v for n,v in stats.items() if n!='stages'},{n:v for n,v in plain.items() if n!='stages'})
                delta=W.flat(list(m.parameters()),list(m.parameters()))-W.flat(list(initial.parameters()),list(initial.parameters()));assert bool(torch.isfinite(delta).all()) and float(delta.norm())>0
                def infer():
                    with torch.no_grad():batched_logits(m,evaluation,314159)
                inference=capture(infer);summary=W.S.merge(list(stats['stages'].values()));ops=summary['arithmetic_flops']+summary['special_function_evaluations'];iops=inference['arithmetic_flops']+inference['special_function_evaluations'];pred=W.predict(m,evaluation)
                arms.append(dict(k=k,credit='factorized no-choice' if k==0 else 'existing sampled choice' if k<168 else 'existing full choice',targets=4,updates=1,parameters=sum(p.numel() for p in m.parameters()),available_receivers=16,noise_seed=seed,factual_pre_update_ce=stats['loss_sum']/4,actual_parameter_change_L2=float(delta.norm()),predictions=pred,whole_step_unit_special_flops=ops,fit_unit_special_flops_per_target=ops/4,inference_unit_special_flops_per_target=iops/12,inference_targets=12,stats=stats,inference=inference,scope='ONE traced actual warm update, plus one exact untraced verification; discarded, no benchmark'))
                if k==168:
                    final,fo=W.restored(saved['model'],saved['optimizer']);BL.train_window(final,fo,batch,a,0,False);W.equal(m.state_dict(),final.state_dict());W.equal(o.state_dict(),fo.state_dict())
                checks.append(f'k{k}: EVERY-weight/moment exact traced/untraced update, driver counts, finite movement and full operator coverage');print(json.dumps(dict(k=k,status='completed',step_flops=ops)),flush=True)
            assert all(x['factual_pre_update_ce']==arms[0]['factual_pre_update_ce'] for x in arms);assert arms[1]['factual_pre_update_ce']==parent['cases'][0]['precision']['float32_factual_ce']
            for name,item in saved['model'].items():assert torch.equal(item,initial.state_dict()[name])
            comparisons=[];reference=arms[1]['whole_step_unit_special_flops']
            for arm,s in zip(arms[1:],parent['summaries']):
                assert arm['k']==s['k'];variance=s['total_trace_estimate']/parent['summaries'][0]['total_trace_estimate'];work=arm['whole_step_unit_special_flops']/reference;comparisons.append(dict(k=arm['k'],raw_total_variance_ratio_estimate=variance,actual_step_work_ratio=work,variance_times_work_ratio=variance*work,scope='Four-history projected double variance times one-window actual float32 work; heuristic, not Adam or quality theorem'))
    finally:BL.REPLAYS[0]=replays
    assert torch.equal(caller,torch.get_rng_state())
    for n,h in sources.items():assert W.N.sha(ROOT/n)==h,n
    assert W.N.sha(ROOT/parent['snapshot'])==parent['snapshot_sha256'];checks.append('Exact factual CE/source-bound warm state, serialized full recovery and caller RNG/replay counter preserved')
    result=dict(status='completed',args=vars(args),contracts_passed=len(checks),contracts=checks,parent=PARENT,parent_sha256=W.N.sha(ROOT/PARENT),snapshot=parent['snapshot'],snapshot_sha256=parent['snapshot_sha256'],source_sha256=sources,data=data,baseline=baseline,arms=arms,comparisons=comparisons,execution=dict(traced_updates=5,untraced_verification_updates=5,serialized_full_verification_updates=1,discarded_updates=11,traced_full_return_shadow_lanes=sum(x['stats']['route_replays'] for x in arms),traced_full_return_shadow_events=sum(x['stats']['route_replays']*21 for x in arms),verification_full_return_shadow_lanes=sum(x['stats']['route_replays'] for x in arms)+1344,prediction_targets=72,scope='Extra inference traces/admission and11discarded actual updates; total campaign FLOPs/traffic/energy unknown, not zero'),wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Exactly counted existing teacher one-window coverage tradeoff at one tiny trained state; zero-choice changes teacher mean, no quality/policy/Adam variance promotion')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='completed',contracts=len(checks),wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'],comparisons=comparisons)))
if __name__=='__main__':main()
