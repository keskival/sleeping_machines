"""Source-bound trained native route-site/race covariance separation."""
import argparse,copy,itertools,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import trained_content_batch_variance as W
import depth_route_sampling_variance_probe as V
import projected_batch_credit_variance as J
import dvs_batched_le_benchmark as BL
import conditional_branch_content_contracts as Q
from sleeping_machines.batched_episodes import batched_logits
PARENT='experiments/results/diagnostics/local_trained_content_batch_variance_20261003T052200Z.json'
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args();out=ROOT/'experiments/results/diagnostics'/(args.tag+'.json');assert Path(args.tag).name==args.tag and not out.exists();torch.set_num_threads(1);begin=time.perf_counter();caller=torch.get_rng_state().clone();parent=json.loads((ROOT/PARENT).read_text());assert parent['status']=='completed' and parent['contracts_passed']==5
    for name,digest in parent['source_sha256'].items():assert W.N.sha(ROOT/name)==digest,name
    assert W.N.sha(ROOT/parent['snapshot'])==parent['snapshot_sha256'];saved=torch.load(ROOT/parent['snapshot'],weights_only=False,map_location='cpu')
    sources={**parent['source_sha256'],**{n:W.N.sha(ROOT/n) for n in ('experiments/trained_route_site_noise.py','experiments/theory/134_trained_route_site_noise.md','experiments/projected_batch_credit_variance.py','experiments/depth_route_sampling_variance_probe.py')}};cases=[];arrays={};checks=[];oldseed=V.SEED;oldreplays=BL.REPLAYS[0]
    try:
        with torch.random.fork_rng():
            rows,data=W.S.inputs();assert data==parent['data'];rows=rows[:4];m,_=W.restored(saved['model'],saved['optimizer']);m.double();weights=copy.deepcopy(m.state_dict());tiny=[dict(r,events=r['events'][:T]) for r,T in zip(rows,[2,1])];V.SEED=parent['schedule'][0]['noise_seed'];d=V.prepare(m,tiny);parameters=sum(p.numel() for p in m.parameters());bank=np.zeros((2,len(d['scores']),parameters))
            for j,R in enumerate(d['lengths']):
                for race in range(R):bank[j,race]=V.grad(d['terms'][j,race],d['parameters']).numpy()
            np.testing.assert_allclose(bank.sum((0,1)),d['full'].numpy(),rtol=3e-7,atol=3e-9);apply=J.projector(m,d['scores'],d['q'],2);rng=np.random.default_rng(13417007);errors=[]
            for _ in range(3):
                sign=(2*rng.integers(0,2,parameters)-1).astype(np.float64);actual=apply(sign);expected=bank@sign;np.testing.assert_allclose(actual,expected,rtol=3e-7,atol=3e-9);errors.append(float(np.max(np.abs(actual-expected))))
            old,_=BL.route_term(m,tiny,V.SEED,d['scores']);np.testing.assert_allclose(V.grad(old/2,d['parameters']).numpy(),d['full'].numpy(),rtol=3e-7,atol=3e-9)
            enumerated=0.
            for j,R in enumerate(d['lengths']):
                samples=np.stack([(R/2)*bank[j,list(s)].sum(0) for s in itertools.combinations(range(R),2)]);full=bank[j,:R].sum(0);np.testing.assert_allclose(samples.mean(0),full,rtol=3e-7,atol=3e-9);exact=float(np.square(samples-full).sum()/len(samples));analytic=V.covariance(bank[j,:R],2);assert abs(exact-analytic)<1e-10*max(1.,exact);enumerated+=exact
            contract=dict(races=d['lengths'],all_route_projection_max_errors=errors,exact_enumerated_k2_trace=enumerated,full_return_shadow_lanes=2*d['lanes'],full_return_shadow_events=2*d['events'],individual_route_VJPs=sum(d['lengths']),projected_pullbacks=3,scope='Both bank prepare AND original BL equivalence execute full returns; all charged')
            checks.append('Trained EVERY-route projections, full original teacher and explicit finite-population mean/covariance agree');del d,bank,apply,old
            for number,s in enumerate(parent['schedule'][:4]):
                V.SEED=s['noise_seed'];float_model,_=W.restored(saved['model'],saved['optimizer'])
                with torch.no_grad():
                    with Q.clock_reference() as fh:zf=batched_logits(float_model,rows,V.SEED)
                    with Q.clock_reference() as dh:zd=batched_logits(m,rows,V.SEED)
                actual_ce=float(F.cross_entropy(zf,torch.tensor([r['target'] for r in rows]),reduction='none').mean());assert actual_ce==parent['draws'][number]['actual_factual_ce'];agree=all(torch.equal(x['winner'],y['winner']) for x,y in zip(fh,dh));assert len(fh)==len(dh)==168 and agree
                precision=dict(factual_winner_entries=672,all_factual_winners_agree=agree,maximum_factual_logit_abs_error=float((zf.double()-zd).abs().max()),float32_factual_ce=actual_ce,scope='Factual histories only; no forced-branch cross-precision equivalence claim');del float_model,fh,dh,zf
                d=V.prepare(m,rows);assert torch.equal(zd,d['z'].detach());apply=J.projector(m,d['scores'],d['q'],4);probes={k:[] for k in (1,8,32,168)};projected=[];signs=[]
                for index in range(32):
                    sign=(2*rng.integers(0,2,parameters)-1).astype(np.float64);projection=apply(sign);np.testing.assert_allclose(projection.sum(),d['full'].numpy()@sign,rtol=3e-7,atol=3e-9);projected.append(projection);signs.append(sign.astype(np.int8))
                    for k in probes:probes[k].append(sum(V.covariance(projection[j,:R,None],k) for j,R in enumerate(d['lengths'])))
                combined=d['factual']+d['full'];denominator=float(combined.square().sum());assert denominator>0
                cases.append(dict(noise_seed=V.SEED,parameters=parameters,batch=4,races=d['lengths'],precision=precision,factual_ce=float(d['loss'].detach()),factual_gradient_norm=float(d['factual'].norm()),full_route_gradient_norm=float(d['full'].norm()),combined_gradient_norm=float(combined.norm()),full_return_shadow_lanes=d['lanes'],full_return_shadow_events=d['events'],estimates=[dict(k=k,**J.estimate(v,denominator)) for k,v in probes.items()]))
                arrays[f'noise{number}_factual']=d['factual'].numpy();arrays[f'noise{number}_full_route']=d['full'].numpy();arrays[f'noise{number}_combined']=combined.numpy();arrays[f'noise{number}_returns']=d['q'].numpy();arrays[f'noise{number}_projections']=np.asarray(projected);arrays[f'noise{number}_signs']=np.asarray(signs)
                W.equal(m.state_dict(),weights);print(json.dumps(dict(noise_history=number,status='completed',k1_trace_estimate=cases[-1]['estimates'][0]['parameter_trace_covariance_estimate'])),flush=True);del d,apply,zd
            checks.append('All128 main projected sums match actual full parameter gradients; factual float32 histories reproduce parent and agree with double')
            means=np.stack([arrays[f'noise{i}_combined'] for i in range(4)]);between=W.ensemble(means)['sample_covariance_trace'];summaries=[]
            for index,k in enumerate((1,8,32,168)):
                conditional=[c['estimates'][index]['parameter_trace_covariance_estimate'] for c in cases];within=float(np.mean(conditional));summaries.append(dict(k=k,mean_site_trace_estimate=within,between_history_full_mean_sample_trace=between,total_trace_estimate=within+between,site_share_descriptive_ratio=within/max(within+between,1e-300),histories=4,scope='Law of total covariance;4 noise histories and randomized conditional traces, ratios descriptive, no guaranteed uncertainty or Adam variance'))
            assert all(c['estimates'][-1]['parameter_trace_covariance_estimate']==0 for c in cases)
            checks.append('Full-site sampling variance zero; exact finite-population k scaling and full-gradient history decomposition retained')
            artifact=out.with_suffix('.projections.npz');assert not artifact.exists();np.savez_compressed(artifact,**arrays);checks.append('All detached returns/full parameter vectors/signs/projections saved; weights and sources preserved')
    finally:V.SEED=oldseed;BL.REPLAYS[0]=oldreplays
    assert torch.equal(caller,torch.get_rng_state()) and V.SEED==oldseed and BL.REPLAYS[0]==oldreplays
    for n,h in sources.items():assert W.N.sha(ROOT/n)==h,n
    assert W.N.sha(ROOT/parent['snapshot'])==parent['snapshot_sha256'];checks.append('Caller RNG, source bytes, parent snapshot and original helper seed/replay counter retained')
    result=dict(status='completed',args=vars(args),contracts_passed=len(checks),contracts=checks,parent=PARENT,parent_sha256=W.N.sha(ROOT/PARENT),snapshot=parent['snapshot'],snapshot_sha256=parent['snapshot_sha256'],source_sha256=sources,data=data,contract=contract,cases=cases,summaries=summaries,projections=str(artifact.relative_to(ROOT)),projections_sha256=W.N.sha(artifact),work=dict(main_full_return_shadow_lanes=sum(c['full_return_shadow_lanes'] for c in cases),main_full_return_shadow_events=sum(c['full_return_shadow_events'] for c in cases),main_projected_pullbacks=128,main_factual_targets=16,additional_main_precision_factual_targets=32,tiny_full_return_shadow_lanes=contract['full_return_shadow_lanes'],tiny_full_return_shadow_events=contract['full_return_shadow_events'],executed_optimizer_steps=0,total_FLOPs=None,traffic=None,energy=None,scope='Plus tiny explicit VJPs/full backwards/precision checks; total campaign work unknown, not zero'),wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Source-bound p4D4 trained-state controlled DOUBLE precision,4 race histories and32 parameter signs each; conditional site covariance estimate, not every trained failure, Adam variance or quality/policy promotion')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='completed',contracts=len(checks),wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'],summaries=summaries)))
if __name__=='__main__':main()
