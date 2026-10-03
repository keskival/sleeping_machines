"""Conditioned native full-write utility and completed bounded-score smoke audit."""
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
import dvs_bridge_batched_benchmark as D
import sleeping_machines.bridge_batched_episodes as BB
from dvs_key_score_decomposition import replay
from dvs_native_window_intervention import equal_state
from sleeping_machines.native_bounded_score_diagnostic import NativeBoundedScoreDiagnostic
from sleeping_machines.bounded_score_sensitivity import bounded_score_bridge


class ActualSuffix(NativeBoundedScoreDiagnostic):
    def __init__(self, **kw):
        super().__init__(**kw);self.prefix_trace=None;self.forced_site=None;self.forced_winner=None
    def race(self, scores, values=None):
        idx=len(self.records)
        value,delay,winner=super().race(scores,values)
        if self.prefix_trace is not None and idx<=self.forced_site:
            reference=self.prefix_trace[idx];delay=reference['delay'];winner=reference['winner']
            if idx==self.forced_site:winner=winner.new_tensor(self.forced_winner)
        return value,delay,winner


def exposure(model, rows):
    old=BB.bounded_score_bridge;record=[]
    def tap(raw,alpha,bound=12.):
        record.append(raw.detach().double().clone());return old(raw,alpha,bound)
    BB.bounded_score_bridge=tap
    try:
        with torch.no_grad():D.batched_logits(model,rows,314159)
    finally:BB.bounded_score_bridge=old
    candidates=races=capped_candidates=capped_races=0;maximum=0.
    for idx,raw in enumerate(record):
        event=idx//model.depth
        active=torch.tensor([len(r['events'])>event for r in rows])
        value=raw[active];mask=value.abs()>12
        candidates+=value.numel();races+=value.numel()//model.pool
        capped_candidates+=int(mask.sum());capped_races+=int(mask.any(-1).sum())
        maximum=max(maximum,float(value.abs().max()))
    return dict(races=races,candidates=candidates,capped_races=capped_races,
        capped_candidates=capped_candidates,max_abs_raw_score=maximum)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--bridge',required=True)
    p.add_argument('--hard-smoke',required=True);p.add_argument('--bridge-smoke',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    torch.set_num_threads(1);begin=time.perf_counter();base=json.loads((ROOT/a.bridge).read_text())
    assert base['status']=='completed';parents=[];sources=dict(base['source_sha256'])
    for name,digest in sources.items():assert N.sha(ROOT/name)==digest
    parent_ref=next(r for r in base['parents'] if '_s7_' in r['result'])
    parent=json.loads((ROOT/parent_ref['result']).read_text());assert N.sha(ROOT/parent_ref['result'])==parent_ref['result_sha256']
    cp=(ROOT/parent_ref['result']).with_suffix('.progress.pt');assert N.sha(cp)==parent_ref['checkpoint_sha256']
    config=argparse.Namespace(**parent['args']);expanded=copy.copy(config);expanded.fit=984
    fit,_,data=N.load(expanded);assert data==parent['data'];saved=torch.load(cp,weights_only=False)
    model=ActualSuffix(sources=1,content_dim=33,classes=11,payload=config.payload,depth=config.depth,
        heads=config.heads,pool=config.pool);model.load_state_dict(saved['online_model']);model.eval()
    seed=510173;rows=[];replays=0
    cases=[r for r in base['rows'] if r['original_clamped']]
    assert len(cases)==5 and all(r['seed']==7 and r['encoder']=='fixed_pass4' for r in cases)
    for case in cases:
        row=fit[case['fit_index']];model.clear_trace();model.prefix_trace=None
        factual,state,rng=replay(model,row,seed);trace=copy.deepcopy(model.records)
        site=case['site'][0]*model.depth*model.heads+case['site'][1]*model.heads+case['site'][2]
        assert bool((trace[site]['raw'].abs()>12).any())
        utilities=[];state_bytes=[]
        for winner in range(model.pool):
            model.clear_trace();model.prefix_trace=trace;model.forced_site=site;model.forced_winner=winner
            z,s,end_rng=replay(model,row,seed);replays+=1
            assert torch.equal(end_rng,rng)
            assert s.selected_updates==84 and s.candidate_scores==168 and s.counterfactual_values==0
            assert (case['site'][1],case['site'][2],0,winner) in s.visited_units
            if winner==int(trace[site]['winner']):
                assert torch.equal(z,factual);equal_state(s,state)
            utilities.append(float(F.cross_entropy(z[None],torch.tensor([row['target']]))))
            state_bytes.append(s.storage()['persistent_tensor_bytes'])
        raw=trace[site]['raw'].double().requires_grad_(True);losses=torch.tensor(utilities,dtype=torch.float64)
        estimates=[]
        for alpha in (0.,.1):
            def risk(value):return (bounded_score_bridge(value,alpha).softmax(0)*losses).sum()
            assert torch.autograd.gradcheck(risk,(raw,),eps=1e-5,atol=1e-8,rtol=1e-5)
            risk_value=risk(raw);grad,=torch.autograd.grad(risk_value,raw)
            pi=bounded_score_bridge(raw.detach(),alpha).softmax(0)
            slope=(1-alpha)*(raw.detach().abs()<12).double()+alpha*(1+(raw.detach()/12).square()).pow(-1.5)
            analytic=pi.prod()*(losses[0]-losses[1])*slope*torch.tensor([1.,-1.],dtype=torch.float64)
            torch.testing.assert_close(grad,analytic,rtol=1e-6,atol=1e-14)
            estimates.append(dict(alpha=alpha,probabilities=pi.tolist(),conditional_risk=float(risk_value.detach()),
                raw_score_choice_derivative=grad.tolist(),analytic_derivative=analytic.tolist()))
        rows.append(dict(fit_index=case['fit_index'],target=int(row['target']),site=case['site'],
            raw_scores=raw.detach().tolist(),factual_winner=int(trace[site]['winner']),
            first_delay=float(trace[site]['delay']),actual_suffix_losses=utilities,
            utility_gap=utilities[0]-utilities[1],conditional_choice=estimates,live_state_bytes=state_bytes))
    ledger=[];smoke_checks=[]
    for name in (a.hard_smoke,a.bridge_smoke):
        r=json.loads((ROOT/name).read_text());assert r['status']=='completed'
        for path,digest in r['source_sha256'].items():assert N.sha(ROOT/path)==digest
        sources.update(r['source_sha256']);args=argparse.Namespace(**r['args']);fit_smoke,dev_smoke,info=N.load(args)
        assert info==r['data'] and len(fit_smoke)==32 and len(dev_smoke)==16
        assert r['work']['fitting_targets']==64 and r['work']['optimizer_updates']==8
        for sample in r['work_samples']:
            assert all(t['formula_coverage_complete'] for t in sample['stages'].values())
        cp=(ROOT/name).with_suffix('.progress.pt');snap=torch.load(cp,weights_only=False)
        m=D.make_model(args);initial=exposure(m,fit_smoke);m.load_state_dict(snap['best_state'])
        final=exposure(m,fit_smoke)
        live=[];writes=[];scores=[]
        with torch.no_grad():
            for row in dev_smoke[:11]:
                _,s=N.predict(m,row,314159,False);live.append(s.storage()['persistent_tensor_bytes'])
                writes.append(s.selected_updates);scores.append(s.candidate_scores)
        w=r['work'];ledger.append(dict(result=name,result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(cp),
            alpha=args.score_bridge,parameters=r['parameters'],fit_inputs=32,dev_inputs=16,passes=2,
            fitting_presentations=64,optimizer_updates=8,available_receivers=w['native_available_receivers'],
            selected_writes_per_event=args.depth*args.heads,key_scores_per_event=args.depth*args.heads*args.pool,
            initial_fit_nll=r['initial_fit']['nll'],final_fit_nll=r['final_fit_diagnostic']['nll'],
            dev_accuracy=r['final']['accuracy'],dev_nll=r['final']['nll'],selected_epoch=r['selected_epoch'],
            whole_fit_gflops=w['whole_fit_unit_special_flops_estimate']/1e9,
            fit_mflops_per_presentation=w['fit_unit_special_flops_per_target_estimate']/1e6,
            inference_mflops_per_prefix=w['inference_unit_special_flops_per_target_estimate']/1e6,
            wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb'],initial_cap_exposure=initial,final_cap_exposure=final,
            native_live_state_bytes_range=[min(live),max(live)],selected_writes_per_prefix_range=[min(writes),max(writes)],
            key_scores_per_prefix_range=[min(scores),max(scores)]))
        smoke_checks.append(dict(result=name,source_and_budget_and_all_stage_coverage=True,
            small_fit_learning_passed=r['small_fit_learning_passed']))
        parents.append(dict(result=name,result_sha256=N.sha(ROOT/name)))
    assert ledger[0]['alpha']==0. and ledger[1]['alpha']==.1
    reference_names=['curie_dvs_batched_p16d2pool2_leall_s7_20261002T224500Z.json',
        'curie_dvs_batched_p16d4pool2_le8_s7_20261002T224500Z.json']
    refs=[]
    for name in reference_names:
        path='experiments/results/dvs_native/'+name;r=json.loads((ROOT/path).read_text());assert r['status']=='completed'
        w=r['work'];refs.append(dict(result=path,result_sha256=N.sha(ROOT/path),
            dev_accuracy=r['final']['accuracy'],dev_nll=r['final']['nll'],
            fitting_presentations=w['fitting_targets'],whole_fit_gflops=w['whole_fit_unit_special_flops_estimate']/1e9,
            fit_mflops_per_presentation=w['fit_unit_special_flops_per_target_estimate']/1e6,
            inference_mflops_per_prefix=w['inference_unit_special_flops_per_target_estimate']/1e6,
            scope='Unequal width/data/passes/quality, raw work comparison only; not advantage'))
    names=['experiments/dvs_bridge_utility_and_smoke_audit.py','experiments/theory/107_bounded_information_and_native_route_utility.md',
        'experiments/dvs_bridge_training_contracts.py']
    sources.update({name:N.sha(ROOT/name) for name in names})
    result=dict(status='completed',args=vars(a),source_sha256=sources,parents=parents+[parent_ref],
        utility_rows=rows,forced_suffix_replays=replays,factual_native_forwards=len(cases),
        additional_raw_dot_products_per_native_forward=168,smoke_checks=smoke_checks,common_unit_ledger=ledger,
        saved_references=refs,optimizer_steps=0,development_quality_evaluations=0,
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(host=platform.node(),device='CPU',threads=1),whole_audit_flops=None,
        scope='Actual hard-map suffix utilities at five old saturated histories; bridge only in conditional choice derivative, '
            'factual first time held, suffix hard maps retained. Revealed FIT labels258/981, no longer label-unused. '
            'Smoke initial/final cap and native state audit; no optimizer, official test or practical advantage claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(utility_rows=rows,common_unit_ledger=ledger,wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'])))


if __name__=='__main__':main()
