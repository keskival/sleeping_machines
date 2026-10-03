"""Frozen actual-text8 causal16/32-return audit, not a new fit."""
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import causal_language_replay_helpers as H
import causal_language_replay_contracts as C
from dvs_conditional_clock_choice_geometry import flat_grad
from e120_shared_tasks import text_slice
from sleeping_machines.fast_native_core import FastNativeStreamLanguageModel


def cosine(a,b):
    denominator=float(a.norm()*b.norm())
    return float(a@b)/denominator if denominator else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--native',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    parent=json.loads((ROOT/a.native).read_text());assert parent['status']=='completed'
    for name,digest in parent['source_sha256'].items():assert N.sha(ROOT/name)==digest
    assert parent['args']['fit']==8192 and parent['args']['epochs']==4 and parent['args']['seed']==6
    cp=(ROOT/a.native).with_suffix('.progress.pt');snap=torch.load(cp,weights_only=False)
    torch.set_num_threads(1);begin=time.perf_counter();model=FastNativeStreamLanguageModel(payload=16,depth=8,pool=2,heads=2)
    model.load_state_dict(snap['model']);model.double();params=tuple(model.parameters());before=copy.deepcopy(model.state_dict())
    rows=[];contracts=[];arrays={};aggregates=[];seed=111329
    for start in (8448,8576):
        for seed in (111330,111331,111332):
            tokens=torch.tensor(text_slice(start,49));assert len(tokens)==49
            with torch.random.fork_rng(),torch.no_grad():
                torch.manual_seed(111329);model.eval();_,entering=model.forward_chunk(tokens[:16]);entering.detach()
            inputs=tokens[16:48];targets=tokens[17:49];records=[]
            loss,factual,state,rng=H.sequential(model,inputs,targets,entering,seed,record=records)
            sum16=torch.zeros(sum(p.numel() for p in params),dtype=torch.float64);sum32=sum16.clone()
            for event,depth,head in [(0,0,0),(0,7,0),(8,0,0),(8,7,0)]:
                race=event*16+depth*2+head;rec=records[race]
                probabilities=rec['scores'].detach().double().softmax(0)
                direction=flat_grad(rec['scores'][0]-rec['scores'][1],params)
                returns=[];bytes_values=[]
                for candidate in (0,1):
                    forced=[]
                    with torch.no_grad():
                        alt,z,s,end_rng=H.sequential(model,inputs,targets,entering,seed,force=(race,candidate),record=forced)
                    C.close(rng,end_rng,True);C.close(rec['delay'],forced[race]['delay'],True)
                    C.close(z[:event],factual.detach()[:event],True)
                    assert int(forced[race]['winner'])==candidate
                    if candidate==int(rec['winner']):
                        C.close(z,factual.detach(),True);C.state_close(s,state,True)
                    assert (depth,head,0,candidate) in s.visited_units
                    returns.append([float(alt[event:16].sum()),float(alt[16:].sum()),float(alt[event:].sum())])
                    bytes_values.append(s.storage()['persistent_tensor_bytes'])
                delta16=returns[0][0]-returns[1][0];delta32=returns[0][2]-returns[1][2];late=returns[0][1]-returns[1][1]
                assert abs(delta32-(delta16+late))<1e-10
                coefficient16=float(probabilities.prod())*delta16;coefficient32=float(probabilities.prod())*delta32
                g16=coefficient16*direction;g32=coefficient32*direction;sum16+=g16;sum32+=g32
                key=f'offset{start}_noise{seed}_e{event}d{depth}h{head}';arrays[key+'_direction']=direction.numpy()
                arrays[key+'_credit16']=g16.numpy();arrays[key+'_credit32']=g32.numpy()
                rows.append(dict(fit_start=start,noise_seed=seed,evaluated_start=start+16,site=[event,depth,head],
                    factual_winner=int(rec['winner']),scores=rec['scores'].detach().tolist(),probabilities=probabilities.tolist(),
                    capped_score_proxy=bool((rec['scores'].detach().abs()>=12).any()),
                    utilities_by_candidate_early_late_total=returns,utility_contrast16=delta16,
                    utility_contrast32=delta32,late_utility_contrast=late,
                    choice_coefficient16=coefficient16,choice_coefficient32=coefficient32,
                    parameter_direction_norm=float(direction.norm()),parameter_credit_norm16=float(g16.norm()),
                    parameter_credit_norm32=float(g32.norm()),horizon_sign_reversal=coefficient16*coefficient32<0,
                    parameter_credit_cosine=cosine(g16,g32),live_state_bytes=bytes_values))
            # Causal future input/labels, actual fixed frozen producer unchanged.
            with torch.no_grad():
                future=inputs.clone();future[16:]=(future[16:]+7)%27
                _,z,_,_=H.sequential(model,future,targets,entering,seed)
                C.close(factual.detach()[:16],z[:16],True)
                _,z,_,_=H.sequential(model,inputs,(targets+3)%27,entering,seed)
                C.close(factual.detach(),z,True)
            assert all(torch.equal(before[n],v.detach()) for n,v in model.state_dict().items())
            contracts.append(dict(start=start,noise_seed=seed,original_winner_all_logits_state_rng_exact=True,
                alternate_first_time_real_state_rng=True,future_input_and_target_causality=True,parameters_unchanged=True,
                factual_targets=32,actual_key_scores=1024,actual_writes=512,entering_state_bytes=entering.storage()['persistent_tensor_bytes']))
            arrays[f'offset{start}_noise{seed}_aggregate16']=sum16.numpy();arrays[f'offset{start}_noise{seed}_aggregate32']=sum32.numpy()
            aggregates.append(dict(fit_start=start,noise_seed=seed,selected_sites=4,credit_norm16=float(sum16.norm()),
                credit_norm32=float(sum32.norm()),credit_cosine=cosine(sum16,sum32),
                change_norm=float((sum32-sum16).norm()),scope='Four prescribed sites, not full-chunk gradient'))
    confirmation=[]
    for span in (8448,8576):
        mean16=torch.tensor(np.stack([arrays[f'offset{span}_noise{noise}_aggregate16'] for noise in (111330,111331,111332)]).mean(0))
        mean32=torch.tensor(np.stack([arrays[f'offset{span}_noise{noise}_aggregate32'] for noise in (111330,111331,111332)]).mean(0))
        sign_cases=sum(r['horizon_sign_reversal'] for r in rows if r['fit_start']==span)
        angle=cosine(mean16,mean32)
        arrays[f'offset{span}_mean16']=mean16.numpy();arrays[f'offset{span}_mean32']=mean32.numpy()
        confirmation.append(dict(fit_start=span,noise_draws=3,prescribed_site_cases=12,sign_reversals=sign_cases,
            mean_credit_norm16=float(mean16.norm()),mean_credit_norm32=float(mean32.norm()),mean_credit_cosine=angle,
            descriptive_gate_passed=angle is not None and angle<0 and sign_cases>0))
    artifact=out.with_suffix('.credits.npz');np.savez_compressed(artifact,**arrays)
    names=['experiments/native_language_horizon_return_confirmation.py',
        'experiments/theory/112_native_language_horizon_confirmation.md',
        'experiments/causal_language_replay_helpers.py','experiments/causal_language_replay_contracts.py',
        'sleeping_machines/causal_language_shadow.py','sleeping_machines/fast_native_core.py',
        'sleeping_machines/factorized_race.py','experiments/dvs_conditional_clock_choice_geometry.py']
    result=dict(status='completed',args=vars(a),source_sha256={**parent['source_sha256'],**{n:N.sha(ROOT/n) for n in names}},
        parent=dict(result=a.native,result_sha256=N.sha(ROOT/a.native),checkpoint_sha256=N.sha(cp),
            encoder='fixed final-pass4 online model, not DEV-selected',original_fit_targets=parent['work']['fitting_targets']),
        rows=rows,aggregates=aggregates,native_contracts=contracts,forced_forward_replays=48,parameter_vjps=24,
        confirmation=confirmation,descriptive_confirmation_gate_passed=all(r['descriptive_gate_passed'] for r in confirmation),
        optimizer_steps=0,development_evaluations=0,whole_audit_flops=None,
        artifact=str(artifact.relative_to(ROOT)),artifact_sha256=N.sha(artifact),
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(host=platform.node(),device='CPU',threads=1),
        scope='Two FRESH producer-unseen FIT spans, three evaluated race draws each,24 prescribed native site cases; factual-first-time actual-write '
              'utility compares16/32 suffix returns. Realized branch score-contrast parameter Jacobian includes prior pathwise '
              'times; not fixed-clock likelihood geometry, full512-site gradient, expected-risk quality or benchmark advantage.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(confirmation=confirmation,aggregates=aggregates,wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'])))


if __name__=='__main__':main()
