"""Production equivalence, actual counted T4 work and replicated T16 wall."""
import argparse
import copy
import json
from pathlib import Path
import resource
import statistics
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import causal_language_replay_contracts as C
import causal_language_replay_resource_admission as P
import causal_language_precision_audit as Precision
import causal_language_replay_helpers_rng as Full
import causal_language_replay_winner_reuse as Winner
import causal_language_replay_cached_prefix as Cached
import causal_language_cached_prefix_contracts as Contract
from causal_language_replay_accumulator import ReplayAccumulator
from causal_language_replay_winner_accumulator import WinnerReuseAccumulator
from causal_language_replay_cached_accumulator import CachedPrefixAccumulator
from native_language_replay_driver_contracts import exact
from race_language_screen import capture
ARMS={'full':(Full,ReplayAccumulator),'winner':(Winner,WinnerReuseAccumulator),'cached':(Cached,CachedPrefixAccumulator)}


def vector(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().double().flatten() for p in model.parameters()])


def weights(model):return torch.cat([p.detach().double().flatten() for p in model.parameters()])


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--contracts',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    prior=json.loads((ROOT/a.contracts).read_text());assert prior['status']=='completed' and prior['contracts_passed']==36
    for n,digest in prior['source_sha256'].items():assert N.sha(ROOT/n)==digest,n
    torch.set_num_threads(1);begin=time.perf_counter();checks=[];families=[];ledger=[];audits={}
    observed=torch.tensor([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3,2]);ins=observed[:16];targets=observed[1:17]
    for family in ('private','depth'):
        base=P.make(family);st=P.initial(base);double=copy.deepcopy(base).double();ds=Precision.promote(st)
        torch.manual_seed(119329);rng=torch.get_rng_state().clone();reference=None;counts=[]
        for name,(helper,_) in ARMS.items():
            m=copy.deepcopy(double);obj,z,state,act=helper.batched_objective(m,ins,targets,copy.deepcopy(ds),rng)
            obj.backward();grad=C.grads(m)
            if reference is None:reference=dict(grad=grad,z=z.detach(),state=state.detach(),objective=obj.detach(),rng=act['factual_end_rng'])
            else:
                C.grads_close(grad,reference['grad']);C.close(obj.detach(),reference['objective']);C.close(z,reference['z'],True)
                C.state_close(state,reference['state'],True);C.close(act['factual_end_rng'],reference['rng'],True)
            assert act['shadow_lanes']==(512 if name=='full' else 256)
            assert act['shadow_events']==dict(full=8192,winner=4096,cached=2176)[name]
            counts.append(dict(arm=name,targets=16,shadow_lanes=act['shadow_lanes'],shadow_events=act['shadow_events'],
                snapshot_tensor_bytes=act.get('snapshot_tensor_bytes',0),scope='Double T16 gradient test; snapshots only for cached, no optimizer'))
        checks.append(f'{family}: productionT16 EVERYdouble gradient/objective equals both originals; factual logits/private state/endRNG exact')
        # All arithmetic auditing is restricted to explicitly labelled T4 steps.
        for name,(_,Learner) in ARMS.items():
            m=copy.deepcopy(base);opt=torch.optim.Adam(m.parameters(),lr=.002);learner=Learner(m,opt,.002,32)
            torch.set_rng_state(rng);box={}
            def accumulate():box['data']=learner.accumulate(ins[:4],targets[:4],copy.deepcopy(st))
            stages=dict(full_credit_accumulate=capture(accumulate),normalize=capture(learner.normalize),
                clip=capture(lambda:torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)),Adam=capture(learner.step))
            assert all(s['formula_coverage_complete'] for s in stages.values());audits[family+'/'+name]=stages
            total=sum(s['arithmetic_flops']+s['special_function_evaluations'] for s in stages.values())
            m.eval();state=m.new_state()
            with torch.no_grad(),torch.random.fork_rng():
                torch.manual_seed(314159);itr=capture(lambda:m.forward_chunk(ins[:4],state))
            ledger.append(dict(family=family,arm=name,targets=4,optimizer_updates=1,parameters=sum(p.numel() for p in m.parameters()),
                available_receivers=32,key_scores_per_target=32,selected_updates_per_target=16,
                shadow_lanes=learner.shadow_lanes,shadow_events=learner.shadow_events,
                whole_fit_unit_special_flops=total,fit_unit_special_flops_per_target=total/4,
                inference_unit_special_flops_per_target=(itr['arithmetic_flops']+itr['special_function_evaluations'])/4,
                scope='ONEactual synthetic T4testupdate; actual cache/shadow backward/normalize/clip/Adam paid; noT16projection'))
        checks.append(f'{family}: THREEactual T4fits plus selected-value inference have complete accounting with matching denominators')
        # Rotate order across three fresh T16 test-step replicates.
        times={k:[] for k in ARMS};gradient_errors={k:[] for k in ARMS};update_errors={k:[] for k in ARMS}
        order=list(ARMS);first=None
        for repeat in range(3):
            outcomes={}
            for name in order[repeat:]+order[:repeat]:
                m=copy.deepcopy(base);opt=torch.optim.Adam(m.parameters(),lr=.002);learner=ARMS[name][1](m,opt,.002,32)
                torch.set_rng_state(rng);start=time.perf_counter()
                loss,state,z=learner.accumulate(ins,targets,copy.deepcopy(st));g=vector(m);learner.update()
                times[name].append(time.perf_counter()-start)
                outcomes[name]=dict(loss=loss,state=state,logits=z,rng=torch.get_rng_state().clone(),grad=g,weights=weights(m))
            old=outcomes['full'];movement=float((old['weights']-weights(base)).norm())
            for name,result in outcomes.items():
                assert result['loss']==old['loss'];C.close(result['logits'],old['logits'],True)
                C.state_close(result['state'],old['state'],True);C.close(result['rng'],old['rng'],True)
                ge=float((result['grad']-old['grad']).norm()/old['grad'].norm().clamp_min(1e-300))
                ue=float((result['weights']-old['weights']).norm())/max(movement,1e-300)
                assert ge<=3e-5 and ue<=.005;gradient_errors[name].append(ge);update_errors[name].append(ue)
            if first is None:first=outcomes
            else:
                for name in ARMS:
                    exact(outcomes[name]['weights'],first[name]['weights']);exact(outcomes[name]['grad'],first[name]['grad'])
        checks.append(f'{family}: THREEfresh rotated T16actualAdam replicates reproduce every own weight/gradient bitwise and meet globalgradient/update bounds')
        whole={r['arm']:r['whole_fit_unit_special_flops'] for r in ledger if r['family']==family}
        families.append(dict(family=family,production_double_activity=counts,
            t4_cached_work_saving_against_full=1-whole['cached']/whole['full'],t4_cached_work_saving_against_winner=1-whole['cached']/whole['winner'],
            t16_learning_wall_s=times,t16_median_learning_wall_s={k:statistics.median(v) for k,v in times.items()},
            t16_relative_gradient_errors=gradient_errors,t16_relative_Adam_update_errors=update_errors,
            scope='T16wall includes gradient-vector measurement equally in eacharm; diagnosticcounter extraction, initialization/DEV omitted; no throughput/energyclaim'))
    names=['experiments/causal_language_cached_prefix_resource.py','experiments/theory/119_cached_prefix_production_admission.md',
        'experiments/causal_language_precision_audit.py','experiments/causal_language_replay_resource_admission.py',
        'experiments/causal_language_replay_winner_reuse.py','experiments/causal_language_replay_winner_accumulator.py',
        'sleeping_machines/causal_language_shadow_winner_reuse.py']
    r=dict(status='completed',args=vars(a),contracts_passed=len(checks),contracts=checks,families=families,
        common_unit_ledger=ledger,work_audits=audits,source_sha256={**Contract.sources(),**{n:N.sha(ROOT/n) for n in names}},
        contract_result_sha256=N.sha(ROOT/a.contracts),wall_s=time.perf_counter()-begin,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Grouped cached-prefix SAMEestimator production admission and syntheticresource tradeoff; '
            'six auditedT4 andeighteen untracedT16 testAdamupdates, sixdoubleT16gradientfits; '
            'no data/DEV/test/longfitgain. OnlyT4has countedfullfittingwork; campaign/T16totalFLOPs/traffic/energyunknown.')
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r[k] for k in ('status','contracts_passed','families','wall_s','max_rss_kb')}))


if __name__=='__main__':main()
