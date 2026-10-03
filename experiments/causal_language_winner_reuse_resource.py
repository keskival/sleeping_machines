"""Production-shape same-gradient full-work saving; no real-data quality fit."""
import argparse
import copy
import io
import json
from pathlib import Path
import resource
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import causal_language_replay_contracts as C
import causal_language_replay_resource_admission as P
import causal_language_winner_reuse_contracts as W
from causal_language_replay_accumulator import ReplayAccumulator
from causal_language_replay_winner_accumulator import WinnerReuseAccumulator
from native_language_replay_driver_contracts import exact
from race_language_screen import capture


def gradients_error(left,right):
    names=dict(left.named_parameters());others=dict(right.named_parameters());err=0.;norm=0.;maximum=0.;failed=[]
    for n,p in names.items():
        if p.grad is None and others[n].grad is None:continue
        x=p.grad if p.grad is not None else torch.zeros_like(others[n].grad)
        y=others[n].grad if others[n].grad is not None else torch.zeros_like(x)
        try:torch.testing.assert_close(x,y,rtol=3e-4,atol=3e-6)
        except AssertionError:
            failed.append(dict(parameter=n,maximum_absolute=float((x-y).abs().max()),
                               original_maximum_absolute=float(x.abs().max()),reuse_maximum_absolute=float(y.abs().max())))
        delta=(x-y).double();err+=float(delta.square().sum());norm+=float(x.double().square().sum())
        maximum=max(maximum,float(delta.abs().max()))
    relative=(err/max(norm,1e-300))**.5
    return dict(relative_l2=relative,maximum_absolute=maximum,failed_parameter_tolerances=failed,
        all_parameter_tolerances_passed=not failed,global_relative_tolerance_passed=relative<=3e-5)


def trace(action):
    start=time.perf_counter();result=capture(action);return result,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--contracts',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    prior=json.loads((ROOT/a.contracts).read_text());assert prior['status']=='completed' and prior['contracts_passed']==28
    for n,digest in prior['source_sha256'].items():assert N.sha(ROOT/n)==digest,n
    torch.set_num_threads(1);begin=time.perf_counter();rows=[];checks=[];audits={};families=[]
    observed=torch.tensor([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3,2,1,4,2,3]);inputs=observed[:16];targets=observed[1:17]
    for family in ('private','depth'):
        old=P.make(family);new=copy.deepcopy(old);st=P.initial(old);initial={n:p.detach().clone() for n,p in old.named_parameters()}
        learners={};models={'original':old,'reuse':new};states={};outputs={};stage_times={}
        torch.manual_seed(116329);rng=torch.get_rng_state().clone()
        for name,model in models.items():
            opt=torch.optim.Adam(model.parameters(),lr=.002)
            learner=(ReplayAccumulator if name=='original' else WinnerReuseAccumulator)(model,opt,.002,32)
            learners[name]=learner;torch.set_rng_state(rng);box={}
            def accumulate():box['data']=learner.accumulate(inputs,targets,copy.deepcopy(st))
            stages={};times={};stages['full_credit_accumulate'],times['full_credit_accumulate']=trace(accumulate)
            loss,state,z=box['data'];states[name]=state;outputs[name]=dict(loss=loss,logits=z,rng=torch.get_rng_state().clone())
            audits[family+'/'+name]=stages;stage_times[name]=times
        exact(outputs['original'],outputs['reuse']);C.state_close(states['original'],states['reuse'],True)
        ge=gradients_error(old,new)
        checks.append(f'{family}: production float32 factual logits/state/RNG exact; every-gradient comparison measured with declared tolerances')
        for name,learner in learners.items():
            stages=audits[family+'/'+name];times=stage_times[name]
            for label,action in [('normalize',learner.normalize),('clip',lambda:torch.nn.utils.clip_grad_norm_(learner.model.parameters(),1.,error_if_nonfinite=True)),('Adam',learner.step)]:
                stages[label],times[label]=trace(action)
            assert all(s['formula_coverage_complete'] for s in stages.values())
            whole=sum(s['arithmetic_flops']+s['special_function_evaluations'] for s in stages.values())
            model=learner.model;model.eval();state=model.new_state()
            with torch.no_grad(),torch.random.fork_rng():
                torch.manual_seed(314159);itr=capture(lambda:model.forward_chunk(inputs,state))
            inference=(itr['arithmetic_flops']+itr['special_function_evaluations'])/16
            rows.append(dict(family=family,arm=name,parameters=sum(p.numel() for p in model.parameters()),
                targets=16,optimizer_updates=1,shadow_lanes=learner.shadow_lanes,shadow_events=learner.shadow_events,
                available_receivers=32,key_scores_per_target=32,selected_updates_per_target=16,
                whole_step_unit_special_flops=whole,fit_unit_special_flops_per_target=whole/16,
                inference_unit_special_flops_per_target=inference,traced_stage_wall_s=times,
                scope='One synthetic sixteen-target update; all factual/shadow backward/normalize/clip/Adam charged'))
        error=sum(float((p-dict(new.named_parameters())[n]).double().square().sum()) for n,p in old.named_parameters())
        movement=sum(float((p-initial[n]).double().square().sum()) for n,p in old.named_parameters())
        relative_update=(error/max(movement,1e-300))**.5
        before,after=rows[-2:];saving=1-after['whole_step_unit_special_flops']/before['whole_step_unit_special_flops']
        assert saving>.4 and before['shadow_lanes']==512 and after['shadow_lanes']==256
        assert before['shadow_events']==8192 and after['shadow_events']==4096
        assert before['inference_unit_special_flops_per_target']==after['inference_unit_special_flops_per_target']
        checks.append(f'{family}: actual target-normalize/clip/warmup/Adam update difference measured; complete fitting work falls >40%')
        # Partial pending second window with nonempty Adam and persistent stream state.
        learn=learners['reuse'];torch.set_rng_state(outputs['reuse']['rng'])
        _,following,_=learn.accumulate(observed[16:19],observed[17:20],states['reuse'])
        buf=io.BytesIO();torch.save(dict(model=new.state_dict(),opt=learn.optimizer.state_dict(),grads=C.grads(new),
            state=following,counters=learn.counters(),rng=torch.get_rng_state()),buf);buf.seek(0);saved=torch.load(buf,weights_only=False)
        recovered=P.make(family);recovered.load_state_dict(saved['model']);opt=torch.optim.Adam(recovered.parameters(),lr=.002)
        opt.load_state_dict(saved['opt']);other=WinnerReuseAccumulator(recovered,opt,.002,32);other.restore_counters(saved['counters'])
        for n,par in recovered.named_parameters():par.grad=saved['grads'].get(n)
        torch.set_rng_state(saved['rng']);l,following,z=learn.accumulate(observed[19:20],observed[20:21],following)
        end=torch.get_rng_state().clone();torch.set_rng_state(saved['rng'])
        ll,fs,zz=other.accumulate(observed[19:20],observed[20:21],saved['state'])
        assert l==ll;C.close(z,zz,True);C.state_close(following,fs,True);C.grads_close(C.grads(new),C.grads(recovered),True)
        C.close(torch.get_rng_state(),end,True);learn.update();other.update()
        exact(new.state_dict(),recovered.state_dict());exact(learn.optimizer.state_dict(),other.optimizer.state_dict());exact(learn.counters(),other.counters())
        checks.append(f'{family}: nonempty Adam/private state/pending gradients/RNG and second partial warmup update recover bitwise')
        families.append(dict(family=family,gradient_error=ge,relative_actual_Adam_update_error=relative_update,
            counted_full_fitting_work_saving_fraction=saving,second_partial_recovery_exact=True,
            production_gradient_and_update_admission_passed=ge['all_parameter_tolerances_passed'] and ge['global_relative_tolerance_passed'] and relative_update<=.005,
            diagnostic_optimizer_updates=4,scope='Two matched first updates plus optimized/recovered second update; table describes first updates only'))
    names=['experiments/causal_language_winner_reuse_resource.py','experiments/theory/116_winner_reuse_production_work_contract.md']
    r=dict(status='completed',args=vars(a),contracts_passed=len(checks),contracts=checks,common_unit_ledger=rows,
        production_gradient_and_update_admission_passed=all(f['production_gradient_and_update_admission_passed'] for f in families),
        families=families,work_audits=audits,source_sha256={**W.sources(),**{n:N.sha(ROOT/n) for n in names}},
        contract_result_sha256=N.sha(ROOT/a.contracts),wall_s=time.perf_counter()-begin,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Same-estimator synthetic production-shape complete fitting-work reduction and actual Adam recovery; '
            'no real-data BPC, no Transformer/count comparison, no energy or benchmark supremacy claim.')
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r[k] for k in ('status','contracts_passed','production_gradient_and_update_admission_passed','families','wall_s','max_rss_kb')}))


if __name__=='__main__':main()
