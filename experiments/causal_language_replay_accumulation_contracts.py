"""Target-weighted stateful replay accumulation and exact stream-RNG contracts."""
import argparse
import copy
import io
import json
from pathlib import Path
import resource
import sys
import time
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import causal_language_replay_contracts as C
import causal_language_replay_helpers_rng as R
from causal_language_replay_accumulator import ReplayAccumulator
from race_language_screen import capture


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    torch.set_num_threads(1);begin=time.perf_counter();checks=[];rows=[];audits={}
    tokens=torch.tensor([3,1,4,2]);lr=.002;warmup=4
    for family in ('private','depth'):
        model=C.make(family);original=copy.deepcopy(model);reference=copy.deepcopy(model)
        state=C.entering(model);old=copy.deepcopy(state);rs=copy.deepcopy(state)
        opt=torch.optim.Adam(model.parameters(),lr=lr);ropt=torch.optim.Adam(reference.parameters(),lr=lr)
        learner=ReplayAccumulator(model,opt,lr,warmup)
        torch.manual_seed(113329);enter_rng=torch.get_rng_state().clone()
        z_rngs=[]
        # First factual predictions/state/RNG exactly match original chronological teacher.
        loss,state,z=learner.accumulate(tokens[:2],tokens[1:3],state);factual_end=torch.get_rng_state().clone()
        torch.set_rng_state(enter_rng);original.train()
        with torch.no_grad():oz,old=original.forward_chunk(tokens[:2],old)
        C.close(torch.get_rng_state(),factual_end,True);C.close(z,oz);C.state_close(state,old)
        objective,sz,rs,activity=R.sequential_objective(reference,tokens[:2],tokens[1:3],rs,enter_rng)
        objective.backward();rs.detach();C.close(sz,z)
        C.grads_close(C.grads(model),C.grads(reference))
        checks.append(f'{family}: factual chronological teacher output/private state/end RNG and every first-chunk replay gradient agree')
        torch.set_rng_state(factual_end)
        saved=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,
            gradients=C.grads(model),counters=learner.counters(),cursor=dict(next_target=2),rng=factual_end),saved)
        saved.seek(0);checkpoint=torch.load(saved,weights_only=False)
        recovered=C.make(family);recovered.load_state_dict(checkpoint['model'])
        recovered_opt=torch.optim.Adam(recovered.parameters(),lr=lr);recovered_opt.load_state_dict(checkpoint['optimizer'])
        other=ReplayAccumulator(recovered,recovered_opt,lr,warmup);other.restore_counters(checkpoint['counters'])
        for name,par in recovered.named_parameters():par.grad=checkpoint['gradients'].get(name)
        torch.set_rng_state(checkpoint['rng']);loss,state,z=learner.accumulate(tokens[2:3],tokens[3:4],state)
        next_rng=torch.get_rng_state().clone()
        torch.set_rng_state(checkpoint['rng']);loss2,restored,z2=other.accumulate(tokens[2:3],tokens[3:4],checkpoint['state'])
        assert loss==loss2;C.close(z,z2,True);C.state_close(state,restored,True)
        C.close(torch.get_rng_state(),next_rng,True);assert learner.counters()==other.counters()
        C.grads_close(C.grads(model),C.grads(recovered),True)
        checks.append(f'{family}: pending gradients/private state/cursor/RNG/replay counters recover bitwise before delayed update')
        # Independent teacher chronology through the second microchunk.
        torch.set_rng_state(factual_end)
        with torch.no_grad():oz,old=original.forward_chunk(tokens[2:3],old)
        C.close(oz,z);C.state_close(state,old);C.close(torch.get_rng_state(),next_rng,True)
        objective,sz,rs,activity=R.sequential_objective(reference,tokens[2:3],tokens[3:4],rs,factual_end)
        objective.backward();rs.detach();C.grads_close(C.grads(model),C.grads(reference))
        assert learner.pending_targets==learner.total_targets==3
        # Match actual partial normalization/clipping/warmup/Adam.
        for par in reference.parameters():
            if par.grad is not None:par.grad.div_(3)
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True)
        for group in ropt.param_groups:group['lr']=lr*3/warmup
        ropt.step();learner.update();other.update()
        for (name,par),(rn,q) in zip(model.named_parameters(),recovered.named_parameters()):
            assert name==rn;C.close(par,q,True)
            C.close(par,dict(reference.named_parameters())[name])
        assert learner.updates==1 and learner.pending_targets==0 and opt.param_groups[0]['lr']==lr*.75
        assert learner.counters()==other.counters()
        for key,values in opt.state_dict()['state'].items():
            for name,v in values.items():C.close(v,recovered_opt.state_dict()['state'][key][name],True)
        checks.append(f'{family}: independent summed full-return gradients, target normalization/clip/warmup/Adam and recovery exactly agree')
        # Target labels alter pending gradients, never factual states/noise/predictions before update.
        left=C.make(family);right=copy.deepcopy(left);ls=C.entering(left);rr=copy.deepcopy(ls)
        la=ReplayAccumulator(left,torch.optim.Adam(left.parameters(),lr=lr),lr,warmup)
        rb=ReplayAccumulator(right,torch.optim.Adam(right.parameters(),lr=lr),lr,warmup)
        torch.set_rng_state(enter_rng);_,ls,lz=la.accumulate(tokens[:2],tokens[1:3],ls);lrng=torch.get_rng_state().clone()
        torch.set_rng_state(enter_rng);_,rr,rz=rb.accumulate(tokens[:2],(tokens[1:3]+7)%27,rr)
        C.close(lz,rz,True);C.state_close(ls,rr,True);C.close(lrng,torch.get_rng_state(),True)
        torch.set_rng_state(lrng);_,ls,lz=la.accumulate(tokens[2:3],tokens[3:4],ls)
        torch.set_rng_state(lrng);_,rr,rz=rb.accumulate(tokens[2:3],tokens[3:4],rr)
        C.close(lz,rz,True);C.state_close(ls,rr,True)
        checks.append(f'{family}: target-label changes leave both pre-update factual chunks/private states/RNG exact')
        # Complete numerical work includes all real shadows and optimizer.
        traced=C.make(family);ts=C.entering(traced);to=torch.optim.Adam(traced.parameters(),lr=lr)
        trial=ReplayAccumulator(traced,to,lr,warmup);box={}
        def fit():box['data']=trial.accumulate(tokens[:2],tokens[1:3],ts)
        stages={'full_credit_accumulate':capture(fit),'target_normalize_clip_warmup_Adam':capture(trial.update)}
        assert all(tr['formula_coverage_complete'] for tr in stages.values());audits[family]=stages
        checks.append(f'{family}: full shadow/backward plus normalize/clip/warmup/Adam work completely covered')
        rows.append(dict(family=family,parameters=sum(p.numel() for p in model.parameters()),
            depth=8,heads=2,payload=4,pool=2,targets=3,credit_chunks=2,shadow_lanes=learner.shadow_lanes,
            shadow_events=learner.shadow_events,actual_partial_learning_rate=opt.param_groups[0]['lr'],
            optimizer_updates=learner.updates,final_live_state_bytes=state.storage()['persistent_tensor_bytes']))
    names=['experiments/causal_language_replay_accumulation_contracts.py','experiments/causal_language_replay_accumulator.py',
        'experiments/causal_language_replay_helpers_rng.py','sleeping_machines/causal_language_shadow_rng.py',
        'experiments/causal_language_replay_contracts.py','experiments/parallel_head_gradient_accumulation.py',
        'sleeping_machines/fast_native_core.py','sleeping_machines/native_stream_language.py',
        'sleeping_machines/addressed_event_heads.py','sleeping_machines/factorized_race.py',
        'sleeping_machines/sparse_race_language.py','sleeping_machines/parallel_head_race_language.py',
        'sleeping_machines/parallel_stream_language.py','sleeping_machines/operation_audit.py',
        'experiments/race_language_screen.py','experiments/theory/113_stateful_language_replay_accumulation.md']
    result=dict(status='completed',args=vars(a),contracts_passed=len(checks),contracts=checks,
        families=rows,work_audits=audits,source_sha256={name:N.sha(ROOT/name) for name in names},
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Stateful chronological RNG and target-weighted actual language replay accumulator numerical contracts; '
              'test optimizer steps only, no data/DEV/test/benchmark gain or changed active AWS10M source.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ('status','contracts_passed','families','wall_s','max_rss_kb')}))


if __name__=='__main__':main()
