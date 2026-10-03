"""Small deep native all-target language shadow credit contracts; no fit claim."""
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
import causal_language_replay_helpers as H
from sleeping_machines.fast_native_core import FastNativeStreamLanguageModel
from sleeping_machines.causal_language_shadow import batched_chunks
from race_language_screen import capture


def close(a,b,exact=False):torch.testing.assert_close(a,b,rtol=0 if exact else 3e-7,atol=0 if exact else 3e-9)


def state_close(a,b,exact=False):
    for field in ('events','candidate_scores','selected_updates','counterfactual_values','last_input_time','visited_units'):
        assert getattr(a,field)==getattr(b,field),field
    for field in ('memories','arrivals','contexts'):
        left,right=getattr(a,field),getattr(b,field);assert left.keys()==right.keys()
        for key in left:
            values=zip(left[key],right[key]) if field=='contexts' else [(left[key],right[key])]
            for x,y in values:close(x,y,exact)
    assert abs(a.queue_wait_sum-b.queue_wait_sum)<1e-12


def grads(model):return {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}


def grads_close(a,b,exact=False):
    if exact:assert a.keys()==b.keys()
    for name in a.keys()|b.keys():
        left=a[name] if name in a else torch.zeros_like(b[name])
        right=b[name] if name in b else torch.zeros_like(a[name])
        try:close(left,right,exact)
        except AssertionError as e:raise AssertionError(name) from e


def make(family):
    torch.manual_seed(108);m=FastNativeStreamLanguageModel(payload=4,depth=8,pool=2,heads=2).double()
    if family=='depth':
        for head in range(m.heads):
            master=m.units[0][head][0][0]
            for layer in m.units:
                for unit in layer[head][0]:
                    for name in ('input','output','gate','control','key_read'):setattr(unit,name,getattr(master,name))
    with torch.no_grad():
        for p in m.parameters():p.add_(torch.randn_like(p)*.03)
    return m


def entering(model):
    with torch.random.fork_rng(),torch.no_grad():
        torch.manual_seed(1908);model.train();_,state=model.forward_chunk(torch.tensor([1,2]))
    return state.detach()


def accumulate(model,tokens,targets,st,seed):
    objective,logits,following,activity=H.batched_objective(model,tokens,targets,st,seed)
    objective.backward();return logits.detach().clone(),following.detach(),activity


def finish(model,optimizer,count):
    for p in model.parameters():
        if p.grad is not None:p.grad.div_(count)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    torch.set_num_threads(1);begin=time.perf_counter();checks=[];families=[];audits={}
    inputs=torch.tensor([3,1,4]);targets=torch.tensor([1,4,2]);seed=192810
    for family in ('private','depth'):
        m=make(family);st=entering(m);original=copy.deepcopy(m)
        with torch.random.fork_rng(),torch.no_grad():
            torch.manual_seed(seed);original.train();z,s=original.forward_chunk(inputs,copy.deepcopy(st))
            factorized,zf,sf,rng=H.sequential(m,inputs,targets,st,seed)
            close(z,zf,True);state_close(s,sf,True)
        checks.append(f'{family}: original-teacher/factorized logits and ALL native state exactly agree')
        m.zero_grad();seq_objective,zs,ss,counts=H.sequential_objective(m,inputs,targets,st,seed)
        seq_objective.backward();seq_grad=grads(m);m.zero_grad()
        objective,zb,sb,bcounts=H.batched_objective(m,inputs,targets,st,seed)
        objective.backward();close(objective,seq_objective);close(zb,zs);state_close(sb,ss)
        grads_close(grads(m),seq_grad)
        assert counts['shadow_lanes']==bcounts['shadow_lanes']==96
        assert bcounts['shadow_events']==288
        checks.append(f'{family}: EVERY-parameter all-target full-write replay gradient/state equals independent sequential return')
        # Variable lengths retain nonempty actual persistent state.
        with torch.no_grad():
            observed=[H.row(m,inputs[:2],st),H.row(m,inputs,st)]
            batch,states,brng=batched_chunks(m,observed,seed,[st,st])
            for i,length in enumerate((2,3)):
                loss,seq,sseq,srng=H.sequential(m,inputs[:length],targets[:length],st,seed)
                close(batch[i,:length],seq);state_close(states[i],sseq)
                if length==3:close(brng,srng,True)
            before=torch.get_rng_state().clone();batched_chunks(m,observed,seed,[st,st]);close(before,torch.get_rng_state(),True)
        checks.append(f'{family}: variable-length lanes/native entering state match, longest RNG and caller RNG preserved')
        # Lose at first race after some history; common time exactly retained.
        records=[]
        with torch.no_grad():
            _,zf,sf,frng=H.sequential(m,inputs,targets,st,seed,record=records)
            forced_site=17;other=1-int(records[forced_site]['winner']);changed=[]
            loss,zalt,salt,arng=H.sequential(m,inputs,targets,st,seed,force=(forced_site,other),record=changed)
            close(records[forced_site]['delay'],changed[forced_site]['delay'],True)
            assert int(changed[forced_site]['winner'])==other;close(frng,arng,True)
            assert (0,1,0,other) in salt.visited_units
            assert not torch.equal(zalt[1:],zf[1:])
            close(zalt[0],zf[0],True)
            alt,asts,arng2=batched_chunks(m,[H.row(m,inputs,st)],seed,[st],[(forced_site,other)])
            close(alt[0],zalt);state_close(asts[0],salt);close(arng,arng2,True)
        checks.append(f'{family}: losing identity at factual FIRST time, real changed suffix state/logits and unchanged earlier prediction')
        # Future input and targets cannot change preceding/current factual predictions.
        with torch.no_grad():
            future=inputs.clone();future[-1]=9
            _,zfuture,_,_=H.sequential(m,future,targets,st,seed)
            close(zfuture[:2],zf[:2],True)
            _,zlabels,_,_=H.sequential(m,inputs,(targets+7)%27,st,seed)
            close(zlabels,zf,True)
        checks.append(f'{family}: future-token and label-independent causal predictions')
        # Actual accumulated partial-window checkpoint must retain gradients and private state.
        left=make(family);st0=entering(left);opt=torch.optim.Adam(left.parameters(),lr=.002);left.zero_grad()
        first,st1,c1=accumulate(left,inputs[:2],targets[:2],st0,seed)
        payload=dict(model=left.state_dict(),optimizer=opt.state_dict(),state=st1,gradients=grads(left),
                     pending=2,total=2,cursor=dict(event=4),rng=torch.get_rng_state())
        buf=io.BytesIO();torch.save(payload,buf);buf.seek(0);saved=torch.load(buf,weights_only=False)
        recovered=make(family);recovered.load_state_dict(saved['model']);ro=torch.optim.Adam(recovered.parameters(),lr=.002)
        ro.load_state_dict(saved['optimizer'])
        for name,param in recovered.named_parameters():param.grad=saved['gradients'].get(name)
        torch.set_rng_state(saved['rng']);second,ls,c2=accumulate(left,inputs[2:],targets[2:],st1,seed+1)
        torch.set_rng_state(saved['rng']);again,rs,rc2=accumulate(recovered,inputs[2:],targets[2:],saved['state'],seed+1)
        close(second,again,True);state_close(ls,rs,True)
        assert all(not p.requires_grad for p in ls.memories.values())
        finish(left,opt,3);finish(recovered,ro,3)
        for (name,p),(rn,r) in zip(left.named_parameters(),recovered.named_parameters()):
            assert name==rn;close(p,r,True)
        grads_close(grads(left),grads(recovered),True)
        for key,values in opt.state_dict()['state'].items():
            for name,v in values.items():close(v,ro.state_dict()['state'][key][name],True)
        assert saved['pending']==saved['total']==2 and saved['cursor']['event']==4
        checks.append(f'{family}: actual partial accumulated gradients/private state/Adam/cursor/RNG recovery bitwise exact')
        # Independent normalization/Adam against the sum of sequential causal terms.
        ref=make(family);rstate=entering(ref);r_opt=torch.optim.Adam(ref.parameters(),lr=.002);ref.zero_grad()
        for ins,tar,s0,snoise in [(inputs[:2],targets[:2],rstate,seed),(inputs[2:],targets[2:],st1,seed+1)]:
            obj,_,_,_=H.sequential_objective(ref,ins,tar,s0,snoise);obj.backward()
        finish(ref,r_opt,3);grads_close(grads(left),grads(ref))
        for name,p in left.named_parameters():close(p,dict(ref.named_parameters())[name])
        checks.append(f'{family}: partial-window normalization/clipping/Adam equals independently summed sequential full-return gradients')
        # Paid work for complete factual + shadow + backward + real optimizer.
        traced=make(family);tst=entering(traced);to=torch.optim.Adam(traced.parameters(),lr=.002)
        box={}
        def forward():
            obj,z,state,count=H.batched_objective(traced,inputs,targets,tst,seed);box.update(objective=obj,activity=count)
        stages={'factual_and_full_shadow':capture(forward)}
        stages['backward']=capture(lambda:box['objective'].backward())
        stages['normalize_clip_Adam']=capture(lambda:finish(traced,to,len(inputs)))
        assert all(tr['formula_coverage_complete'] for tr in stages.values())
        audits[family]=stages
        checks.append(f'{family}: complete factual/full-shadow/backward/normalization/clipping/Adam operation coverage')
        families.append(dict(family=family,parameters=sum(p.numel() for p in m.parameters()),
            input_tokens=inputs.tolist(),targets=targets.tolist(),credit_targets=3,depth=8,heads=2,pool=2,payload=4,
            entering_events=st.events,entering_bytes=st.storage()['persistent_tensor_bytes'],
            final_bytes=sb.storage()['persistent_tensor_bytes'],all_route_shadow_lanes=96,shadow_events=288,
            factual_loss_sum=bcounts['factual_loss_sum'],all_target_objective=float(objective.detach()),
            compared_nonzero_parameter_gradients=len(seq_grad)))
    names=['experiments/causal_language_replay_contracts.py','experiments/causal_language_replay_helpers.py',
        'sleeping_machines/causal_language_shadow.py','sleeping_machines/factorized_race.py',
        'sleeping_machines/batched_episodes.py','sleeping_machines/fast_native_core.py',
        'sleeping_machines/native_stream_language.py','sleeping_machines/addressed_event_heads.py',
        'sleeping_machines/parallel_head_race_language.py','sleeping_machines/parallel_stream_language.py',
        'sleeping_machines/sparse_race_language.py','sleeping_machines/operation_audit.py',
        'experiments/race_language_screen.py','experiments/theory/108_causal_language_shadow_credit_contract.md']
    r=dict(status='completed',args=vars(a),contracts_passed=len(checks),contracts=checks,families=families,
        work_audits=audits,source_sha256={name:N.sha(ROOT/name) for name in names},
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Small p4/L8 numerical full-write all-target causal language replay port; actual partial-window optimizer updates '
              'are correctness tests, not trained benchmark quality. Entering state detached; no full-stream exact-gradient claim, '
              '10M fit or changed AWS original-teacher source.')
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],contracts_passed=r['contracts_passed'],families=families,wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb'])))


if __name__=='__main__':main()
