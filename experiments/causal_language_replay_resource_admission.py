"""Production-shape resource and exact next-step admission, not a language fit."""
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
import causal_language_replay_contracts as C
from sleeping_machines.fast_native_core import FastNativeStreamLanguageModel
from race_language_screen import capture


def make(family):
    torch.manual_seed(7);m=FastNativeStreamLanguageModel(payload=16,depth=8,pool=2,heads=2)
    if family=='depth':
        for head in range(m.heads):
            master=m.units[0][head][0][0]
            for layer in m.units:
                for unit in layer[head][0]:
                    for name in ('input','output','gate','control','key_read'):setattr(unit,name,getattr(master,name))
    return m


def initial(model):
    with torch.random.fork_rng(),torch.no_grad():
        torch.manual_seed(20109);model.train();_,st=model.forward_chunk(torch.tensor([1,2,1,3,1]))
    return st.detach()


def traced_step(model,opt,ins,targets,st,seed,cache=None):
    opt.zero_grad(set_to_none=True);box={};stages={};old=H.batched_chunks
    if cache is not None:
        def tapped(*args,**kwargs):
            result=old(*args,**kwargs)
            if len(args[1])>1:cache['shadow_logits']=result[0].detach().clone()
            return result
        H.batched_chunks=tapped
    def forward():
        obj,z,new,act=H.batched_objective(model,ins,targets,st,seed)
        box.update(objective=obj,logits=z.detach().clone(),state=new,activity=act)
    try:stages['factual_and_all_shadows']=capture(forward)
    finally:H.batched_chunks=old
    stages['backward']=capture(lambda:box['objective'].backward())
    stages['normalize_clip_Adam']=capture(lambda:C.finish(model,opt,len(ins)))
    assert all(tr['formula_coverage_complete'] for tr in stages.values())
    box['state'].detach();box['stages']=stages
    return box


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--contracts',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    prior=json.loads((ROOT/a.contracts).read_text());assert prior['status']=='completed' and prior['contracts_passed']==16
    for name,digest in prior['source_sha256'].items():assert N.sha(ROOT/name)==digest
    torch.set_num_threads(1);begin=time.perf_counter();families=[];ledger=[];work_audits={}
    observed=torch.tensor([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3,2,1,4,2])
    inputs=observed[:16];targets=observed[1:17];seed=2019109
    for family in ('private','depth'):
        model=make(family);st=initial(model);reference=copy.deepcopy(model)
        original_params={n:v.detach().clone() for n,v in model.state_dict().items()}
        # Independent factual and four first-time alternatives BEFORE optimizer update.
        records=[]
        with torch.no_grad():
            losses,seq,sseq,srng=H.sequential(reference,inputs,targets,st,seed,record=records)
        cache={};opt=torch.optim.Adam(model.parameters(),lr=.002)
        box=traced_step(model,opt,inputs,targets,st,seed,cache)
        torch.testing.assert_close(box['logits'],seq,rtol=3e-5,atol=3e-6)
        C.state_close(box['state'],sseq)
        assert box['activity']['shadow_lanes']==512 and box['activity']['shadow_events']==8192
        alternative_checks=[]
        with torch.no_grad():
            for race in (0,15,127,255):
                other=1-int(records[race]['winner']);forced=[]
                alt_loss,alt,alt_state,end_rng=H.sequential(reference,inputs,targets,st,seed,
                    force=(race,other),record=forced)
                lane=race*2+other;event=race//16
                torch.testing.assert_close(cache['shadow_logits'][lane],alt,rtol=3e-5,atol=3e-6)
                C.close(records[race]['delay'],forced[race]['delay'],True);C.close(srng,end_rng,True)
                C.close(seq[:event],alt[:event],True)
                bat_loss=F.cross_entropy(cache['shadow_logits'][lane],targets,reduction='none')
                torch.testing.assert_close(bat_loss[event:].sum(),alt_loss[event:].sum(),rtol=3e-5,atol=3e-6)
                alternative_checks.append(dict(race=race,event=event,forced_winner=other,
                    true_suffix_loss=float(alt_loss[event:].sum()),batch_suffix_loss=float(bat_loss[event:].sum()),
                    first_time_preserved=True,earlier_losses_unchanged=True,end_rng_preserved=True,
                    actual_native_writes=alt_state.selected_updates-st.selected_updates))
        assert any(not torch.equal(original_params[n],v.detach()) for n,v in model.state_dict().items())
        # Nonempty Adam/current state checkpoint, then actual partial update both paths.
        buf=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),state=box['state'],
            cursor=dict(next_target=16,updates=1),rng=torch.get_rng_state()),buf);buf.seek(0)
        saved=torch.load(buf,weights_only=False);recovered=make(family);recovered.load_state_dict(saved['model'])
        ro=torch.optim.Adam(recovered.parameters(),lr=.002);ro.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        partial=traced_step(model,opt,observed[16:19],observed[17:20],box['state'],seed+1)
        torch.set_rng_state(saved['rng'])
        resumed=traced_step(recovered,ro,observed[16:19],observed[17:20],saved['state'],seed+1)
        C.close(partial['logits'],resumed['logits'],True);C.state_close(partial['state'],resumed['state'],True)
        for (name,p),(rn,r) in zip(model.named_parameters(),recovered.named_parameters()):assert name==rn;C.close(p,r,True)
        C.grads_close(C.grads(model),C.grads(recovered),True)
        for key,val in opt.state_dict()['state'].items():
            for name,v in val.items():C.close(v,ro.state_dict()['state'][key][name],True)
        assert saved['cursor']==dict(next_target=16,updates=1)
        # Audit sparse native inference from the same detached credit-boundary state.
        inference={}
        def infer():
            with torch.no_grad(),torch.random.fork_rng():
                torch.manual_seed(seed);model.eval();z,s=model.forward_chunk(inputs,copy.deepcopy(st))
                inference.update(logits=z,actual_state=s)
        itr=capture(infer);assert itr['formula_coverage_complete']
        work_audits[family]=dict(full=box['stages'],partial=partial['stages'],inference=itr)
        for label,count,traces,activity in [('full16',16,box['stages'],box['activity']),('partial3',3,partial['stages'],partial['activity'])]:
            total=sum(t['arithmetic_flops']+t['special_function_evaluations'] for t in traces.values())
            ledger.append(dict(family=family,window=label,targets=count,optimizer_updates=1,
                whole_step_gflops=total/1e9,fit_mflops_per_target=total/count/1e6,
                inference_mflops_per_target=(itr['arithmetic_flops']+itr['special_function_evaluations'])/16/1e6,
                **activity,scope='Synthetic correctness/resource step, no benchmark quality'))
        families.append(dict(family=family,parameters=sum(p.numel() for p in model.parameters()),depth=8,
            heads=2,payload=16,pool=2,credit_targets=16,available_receivers=32,key_scores_per_target=32,
            selected_writes_per_target=16,alternative_checks=alternative_checks,
            exact_nonempty_Adam_cursor_rng_private_state_recovery=True,
            full_factual_logits_equal_sequential=True,entering_state_bytes=st.storage()['persistent_tensor_bytes'],
            final_state_bytes=partial['state'].storage()['persistent_tensor_bytes']))
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1000000
    names=['experiments/causal_language_replay_resource_admission.py',
           'experiments/theory/109_production_language_replay_resource_admission.md']
    r=dict(status='completed',args=vars(a),source_sha256={**prior['source_sha256'],**{name:N.sha(ROOT/name) for name in names}},
        contract_result_sha256=N.sha(ROOT/a.contracts),families=families,common_unit_ledger=ledger,
        work_audits=work_audits,wall_s=time.perf_counter()-begin,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Production width/T16 full512-lane execution and actual next3-target recovery; eight alternative returns independently checked. '
              'ALL-parameter full sequential comparison remains the prior smaller T3/p4 contract, not full T16. '
              'Four test optimizer updates plus two repeated recovery updates across both arms are paid; no DEV/test/fit advantage.')
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',families=families,common_unit_ledger=ledger,wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb'])))


if __name__=='__main__':main()
