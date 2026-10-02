"""Read-only native static/dynamic-key score decomposition and explicit RNG audit."""
import argparse
import copy
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import types
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from dvs_native_window_intervention import equal_state
from dvs_race_support_profile import collector


@torch.no_grad()
def replay(model,row,seed):
    model.eval();state=model.new_state()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for timestamp,content in row['events']:z,_=model.consume_event(0,timestamp,content,state)
        rng=torch.get_rng_state().clone()
    return z,state,rng


def instrument(model,records):
    query={};memory={};hooks=[]
    for depth in range(model.depth):
        for head in range(model.heads):
            def qhook(module,args,out,d=depth,h=head):query[d,h]=out.detach()
            hooks.append(model.queries[depth][head].register_forward_hook(qhook))
            for index,unit in enumerate(model.units[depth][head][0]):
                def mhook(module,args,out,d=depth,h=head,i=index):memory[d,h,i]=(args[0].detach(),out.detach())
                hooks.append(unit.key_read.register_forward_hook(mhook))
    original=model.race
    def race(self,scores,values=None):
        event=len(records)//4;depth=(len(records)%4)//2;head=len(records)%2
        q=query[depth,head];static=[];dynamic=[];norms=[]
        for i,unit in enumerate(self.units[depth][head][0]):
            m,read=memory[depth,head,i]
            static.append(q@unit.key/math.sqrt(self.payload)+unit.clock_bias)
            dynamic.append(q@read/math.sqrt(self.payload));norms.append(m.norm())
        static=torch.stack(static);dynamic=torch.stack(dynamic);unclamped=static+dynamic
        torch.testing.assert_close(unclamped.clamp(-12,12),scores,rtol=2e-5,atol=2e-5)
        records.append(dict(site=(event,depth,head),scores=scores.detach().numpy().copy(),
            static=static.numpy().copy(),dynamic=dynamic.numpy().copy(),memory_norm=torch.stack(norms).numpy().copy(),
            reconstruction_error=float((unclamped.clamp(-12,12)-scores).abs().max())))
        return original(scores,values)
    model.race=types.MethodType(race,model)
    return hooks


def summarize(records):
    score=np.array([r['scores'] for r in records],dtype=np.float64)
    static=np.array([r['static'] for r in records],dtype=np.float64)
    dynamic=np.array([r['dynamic'] for r in records],dtype=np.float64)
    norm=np.array([r['memory_norm'] for r in records],dtype=np.float64);site=np.array([r['site'] for r in records])
    def stats(mask):
        result=dict(races=int(mask.sum()),mean_absolute_static_gap=float(np.abs(np.diff(static[mask],axis=1)).mean()),
            mean_absolute_dynamic_gap=float(np.abs(np.diff(dynamic[mask],axis=1)).mean()),
            mean_memory_norm=float(norm[mask].mean()),memory_norm_q05_q50_q95=np.quantile(norm[mask],[.05,.5,.95]).tolist(),
            memory_gap_larger_than_static_fraction=float((np.abs(np.diff(dynamic[mask],axis=1))>np.abs(np.diff(static[mask],axis=1))).mean()))
        for name,x in [('factual',score),('static_only',np.clip(static,-12,12)),('dynamic_only',np.clip(dynamic,-12,12))]:
            pi=torch.tensor(x[mask]).softmax(-1)
            result[name]=dict(mean_normalized_entropy=float((-(pi*pi.log()).sum(-1)/np.log(2)).mean()),
                top_probability99_fraction=float((pi.max(-1).values>=.99).double().mean()))
        return result
    summaries=dict(all_sites=stats(np.ones(len(records),dtype=bool)),first_event=stats(site[:,0]==0),later_events=stats(site[:,0]>0),
        by_layer_head={f'layer{d}_head{h}':stats((site[:,1]==d)&(site[:,2]==h)) for d in range(2) for h in range(2)})
    return summaries,dict(scores=score,static=static,dynamic=dynamic,memory_norm=norm,site=site)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);p.add_argument('--profile',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    begin=time.perf_counter();torch.set_num_threads(1)
    profile=json.loads((ROOT/a.profile).read_text());assert profile['status']=='completed'
    for path,digest in profile['source_sha256'].items():assert N.sha(ROOT/path)==digest
    indices=profile['indices'];rows=[];sources={};parents=[];arrays={};rngchecks=0
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        for path,digest in parent['source_sha256'].items():assert N.sha(ROOT/path)==digest
        sources.update(parent['source_sha256']);config=argparse.Namespace(**parent['args'])
        expanded=copy.copy(config);expanded.fit=984;fit,_,metadata=N.load(expanded);assert metadata==parent['data']
        selected=[fit[i] for i in indices];cp=(ROOT/name).with_suffix('.progress.pt');saved=torch.load(cp,weights_only=False)
        assert saved['cursor']['epoch']==5 and saved['source_sha256']==parent['source_sha256']
        initial=N.make_model(config,fast=False).state_dict()
        parents.append(dict(result=name,result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(cp)))
        for encoder,weights in [('initial',initial),('fixed_pass4',saved['online_model'])]:
            model=N.make_model(config,fast=False);model.load_state_dict(weights)
            z,st,rng=replay(model,selected[0],314159);before={n:v.detach().clone() for n,v in model.state_dict().items()}
            records=[];hooks=instrument(model,records)
            zi,sti,rngi=replay(model,selected[0],314159)
            assert torch.equal(z,zi) and torch.equal(rng,rngi);equal_state(st,sti);rngchecks+=1
            records.clear();work=dict(prefixes=0,events=0,key_scores=0,selected_updates=0)
            for row in selected:
                local=[];old=len(records);_,state,_=replay(model,row,314159)
                assert len(records)-old==84
                # The collector's event counter is local to each prefix.
                for record in records[old:]:
                    e,d,h=record['site'];record['site']=(e-old//4,d,h);record['fit_index']=row['index']
                work['prefixes']+=1;work['events']+=state.events
                work['key_scores']+=state.candidate_scores;work['selected_updates']+=state.selected_updates
            for hook in hooks:hook.remove()
            assert all(torch.equal(before[n],v.detach()) for n,v in model.state_dict().items())
            summary,numeric=summarize(records);key=f's{config.seed}_{encoder}'
            for suffix,value in numeric.items():arrays[key+'_'+suffix]=value
            arrays[key+'_fit_index']=np.array([r['fit_index'] for r in records])
            # Independent end-RNG check of the already completed99 collector.
            model=N.make_model(config,fast=False);model.load_state_dict(weights);buffer=[];collector(model,buffer)
            zp,stp,rngp=replay(model,selected[0],314159)
            assert torch.equal(z,zp) and torch.equal(rng,rngp);equal_state(st,stp);rngchecks+=1
            rows.append(dict(seed=config.seed,encoder=encoder,observed_work=work,
                maximum_clamped_score_reconstruction_error=max(r['reconstruction_error'] for r in records),**summary))
            print(json.dumps(dict(seed=config.seed,encoder=encoder,all_sites=summary['all_sites'],first=summary['first_event'])),flush=True)
    artifact=out.with_suffix('.decomposition.npz');np.savez_compressed(artifact,**arrays)
    own=['experiments/dvs_key_score_decomposition.py','experiments/theory/100_memory_keys_and_clock_safe_calibration.md',
        'experiments/dvs_race_support_profile.py','experiments/dvs_native_window_intervention.py']
    sources.update({name:N.sha(ROOT/name) for name in own})
    result=dict(status='completed',args=vars(a),profile_result_sha256=N.sha(ROOT/a.profile),parents=parents,rows=rows,
        exact_logit_state_end_rng_contracts_passed=rngchecks,all_weights_bitwise_preserved=True,
        scalar_score_decompositions=5376,indices=indices,noise_seed=314159,optimizer_steps=0,development_evaluations=0,
        decomposition_artifact=str(artifact.relative_to(ROOT)),decomposition_artifact_sha256=N.sha(artifact),
        source_sha256=sources,whole_audit_flops=None,wall_s=time.perf_counter()-begin,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(host=platform.node(),device='CPU',torch_threads=1),
        scope='Read-only exact local native score decomposition into static key+bias and persistent-memory key read. '
            'All component-only probabilities hold entering query fixed, not an actual altered prefix or quality claim. '
            'One noise history,16 producer-unseen FIT examples,two fixed-pass producers+initial reservoirs. '
            'Eight explicit final-logit/state/end-RNG contracts supplement99 without revising it. '
            'No target loss, optimizer, normalization fit, DEV/test evaluation or inferred bad routing. '
            'Audit and original fits paid; full FLOPs/traffic/energy unmeasured, not zero.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
