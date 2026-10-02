"""Frozen native one-site multi-message reception on producer-unseen FIT inputs."""
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
from sleeping_machines.native_window_diagnostic import NativeWindowDiagnostic


def equal_state(a,b):
    assert (a.events,a.candidate_scores,a.selected_updates,a.visited_units)==(b.events,b.candidate_scores,b.selected_updates,b.visited_units)
    for key in ('memories','arrivals','contexts'):
        x,y=getattr(a,key),getattr(b,key);assert x.keys()==y.keys()
        for name in x:
            if key=='contexts':
                for left,right in zip(x[name],y[name]):assert torch.equal(left,right)
            else:assert torch.equal(x[name],y[name])


def make(config,state):
    torch.manual_seed(config.seed)
    m=NativeWindowDiagnostic(sources=1,content_dim=33,classes=11,payload=config.payload,depth=config.depth,heads=config.heads,pool=config.pool)
    m.load_state_dict(state);m.eval();return m


def contracts(config,weights,row):
    reference=N.make_model(config,fast=False);reference.load_state_dict(weights)
    candidate=make(config,weights)
    z,st=N.predict(reference,row,314159,False);zc,sc=N.predict(candidate,row,314159,False)
    assert torch.equal(z,zc);equal_state(st,sc)
    for model in (reference,candidate):model.zero_grad(set_to_none=True)
    z,_=N.predict(reference,row,314159,True);zc,_=N.predict(candidate,row,314159,True)
    F.cross_entropy(z[None],torch.tensor([row['target']])).backward()
    F.cross_entropy(zc[None],torch.tensor([row['target']])).backward()
    assert torch.equal(z,zc)
    for (name,x),(other,y) in zip(reference.named_parameters(),candidate.named_parameters()):
        assert name==other and (x.grad is None)==(y.grad is None)
        if x.grad is not None:assert torch.equal(x.grad,y.grad),name
    candidate.window_width=.011;candidate.window_mode='window_sum'
    with torch.no_grad():
        full,sfull=N.predict(candidate,row,314159,False)
        assert sfull.window_diagnostic['heard']==[0,1] and sfull.selected_updates==st.selected_updates+1
        candidate.window_mode='delivery_only_sum';_,sdelivery=N.predict(candidate,row,314159,False)
        assert sdelivery.selected_updates==st.selected_updates and sdelivery.window_diagnostic['projected_values']==2
        # At the immediate intervention boundary only the full mode commits the
        # second real receiver. Later query feedback is permitted to differ.
        prefix=copy.deepcopy(row);prefix['events']=prefix['events'][:20]
        _,one=N.predict(candidate,prefix,314159,False)
        candidate.window_mode='window_sum';_,both=N.predict(candidate,prefix,314159,False)
        assert any(not torch.equal(both.memories[k],one.memories[k]) for k in both.memories)
        early=copy.deepcopy(row);early['events']=early['events'][:19]
        r,sr=N.predict(reference,early,314159,False);c,ss=N.predict(candidate,early,314159,False)
        assert torch.equal(r,c);equal_state(sr,ss)
        swapped=copy.deepcopy(row);swapped['target']=(row['target']+1)%11
        zs,_=N.predict(candidate,swapped,314159,False);assert torch.equal(full,zs)
    candidate.train(True)
    try:N.predict(candidate,row,314159,True)
    except ValueError as e:assert 'training not installed' in str(e)
    else:raise AssertionError('Positive width training must be refused')
    return 6


@torch.no_grad()
def evaluate(model,rows):
    cases=[];trace=None;prob=[];loss=[]
    for index,row in enumerate(rows):
        box={}
        def predict():box['z'],box['state']=N.predict(model,row,314159,False)
        if index==0:
            trace=N.capture(predict);assert trace['formula_coverage_complete']
        else:predict()
        z=box['z'];state=box['state'];p=z.softmax(-1)
        nll=float(F.cross_entropy(z[None],torch.tensor([row['target']])));loss.append(nll);prob.append(p.tolist())
        ready=float(state.contexts[0][1].max());di=getattr(state,'window_diagnostic',None)
        cases.append(dict(index=row['index'],identity=row['identity'],target=row['target'],nll=nll,
            predicted=int(z.argmax()),key_scores=state.candidate_scores,selected_updates=state.selected_updates,
            final_ready_time=ready,window=di,state_tensor_bytes=state.storage()['persistent_tensor_bytes']))
    return dict(accuracy=float(np.mean([x['predicted']==x['target'] for x in cases])),nll=float(np.mean(loss)),
        probabilities=prob,cases=cases,representative_first_prefix_core_arithmetic_flops=trace['arithmetic_flops'],
        representative_first_prefix_special_evaluations=trace['special_function_evaluations'],
        representative_formula_coverage_complete=trace['formula_coverage_complete'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    torch.set_num_threads(1);begin=time.perf_counter();allrows=[];parents=[];checks=0;sources={};gates=[]
    configurations=[(0.,'winner_only')]+[(h,mode) for h in (.001,.003) for mode in NativeWindowDiagnostic.modes]
    indices=np.linspace(256,983,16,dtype=int).tolist()
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        config=argparse.Namespace(**parent['args'])
        assert (config.fit,config.dev,config.epochs,config.pool,config.depth,config.heads)==(256,192,4,2,2,2)
        for filename,digest in parent['source_sha256'].items():assert N.sha(ROOT/filename)==digest
        sources.update(parent['source_sha256']);expanded=copy.copy(config);expanded.fit=984
        fit,_,data=N.load(expanded);assert data==parent['data'];rows=[fit[i] for i in indices]
        assert all(len(x['events'])==21 and x['events'][19][0]==1. and x['events'][-1][1][-1]==1 for x in rows)
        checkpoint=(ROOT/name).with_suffix('.progress.pt');saved=torch.load(checkpoint,weights_only=False)
        assert saved['cursor']['epoch']==5 and saved['source_sha256']==parent['source_sha256']
        initial=N.make_model(config,fast=False).state_dict()
        parents.append(dict(result=name,result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(checkpoint),
            seed=config.seed,original_trained_core_fit_gflops=parent['work']['whole_fit_unit_special_flops_estimate']/1e9))
        checks+=contracts(config,saved['online_model'],rows[0])
        for encoder,weights in [('initial',initial),('fixed_pass4',saved['online_model'])]:
            model=make(config,weights);before={n:v.detach().clone() for n,v in model.state_dict().items()}
            outcomes=[]
            for width,mode in configurations:
                model.window_width=width;model.window_mode=mode
                quality=evaluate(model,rows)
                outcomes.append(dict(seed=config.seed,encoder=encoder,width=width,mode=mode,**quality))
                print(json.dumps(dict(seed=config.seed,encoder=encoder,width=width,mode=mode,nll=quality['nll'],accuracy=quality['accuracy'])),flush=True)
            assert all(torch.equal(before[n],v.detach()) for n,v in model.state_dict().items())
            allrows.extend(outcomes)
            if encoder=='fixed_pass4':
                w=next(r for r in outcomes if r['width']==.001 and r['mode']=='window_mean')
                control=next(r for r in outcomes if r['width']==.001 and r['mode']=='winner_wait')
                gain=control['nll']-w['nll'];accuracy_gain=w['accuracy']-control['accuracy']
                gates.append(dict(seed=config.seed,window_mean_nll_gain_over_wait=gain,accuracy_gain=accuracy_gain,
                    smoke_admission_passed=gain>=.02 and accuracy_gain>=0.))
    own=['sleeping_machines/native_window_diagnostic.py','sleeping_machines/race_window.py',
        'experiments/dvs_native_window_intervention.py','experiments/theory/98_frozen_native_reception_intervention.md']
    sources.update({f:N.sha(ROOT/f) for f in own})
    result=dict(status='completed',args=vars(a),parents=parents,rows=allrows,gates=gates,
        integrated_smoke_admission_passed=all(g['smoke_admission_passed'] for g in gates),
        contract_checks_passed=checks,indices=indices,unused_fit_targets=16,development_evaluations=0,optimizer_steps=0,
        configuration_count=len(configurations),model_configurations=len(allrows),source_sha256=sources,
        whole_audit_flops=None,wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(host=platform.node(),device='CPU',torch_threads=1),
        scope='Frozen one-site actual-native inference intervention, two fixed-pass trained producers and initial reservoirs, '
            '16 producer-unseen FIT inputs, all eleven configurations, waiting/amplitude/delivery-only controls. '
            'Zero-width forward/state/all-gradient nesting; real extra writes and causal transport. No fitting/dev/test '
            'evaluation, full native boundary estimator, natural-silence result or benchmark advantage. '
            'Representative traced prefix costs are not total-audit FLOPs; original core fits retained, '
            'remaining arithmetic/traffic/energy unmeasured, not zero.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
