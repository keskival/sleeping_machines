"""Select affine decoders on fitting labels unseen by fixed-pass native producers."""
import argparse
import copy
import json
from pathlib import Path
import resource
import platform
import sys
import time
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import sleeping_machines.batched_addressed_fit as K
from dvs_frozen_feature_diagnostic import nll


def fit_head(x,y,lam):
    center=x.mean(0);scale=np.maximum(x.std(0),.5)
    C=1/(len(x)*lam)
    decoder=LogisticRegression(C=C,max_iter=1000,random_state=681)
    decoder.fit((x-center)/scale,y)
    assert list(decoder.classes_)==list(range(11))
    assert abs(C*len(x)*lam-1)<1e-12
    return dict(decoder=decoder,feature_center=center,feature_scale=scale,mean_l2=lam,C=C)


def predictions(head,x):
    return head['decoder'].predict_proba((x-head['feature_center'])/head['feature_scale'])


@torch.no_grad()
def features(model,rows,traced=False):
    xs=[];ps=[];work=[];captured=set();last=[]
    handle=model.head.register_forward_pre_hook(lambda module,args:last.append(args[0].detach()))
    try:
        for start in range(0,len(rows),32):
            batch=rows[start:start+32];last.clear();box={}
            def compute():box['z'],_,_=K.forward(model,batch,314159)
            if traced and len(batch) not in captured:
                tr=N.capture(compute);assert tr['formula_coverage_complete']
                work.append(dict(targets=len(batch),arithmetic_flops=tr['arithmetic_flops'],
                    special_function_evaluations=tr['special_function_evaluations'],random_draws=tr['exponential_random_draws']))
                captured.add(len(batch))
            else:compute()
            xs.append(last[-1].numpy().copy());ps.append(box['z'].softmax(-1).numpy().copy())
    finally:handle.remove()
    return np.concatenate(xs),np.concatenate(ps),work


@torch.no_grad()
def port_head(model,head,rows,expected):
    old={n:v.detach().clone() for n,v in model.state_dict().items()}
    coef=head['decoder'].coef_;center=head['feature_center'];scale=head['feature_scale']
    model.head.weight.copy_(torch.as_tensor(coef/scale,dtype=model.head.weight.dtype))
    model.head.bias.copy_(torch.as_tensor(head['decoder'].intercept_-(coef*center/scale).sum(1),dtype=model.head.bias.dtype))
    _,actual,work=features(model,rows,True)
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
    for index,(row,reference) in enumerate(zip(rows[:8],expected[:8])):
        if index==0:
            box={}
            def predict():box['z'],_=N.predict(model,row,314159,False)
            tr=N.capture(predict);assert tr['formula_coverage_complete'];z=box['z']
            work.append(dict(targets=1,arithmetic_flops=tr['arithmetic_flops'],
                special_function_evaluations=tr['special_function_evaluations'],random_draws=tr['exponential_random_draws']))
        else:z,_=N.predict(model,row,314159,False)
        np.testing.assert_allclose(z.softmax(-1).numpy(),reference,rtol=2e-5,atol=2e-6)
    for n,v in model.state_dict().items():
        if not n.startswith('head.'):assert torch.equal(old[n],v.detach())
    total=sum((w['arithmetic_flops']+w['special_function_evaluations'])*(6 if w['targets']==32 else 8) for w in work)
    return {n:v.detach().clone() for n,v in model.head.state_dict().items()},total


def selection_leakage_contract():
    # A binary producer stores seen labels as features. On unseen inputs the
    # feature is independent of the label: conditional CV and new-input risk
    # prefer opposite confidence. Enumerate the exact binary risks.
    ws=np.array([.5,2.,5.])
    conditional=np.logaddexp(0,-ws)
    held=.5*(np.logaddexp(0,ws)+np.logaddexp(0,-ws))
    assert np.all(np.diff(conditional)<0) and np.all(np.diff(held)>0)
    return dict(weights=ws.tolist(),conditional_seen_label_risks=conditional.tolist(),
        independent_new_input_risks=held.tolist(),opposite_confidence_preferences=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    started=time.perf_counter();torch.set_num_threads(1);toy=selection_leakage_contract()
    rows=[];artifacts={};source={};replay_work=[]
    grid=[1/(984*C) for C in (.1,1.,10.)]
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        config=argparse.Namespace(**parent['args'])
        if (config.fit,config.dev,config.epochs,config.update_targets)!=(256,192,4,16):raise ValueError('Fixed pilot required')
        for path,h in parent['source_sha256'].items():assert N.sha(ROOT/path)==h,('Changed source',path)
        source.update(parent['source_sha256']);cp=(ROOT/name).with_suffix('.progress.pt');saved=torch.load(cp,weights_only=False)
        assert saved['source_sha256']==parent['source_sha256'] and saved['cursor']['epoch']==5
        model=N.make_model(config);model.load_state_dict(saved['online_model']);model.eval()
        expanded=copy.copy(config);expanded.fit=984
        fit,dev,metadata=N.load(expanded);assert metadata==parent['data']
        before={n:v.detach().clone() for n,v in model.state_dict().items()};begin=time.perf_counter()
        xf,_,work=features(model,fit,True);xd,original,_=features(model,dev)
        np.testing.assert_allclose(original,parent['curve'][-1]['dev']['probabilities'],rtol=1e-5,atol=1e-6)
        assert all(torch.equal(before[n],v.detach()) for n,v in model.state_dict().items())
        labels=np.array([r['target'] for r in fit]);dev_labels=np.array([r['target'] for r in dev])
        folds=list(StratifiedKFold(3,shuffle=True,random_state=681).split(xf[:256],labels[:256]))
        candidates=[];begin_solver=time.perf_counter()
        for lam in grid:
            cv=np.empty((256,11))
            for training,held in folds:
                head=fit_head(xf[training],labels[training],lam);cv[held]=predictions(head,xf[held])
            head=fit_head(xf[:256],labels[:256],lam)
            untouched=predictions(head,xf[256:])
            candidates.append(dict(mean_l2=lam,nominal_C_for984=1/(984*lam),
                conditional_decoder_cv_nll=nll(cv,labels[:256]),producer_held_nll=nll(untouched,labels[256:]),
                producer_held_accuracy=float((untouched.argmax(1)==labels[256:]).mean())))
        selections={kind:min(candidates,key=lambda c:c[key]) for kind,key in
            [('conditional_cv','conditional_decoder_cv_nll'),('producer_held','producer_held_nll')]}
        qualities={};heads={};ported={};verification_work=0
        for kind,chosen in selections.items():
            head=fit_head(xf,labels,chosen['mean_l2']);prob=predictions(head,xd)
            qualities[kind]=dict(accuracy=float((prob.argmax(1)==dev_labels).mean()),nll=nll(prob,dev_labels),probabilities=prob.tolist())
            calibrated=copy.deepcopy(model);ported[kind],validation_work=port_head(calibrated,head,dev,prob)
            verification_work+=validation_work;heads[kind]=head
        known_replay=sum((w['arithmetic_flops']+w['special_function_evaluations'])*(36 if w['targets']==32 else 1) for w in work)
        replay_work.extend([dict(native=name,**w) for w in work]);artifacts[str(config.seed)]=dict(decoders=heads,ported_heads=ported)
        rows.append(dict(native=name,result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(cp),seed=config.seed,
            encoder_state='online_model after four fixed passes; no dev epoch selection',producer_fit_targets=256,
            producer_held_targets=728,final_decoder_fit_targets=984,development_targets=192,candidates=candidates,selections=selections,
            development=qualities,original_fixed_pass_development={k:parent['curve'][-1]['dev'][k] for k in ('accuracy','nll')},
            known_encoder_fit_gflops_estimate=parent['work']['whole_fit_unit_special_flops_estimate']/1e9,
            known_feature_replay_gflops_estimate=known_replay/1e9,additional_solver_fit_flops=None,
            known_validation_replay_gflops_estimate=verification_work/1e9,
            replay_wall_s=begin_solver-begin,decoder_and_port_wall_s=time.perf_counter()-begin_solver,
            nonhead_parameters_preserved=True,inference_architecture_unchanged=True,parameters=parent['parameters']))
        print(json.dumps(dict(seed=config.seed,selections={k:v['nominal_C_for984'] for k,v in selections.items()},
            dev={k:{m:v[m] for m in ('accuracy','nll')} for k,v in qualities.items()})),flush=True)
    artifact=out.with_suffix('.heads.joblib');joblib.dump(artifacts,artifact)
    own=['experiments/dvs_producer_held_decoder.py','experiments/dvs_frozen_feature_diagnostic.py',
        'sleeping_machines/batched_addressed_fit.py','experiments/theory/96_producer_held_selection_and_mean_regularization.md']
    source.update({path:N.sha(ROOT/path) for path in own})
    result=dict(status='completed',args=vars(a),rows=rows,selection_leakage_witness=toy,
        mean_l2_grid=grid,contracts_passed=4,source_sha256=source,stage_work=replay_work,formula_coverage_complete=True,
        decoder_artifact=str(artifact.relative_to(ROOT)),decoder_artifact_sha256=N.sha(artifact),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(host=platform.node(),platform=platform.platform(),torch_threads=torch.get_num_threads(),device='CPU'),
        scope='Frozen fixed-four-pass integrated native producer, conditional CV versus producer-unseen FIT selection, '
            'mean-regularization coefficient invariant across sample sizes. Final affine heads use all984 FIT labels, '
            'so unequal-data versus saved256-only pilot; no iso-work superiority. Head ports preserve all nonhead parameters and '
            'native architecture; generic solver/transform/calibration arithmetic unknown. Dev still exploratory, no official test. '
            'Feature and additional head-port verification core replays recorded separately, not zero.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
