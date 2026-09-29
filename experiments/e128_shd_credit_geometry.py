"""Fitting-only class/speaker credit geometry and finite-direction checks.

No official or held-out utterances are opened. Two disjoint fitting-speaker
groups supply equal numbers of examples from each class. The checkpoint is
fixed; only zero-initialized context columns are probed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from e117_serial_event_shd import batch,load_items
import e51_shd_world as S


def cosine(a,b):
    return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))


def common_descent_certificate(gradients):
    """Feasible direction lower bound and feasible convex-mixture upper bound.

    Frank-Wolfe approximates the minimum-norm point. No optimizer convergence
    assumption is needed for the reported two finite-iterate bounds.
    """
    g=gradients.astype(np.float64);gram=g@g.T;n=len(g)
    weights=np.ones(n)/n
    for iteration in range(10000):
        dots=gram@weights;norm2=float(weights@dots);j=int(np.argmin(dots))
        gap=norm2-float(dots[j])
        if gap <= 1e-10*max(1.,norm2):break
        denom=norm2-2*dots[j]+gram[j,j]
        step=np.clip(gap/max(denom,1e-30),0,1)
        weights*=1-step;weights[j]+=step
    v=weights@g;norm=float(np.linalg.norm(v))
    lower=max(0.,float((g@v).min()/max(norm,1e-30)))
    return {"lower_common_improvement_per_unit_radius":lower,
            "upper_common_improvement_per_unit_radius":norm,
            "mixture_weights":weights.tolist(),"iterations":iteration+1,
            "strict_common_direction_certified":lower>1e-8,
            "scope":"Exact pathwise gradients only; fixed differentiable route/clock region; finite-iterate bounds"}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--tag",required=True)
    ap.add_argument("--per-class-group",type=int,default=6)
    ap.add_argument("--bs",type=int,default=4);args=ap.parse_args()
    if Path(args.tag).name!=args.tag or min(args.per_class_group,args.bs)<1:raise ValueError("Invalid settings")
    out=Path("experiments/results/e128")/(args.tag+".json");out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.manual_seed(6);started=time.perf_counter()
    source=Path("experiments/results/e122/d8_n4096_invariance_continue_s6_e2.pt")
    saved=torch.load(source,weights_only=False,map_location="cpu")
    net=SharedEventModel(global_context_layers=(3,7))
    missing,unexpected=net.load_state_dict(saved['state_dict'],strict=False)
    assert not unexpected and len(missing)==4 and all('.bridge_' in n for n in missing)
    named=dict(net.named_parameters());params=[named[n] for n in missing]
    for n,p in named.items():p.requires_grad_(n in missing)
    slices={};start=0
    for name,p in zip(missing,params):slices[name]=(start,start+p.numel());start+=p.numel()
    old_state={n:p.detach().clone() for n,p in net.named_parameters() if n not in missing}
    fit=load_items(40,.01,4096,"fit_spk",6)
    with h5py.File(Path(S.ROOT)/'shd_train.h5','r') as f:
        speakers=np.array(f['extra']['speaker']);labels=np.array(f['labels'])
    mask=~np.isin(speakers,S.VAL_SPEAKERS);speaker_by_id=speakers[mask];label_by_id=labels[mask]
    speaker_ids=sorted(set(int(speaker_by_id[r[4]]) for r in fit))
    groups=[speaker_ids[::2],speaker_ids[1::2]]
    selected=[];group_index=[]
    for label in range(20):
        for group,spks in enumerate(groups):
            candidates=[r for r in fit if r[3]==label and int(speaker_by_id[r[4]]) in spks]
            if len(candidates)<args.per_class_group:raise ValueError((label,group,len(candidates)))
            selected.extend(candidates[:args.per_class_group]);group_index.extend([group]*args.per_class_group)
    y=np.array([r[3] for r in selected]);group_index=np.array(group_index)
    assert all(label_by_id[r[4]]==r[3] for r in selected)
    assert not set(groups[0])&set(groups[1]) and not set(speaker_ids)&set(S.VAL_SPEAKERS)
    record={"status":"running","args":vars(args),"checkpoint":str(source),
            "checkpoint_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "fitting_speaker_groups":groups,"sample_ids":[r[4] for r in selected],
            "sample_labels":y.tolist(),"sample_group":group_index.tolist(),"new_parameters":missing,
            "scope":"Fitting-only equal-class samples in disjoint fitting-speaker groups; no held-out or official-test access; diagnostic perturbations, no fitted model claim"}
    def persist():
        record.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out)
    persist()
    gradients={};losses={}
    for mode in ('pathwise','surrogate'):
        net.train()
        for layer in net.layers:layer.cf_credit=mode=='surrogate'
        rows=[];loss_rows=[]
        for offset in range(0,len(selected),args.bs):
            part=selected[offset:offset+args.bs];packed=batch(part)
            z=net(*packed[:4],len(part),trace=True)[0]
            ll=F.cross_entropy(z,packed[-1],reduction='none');loss_rows.extend(ll.detach().tolist())
            for i,loss in enumerate(ll):
                grad=torch.autograd.grad(loss,params,retain_graph=i<len(ll)-1,allow_unused=True)
                rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                       for p,g in zip(params,grad)]).detach().numpy())
            if offset%(10*args.bs)==0:print(json.dumps({'mode':mode,'examples':offset+len(part)}),flush=True)
        gradients[mode]=np.stack(rows).astype(np.float64);losses[mode]=np.array(loss_rows)
    assert np.array_equal(losses['pathwise'],losses['surrogate'])
    class_grad={mode:np.stack([g[y==c].mean(0) for c in range(20)]) for mode,g in gradients.items()}
    class_group_grad={mode:np.stack([np.stack([g[(y==c)&(group_index==j)].mean(0) for c in range(20)])
                                    for j in range(2)]) for mode,g in gradients.items()}
    # Hard classes are selected from fitting accuracy at the saved parent, never held-out scores.
    findings=Path('experiments/results/e125/class_summary_20260929.json')
    fit_classes=json.loads(findings.read_text())['rows']['starting_checkpoint']['fit']['classes']
    hard=[r['class'] for r in sorted(fit_classes,key=lambda r:r['accuracy'])[:2]]
    record['hard_classes_from_parent_fit']=hard
    record['class_mean_loss']=[float(losses['pathwise'][y==c].mean()) for c in range(20)]
    record['geometry']={}
    for mode,g in gradients.items():
        cg=class_grad[mode];a,b=class_group_grad[mode]
        mean=cg.mean(0);norms=np.linalg.norm(g,axis=1)
        record['geometry'][mode]={
            'class_gradient_norm':np.linalg.norm(cg,axis=1).tolist(),
            'class_gram':(cg@cg.T).tolist(),
            'speaker_group_cross_gram':(a@b.T).tolist(),
            'same_class_speaker_cosine':[cosine(a[c],b[c]) for c in range(20)],
            'balanced_speaker_group_cosine':cosine(a.mean(0),b.mean(0)),
            'mean_gradient_norm':float(np.linalg.norm(mean)),
            'cancellation_ratio_mean_norm_over_mean_individual_norm':float(np.linalg.norm(mean)/norms.mean()),
            'class_derivative_under_balanced_unit_descent':(-cg@mean/max(np.linalg.norm(mean),1e-30)).tolist(),
            'class_derivative_under_hard_unit_descent':(-cg@cg[hard].mean(0)/max(np.linalg.norm(cg[hard].mean(0)),1e-30)).tolist()}
    record['surrogate_pathwise_class_cosine']=[cosine(class_grad['pathwise'][c],class_grad['surrogate'][c]) for c in range(20)]
    record['common_descent_certificate']=common_descent_certificate(class_grad['pathwise'])
    record['hard_common_descent_certificate']=common_descent_certificate(class_grad['pathwise'][hard])
    record['parameter_block_norms']={mode:{name:float(np.linalg.norm(g.mean(0)[a:b]))
                                          for name,(a,b) in slices.items()} for mode,g in gradients.items()}
    persist()

    @torch.no_grad()
    def finite_evaluate():
        net.eval();loss_values=[];winner_values=[];orders=[]
        for offset in range(0,len(selected),args.bs):
            part=selected[offset:offset+args.bs];packed=batch(part)
            z,_,_,trace=net(*packed[:4],len(part),trace=True)
            loss_values.extend(F.cross_entropy(z,packed[-1],reduction='none').tolist())
            winner_values.append(torch.stack([r['winner'] for r in trace]))
            for r in trace:
                for j in range(len(part)):
                    tt=r['times'][packed[3]==j]
                    orders.append(torch.argsort(tt,stable=True))
        return np.array(loss_values),torch.cat(winner_values,1),orders
    baseline,base_winners,base_orders=finite_evaluate()
    assert np.array_equal(baseline,losses['pathwise'])
    directions={
        'balanced_pathwise':-class_grad['pathwise'].mean(0),
        'balanced_surrogate':-class_grad['surrogate'].mean(0),
        'hard_pathwise':-class_grad['pathwise'][hard].mean(0)}
    probes=[]
    for name,direction in directions.items():
        direction=direction/max(np.linalg.norm(direction),1e-30)
        predicted=class_grad['pathwise']@direction
        for radius in (.001,.01):
            with torch.no_grad():
                for param,n in zip(params,missing):
                    a,b=slices[n];param.copy_(torch.as_tensor(radius*direction[a:b],dtype=param.dtype).reshape_as(param))
            changed,winners,orders=finite_evaluate()
            delta=np.array([(changed-baseline)[y==c].mean() for c in range(20)])
            probes.append({'direction':name,'radius_l2_new_parameters':radius,
                           'predicted_class_loss_change':(radius*predicted).tolist(),
                           'observed_class_loss_change':delta.tolist(),
                           'balanced_loss_change':float((changed-baseline).mean()),
                           'hard_class_loss_change':float(delta[hard].mean()),
                           'winner_disagreements':int((winners!=base_winners).sum()),
                           'query_layer_time_order_changes':sum(not torch.equal(a,b) for a,b in zip(orders,base_orders))})
    with torch.no_grad():
        for p in params:p.zero_()
    assert all(torch.equal(p,old_state[n]) for n,p in net.named_parameters() if n not in missing)
    record.update(status='completed',finite_probes=probes,old_parameters_exactly_unchanged=True,
                  hardware={'platform':platform.platform(),'torch':torch.__version__,'threads':1,'device':'cpu'},
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__),Path('sleeping_machines/shared_event.py'),Path('sleeping_machines/event_memory.py'),findings)})
    persist()
    np.savez_compressed(out.with_suffix('.npz'),**gradients,labels=y,group=group_index)
    print(json.dumps({'hard_classes':hard,'mean_gradient_norms':{k:v['mean_gradient_norm'] for k,v in record['geometry'].items()},
                      'speaker_cosines':{k:v['balanced_speaker_group_cosine'] for k,v in record['geometry'].items()},
                      'common_descent_certificate':record['common_descent_certificate'],
                      'finite_balanced_loss_changes':[r['balanced_loss_change'] for r in probes],
                      'wall_s':record['wall_s'],'max_rss_kb':record['max_rss_kb']}),flush=True)


if __name__=='__main__':main()
