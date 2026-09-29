"""A fitting-only class/speaker constrained teacher for the new causal context.

Retain counterfactual credit up to the amount allowed by exact local descent
constraints. Verify each finite update on the hard computation. Select no
hyperparameter or checkpoint from held-out results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from e117_serial_event_shd import batch,load_items
from e118_race_carrier_shd import evaluate
from e128_shd_credit_geometry import common_descent_certificate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--steps',type=int,default=4);ap.add_argument('--radius',type=float,default=.05)
    ap.add_argument('--bs',type=int,default=4);a=ap.parse_args()
    if Path(a.tag).name!=a.tag or min(a.steps,a.bs)<1 or a.radius<=0:raise ValueError('Invalid settings')
    out=Path('experiments/results/e129')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.manual_seed(6);started=time.perf_counter()
    audit_path=Path('experiments/results/e128/class_speaker_geometry_20260929.json')
    audit=json.loads(audit_path.read_text());assert audit['status']=='completed'
    source=Path(audit['checkpoint']);saved=torch.load(source,weights_only=False,map_location='cpu')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==audit['checkpoint_sha256']
    net=SharedEventModel(global_context_layers=(3,7))
    missing,unexpected=net.load_state_dict(saved['state_dict'],strict=False)
    assert not unexpected and missing==audit['new_parameters']
    named=dict(net.named_parameters());params=[named[n] for n in missing]
    for n,p in named.items():p.requires_grad_(n in missing)
    old_state={n:p.detach().clone() for n,p in named.items() if n not in missing}
    fit=load_items(40,.01,4096,'fit_spk',6);by_id={r[4]:r for r in fit}
    selected=[by_id[j] for j in audit['sample_ids']]
    labels=np.array(audit['sample_labels']);groups=np.array(audit['sample_group'])
    assert all(r[3]==y for r,y in zip(selected,labels))
    masks=[(labels==c)&(groups==j) for c in range(20) for j in range(2)]
    counts=np.array([m.sum() for m in masks]);assert bool((counts==counts[0]).all())
    archive=np.load(audit_path.with_suffix('.npz'))
    initial_grad={k:archive[k].astype(np.float64) for k in ('pathwise','surrogate')}
    assert np.array_equal(archive['labels'],labels) and np.array_equal(archive['group'],groups)
    source_json=json.loads(source.with_suffix('.json').read_text())
    result={'status':'running','args':vars(a),'curve':[],
            'initial':{k:source_json['final'][k] for k in ('fit','dev_original','dev_additional')},
            'fitting_sample_ids':audit['sample_ids'],'fitting_sample_classes':labels.tolist(),
            'fitting_speaker_groups':audit['fitting_speaker_groups'],
            'protocol':'One fixed fitting-only balanced sample, 40 class/speaker conditions; old parameters fixed; deterministic finite hard-loss line search; held-out evaluated once at final state; official test untouched',
            'teacher':'minimum-norm convex mixture of normalized exact conditional gradients, plus bounded mixture of counterfactual increments',
            'line_search':'dyadic radii from declared maximum; at most 12 trials; all 40 conditional loss changes <= 1e-7 and balanced loss strictly decreases',
            'checkpoint_sha256':audit['checkpoint_sha256'],
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (Path(__file__),audit_path,audit_path.with_suffix('.npz'),Path('sleeping_machines/shared_event.py'))}}
    def persist():
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out)
    persist()

    @torch.no_grad()
    def fitting_losses():
        net.eval();values=[]
        for start in range(0,len(selected),a.bs):
            part=selected[start:start+a.bs];packed=batch(part)
            z=net(*packed[:4],len(part),trace=True)[0]
            values.extend(F.cross_entropy(z,packed[-1],reduction='none').tolist())
        values=np.array(values)
        return np.array([values[m].mean() for m in masks])

    def parameter_vector():return torch.cat([p.detach().flatten() for p in params]).clone()

    @torch.no_grad()
    def set_parameters(vector):
        start=0
        for p in params:
            p.copy_(vector[start:start+p.numel()].reshape_as(p));start+=p.numel()

    def teachers():
        vectors={}
        for mode in ('pathwise','surrogate'):
            net.train()
            for layer in net.layers:layer.cf_credit=mode=='surrogate'
            rows=[]
            for start in range(0,len(selected),a.bs):
                part=selected[start:start+a.bs];packed=batch(part)
                z=net(*packed[:4],len(part),trace=True)[0]
                losses=F.cross_entropy(z,packed[-1],reduction='none')
                for j,loss in enumerate(losses):
                    gg=torch.autograd.grad(loss,params,retain_graph=j<len(losses)-1,allow_unused=True)
                    rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                           for p,g in zip(params,gg)]).detach().numpy())
            vectors[mode]=np.stack(rows).astype(np.float64)
        return vectors

    previous=fitting_losses()
    for step in range(1,a.steps+1):
        grads=initial_grad if step==1 else teachers()
        g=np.stack([grads['pathwise'][m].mean(0) for m in masks])
        s=np.stack([grads['surrogate'][m].mean(0) for m in masks])
        norms=np.linalg.norm(g,axis=1).clip(1e-12)
        normalized=g/norms[:,None]
        certificate=common_descent_certificate(normalized)
        if not certificate['strict_common_direction_certified']:
            result['stop_reason']='No strict common pathwise direction certified in this fitting sample';break
        weights=np.array(certificate['mixture_weights'])
        direct=-(weights@normalized)
        counterfactual=-(weights@((s-g)/norms[:,None]))
        first=g@direct;extra=g@counterfactual
        assert bool((first<0).all())
        # Preserve at least half of each exact first-order improvement.
        positive=extra>0
        strength=min(1.,float(np.min(-first[positive]/(2*extra[positive])))) if positive.any() else 1.
        direction=direct+strength*counterfactual
        direction/=np.linalg.norm(direction)
        assert bool((g@direction<0).all())
        baseline=parameter_vector();trials=[];accepted=False
        for trial in range(12):
            radius=a.radius/(2**trial)
            proposal=baseline+torch.as_tensor(radius*direction,dtype=baseline.dtype)
            set_parameters(proposal);current=fitting_losses();change=current-previous
            okay=bool((change<=1e-7).all()) and float(change.mean()) < -1e-8
            trials.append({'radius':radius,'worst_condition_loss_change':float(change.max()),
                           'mean_loss_change':float(change.mean()),'accepted':okay})
            if okay:accepted=True;break
        if not accepted:set_parameters(baseline)
        row={'step':step,'accepted':accepted,'common_direction_certificate':certificate,
             'counterfactual_strength':strength,'conditional_gradient_norms':norms.tolist(),
             'predicted_condition_loss_derivative':(g@direction).tolist(),
             'line_search':trials,'fitting_condition_losses':(current if accepted else previous).tolist(),
             'fitting_condition_loss_changes':(change if accepted else np.zeros_like(change)).tolist(),
             'new_parameter_l2':float(parameter_vector().norm()),
             'fitting_gradient_presentations':len(selected)*(0 if step==1 else 2),
             'fitting_line_search_presentations':len(selected)*len(trials)}
        result['curve'].append(row);persist()
        print(json.dumps({'step':step,'accepted':accepted,'radius':trials[-1]['radius'],
                          'counterfactual_strength':strength,'worst_loss_change':trials[-1]['worst_condition_loss_change'],
                          'mean_loss_change':trials[-1]['mean_loss_change'],'wall_s':result['wall_s']}),flush=True)
        if not accepted:
            result['stop_reason']='No finite hard-computation step passed all 40 fitting constraints';break
        previous=current
    assert all(torch.equal(p,old_state[n]) for n,p in named.items() if n not in missing)
    for layer in net.layers:layer.cf_credit=True
    held=load_items(40,.01,512,'val_spk',7)
    assert [r[4] for r in held[:256]]==source_json['dev_original_ids']
    assert [r[4] for r in held[256:]]==source_json['dev_additional_ids']
    final={'fit':evaluate(net,fit,a.bs),'dev_original':evaluate(net,held[:256],a.bs),
           'dev_additional':evaluate(net,held[256:],a.bs)}
    result.update(status='completed',final=final,old_parameters_exactly_unchanged=True,
                  accepted_steps=sum(r['accepted'] for r in result['curve']),
                  new_parameter_norms={n:float(named[n].detach().norm()) for n in missing},
                  hardware={'platform':platform.platform(),'torch':torch.__version__,'threads':1,'device':'cpu'})
    persist()
    torch.save({'state_dict':net.state_dict(),'config':{'global_context_layers':(3,7)},
                'args':vars(a),'source_sha256':result['source_sha256']},out.with_suffix('.pt'))
    print(json.dumps({'accepted_steps':result['accepted_steps'],
                      'final':{k:v['accuracy'] for k,v in final.items()},'wall_s':result['wall_s'],
                      'max_rss_kb':result['max_rss_kb']}),flush=True)


if __name__=='__main__':main()
