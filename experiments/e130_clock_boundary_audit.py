"""Separate winner and arrival-order boundary effects from interior credit.

Frozen schedules are diagnostic graphs, not causal deployment models. Fitting
data and a direction determined before inspecting these losses are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from e117_serial_event_shd import batch,load_items
from e128_shd_credit_geometry import common_descent_certificate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e130')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    audit_path=Path('experiments/results/e128/class_speaker_geometry_20260929.json')
    audit=json.loads(audit_path.read_text());archive=np.load(audit_path.with_suffix('.npz'))
    net=SharedEventModel(global_context_layers=(3,7))
    saved=torch.load(audit['checkpoint'],weights_only=False,map_location='cpu')
    names,unexpected=net.load_state_dict(saved['state_dict'],strict=False);assert not unexpected
    params=[dict(net.named_parameters())[n] for n in names]
    old={n:p.detach().clone() for n,p in net.named_parameters() if n not in names}
    fit=load_items(40,.01,4096,'fit_spk',6);by_id={r[4]:r for r in fit}
    selected=[by_id[j] for j in audit['sample_ids']]
    y=archive['labels'];group=archive['group']
    masks=[(y==c)&(group==j) for c in range(20) for j in range(2)]
    g=np.stack([archive['pathwise'][m].mean(0) for m in masks]).astype(np.float64)
    s=np.stack([archive['surrogate'][m].mean(0) for m in masks]).astype(np.float64)
    norms=np.linalg.norm(g,axis=1).clip(1e-12)
    cert=common_descent_certificate(g/norms[:,None]);weights=np.array(cert['mixture_weights'])
    direct=-(weights@(g/norms[:,None]));extra=-(weights@((s-g)/norms[:,None]))
    aa=g@direct;bb=g@extra;positive=bb>0
    strength=min(1.,float(np.min(-aa[positive]/(2*bb[positive])))) if positive.any() else 1.
    direction=direct+strength*extra;direction/=np.linalg.norm(direction)
    assert bool((g@direction<0).all())
    net.eval();batches=[];baseline=[];margin_rows=[]
    with torch.no_grad():
        for offset in range(0,len(selected),4):
            part=selected[offset:offset+4];packed=batch(part)
            z,_,_,tr=net(*packed[:4],len(part),trace=True)
            loss=F.cross_entropy(z,packed[-1],reduction='none')
            baseline.extend(loss.tolist())
            # Freeze every realized choice; the override API also accepts tensors.
            overrides={j:(torch.arange(len(packed[0])),r['winner']) for j,r in enumerate(tr)}
            orders={j:r['time_order'] for j,r in enumerate(tr)}
            batches.append((packed,overrides,orders,tr))
            for j,r in enumerate(tr):
                width=net.bands//net.groups
                keys=packed[3]*(net.groups+1)+(packed[0]+(width//2 if j%2 else 0))//width
                # Incoming times at this layer are previous outgoing times.
                tt=packed[1] if j==0 else tr[j-1]['times']
                order=r['time_order'];secondary=torch.argsort(keys[order],stable=True)
                order=order[secondary];same=keys[order][1:]==keys[order][:-1]
                gaps=(tt[order][1:]-tt[order][:-1])[same]
                clock_gap=r['delays'].sort(-1).values
                winner_gap=clock_gap[:,1]-clock_gap[:,0]
                margin_rows.append({'layer':j,'adjacent_same_receiver_pairs':len(gaps),
                  'arrival_exact_ties':int((gaps==0).sum()),
                  'arrival_gaps_at_most_1us':int((gaps<=1e-6).sum()),
                  'winner_exact_ties':int((winner_gap==0).sum()),
                  'winner_gaps_at_most_1us':int((winner_gap<=1e-6).sum())})
    baseline=np.array(baseline)
    cases=[]
    with torch.no_grad():
        for radius in (.01,.001,.0001,.0000244140625):
            start=0
            for p in params:
                p.copy_(torch.as_tensor(radius*direction[start:start+p.numel()],dtype=p.dtype).reshape_as(p));start+=p.numel()
            for mode in ('realized','frozen_winners','frozen_orders','frozen_both'):
                losses=[];winner_changes=0;time_order_changes=0
                for packed,overrides,orders,base_trace in batches:
                    z,_,_,tr=net(*packed[:4],len(packed[-1]),trace=True,
                                 overrides=overrides if mode in ('frozen_winners','frozen_both') else None,
                                 replay_orders=orders if mode in ('frozen_orders','frozen_both') else None)
                    losses.extend(F.cross_entropy(z,packed[-1],reduction='none').tolist())
                    winner_changes+=sum(int((r['winner']!=base['winner']).sum()) for r,base in zip(tr,base_trace))
                    time_order_changes+=sum(not torch.equal(r['time_order'],base['time_order']) for r,base in zip(tr,base_trace))
                delta=np.array(losses)-baseline
                observed=np.array([delta[m].mean() for m in masks]);predicted=radius*(g@direction)
                cases.append({'radius':radius,'mode':mode,'conditional_loss_change':observed.tolist(),
                              'predicted_conditional_loss_change':predicted.tolist(),
                              'worst_condition_loss_change':float(observed.max()),'mean_loss_change':float(delta.mean()),
                              'first_order_prediction_error_l2':float(np.linalg.norm(observed-predicted)),
                              'winner_changes':winner_changes,'global_layer_time_order_changes':time_order_changes})
            print(json.dumps({'radius':radius,'cases':[{k:r[k] for k in
                ('mode','worst_condition_loss_change','mean_loss_change','first_order_prediction_error_l2','winner_changes')}
                 for r in cases[-4:]]}),flush=True)
        for p in params:p.zero_()
    assert all(torch.equal(p,old[n]) for n,p in net.named_parameters() if n not in names)
    margins=[]
    for j in range(8):
        rows=[r for r in margin_rows if r['layer']==j]
        margins.append({'layer':j,**{k:sum(r[k] for r in rows) for k in rows[0] if k!='layer'}})
    result={'status':'completed','cases':cases,'baseline_loss':float(baseline.mean()),
            'counterfactual_strength':strength,'margin_counts':margins,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 (Path(__file__),Path('sleeping_machines/shared_event.py'),audit_path,audit_path.with_suffix('.npz'))},
            'max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope':'Fitting-only frozen checkpoint; identical preselected direction; frozen schedules can be noncausal and are only diagnostics; no trained model or held-out improvement'}
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
