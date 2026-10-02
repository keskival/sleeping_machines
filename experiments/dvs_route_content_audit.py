"""Frozen route movement, content learning and policy/time gradient-path audit."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import sleeping_machines.batched_addressed_fit as K


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group(name):
    if name.startswith('head.'):return 'decoder'
    if any(x in name for x in ('raw_rate','frequency','transport_rate')):return 'time_maps'
    if name.startswith('queries.') or any(x in name for x in ('.key','.clock_bias')):return 'route_maps'
    if name.startswith('units.'):return 'message_maps'
    return 'incoming_context_maps'


@contextmanager
def trace_routes(block_policy_time=False):
    original=K.BatchedTemporalRoute;records=[]
    def traced(scores,values,noise):
        s=scores.detach();pi=s.softmax(-1)
        row=dict(entropy=-(pi*pi.log()).sum(-1).detach(),
            clamped=(s.abs()>=12).detach(),scores=s,
            value_gradient=None,score_gradient=None)
        if values.requires_grad:
            values.register_hook(lambda g:row.update(value_gradient=g.detach()))
        if scores.requires_grad and not block_policy_time:
            scores.register_hook(lambda g:row.update(score_gradient=g.detach()))
        result=original.apply(scores.detach() if block_policy_time else scores,values,noise)
        row['winner']=result[2].detach();records.append(row);return result
    K.BatchedTemporalRoute=SimpleNamespace(apply=traced)
    try:yield records
    finally:K.BatchedTemporalRoute=original


def gradients(model,rows,block=False,pairs=False):
    model.zero_grad(set_to_none=True)
    with trace_routes(block) as records:
        logits,state,options=K.forward(model,rows,314159,terminal_pairs=pairs)
        target=torch.tensor([x['target'] for x in rows])
        if options is None:loss=F.cross_entropy(logits,target)
        else:
            zs,ps=options;B,C,classes=zs.shape
            ls=F.cross_entropy(zs.reshape(-1,classes),target[:,None].expand(B,C).reshape(-1),reduction='none').reshape(B,C)
            loss=(ps*ls).sum()/B
        loss.backward()
    gradient={n:torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
    score_squared=sum(float(x['score_gradient'].double().square().sum()) for x in records if x['score_gradient'] is not None)
    value_squared=sum(float(x['value_gradient'].double().square().sum()) for x in records if x['value_gradient'] is not None)
    last=records[-1];value_gradient=last['value_gradient']
    if value_gradient is not None:
        mask=F.one_hot(last['winner'],model.pool).bool()
        losing_norm=float(value_gradient[~mask].double().norm())
    else:losing_norm=0.
    return gradient,logits.detach(),dict(mean_loss=float(loss.detach()),
        score_tensor_gradient_l2=score_squared**.5,candidate_value_tensor_gradient_l2=value_squared**.5,
        final_query_losing_value_gradient_l2=losing_norm)


@torch.no_grad()
def development(model,rows,expected):
    winners=[];entropies=[];clamps=[];probabilities=[]
    for k in range(0,len(rows),16):
        with trace_routes() as records:
            zs,_,_=K.forward(model,rows[k:k+16],314159)
        probabilities.extend(zs.softmax(-1).tolist())
        winners.append(torch.stack([x['winner'] for x in records]).permute(1,0,2).reshape(len(zs),-1))
        entropies.extend(float(x['entropy'].mean()) for x in records)
        clamps.extend(float(x['clamped'].float().mean()) for x in records)
    np.testing.assert_allclose(probabilities,expected['probabilities'],rtol=1e-4,atol=1e-5)
    return torch.cat(winners),dict(targets=len(rows),probabilities_reproduce_completed_result=True,
        mean_race_entropy_nats=float(np.mean(entropies)),maximum_race_entropy_nats=float(np.log(model.pool)),
        fraction_clamped_candidate_scores=float(np.mean(clamps)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);a=p.parse_args()
    started=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused plain tag required')
    rows=[];sources={}
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        for f,h in parent['source_sha256'].items():
            if sha(ROOT/f)!=h:raise ValueError('Changed frozen model source')
        sources.update(parent['source_sha256']);args=argparse.Namespace(**parent['args'])
        fit,dev,info=N.load(args);assert info==parent['data']
        initial=N.make_model(args);selected=N.make_model(args)
        cp=(ROOT/name).with_suffix('.progress.pt');ck=torch.load(cp,weights_only=False)
        assert ck['source_sha256']==parent['source_sha256'] and ck['data']==parent['data']
        selected.load_state_dict(ck['best_state'])
        before={n:x.detach().clone() for n,x in selected.named_parameters()}
        initial_choices,initial_stats=development(initial,dev,parent['initial_dev'])
        final_choices,final_stats=development(selected,dev,parent['final'])
        full,z,full_stats=gradients(selected,fit[:16])
        blocked,bz,blocked_stats=gradients(selected,fit[:16],block=True)
        torch.testing.assert_close(z,bz,rtol=0,atol=0)
        pair_gradient,_,pair_stats=gradients(selected,fit[:16],pairs=True)
        groups={}
        for g in sorted({group(n) for n in before}):
            names=[n for n in before if group(n)==g]
            start=dict(initial.named_parameters())
            change=sum(float((before[n]-start[n].detach()).double().square().sum()) for n in names)**.5
            base=sum(float(start[n].detach().double().square().sum()) for n in names)**.5
            total=sum(float(full[n].double().square().sum()) for n in names)**.5
            payload=sum(float(blocked[n].double().square().sum()) for n in names)**.5
            removed=sum(float((full[n]-blocked[n]).double().square().sum()) for n in names)**.5
            dot=sum(float((full[n].double()*blocked[n].double()).sum()) for n in names)
            groups[g]=dict(parameter_elements=sum(before[n].numel() for n in names),
                initial_to_selected_parameter_update_l2=change,initial_parameter_l2=base,
                relative_update_l2=None if base==0 else change/base,
                full_local_gradient_l2=total,policy_time_blocked_gradient_l2=payload,
                removed_policy_time_path_gradient_l2=removed,
                full_blocked_gradient_cosine=None if total*payload==0 else dot/(total*payload))
        assert all(torch.equal(p.detach(),before[n]) for n,p in selected.named_parameters())
        rows.append(dict(native=name,native_result_sha256=sha(ROOT/name),checkpoint_sha256=sha(cp),
            selected_epoch=parent['selected_epoch'],development_accuracy=parent['final']['accuracy'],
            development_nll=parent['final']['nll'],initial=initial_stats,selected=final_stats,
            compared_hard_choices=initial_choices.numel(),
            fraction_initial_to_selected_hard_route_changes=float((initial_choices!=final_choices).double().mean()),
            parameter_groups=groups,full_local_objective=full_stats,
            local_objective_with_all_route_policy_time_edges_blocked=blocked_stats,
            terminal_pair_objective=pair_stats,
            frozen_forward_preserved_during_gradient_edge_intervention=True,
            parameter_weights_preserved=True,optimizer_updates=0))
    for f in ['experiments/dvs_route_content_audit.py','sleeping_machines/batched_addressed_fit.py']:
        sources[f]=sha(ROOT/f)
    result=dict(status='completed',args=vars(a),rows=rows,source_sha256=sources,
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen initial/selected dev route movement and preserved probability reproduction; first16 fit loss gradients by parameter groups with all sampled route score/time edges removed. Removing edges measures path sensitivity, not causal importance or an orthogonal gradient decomposition. Terminal-pair losing-value credit measured separately; prior commits remain sampled. No refit, accuracy intervention, exact whole-history gradient or inferred dominant bottleneck.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
