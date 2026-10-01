"""Guarded frozen-checkpoint diagnosis; no fitting or parameter updates.

Scores are short-window interventions, not refitted ablations/benchmark claims.
Categorical loss replays condition on one realized race time and continuation
seed; they audit local content credit, not whole-history expected gradients.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from e120_shared_tasks import text_slice
from parallel_head_race_language_screen import source_hashes
from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(record):
    path=ROOT/record;r=json.loads(path.read_text())
    assert r['status']=='completed' and not r['protocol']['official_test_read']
    for name,sha in r['source_sha256'].items():assert digest(ROOT/name)==sha,name
    a=r['args'];checkpoint=path.with_suffix('.progress.pt')
    saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
    assert saved['result']['source_sha256']==r['source_sha256']
    model=ParallelHeadRaceLanguageModel(a['payload'],a['depth'],a['pool'],heads=a['heads'],matching=a['matching'],recent=a['recent'])
    model.load_state_dict(saved['best_state']);model.eval()
    return model,r,dict(record=record,record_sha256=digest(path),checkpoint_sha256=digest(checkpoint))


@torch.no_grad()
def short_intervention(model,tokens,mode):
    hooks=[]
    if mode in ('source_off','kv_off'):
        modules=[model.source_gate] if mode=='source_off' else list(model.kv_gate)
        hooks=[m.register_forward_hook(lambda module,args,out:torch.full_like(out,-60.)) for m in modules]
    elif mode=='channel_identity':
        hooks=[m.register_forward_hook(lambda module,args,out:args[0]) for m in model.channel_mix]
    elif mode!='unaltered':raise ValueError(mode)
    try:
        torch.manual_seed(314159);state=model.new_state();total=0.;norms=[]
        for t,target in zip(tokens[:-1],tokens[1:]):
            logits=model.consume(t,state)
            total+=float(F.cross_entropy(logits[None],target[None],reduction='sum'))
            norms.append(float(state.context.square().mean().sqrt()))
        return dict(mode=mode,bpc=total/(len(tokens)-1)/math.log(2),context_rms_mean=sum(norms)/len(norms),context_rms_max=max(norms))
    finally:
        for hook in hooks:hook.remove()


def route_credit(model,tokens,probes):
    original=model.race;state=model.new_state();rows=[]
    model.eval();torch.manual_seed(314159)
    with torch.no_grad():
        for token in tokens[:64]:model.consume(token,state)
    for position in range(64,64+probes):
        token,target=tokens[position:position+2]
        snapshot=copy.deepcopy(state);rng=torch.get_rng_state()
        for layer in sorted(set((0,model.depth//2,model.depth-1))):
            slot=2*(layer*model.heads)+1  # head0 historical race; all banks nonempty
            counter=[0];box={}
            def observe(scores,values=None):
                result=original(scores,values);call=counter[0];counter[0]+=1
                if call==slot:
                    assert values is not None
                    box.update(scores=scores,values=values,delay=result[1],winner=result[2])
                    result[0].register_hook(lambda error:box.update(error=error.detach()))
                return result
            try:
                model.race=observe;model.train();model.zero_grad(set_to_none=True)
                torch.set_rng_state(rng)
                z=model.consume(token,copy.deepcopy(snapshot))
                F.cross_entropy(z[None],target[None]).backward()
                s=box['scores'].detach().double();v=box['values'].detach().double()
                error=box['error'].double();winner=int(box['winner'])
                u=(box['delay'].detach().double()-.001)/.010;waiting=u/(1-u)
                teacher=s.exp()*waiting*((v-v.mean(0))@error);teacher[winner]-=teacher.sum()
                losses=[]
                for choice in range(len(s)):
                    count=[0]
                    def force(scores,values=None):
                        result=original(scores,values);call=count[0];count[0]+=1
                        if call==slot:
                            assert values is not None
                            return values[choice],result[1],result[2].new_tensor(choice)
                        return result
                    model.race=force;torch.set_rng_state(rng)
                    # Teacher/inference forward equality allows no-grad training
                    # mode replays with the same candidate proposal convention.
                    with torch.no_grad():
                        zz=model.consume(token,copy.deepcopy(snapshot))
                        losses.append(float(F.cross_entropy(zz[None],target[None])))
                loss=torch.tensor(losses,dtype=torch.float64);p=s.softmax(0)
                categorical=p*(loss-(p*loss).sum())
                denominator=float(teacher.norm()*categorical.norm())
                rows.append(dict(position=position,layer=layer,head=0,candidates=len(s),winner=winner,
                    probability_max=float(p.max()),entropy_nats=float(-(p*p.log()).sum()),
                    realized_loss=losses[winner],minimum_candidate_loss=min(losses),
                    exact_conditional_content_gradient=categorical.tolist(),local_content_teacher=teacher.tolist(),
                    cosine=(float(teacher@categorical)/denominator if denominator>1e-15 else None),
                    teacher_sum=float(teacher.sum()),candidate_losses=losses))
            finally:model.race=original
        model.eval();torch.set_rng_state(rng)
        with torch.no_grad():model.consume(token,state)
    model.zero_grad(set_to_none=True)
    defined=[r for r in rows if r['cosine'] is not None]
    return dict(rows=rows,summary=dict(probes=len(rows),defined_cosines=len(defined),
        mean_cosine=sum(r['cosine'] for r in defined)/len(defined) if defined else None,
        negative_cosines=sum(r['cosine']<0 for r in defined),
        mean_probability_max=sum(r['probability_max'] for r in rows)/len(rows),
        mean_oracle_gap_nats=sum(r['realized_loss']-r['minimum_candidate_loss'] for r in rows)/len(rows)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--records',nargs='+',required=True);p.add_argument('--n',type=int,default=257);p.add_argument('--probes',type=int,default=8)
    a=p.parse_args();out=ROOT/'experiments/results/diagnostics'/f'{a.tag}.json'
    if Path(a.tag).name!=a.tag or out.exists() or not 66+a.probes<=a.n<=1025:raise ValueError('Unique tag and bounded diagnostic window required')
    out.parent.mkdir(exist_ok=True);started=time.perf_counter();torch.set_num_threads(1)
    tokens=torch.tensor(text_slice(90_000_000,a.n));rows=[]
    for record in a.records:
        model,r,provenance=load(record)
        before=copy.deepcopy(model.state_dict())
        variants=[short_intervention(model,tokens,mode) for mode in ('unaltered','source_off','kv_off','channel_identity')]
        spectra=[dict(layer=i,spectral_norm=float(torch.linalg.matrix_norm(m.weight,ord=2)),
            departure_from_identity=float((m.weight-torch.eye(model.total_payload)).norm())) for i,m in enumerate(model.channel_mix)]
        gates=torch.sigmoid(model.source_gate(model.embedding.weight)).detach()
        credit=route_credit(model,tokens,a.probes) if model.heads==2 else None
        for name,tensor in model.state_dict().items():torch.testing.assert_close(tensor,before[name],rtol=0,atol=0)
        rows.append(dict(heads=model.heads,provenance=provenance,selected_epoch=r['selected_epoch'],variants=variants,
            channel_spectra=spectra,source_gate_embedding_mean=float(gates.mean()),route_credit=credit))
    hashes=source_hashes();hashes['experiments/diagnose_parallel_head_language.py']=digest(Path(__file__))
    result=dict(status='completed',args=vars(a),source_sha256=hashes,rows=rows,
        protocol=dict(official_test_read=False,fitting=False,parameter_updates=0,development=[90_000_000,90_000_000+a.n],
            scope='Frozen single-seed short-window interventions, not refitted model comparisons or promotion scores',
            route_credit='Head0, three depths, next-token losses at one realized fixed race time and continuation seed; local content teacher versus conditional categorical loss gradient; not full-history/noise-averaged gradient'),
        numerical_contracts=dict(parameters_unchanged=True,completed_saved_source_identity=True),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(device='cpu',threads=1,platform=platform.platform(),torch=torch.__version__))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=a.tag,wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()
