"""Frozen native credit horizons with identical context, target suffix and race noise."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))
from e120_shared_tasks import text_slice
from language_learning_audit import fingerprint, piece
from sleeping_machines.native_stream_language import NativeStreamLanguageModel


def vector(model, prefix=None):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten()
        for n,p in model.named_parameters() if prefix is None or any(n.startswith(s) for s in prefix)])


def contrast(short, long):
    sn, ln = float(short.norm()), float(long.norm())
    return dict(short_norm=sn, long_norm=ln, difference_norm=float((long-short).norm()),
        relative_difference_to_long=float((long-short).norm())/max(ln, 1e-30),
        cosine_similarity=float(F.cosine_similarity(short[None],long[None],eps=1e-30)))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--tag',required=True)
    args=parser.parse_args(); out=ROOT/'experiments/results/diagnostics'/(args.tag+'.json')
    if Path(args.tag).name!=args.tag or out.exists(): raise ValueError('Unused plain tag required')
    started=time.perf_counter(); torch.set_num_threads(1)
    name='experiments/results/native_language/local_native_language_D8192_H2_d16_depth8_s6_20261001T174000Z.json'
    path=ROOT/name; row=json.loads(path.read_text()); assert row['status']=='completed'
    for source,sha in row['source_sha256'].items():
        if source.startswith('sleeping_machines/'):
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha, source
    ckpt=path.with_suffix('.progress.pt'); saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert saved['result']['status']=='completed' and saved['result']['final']==row['final']
    a=row['args']; model=NativeStreamLanguageModel(payload=a['payload'],depth=a['depth'],heads=a['heads'],pool=a['pool'])
    model.load_state_dict(saved['model']); model.train(); before=fingerprint(model)
    tokens=torch.tensor(text_slice(90_000_000,129)); outputs=[]; gradients={}; rows=[]
    for horizon in (16,32,64):
        model.zero_grad(set_to_none=True)
        boundary=128-horizon
        with torch.no_grad(),torch.random.fork_rng():
            _,state=piece(model,tokens,0,boundary,True)
        state.detach()
        with torch.random.fork_rng():
            z,_=piece(model,tokens,boundary,128,True,state)
            target_logits=z[-16:]
            loss=F.cross_entropy(target_logits,tokens[113:129]); loss.backward()
        outputs.append(target_logits.detach()); whole=vector(model)
        assert bool(torch.isfinite(whole).all()) and float(whole.norm())>0
        layers=[vector(model,[f'units.{d}.',f'queries.{d}.',f'channel_mix.{d}.']) for d in range(model.depth)]
        gradients[horizon]=(whole,layers)
        rows.append(dict(credit_targets=horizon,prediction_bpc=float(loss.detach())/math.log(2),
            gradient_norm=float(whole.norm()),per_layer_gradient_norms=[float(v.norm()) for v in layers],
            forward_input_tokens=128,graph_input_tokens=horizon,loss_targets=16,backwards=1))
    for z in outputs[1:]: torch.testing.assert_close(z,outputs[0],atol=2e-6,rtol=2e-6)
    assert fingerprint(model)==before
    comparisons=[]
    for short,long in ((16,32),(16,64),(32,64)):
        x,xl=gradients[short]; y,yl=gradients[long]
        comparisons.append(dict(short=short,long=long,whole_model=contrast(x,y),
            per_layer=[dict(depth=d+1,**contrast(s,l)) for d,(s,l) in enumerate(zip(xl,yl))]))
    sources=list(row['source_sha256'])+['experiments/language_credit_horizon_audit.py','experiments/language_learning_audit.py']
    result=dict(status='completed',args=vars(args),result=name,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),horizons=rows,comparisons=comparisons,
        same_context_targets_and_per_position_noise=True,weights_preserved=True,optimizer_steps=0,
        max_target_logit_change=float(max((z-outputs[0]).abs().max() for z in outputs)),
        protocol=dict(development=[90000000,90000129],official_test_read=False,forward_input_tokens=384,
            graph_input_tokens=112,backwards=3,unique_loss_targets=16,repeated_loss_target_evaluations=48,
            floating_arithmetic_instrumented=False,training_surrogate_credit=True),
        hardware=dict(host=__import__('os').uname().nodename,device='cpu',torch=torch.__version__,threads=1),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(set(sources))},
        scope='Fixed saved model, one 16-target slice. Credit difference is not a measured fitting-quality benefit, semantic-feature proof, unbiased route gradient or superiority result. Long64 is a comparison, not all-history truth.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=args.tag,wall_s=result['wall_s'])),flush=True)


if __name__=='__main__': main()
