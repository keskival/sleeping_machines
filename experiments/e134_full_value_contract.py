"""Verify whole-value program teaching and actual-key immutability."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import math
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e134_value_phase import build,SOURCE,hashes
from e117_serial_event_shd import batch,load_items


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e134')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    torch.set_num_threads(1)
    fixed,names,config=build('fixed');coupled,other,_=build('coupled')
    assert names==other
    fixed.eval();coupled.eval()
    rows=load_items(40,.01,4,'fit_spk',6);x=batch(rows)
    z,s,_,tr=fixed(*x[:4],len(rows),trace=True)
    z0,s0,_,tr0=coupled(*x[:4],len(rows),trace=True)
    assert torch.equal(z,z0) and torch.equal(s,s0)
    assert all(torch.equal(p['winner'],q['winner']) and torch.equal(p['times'],q['times']) for p,q in zip(tr,tr0))
    named=dict(fixed.named_parameters());params=[named[n] for n in names]
    base={n:p.detach().clone() for n,p in named.items()}
    loss=F.cross_entropy(z,x[-1]);grad=torch.autograd.grad(loss,params,allow_unused=True)
    flat=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,grad)])
    norm=float(flat.norm());assert norm>0
    radius=1e-5
    with torch.no_grad():
        for p,g in zip(params,grad):
            if g is not None:p.add_(g,alpha=-radius/norm)
        zp,_,_,tp=fixed(*x[:4],len(rows),trace=True)
    observed=float(F.cross_entropy(zp,x[-1])-loss.detach());predicted=-radius*norm
    assert observed<0 and abs(observed-predicted)<max(3e-6,.03*abs(predicted))
    assert all(torch.equal(p['winner'],q['winner']) and torch.equal(p['times'],q['times']) for p,q in zip(tr,tp))
    layer_grads={str(j):float(torch.cat([(torch.zeros_like(named[n]) if g is None else g).flatten()
                  for n,g in zip(names,grad) if n.startswith(f'layers.{j}.')]).norm()) for j in range(8)}
    assert all(g>0 for g in layer_grads.values())
    with torch.no_grad():
        for n,p in fixed.named_parameters():
            if p.requires_grad:p.add_(torch.randn_like(p)*.01)
        _,_,_,tl=fixed(*x[:4],len(rows),trace=True)
    assert all(torch.equal(p['winner'],q['winner']) and torch.equal(p['times'],q['times']) for p,q in zip(tr,tl))
    assert all(torch.equal(p,base[n]) for n,p in fixed.named_parameters() if not p.requires_grad)
    a_l=[2 if j in (3,7) else 1 for j in range(8)]
    result={'status':'completed','train_names':names,'train_parameters':sum(p.numel() for p in params),
            'extra_frozen_keys':sum(p.numel() for n,p in named.items() if n.startswith('key_')),
            'same_trainable_parameter_mask':True,'initial_logits_summary_winners_clocks_exact':True,
            'all_value_layer_gradient_norms':layer_grads,'gradient_norm':norm,
            'radius':radius,'predicted_loss_change':predicted,'observed_loss_change':observed,
            'actual_keys_winners_clocks_unchanged_small_large_updates':True,
            'conditional_value_state_transport_bounds':[math.prod(1-b/8 for b in a_l),math.prod(1+b/8 for b in a_l)],
            'checkpoint_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'source_sha256':hashes(__file__),
            'scope':'Fixed key policy; all value, embedding, memory-time and head parameters train. Conditional transport and a fitting finite-step contract, not global convergence or generalization.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
