"""A separate causal key stream protects routing while values learn."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from e117_serial_event_shd import batch,load_items


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e131')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.manual_seed(6)
    source=Path('experiments/results/e122/d8_n4096_invariance_continue_s6_e2.pt')
    saved=torch.load(source,weights_only=False,map_location='cpu')
    parent=SharedEventModel();parent.load_state_dict(saved['state_dict']);parent.eval()
    paired=SharedEventModel(global_context_layers=(3,7))
    missing,unexpected=paired.load_state_dict(saved['state_dict'],strict=False)
    assert not unexpected and len(missing)==4
    paired.freeze_keys_from_values();paired.eval()
    examples=load_items(40,.01,4,'fit_spk',6);packed=batch(examples)
    with torch.no_grad():
        z0,s0,_,tr0=parent(*packed[:4],4,trace=True)
        z1,s1,_,tr1=paired(*packed[:4],4,trace=True)
    assert torch.equal(z0,z1) and torch.equal(s0,s1)
    assert all(torch.equal(x['winner'],y['winner']) and torch.equal(x['times'],y['times']) for x,y in zip(tr0,tr1))
    # Only the two added value maps learn. Route decisions are still computed
    # from each new query by the immutable keys, never from a stored label.
    for n,p in paired.named_parameters():p.requires_grad_('.bridge_value' in n and not n.startswith('key_'))
    params=[p for p in paired.parameters() if p.requires_grad]
    paired.train();z=paired(*packed[:4],4)[0];loss=F.cross_entropy(z,packed[-1])
    gradients=torch.autograd.grad(loss,params)
    g=torch.cat([v.flatten() for v in gradients]);direction=-g/g.norm()
    radius=.0001
    with torch.no_grad():
        start=0
        for p in params:
            p.copy_((radius*direction[start:start+p.numel()]).reshape_as(p));start+=p.numel()
    paired.eval()
    with torch.no_grad():z2,s2,st2,tr2=paired(*packed[:4],4,trace=True)
    changed_loss=float(F.cross_entropy(z2,packed[-1]))
    predicted_change=-radius*float(g.norm())
    observed_change=changed_loss-float(loss.detach())
    assert observed_change<0
    assert abs(observed_change-predicted_change)<max(2e-6,.01*abs(predicted_change))
    assert all(torch.equal(x['winner'],y['winner']) and torch.equal(x['times'],y['times']) for x,y in zip(tr0,tr2))
    assert not torch.equal(z0,z2)
    # A large value perturbation also cannot change the independently computed keys.
    with torch.no_grad():
        for p in params:p.normal_(0,.08)
        combined=paired(*packed[:4],4,trace=True)
        single=torch.cat([paired(*batch([r])[:4],1)[0] for r in examples])
    assert all(torch.equal(x['winner'],y['winner']) and torch.equal(x['times'],y['times']) for x,y in zip(tr0,combined[3]))
    isolation=float((combined[0]-single).abs().max());assert isolation<3e-4
    restored=SharedEventModel(global_context_layers=(3,7),separate_keys=True)
    restored.load_state_dict(paired.state_dict());restored.eval()
    with torch.no_grad():assert torch.equal(combined[0],restored(*packed[:4],4,trace=True)[0])
    result={'status':'completed','exact_initial_logits_summary_winners_clocks':True,
            'value_gradient_norm':float(g.norm()),'radius':radius,
            'predicted_loss_change':predicted_change,'observed_loss_change':observed_change,
            'keys_exact_under_small_and_large_value_updates':True,'query_isolation_error':isolation,
            'checkpoint_roundtrip_exact':True,
            'extra_frozen_key_parameters':sum(p.numel() for n,p in paired.named_parameters() if n.startswith('key_')),
            'trainable_value_parameters':sum(p.numel() for p in params),
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (Path(__file__),Path('sleeping_machines/shared_event.py'),source)},
            'scope':'Frozen routing-key learning phase; two event-driven payload streams; keys execute from actual observations; value updates do not imply learned new routing or benchmark superiority'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
