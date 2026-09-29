"""Disconnected-component obstruction and a causal bridge construction.

The XOR construction proves representability; it is not a learned benchmark.
All audits run through the guarded experiment runner.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from e117_serial_event_shd import batch, load_items


def witness():
    b = torch.tensor([a for u in (0,1) for v in (38,39) for a in (u,v)])
    t = torch.tensor([.1,.2]*4)
    c = torch.ones(8)
    ids = torch.arange(4).repeat_interleave(2)
    return b,t,c,ids,4


def mixed_difference(z):
    return z[0]+z[3]-z[1]-z[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--tag",required=True)
    args=parser.parse_args()
    out=Path("experiments/results/e126")/(args.tag+".json");out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.manual_seed(6)
    inputs=witness()
    old=SharedEventModel(classes=2);old.eval()
    z,_,_,_=old(*inputs)
    additive_error=float(mixed_difference(z).abs().max())
    assert additive_error<2e-6
    weighted=SharedEventModel(classes=2,readout="weighted");weighted.eval()
    with torch.no_grad():weighted.readout_gain.weight.fill_(.1)
    z,_,_,traces=weighted(*inputs,trace=True)
    x=traces[-1]["out"]
    gains=torch.exp(2*torch.tanh(weighted.readout_gain(x).squeeze(-1)/2))
    mass=torch.zeros(4).index_add(0,inputs[3],gains)
    weighted_error=float(mixed_difference((z[:,1]-z[:,0])*mass).abs())
    assert weighted_error<2e-6

    # A constructive bump of global content separates equal/opposite bits.
    bridge=SharedEventModel(dim=8,classes=2,global_context_layers=(3,7));bridge.eval()
    with torch.no_grad():
        for param in bridge.parameters():param.zero_()
        bridge.embedding.weight[[0,38],0]=-.5
        bridge.embedding.weight[[1,39],0]=.5
        for layer in bridge.layers:
            layer.log_tau.fill_(math.log(.2));layer.route_bias[0]=10.
        layer=bridge.layers[3]
        layer.bridge_value[0,4,0]=1.;layer.bridge_value[0,5,0]=1.
        layer.bias[0,4]=.3;layer.bias[0,5]=-.3
        _,summary,_,_=bridge(*inputs)
        bump=summary[:,4]-summary[:,5]
        same=torch.stack((bump[0],bump[3]));different=torch.stack((bump[1],bump[2]))
        assert float(different.min()-same.max())>0
        bridge.head.weight[1,4]=1.;bridge.head.weight[1,5]=-1.
        bridge.head.bias[1]=-(same.max()+different.min())/2
    constructed,_,_,_=bridge(*inputs)
    assert constructed.argmax(-1).tolist()==[0,1,1,0]

    # Copy a real deep checkpoint. The extra route/value columns start at zero.
    source=Path("experiments/results/e122/d8_n4096_invariance_continue_s6_e2.pt")
    saved=torch.load(source,weights_only=False,map_location="cpu")
    plain=SharedEventModel();plain.load_state_dict(saved["state_dict"])
    upgraded=SharedEventModel(global_context_layers=(3,7))
    missing,unexpected=upgraded.load_state_dict(saved["state_dict"],strict=False)
    assert not unexpected and len(missing)==4 and all(".bridge_" in n for n in missing)
    rows=load_items(40,.01,4,"fit_spk",6);packed=batch(rows)
    plain.train();upgraded.train()
    a,sa,sta,_=plain(*packed[:4],len(rows))
    b,sb,stb,_=upgraded(*packed[:4],len(rows))
    assert torch.equal(a,b) and torch.equal(sa,sb)
    assert [s["winner_counts"] for s in sta["layers"]]==[s["winner_counts"] for s in stb["layers"]]
    F.cross_entropy(a,packed[-1]).backward();F.cross_entropy(b,packed[-1]).backward()
    upgraded_names=dict(upgraded.named_parameters())
    gradient_errors={n:float((p.grad-upgraded_names[n].grad).abs().max()) for n,p in plain.named_parameters()}
    assert max(gradient_errors.values())<2e-6
    new_gradients={n:float(upgraded_names[n].grad.norm()) for n in missing}
    assert all(v>0 and math.isfinite(v) for v in new_gradients.values())
    upgraded.eval()
    combined=upgraded(*packed[:4],len(rows))[0]
    separate=torch.cat([upgraded(*batch([r])[:4],1)[0] for r in rows])
    separability_error=float((combined-separate).abs().max())
    assert separability_error<3e-4

    # Future content cannot enter the first message at a context layer.
    before=list(inputs);after=list(inputs);after[0]=inputs[0].clone();after[0][1::2]=37
    first=torch.arange(0,8,2)
    ta=bridge(*before,trace=True)[3][3]["out"][first]
    tb=bridge(*after,trace=True)[3][3]["out"][first]
    assert torch.equal(ta,tb)
    result={"status":"completed","disconnected_additive_error":additive_error,
            "weighted_numerator_additive_error":weighted_error,
            "constructed_xor_predictions":constructed.argmax(-1).tolist(),
            "construction_margin":float((different.min()-same.max())/2),
            "zero_bridge_exact_logits_summary_winners":True,
            "legacy_gradient_max_error":max(gradient_errors.values()),
            "bridge_gradient_norms":new_gradients,"query_separability_error":separability_error,
            "causal_first_payload_exact":True,"extra_parameters":sum(upgraded_names[n].numel() for n in missing),
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [Path(__file__),Path("sleeping_machines/shared_event.py"),source]},
            "scope":"An expressivity obstruction and hand-constructed remedy, not learned XOR or SHD improvement"}
    out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":main()
