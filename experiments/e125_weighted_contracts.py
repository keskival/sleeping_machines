"""Check causal event pooling, exact mean initialization and local score credit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from e117_serial_event_shd import batch,load_items


def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True);a=p.parse_args()
    out=Path("experiments/results/e125")/(a.tag+".json");out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.manual_seed(6)
    checkpoint=Path("experiments/results/e122/d8_n2048_invariance_s6.pt")
    saved=torch.load(checkpoint,weights_only=False,map_location="cpu")
    mean=SharedEventModel();mean.load_state_dict(saved["state_dict"],strict=True)
    weighted=SharedEventModel(readout="weighted")
    missing,unexpected=weighted.load_state_dict(saved["state_dict"],strict=False)
    assert missing==["readout_gain.weight"] and not unexpected
    rows=load_items(40,.01,4,"val_spk",7);b,t,c,ids,y=batch(rows)
    for model in (mean,weighted):model.eval()
    with torch.no_grad():
        reference,summary_ref,_,_=mean(b,t,c,ids,4)
        actual,summary,_,_=weighted(b,t,c,ids,4)
    assert torch.equal(reference,actual) and torch.equal(summary_ref,summary)
    weighted.train()
    z,summary,_,trace=weighted(b,t,c,ids,4,trace=True)
    loss=torch.nn.functional.cross_entropy(z,y)
    g,=torch.autograd.grad(loss,summary,retain_graph=True)
    actual,=torch.autograd.grad(loss,weighted.readout_gain.weight)
    x=trace[-1]["out"].detach()
    mass=x.new_zeros(4).index_add(0,ids,c)
    weights=c/mass[ids]
    centered=x-summary[ids,:32].detach()
    local_scalar=(g[ids,:32]*centered).sum(-1)
    expected=(weights[:,None]*local_scalar[:,None]*x).sum(0)[None]
    relative=float((actual-expected).norm()/actual.norm().clamp_min(1e-12))
    assert relative<1e-5 and float(actual.norm())>0
    weighted.eval()
    with torch.no_grad():
        weighted.readout_gain.weight.normal_(std=.05)
        together=weighted(b,t,c,ids,4)[0]
        separate=[]
        for item in rows:
            b1,t1,c1,id1,_=batch([item]);separate.append(weighted(b1,t1,c1,id1,1)[0])
        delta=float((together-torch.cat(separate)).abs().max())
    assert delta<1e-4
    result={"status":"completed","zero_gain_logits_exact":True,"zero_gain_summary_exact":True,
            "local_gain_gradient_relative_error":relative,"gain_gradient_norm":float(actual.norm()),
            "batch_separability_max_error":delta,"extra_parameters":32,
            "source_sha256":hashlib.sha256(Path("sleeping_machines/shared_event.py").read_bytes()).hexdigest(),
            "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":main()
