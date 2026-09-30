"""Observable twelve-layer SHD query and independent-key credit contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e136_scattering_shd_model import ScatteringClassifier
from e117_serial_event_shd import batch,load_items


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e136')/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    torch.set_num_threads(1);model=ScatteringClassifier();rows=load_items(40,.01,4,'fit_spk',6);data=batch(rows)
    model.calibrate(rows,batch);z,summary,stats,tr=model(*data[:4],len(rows),trace=True)
    loss=F.cross_entropy(z,data[-1]);loss.backward()
    norms=model.angle_weight.grad.flatten(1).norm(dim=1)
    assert torch.isfinite(z).all() and (norms>0).all()
    assert summary.shape==(4,6945) and all(p.grad is None for p in model.key.parameters())
    frozen={n:p.detach().clone() for n,p in model.key.named_parameters()}
    with torch.no_grad():
        model.embedding.weight.add_(torch.randn_like(model.embedding.weight)*.01)
        model.angle_weight.add_(torch.randn_like(model.angle_weight)*.01)
        z2,_,_,tr2=model(*data[:4],len(rows),trace=True)
    assert (z2-z).abs().max()>1e-5
    assert all(torch.equal(p['winner'],q['winner']) and torch.equal(p['times'],q['times']) for p,q in zip(tr,tr2))
    assert all(torch.equal(p,frozen[n]) for n,p in model.key.named_parameters())
    result={'status':'completed','depth':12,'all_angle_layer_gradient_norms':norms.tolist(),
      'train_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
      'frozen_key_parameters':sum(p.numel() for p in model.key.parameters()),'summary_features':summary.shape[1],
      'actual_winners_clocks_preserved':True,'query_reads_retained_state':True,
      'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/e136_scattering_shd_model.py'),Path('experiments/e136_event_scattering.py'),Path('sleeping_machines/shared_event.py'),Path('sleeping_machines/event_memory.py')]},
      'scope':'Four fitting utterances; supervised loss only at completed query. Prototype with pretrained independent eight-layer key weights reused through twelve value exchanges. Not an accuracy or convergence claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
