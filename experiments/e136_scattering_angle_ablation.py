"""Withdraw learned angle changes while preserving a completed decoder/key program."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e136_scattering_shd_model import ScatteringClassifier
from e136_scattering_shd import evaluate
from e117_serial_event_shd import load_items


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    root=Path('experiments/results/e136');out=root/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    path=root/'scattering_state_d12_n1024_s6_e3_20260929.json';r=json.loads(path.read_text())
    assert r['status']=='completed';torch.set_num_threads(1)
    model=ScatteringClassifier();initial_w=model.angle_weight.detach().clone();initial_b=model.angle_bias.detach().clone()
    saved=torch.load(path.with_suffix('.pt'),weights_only=False,map_location='cpu');model.load_state_dict(saved['state_dict'])
    trained_w=model.angle_weight.detach().clone();trained_b=model.angle_bias.detach().clone()
    fit=load_items(40,.01,1024,'fit_spk',6);held=load_items(40,.01,512,'val_spk',7)
    assert r['fit_ids']==[x[4] for x in fit] and r['held_ids']==[x[4] for x in held]
    with torch.no_grad():model.angle_weight.copy_(initial_w);model.angle_bias.copy_(initial_b)
    reset={'fit':evaluate(model,fit),'held':evaluate(model,held)}
    for split in reset:
        assert reset[split]['winner_counts']==r['final'][split]['winner_counts']
        assert reset[split]['mean_added_delay_ms']==r['final'][split]['mean_added_delay_ms']
    with torch.no_grad():model.angle_weight.copy_(trained_w);model.angle_bias.copy_(trained_b)
    probe=fit[:64];base=evaluate(model,probe);layer_changes=[]
    for j in range(model.depth):
        with torch.no_grad():model.angle_weight[j].copy_(initial_w[j]);model.angle_bias[j].copy_(initial_b[j])
        p=evaluate(model,probe);layer_changes.append({'layer':j,'fitting_probe_nll_change':p['nll']-base['nll'],
                  'correct':p['correct'],'angle_parameter_l2_change':float((trained_w[j]-initial_w[j]).norm())})
        with torch.no_grad():model.angle_weight[j].copy_(trained_w[j]);model.angle_bias[j].copy_(trained_b[j])
    def compact(p):return {n:p[n] for n in ('correct','n','accuracy','nll')}
    result={'status':'completed','trained':{n:compact(r['final'][n]) for n in ('fit','held')},
      'all_angles_reset_same_decoder':{n:compact(p) for n,p in reset.items()},
      'per_layer_reset_fitting_64':layer_changes,'fitting_probe_baseline':compact(base),
      'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),path,Path('experiments/e136_scattering_shd_model.py'),Path('experiments/e136_event_scattering.py')]},
      'scope':'Frozen-checkpoint intervention; learned embedding, decoder/calibration and key program unchanged. Measures angle contribution/coadaptation, not a retrained fixed-angle baseline.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
