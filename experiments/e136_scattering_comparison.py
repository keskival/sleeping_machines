"""Audit completed twelve-layer observable/packet-only SHD query screens."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    root=Path('experiments/results/e136');out=root/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    paths={k:root/f'scattering_{k}_d12_n1024_s6_e3_20260929.json' for k in ('state','packets')}
    r={k:json.loads(p.read_text()) for k,p in paths.items()}
    assert all(v['status']=='completed' for v in r.values())
    for n in ('fit_ids','held_ids','source_sha256','parent_key_checkpoint_sha256','train_parameters','frozen_key_parameters'):
        assert r['state'][n]==r['packets'][n],n
    for n in ('limit','epochs','lr','depth'):assert r['state']['args'][n]==r['packets']['args'][n],n
    saved={k:torch.load(p.with_suffix('.pt'),weights_only=False,map_location='cpu') for k,p in paths.items()}
    assert saved['state']['rng']==saved['packets']['rng']
    for n,p in saved['state']['state_dict'].items():
        if n.startswith('key.') or n in ('center','scale'):assert torch.equal(p,saved['packets']['state_dict'][n]),n
    rows={}
    for k,v in r.items():
        rows[k]={'initial_fit':v['initial']['fit']['accuracy'],'initial_held':v['initial']['held']['accuracy'],
          'curve':[{'epoch':f['epoch'],'fit_correct':f['fit']['correct'],'fit_n':f['fit']['n'],
             'fit_accuracy':f['fit']['accuracy'],'fit_nll':f['fit']['nll'],
             'held_correct':f['held']['correct'],'held_n':f['held']['n'],'held_accuracy':f['held']['accuracy'],
             'held_nll':f['held']['nll'],'angle_gradient_norms':f['angle_layer_gradient_norms']} for f in v['curve']],
          'wall_s':v['wall_s'],'peak_rss_kib':v['peak_rss_kib'],
          'nominal_train_parameters':v['train_parameters'],
          'active_train_parameters':v['train_parameters'] if k=='state' else 2188}
        assert all(g>0 for f in v['curve'] for g in f['angle_layer_gradient_norms'])
        assert v['final']['train_parameter_l2_changes']['angle_weight']>0
    result={'status':'completed','rows':rows,'same_key_program_and_calibration':True,
      'same_fit_held_order_and_optimizer_budget':True,'all_twelve_angle_layers_receive_credit':True,
      'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*paths.values()]},
      'scope':'Query-boundary/learnability intervention, not a capacity-matched model comparison. Observable states add 138240 active terminal-head weights. Keys inherit supervised E122 training; new fitting set is 1024 unaugmented utterances. No official test or energy claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
