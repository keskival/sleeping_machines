"""Completed content-retrieval phase versus its plain full-value control."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e135')/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    paths={'plain':Path('experiments/results/e134/full_value_fixed_s6_20260929.json'),
           'content':Path('experiments/results/e135/content_full_value_s6_20260929.json')}
    r={k:json.loads(p.read_text()) for k,p in paths.items()}
    assert all(v['status']=='completed' for v in r.values())
    for n in ('checkpoint_sha256','ids','initial'):assert r['plain'][n]==r['content'][n],n
    for n in ('key_mode','limit','epochs','bs','lr','checkpoint_every'):
        assert r['plain']['args'][n]==r['content']['args'][n],n
    common=set(r['plain']['train_names']);extra=set(r['content']['train_names'])-common
    assert set(r['content']['train_names'])==common|extra
    assert extra=={f'layers.{j}.memory.{n}' for j in range(8) for n in ('query','key')}
    assert r['content']['train_parameters']-r['plain']['train_parameters']==2048
    saved={k:torch.load(p.with_suffix('.pt'),weights_only=False,map_location='cpu') for k,p in paths.items()}
    for n in ('order_rng','augmentation_rng'):assert saved['plain'][n]==saved['content'][n],n
    for n,p in saved['plain']['state_dict'].items():
        if n.startswith('key_') or n.endswith(('.route','.route_bias','.bridge_route')):
            assert torch.equal(p,saved['content']['state_dict'][n]),n
    rows={}
    for k,v in r.items():
        f=v['final'];held=[f[n] for n in ('dev_original','dev_additional')]
        n=sum(p['n'] for p in held);correct=sum(p['correct'] for p in held)
        rows[k]={'held_correct':correct,'held_n':n,'held_accuracy':correct/n,
          'held_nll':sum(p['nll']*p['n'] for p in held)/n,
          'fit_accuracy':f['fit']['accuracy'],'fit_nll':f['fit']['nll'],
          'train_parameters':v['train_parameters'],'wall_s':v['wall_s'],'peak_rss_kib':v['peak_rss_kib'],
          'clean_key_schedules_preserved':f['clean_key_schedules_preserved'],
          'value_layer_gradient_norms':f['value_layer_gradient_norms'],
          'content_work':f.get('content_work'),'clipped_steps':f['clipped_steps']}
        assert all(f['clean_key_schedules_preserved'].values()) and f['frozen_parameters_exact']
    f=r['content']['final']
    assert all(f['parameter_l2_changes'][n]>0 for n in extra)
    assert all(g>0 for g in f['query_gradient_norms']) and all(g>0 for g in f['content_key_gradient_norms'])
    result={'status':'completed','rows':rows,'same_parent_ids_augmentation_and_optimizer_budget':True,
      'all_eight_content_query_key_matrices_updated':True,'immutable_policy_programs_exact':True,
      'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*paths.values()]},
      'scope':'Exploratory one-seed continuation; content adds 2048 parameters and extra state/work. Training-file held speakers, no official test. Not a matched parameter/physical-work or state-of-the-art comparison.',
      'work_boundary':'Content ledger is partial forward scalar multiply-adds; multiply by two for contraction FLOPs. Additional nonlinearities, marking, sorting, normalization, backward, optimizer and memory traffic are unmeasured. Wall time excludes initial evaluation equally; no joule claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
