"""Audit completed, matched full-value fixed/coupled SHD phases."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    root=Path('experiments/results/e134');out=root/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    paths={k:root/f'full_value_{k}_s6_20260929.json' for k in ('fixed','coupled')}
    records={k:json.loads(p.read_text()) for k,p in paths.items()}
    fixed,coupled=records['fixed'],records['coupled']
    assert all(r['status']=='completed' for r in records.values())
    for n in ('ids','checkpoint_sha256','source_sha256','initial','train_names','train_parameters','cache_provenance'):
        assert fixed[n]==coupled[n],n
    for n in ('limit','epochs','bs','lr','checkpoint_every'):
        assert fixed['args'][n]==coupled['args'][n],n
    saved={k:torch.load(paths[k].with_suffix('.pt'),map_location='cpu',weights_only=False) for k in paths}
    assert saved['fixed']['order_rng']==saved['coupled']['order_rng']
    assert saved['fixed']['augmentation_rng']==saved['coupled']['augmentation_rng']
    assert all(r['final']['frozen_parameters_exact'] for r in records.values())
    rows={}
    def quality(final):
        held=[final[n] for n in ('dev_original','dev_additional')]
        total=sum(r['n'] for r in held);correct=sum(r['correct'] for r in held)
        return {'fit_correct':final['fit']['correct'],'fit_n':final['fit']['n'],
          'fit_accuracy':final['fit']['accuracy'],'fit_nll':final['fit']['nll'],
          'held_correct':correct,'held_n':total,'held_accuracy':correct/total,
          'held_nll':sum(r['nll']*r['n'] for r in held)/total}
    rows['parent']=quality(fixed['initial'])
    for k,r in records.items():
        f=r['final'];rows[k]=quality(f)
        rows[k].update({n:f[n] for n in ('online_nll','steps','clipped_steps','packets','values','scans',
                   'key_values','key_scans','global_scans','value_layer_gradient_norms','clean_key_schedules_preserved')})
        rows[k].update(wall_s=r['wall_s'],peak_rss_kib=r['peak_rss_kib'],train_parameters=r['train_parameters'])
        assert all(f['parameter_l2_changes'][f'layers.{j}.value']>0 for j in range(8))
    result={'status':'completed','rows':rows,'same_samples_augmentation_mask_and_update_budget':True,
       'all_eight_value_maps_changed':True,'fixed_clean_winner_counts_and_clocks_preserved':True,
       'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*paths.values()]},
       'scope':'One seed; training-file held speakers, no official test. Frozen policy tensors in both; functional keys fixed in only one. Gradient reach and finite-step credit are separate from generalization.',
       'work_boundary':'Both wall times exclude initial evaluation and use identical cached baseline. Value/scan counts include both streams; global scans listed separately. No total FLOP, memory traffic or energy measurement.'}
    assert all(fixed['final']['clean_key_schedules_preserved'].values())
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
