"""Completed matched shared/separate key-value continuations and work ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e131')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    paths={k:Path('experiments/results/e122')/(v+'.json') for k,v in {
       'separate':'d8_n4096_separate_values_recovery_s6',
       'shared':'d8_n4096_shared_values_s6',
       'parent':'d8_n4096_invariance_continue_s6_e2'}.items()}
    records={k:json.loads(p.read_text()) for k,p in paths.items()}
    assert all(r['status']=='completed' for r in records.values())
    separate,shared=records['separate'],records['shared']
    for key in ('fit_ids','dev_original_ids','dev_additional_ids','checkpoint_sha256','source_sha256'):
        assert separate[key]==shared[key],key
    for key in ('augment','limit','epochs','bs','lr','seed','readout','value_backward',
                'train_value_only','global_context_layers','train_new_only','checkpoint','checkpoint_every'):
        assert separate['args'][key]==shared['args'][key],key
    assert separate['args']['separate_keys'] and not shared['args']['separate_keys']
    for split in ('fit','dev_original','dev_additional'):
        assert separate['initial'][split]['predictions']==shared['initial'][split]['predictions']
    saved={k:torch.load(paths[k].with_suffix('.pt'),weights_only=False,map_location='cpu')
           for k in ('separate','shared','parent')}
    old=saved['parent']['state_dict']
    for k in ('separate','shared'):
        for n,p in old.items():assert torch.equal(saved[k]['state_dict'][n],p),(k,n)
        assert all(torch.equal(saved[k]['state_dict'][n],torch.zeros_like(saved[k]['state_dict'][n]))
                   for n in saved[k]['state_dict'] if '.bridge_route' in n)
    for n,p in saved['separate']['state_dict'].items():
        if n.startswith('key_'):assert torch.equal(p,old[n.replace('key_','',1)]),n
    assert saved['separate']['order_rng']==saved['shared']['order_rng']
    assert saved['separate']['augmentation_rng']==saved['shared']['augmentation_rng']
    rows={}
    for k,r in records.items():
        final=r['final'];parts=[final[s] for s in ('dev_original','dev_additional')]
        row={'fit_accuracy':final['fit']['accuracy'],'held_correct':sum(p['correct'] for p in parts),
             'held_n':sum(p['n'] for p in parts),'held_nll':sum(p['nll']*p['n'] for p in parts)/sum(p['n'] for p in parts),
             'predictions':sum((p['predictions'] for p in parts),[])}
        row['held_accuracy']=row['held_correct']/row['held_n']
        if k!='parent':
            epoch=r['curve'][-1]
            row.update(online_nll=epoch['online_nll'],training_input_packets=epoch['training_input_packets'],
              value_map_forward_evaluations=epoch['training_value_evaluations'],
              key_map_forward_evaluations=epoch.get('training_key_value_evaluations',0),
              global_value_scan_compositions=epoch['training_global_scan_compositions'],
              key_local_scan_compositions=epoch.get('training_key_scan_compositions',0),
              measured_wall_s=r['wall_s'],initial_evaluation_reused=bool(r.get('initial_evaluation_reused_from')))
            row['clean_key_winners_clocks_preserved']={s:final[s]['winner_counts']==r['initial'][s]['winner_counts'] and
                   final[s]['mean_added_delay_ms']==r['initial'][s]['mean_added_delay_ms'] for s in ('fit','dev_original','dev_additional')}
        rows[k]=row
    result={'status':'completed','rows':rows,'old_value_and_key_parameters_exactly_preserved':True,
            'same_sample_order_augmentation_and_budget':True,
            'work_boundary':'Value map counts include both streams; global value and key local scans are separate. Wall times have different initial-evaluation reuse and cannot establish a speedup. No joule measurement.',
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*paths.values()]},
            'scope':'One seed; 4096 augmented fitting utterances, one epoch; same 512 held-out training-file speakers; official test untouched'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:r[n] for n in ('fit_accuracy','held_correct','held_nll')} for k,r in rows.items()}),flush=True)


if __name__=='__main__':main()
