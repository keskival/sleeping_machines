"""Completed matched-angle intervention, with paired held decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True); a=ap.parse_args()
    base=Path('experiments/results/e137'); out=base/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists(): raise ValueError('Invalid output')
    paths={mode:base/(f'compact_{mode}_d12_r16_n1024_s6_e3'+('_resume' if mode=='learned' else '')+'_20260930.json')
           for mode in ('learned','frozen')}
    rows={key:json.loads(path.read_text()) for key,path in paths.items()}
    assert all(row['status']=='completed' for row in rows.values())
    learned,frozen=rows['learned'],rows['frozen']
    for key in ('initial_state_sha256','calibration_sha256','fit_ids','held_ids','parent_key_checkpoint_sha256','initial','head_parameters'):
        assert learned[key]==frozen[key], key
    for key in ('rank','depth','limit','epochs','lr'):
        assert learned['args'][key]==frozen['args'][key], key
    assert learned['source_sha256']==frozen['source_sha256']
    def compact(row):
        return {'train_parameters':row['train_parameters'],'head_parameters':row['head_parameters'],
          'head_forward_macs_per_query':row['head_forward_macs_per_query'],
          'completed_wall_s':row['wall_s']+row.get('resume',{}).get('prior_wall_s',0),
          'wall_boundary':'Recorded completed-checkpoint and continuation wall; excludes uncheckpointed lost restart work. Resume repeats setup/calibration.',
          'peak_rss_kib':row['peak_rss_kib'],
          'curve':[{'epoch':x['epoch'],'fit_correct':x['fit']['correct'],'fit_n':x['fit']['n'],
             'fit_accuracy':x['fit']['accuracy'],'fit_nll':x['fit']['nll'],
             'held_correct':x['held']['correct'],'held_n':x['held']['n'],
             'held_accuracy':x['held']['accuracy'],'held_nll':x['held']['nll'],
             'angle_gradient_norms':x['angle_layer_gradient_norms'],'clipped_steps':x['clipped_steps']}
             for x in row['curve']]}
    lp=np.asarray(learned['final']['held']['predictions']); fp=np.asarray(frozen['final']['held']['predictions'])
    # Load only the declared training-speaker development labels.
    from e117_serial_event_shd import load_items
    held=load_items(40,.01,512,'val_spk',7); y=np.asarray([x[3] for x in held])
    assert [x[4] for x in held]==learned['held_ids']
    lc=lp==y; fc=fp==y; differences=lc.astype(float)-fc.astype(float)
    result={'status':'completed','rows':{k:compact(v) for k,v in rows.items()},
       'same_initial_logits_calibration_keys_and_samples':True,'same_decoder_capacity_and_update_budget':True,
       'paired_held_final':{'learned_only_correct':int((lc&~fc).sum()),'frozen_only_correct':int((fc&~lc).sum()),
          'difference_accuracy':float(differences.mean()),'descriptive_standard_error':float(differences.std(ddof=1)/len(differences)**.5),
          'scope':'One seed; uncertainty over this development sample does not include training-seed or speaker-population uncertainty.'},
       'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*paths.values()]},
       'scope':'Pretrained immutable keys, 1024 new unaugmented fitting utterances, 512 held training-file speakers. Only angular adaptation is disabled in the control; both embeddings and identical compact queries learn. No official-test, complete-cost or supremacy claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__':main()
