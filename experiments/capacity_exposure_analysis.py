"""Completed occupied-capacity inventory with equal exposure, unequal total data."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from scripts.run_aws_matrix_recovery import validate_result

ROOT=Path(__file__).resolve().parents[1]
PLAN='experiments/gym/plans/aws_capacity_exposure_20261002T072141Z/manifest.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=ROOT/args.output
    if out.exists():raise ValueError('Never overwrite evidence')
    plan=json.loads((ROOT/PLAN).read_text());rows=[]
    for job in plan['jobs']:
        if job['stage']!='pilot':continue
        d=validate_result(ROOT,job);a=d['args'];w=d['work']
        if (a['sources'],a['fit_targets'],a['dev_targets'],a['epochs'],a['common_source_seed'],a['update_targets'])!=(64,512,1024,4,1,64):
            raise ValueError('Matched-exposure protocol required')
        if w['fitting_query_targets']!=2048:raise ValueError('Fitting denominator changed')
        if (w['selected_updates_per_event'],w['key_scores_per_event'],w['available_receivers'])!=(16,32,2048):
            raise ValueError('Activity/capacity contract changed')
        rows.append(dict(shared_maps=a['shared_maps'],seed=a['seed'],result=job['result'],
            result_sha256=hashlib.sha256((ROOT/job['result']).read_bytes()).hexdigest(),
            accuracy=d['final']['dev']['accuracy'],nll=d['final']['dev']['nll'],parameters=d['parameters'],
            cleared_state_accuracy=d['final']['cleared_state']['accuracy'],
            long_gap_accuracy={k:v['accuracy'] for k,v in d['final']['long_gaps'].items()},
            whole_fit_gflops=w['total_training_arithmetic_flops']/1e9,
            fit_mflops_per_query=w['total_training_arithmetic_flops']/2048/1e6,
            inference_mflops_per_query=w['inference_arithmetic_flops_per_query']/1e6,
            fitting_special_function_evaluations=w['training_special_function_evaluations'],
            fitting_query_presentations=2048,fit_queries_per_source_per_pass=8,passes=4,
            dev_queries=1024,dev_populations=16,available_receivers=2048,
            selected_updates_per_event=16,key_scores_per_event=32,wall_s=d['wall_s'],max_rss_kb=d['max_rss_kb']))
    groups=[]
    for shared in (False,True):
        selected=sorted([r for r in rows if r['shared_maps']==shared],key=lambda r:r['seed'])
        if [r['seed'] for r in selected]!=[6,7,8]:raise ValueError('All declared seeds required')
        groups.append(dict(shared_maps=shared,mean_accuracy=statistics.mean(r['accuracy'] for r in selected),
            sample_std_accuracy=statistics.stdev(r['accuracy'] for r in selected),rows=selected))
    previous=ROOT/'experiments/results/diagnostics/aws_rule_seed_20261002T055545Z_analysis.json'
    reference=json.loads(previous.read_text())
    historical=[r for r in reference['rows'] if r['common_seed']]
    result=dict(status='completed',source64=groups,source16_historical_rows=historical,
        source16_analysis_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
        convention='Whole-fit GFLOPs, fitting/inference MFLOPs per query use identical units for both source counts. 2 FLOPs/MAC; specials separately saved.',
        scope='Development-selected three-seed capacity diagnostic. Equal8 queries/source/pass and16 dev populations, unequal total fit/dev data and optimizer population composition. Fixed observed addresses; no iso-data, iso-quality, discovery or physical-energy supremacy claim.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='completed',means=[g['mean_accuracy'] for g in groups])))


if __name__=='__main__':main()
