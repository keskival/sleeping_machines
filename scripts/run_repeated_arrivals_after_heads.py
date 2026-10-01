"""Finite guarded repeated-arrival trials after the prioritized head campaign."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STEM='local_repeated_arrivals_after_heads_20261001T015000Z'
ALLOWED_STOP='Parallel-head 8K quality requires diagnosis before longer fitting'


def main():
    plan=json.loads((ROOT/f'experiments/queue/{STEM}.json').read_text())
    status_path=ROOT/f'experiments/queue/{STEM}.status.json'
    live=dict(status='waiting_for_prioritized_campaign',completed=[],decisions=[])
    def save():status_path.write_text(json.dumps(live,indent=2)+'\n')
    def frozen():
        for name,digest in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Queued source changed: '+name)
    def validate(path):
        row=json.loads((ROOT/path).read_text())
        if row['status']!='completed':raise ValueError('Completed result required: '+path)
        for name,digest in row['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Result source changed: '+name)
        return row
    def publish(path):
        subprocess.run([sys.executable,'scripts/update_episodic_language_report.py',path],cwd=ROOT,check=True)
    def run(job):
        frozen();live.update(status='running',current_job=job['tag']);save()
        env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},
                 JOB_TIMEOUT_S=str(job['timeout_s']),WAIT='1',WAIT_TIMEOUT_S='86400')
        subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env,check=True)
        row=validate(job['result'])
        if 'work' in row:
            traces=[t[s] for t in row['work']['traces'].values() for s in
                ('forward_and_loss','backward','gradient_normalization','gradient_clipping','optimizer')]
            traces.append(row['work']['inference_trace'])
            if not all(t['formula_coverage_complete'] and not t['unsupported_floating_operators'] for t in traces):
                raise ValueError('Incomplete operator accounting: '+job['tag'])
        live['completed'].append(job['result']);save();publish(job['result']);return row
    try:
        save()
        # Do not acquire the host lock while the previous supervisor has more
        # priority work: its decision/publishing gaps do not release priority.
        while True:
            frozen()
            prior=json.loads((ROOT/plan['prioritized_status']).read_text())
            state=prior['status']
            if state=='completed_bounded_parallel_head_campaign':break
            if state=='needs_review':
                if prior.get('error')!=ALLOWED_STOP:
                    raise ValueError('Prioritized campaign requires review: '+str(prior.get('error')))
                live['decisions'].append('Prior head/data campaign stopped at declared quality gate; test richer retrieval');break
            if state!='running':raise ValueError('Unknown prior campaign status: '+state)
            live['prioritized_current_job']=prior.get('current_job');save();time.sleep(30)
        reference=validate(plan['single_arrival_reference'])
        run(plan['contracts']['1']);run(plan['smoke']['1'])
        candidates={}
        for marks in ('2','4'):
            run(plan['contracts'][marks]);run(plan['smoke'][marks])
            row=run(plan['pilots'][marks]);candidates[marks]=row
        quality=lambda r:r['final']['dev']['bpc']
        acceptable={m:r for m,r in candidates.items() if quality(r)<=quality(reference)+plan['maximum_pilot_regression_bpc']}
        if not acceptable:
            live['decisions'].append('No repeated-arrival pilot passes quality gate; preserve m=1 and diagnose')
        else:
            # Predeclared quality-first selection, work breaks exact ties.
            chosen=min(acceptable,key=lambda m:(quality(acceptable[m]),
                acceptable[m]['work']['cpu_emulator']['total_training_unit_special_flops']))
            row=acceptable[chosen]
            gain=quality(reference)-quality(row)
            live.update(selected_arrivals=int(chosen),pilot_quality_gain_bpc=gain);save()
            if gain>=plan['minimum_pilot_gain_to_scale_bpc']:
                larger=run(plan['eight_k'][chosen])
                saved=validate(plan['single_arrival_eight_k'])
                live['eight_k_quality_gain_bpc']=quality(saved)-quality(larger)
                live['decisions'].append('One matched 8K comparison completed; larger promotion requires review of quality/work curves')
            else:
                live['decisions'].append('Pilot gain below declared promotion threshold; no longer fit')
        live.update(status='completed_bounded_repeated_arrival_campaign',current_job=None);save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise


if __name__=='__main__':main()
