"""Finite diagnosis/credit/data ladder after the active repeated-arrival run."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STEM='local_language_credit_campaign_20261001T074000Z'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-stem',default=STEM)
    stem=parser.parse_args().manifest_stem
    if Path(stem).name!=stem:raise ValueError('Use a local manifest stem')
    plan=json.loads((ROOT/f'experiments/queue/{stem}.json').read_text())
    status=ROOT/f'experiments/queue/{stem}.status.json'
    live=dict(status='waiting_for_repeated_arrivals',completed=[],decisions=[])
    def save():status.write_text(json.dumps(live,indent=2)+'\n')
    def validate(path):
        row=json.loads((ROOT/path).read_text())
        if row['status']!='completed':raise ValueError('Completed record required: '+path)
        for name,sha in row['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Result source changed: '+name)
        return row
    def frozen():
        for name,sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Queued source changed: '+name)
    def run(job,timeout=None):
        frozen();live.update(status='running',current_job=job['tag']);save()
        env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},
            JOB_TIMEOUT_S=str(timeout or job['timeout_s']),WAIT='1',WAIT_TIMEOUT_S='86400')
        subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env,check=True)
        r=validate(job['result']);live['completed'].append(job['result']);save()
        changes=subprocess.check_output(['git','status','--porcelain','--',job['result']],cwd=ROOT,text=True)
        if changes.strip():subprocess.run([sys.executable,'scripts/publish_language_credit_stage.py',job['result']],cwd=ROOT,check=True)
        return r
    quality=lambda r:r['final']['dev']['bpc']
    work=lambda r:r['work']['cpu_emulator']['total_training_unit_special_flops']
    try:
        save()
        while True:
            frozen();prior=json.loads((ROOT/plan['prioritized_status']).read_text())
            if prior['status']=='completed_bounded_repeated_arrival_campaign':break
            if prior['status']=='needs_review':raise ValueError('Prior campaign requires review: '+str(prior.get('error')))
            if prior['status'] not in ('waiting_for_prioritized_campaign','running'):raise ValueError('Unknown prior state')
            live['prioritized_current_job']=prior.get('current_job');save();time.sleep(30)
        run(plan['diagnosis'])
        run(plan['credit_contracts']);run(plan['credit_smoke'])
        pilot=run(plan['credit_pilot']);reference=validate(plan['short_credit_reference'])
        gain=quality(reference)-quality(pilot);live['longer_credit_pilot_gain_bpc']=gain;save()
        short=run(plan['short_credit_eight_k']);eligible={16:short}
        if gain>=plan['minimum_credit_pilot_gain_bpc']:
            eligible[64]=run(plan['long_credit_eight_k'])
        else:
            live['decisions'].append('64-credit pilot misses quality gain gate; no 8K extension');save()
        chosen=min(eligible,key=lambda b:(quality(eligible[b]),work(eligible[b])))
        control=validate(plan['single_head_eight_k']);live['selected_credit']=chosen;save()
        if quality(eligible[chosen])>quality(control)+plan['maximum_full_model_regression_bpc']:
            live.update(status='completed_bounded_credit_campaign',current_job=None)
            live['decisions'].append('Best 8K result misses control gate; stop larger fits and diagnose');save();return
        larger=run(plan['thirtytwo_k'][str(chosen)])
        run(plan['repeat_eight_k'][str(chosen)])
        gain=quality(eligible[chosen])-quality(larger)
        extra_bytes=(131072-32768)*8*2*(2*32*4+16)
        projected_rss=larger['max_rss_kb']+extra_bytes/1024
        timeout=math.ceil(larger['wall_s']*4*1.5+1800)
        live.update(thirtytwo_k_data_gain_bpc=gain,projected_131k_rss_kb=projected_rss,projected_131k_timeout_s=timeout);save()
        if gain>=plan['minimum_larger_data_gain_bpc'] and projected_rss<=plan['maximum_projected_rss_kb'] and timeout<=plan['maximum_timeout_s']:
            run(plan['onehundredthirtyone_k'][str(chosen)],timeout)
        else:
            live['decisions'].append('131K deferred: measured quality, memory or duration gate');save()
        live.update(status='completed_bounded_credit_campaign',current_job=None);save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise


if __name__=='__main__':main()
