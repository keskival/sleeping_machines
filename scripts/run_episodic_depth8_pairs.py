"""Guarded small KV pairs, conditional larger pair, then exact scaling recovery."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM='local_episodic_depth8_pairs_20260930T210000Z'


def main():
    plan=json.loads((ROOT/f'experiments/queue/{STEM}.json').read_text())
    status_path=ROOT/f'experiments/queue/{STEM}.status.json'
    live=dict(status='running',completed=[])
    def save():status_path.write_text(json.dumps(live,indent=2)+'\n')
    def run(job):
        for name,digest in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Preserve run sources: '+name)
        live['current_job']=job['tag'];save()
        env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},
            JOB_TIMEOUT_S=str(job['timeout_s']),WAIT='1',WAIT_TIMEOUT_S='86400')
        subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env,check=True)
        result=json.loads((ROOT/job['result']).read_text())
        if result['status']!='completed' or result['protocol']['official_test_read']:
            raise ValueError('Completed development result required')
        live['completed'].append(job['result']);save()
        return result
    def publish():
        subprocess.run([sys.executable,'scripts/update_episodic_language_report.py',
                        *live['completed']],cwd=ROOT,check=True)
    try:
        smoke=json.loads((ROOT/plan['required_smoke']).read_text())
        if smoke['status']!='completed':raise ValueError('Completed guarded smoke required')
        for name,digest in smoke['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Smoke source changed: '+name)
        small=[run(job) for job in plan['small_pair']]
        publish()
        gain=small[0]['final']['dev']['bpc']-small[1]['final']['dev']['bpc']
        live['small_gain_bpc']=gain;save()
        if gain>=plan['minimum_gain_bpc']:
            for job in plan['larger_pair']:run(job)
            publish()
        else:
            live['larger_pair']='deferred: diagnose small KV intervention before promotion';save()
        live.update(status='completed_pairs',current_job=None);save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise
    # The prior scaling checkpoint/settings are preserved; its manifest now
    # names a distinct recovery queue with --resume, never a direct Python fit.
    subprocess.run([sys.executable,'scripts/run_full_sparse_ladder.py',
        plan['resume_scaling_manifest']],cwd=ROOT,check=True)


if __name__=='__main__':main()
