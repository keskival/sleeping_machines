"""Run the integrated architecture first, serially, with development gates."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');a=p.parse_args()
    path=ROOT/a.manifest;plan=json.loads(path.read_text())
    def save():
        plan['updated_utc']=now();temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(plan,indent=2)+'\n');temporary.replace(path)
    def stop(reason):
        plan.update(status='needs_review',reason=reason);save();raise RuntimeError(reason)
    for source,digest in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:stop('Source changed: '+source)
    for prior in plan['required_completed_checks']:
        result=json.loads((ROOT/prior).read_text())
        if result['status']!='completed':stop('Required check incomplete: '+prior)
        for source,digest in result['source_sha256'].items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:stop('Check source changed: '+source)
    plan.update(status='running');save()
    for job in plan['jobs']:
        if job.get('status')=='completed':continue
        if 'promotion_gate' in job:
            gate=job['promotion_gate'];previous=json.loads((ROOT/gate['previous_result']).read_text())
            score=previous['final']['dev']['bpc']
            if score>gate['maximum_development_bpc']:stop(f'Diagnose integrated model before more data: {score:.3f} bpc')
            if gate.get('compare_result'):
                smaller=json.loads((ROOT/gate['compare_result']).read_text())
                if score>smaller['final']['dev']['bpc']-gate['minimum_gain_bpc']:
                    stop('Integrated model needs a design improvement before the next data stage')
        lines=[x for x in (ROOT/job['queue']).read_text().splitlines() if x.strip() and not x.startswith('#')]
        if len(lines)!=1 or lines[0].split()[0]!=job['tag']:stop('Invalid one-job queue')
        job.update(status='running',started_utc=now());plan['current_job']=job['tag'];save()
        env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},
                 JOB_TIMEOUT_S=str(job['timeout_s']),WAIT='1',WAIT_TIMEOUT_S='86400')
        print(now()+' starting '+job['queue'],flush=True)
        rc=subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env).returncode
        if rc:stop(f'Guarded job exited {rc}; retain its checkpoint and logs')
        result=json.loads((ROOT/job['result']).read_text())
        if result['status']!='completed' or result['protocol']['official_test_read']:stop('Invalid development result')
        score=result['final']['dev']['bpc']
        if not math.isfinite(score) or result['initial_dev']['bpc']-score<.25:
            stop('No adequate finite development learning; preserve result and investigate')
        if result['work']['fitting_targets']!=(job['fit']-1)*job['epochs']:stop('Fitting budget mismatch')
        if result['final']['dev']['active_units_per_token']!=job['depth']:stop('Sparse activity contract changed')
        if result['final']['dev']['capacity_units']!=27*job['depth']*job['pool']:stop('Capacity contract changed')
        job.update(status='completed',finished_utc=now(),development_bpc=score,
            parameters=result['parameters'],training_flops=result['work']['total_training_arithmetic_flops'],
            wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb']);save()
        rc=subprocess.run([sys.executable,'scripts/update_language_report.py',job['result']],cwd=ROOT).returncode
        if rc:stop('Completed evidence preserved; report hook requires review')
    plan.update(status='completed_development_ladder',current_job=None,
        reason='Review full sparse quality/work and comparisons before any official benchmark');save()
    print('Integrated ladder completed; no automatic official-test promotion.',flush=True)


if __name__=='__main__':main()
