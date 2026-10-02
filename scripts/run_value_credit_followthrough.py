"""Wait for matched fits, then serialize frozen audits and report publication."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    path=ROOT/a.plan;status=path.with_suffix('.status.json')
    if os.uname().nodename!=json.loads(path.read_text())['host']:raise ValueError('Followthrough must use the assigned host')
    own=ROOT/'experiments/queue'/(a.tag+'.status.json')
    if Path(a.tag).name!=a.tag or own.exists():raise ValueError('Preserve existing lifecycle; new followthrough tag required')
    live=dict(status='waiting',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=[])
    def save():
        tmp=own.with_suffix('.tmp');tmp.write_text(json.dumps(live,indent=2)+'\n');tmp.replace(own)
    save()
    while True:
        if status.exists():
            parent=json.loads(status.read_text())
            if parent['status']=='completed':break
            if parent['status']=='needs_review':
                live.update(status='needs_review',error='Parent chain: '+str(parent.get('error')));save()
                raise RuntimeError(live['error'])
        time.sleep(10)
    stages=[('local_value_credit_frozen_20261002T090800Z',600),
            ('local_value_credit_publication_20261002T090800Z',180)]
    env=dict(os.environ,MEM_CAP_KB='3000000',MEM_CAP_RSS_KB='1250000',MIN_AVAIL_MB='8192',WAIT='1',WAIT_TIMEOUT_S='120')
    for key in ('AWS_GYM_SLOT','AWS_GYM_HOST_LOCK_FD','AFTER_JOB_HOOK','CONTINUE_ON_FAILURE'):env.pop(key,None)
    try:
        for tag,timeout in stages:
            live.update(status='running',current_job=tag);save()
            queue=f'experiments/queue/{tag}.txt';env['JOB_TIMEOUT_S']=str(timeout)
            rc=subprocess.call(['bash','experiments/queue/run_safe.sh',queue],cwd=ROOT,env=env)
            if rc:raise RuntimeError('Guarded followthrough exit '+str(rc))
            result=f'experiments/results/diagnostics/{tag}.json';r=json.loads((ROOT/result).read_text())
            if r['status']!='completed':raise ValueError('Completed result required')
            paths=[result]
            if 'publication' in tag:paths+=['REPORT.md','report/readable_report.py','report/sleeping_machines_status.pdf','report/figures/value_credit_learning_20261002T090800Z.png']
            subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
            subprocess.run(['git','commit','--only','-m','Complete value-credit '+('publication' if 'publication' in tag else 'frozen audits'),'--',*paths],cwd=ROOT,check=True)
            live['jobs'].append(dict(tag=tag,result=result,wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']));save()
        live.update(status='completed',current_job=None);save()
    except BaseException as error:
        live.update(status='needs_review',error=str(error));save();raise


if __name__=='__main__':main()
