"""Bounded guarded prerequisites/publication after the current credit worker."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_state_credit_boundary import identity
from scripts.run_aws_matrix_recovery import validate_result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',required=True)
    args=parser.parse_args(); path=ROOT/args.plan; plan=json.loads(path.read_text()); state=path.with_suffix('.status.json')
    if state.exists(): raise ValueError('Preserve prior follow-through lifecycle')
    live=dict(status='waiting',completed=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    child=None

    def save():
        temp=state.with_suffix('.tmp'); temp.write_text(json.dumps(live,indent=2)+'\n'); temp.replace(state)

    def verify():
        if os.uname().nodename!=plan['host']: raise ValueError('Host-specific follow-through')
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='main': raise ValueError('Main required')
        for name,sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha: raise ValueError('Frozen source changed: '+name)
        for job in plan['jobs']:
            queue=ROOT/job['queue']
            if hashlib.sha256(queue.read_bytes()).hexdigest()!=job['queue_sha256']: raise ValueError('Queue changed')
            if len([s for s in queue.read_text().splitlines() if s.strip() and not s.startswith('#')])!=1:
                raise ValueError('Exactly one job per queue')

    def interrupted(signum,frame): raise InterruptedError('Signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT): signal.signal(sig,interrupted)
    save()
    try:
        verify(); deadline=time.monotonic()+plan['maximum_wait_s']
        parent=plan['predecessor']; parent_state=ROOT/plan['parent_status']
        while True:
            current=json.loads(parent_state.read_text()); process=identity(parent['pid'])
            if current['status']=='completed' and (not process or process['state']=='Z'): break
            if current['status']=='needs_review': raise ValueError('Preserve failed predecessor; no follow-up admission')
            if process and process['start_ticks']!=parent['start_ticks']: raise ValueError('Predecessor PID reused')
            if not process or process['state']=='Z': raise ValueError('Predecessor interrupted before completion')
            if time.monotonic()>deadline: raise TimeoutError('Bounded follow-through wait expired')
            time.sleep(5)
        for job in plan['jobs']:
            verify(); live.update(status='running',current_job=job['tag']); save()
            env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},
                JOB_TIMEOUT_S=str(job['timeout_s']),WAIT='1',WAIT_TIMEOUT_S='120')
            for key in ('AWS_GYM_SLOT','AWS_GYM_HOST_LOCK_FD','AFTER_JOB_HOOK','CONTINUE_ON_FAILURE'): env.pop(key,None)
            child=subprocess.Popen(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env)
            rc=child.wait(); child=None
            if rc: raise RuntimeError('Guarded follow-through exit '+str(rc))
            r=validate_result(ROOT,job); verify()
            if r['max_rss_kb']>.85*plan['resource_caps']['MEM_CAP_RSS_KB']: raise ValueError('Review high prerequisite RSS; do not scale')
            files=[job['result']]
            if job['stage']=='publication': files+=['REPORT.md','report/sleeping_machines_status.pdf','report/figures']
            subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
            subprocess.run(['git','commit','--only','-m','Complete bounded local credit follow-through: '+job['stage'],'--',*files],cwd=ROOT,check=True)
            live['completed'].append(dict(tag=job['tag'],result=job['result'],wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb'])); save()
        live.update(status='completed',current_job=None); save()
    except BaseException as error:
        live.update(status='needs_review',error=str(error)); save(); raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate(); child.wait(timeout=30)


if __name__=='__main__': main()
