"""Frozen serial 2K credit experiment with the matched minimal-core control."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_aws_matrix_recovery import validate_result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);a=p.parse_args()
    path=ROOT/a.plan;plan=json.loads(path.read_text());status=path.with_suffix('.status.json')
    if status.exists():raise ValueError('Preserve previous lifecycle; new plan required')
    live=dict(status='checking',completed=[],current_job=None,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    child=None
    def save():
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(live,indent=2)+'\n');tmp.replace(status)
    def frozen():
        if os.uname().nodename!=plan['host']:raise ValueError('Wrong host')
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='main':raise ValueError('Main required')
        for name,sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen source changed: '+name)
        for job in plan['jobs']:
            q=ROOT/job['queue'];lines=[s for s in q.read_text().splitlines() if s.strip() and not s.startswith('#')]
            if len(lines)!=1 or lines[0].split()[0]!=job['tag']:raise ValueError('Unique one-job queue required')
            if hashlib.sha256(q.read_bytes()).hexdigest()!=job['queue_sha256']:raise ValueError('Queue changed')
    def stop(signum,frame):raise InterruptedError('Signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,stop)
    save()
    try:
        frozen()
        for name in plan['references']:
            validate_result(ROOT,dict(tag=Path(name).stem,result=name))
        for job in plan['jobs']:
            frozen()
            if not set(job['requires']).issubset({j['tag'] for j in live['completed']}):raise ValueError('Missing prerequisites')
            if (ROOT/job['result']).exists():raise ValueError('Preserve prior output; do not overwrite or silently skip')
            live.update(status='running',current_job=job['tag']);save()
            env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},JOB_TIMEOUT_S=str(job['timeout_s']),WAIT='1',WAIT_TIMEOUT_S='120')
            for key in ('AWS_GYM_SLOT','AWS_GYM_HOST_LOCK_FD','AFTER_JOB_HOOK','CONTINUE_ON_FAILURE'):env.pop(key,None)
            child=subprocess.Popen(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env)
            rc=child.wait();child=None
            if rc:raise RuntimeError('Guarded job exit '+str(rc))
            r=validate_result(ROOT,job);frozen()
            if job['stage'] in ('contracts','smoke') and r['max_rss_kb']>plan['maximum_prerequisite_rss_kb']:
                raise ValueError('Insufficient measured memory margin; no fit admitted')
            subprocess.run(['git','add','--',job['result']],cwd=ROOT,check=True)
            subprocess.run(['git','commit','--only','-m','Complete bounded count-credit stage: '+job['stage'],'--',job['result']],cwd=ROOT,check=True)
            live['completed'].append(dict(tag=job['tag'],result=job['result'],wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']));save()
            if job['stage']=='analysis':live['comparison']={k:r[k] for k in ('followup_gate_passed','full_core_credit_gain_bpc','full_core_advantage_bpc_at64','difference_in_credit_gains_bpc')}
        live.update(status='completed',current_job=None);save()
    except BaseException as error:
        live.update(status='needs_review',error=str(error));save();raise
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)


if __name__=='__main__':main()
