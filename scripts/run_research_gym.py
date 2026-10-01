"""Serial host-local runner for an immutable gym screen, never a remote provisioner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True)
    p.add_argument('--domain',choices=('all','temporal','language','tabular'),default='all')
    p.add_argument('--rss-mib',type=int,required=True);p.add_argument('--vms-mib',type=int,required=True)
    p.add_argument('--smokes-only',action='store_true');p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    if not 1<=a.attempt<=100:raise ValueError('Bounded explicit recovery attempt required')
    if not 512<=a.rss_mib<a.vms_mib:raise ValueError('Measured RSS cap below VMS cap required')
    path=(ROOT/a.manifest).resolve()
    if not path.is_relative_to(ROOT/'experiments/gym/plans'):raise ValueError('Repo gym manifest required')
    plan=json.loads(path.read_text());selected=[j for j in plan['jobs'] if (a.domain=='all' or j['domain']==a.domain) and (not a.smokes_only or j['stage']!='pilot')]
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='main':raise ValueError('Run committed sources on main')
    # Worker-local lifecycle files never mutate shared plans or main-branch report.
    state=path.parent/f'worker_{a.domain}_attempt{a.attempt}.status.json'
    if state.exists():raise ValueError('Inspect previous lifecycle before recovery; do not overwrite worker state')
    env=dict(os.environ,MEM_CAP_KB=str(a.vms_mib*1024),MEM_CAP_RSS_KB=str(a.rss_mib*1024),MIN_AVAIL_MB='8192')
    live=dict(status='checking',current_job=None,completed=[],host=os.uname().nodename,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    def save():
        tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(live,indent=2)+'\n');tmp.replace(state)
    def frozen():
        for source,sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen source changed: '+source)
        for j in plan['jobs']:
            if hashlib.sha256((ROOT/j['queue']).read_bytes()).hexdigest()!=j['queue_sha256']:raise ValueError('Queue changed: '+j['queue'])
    def validate(job):
        r=json.loads((ROOT/job['result']).read_text())
        if r.get('status')!='completed':raise ValueError('Completed result required')
        if r['args']['tag']!=job['tag']:raise ValueError('Result identity differs')
        for source,sha in r.get('source_sha256',{}).items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=sha:raise ValueError('Result source changed: '+source)
        return r
    try:
        frozen();save()
        if shutil.which('nvidia-smi'):
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if gpu:raise ValueError('GPU occupied: preserve the existing job; no concurrent training')
        # Do not run underneath another host coordinator that may start its next job.
        processes=subprocess.check_output(['ps','-eo','args='],text=True)
        for line in processes.splitlines():
            if any(s in line for s in ('scripts/run_aws_non_shd.py','scripts/run_native_research_campaign.py',
                    'scripts/run_delay_feature_campaign.py','scripts/run_delay_feature_recovery.py')):
                raise ValueError('Existing coordinator must finish or be deliberately reserved before starting the gym')
        by_tag={j['tag']:j for j in plan['jobs']}
        for job in selected:
            frozen()
            for prerequisite in job['requires']:validate(by_tag[prerequisite])
            # Existing successful unchanged stages are revalidated; run_safe owns
            # both the host lock and the durable lifecycle marker/skip decision.
            live.update(status='running',current_job=job['tag']);save()
            subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,
                           env=dict(env,JOB_TIMEOUT_S=str(job['timeout_s'])),check=True)
            validate(job);live['completed'].append(job['result']);save()
        live.update(status='completed',current_job=None);save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise


if __name__=='__main__':main()
