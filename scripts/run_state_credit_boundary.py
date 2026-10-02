"""Bounded guarded insertion at an existing job boundary; resume coordinator."""
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
from scripts.run_aws_matrix_recovery import validate_result


def identity(pid):
    path=Path(f'/proc/{pid}')
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        return dict(state=fields[0],ppid=int(fields[1]),start_ticks=fields[19],
            command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    args=parser.parse_args();plan_path=ROOT/args.plan;plan=json.loads(plan_path.read_text())
    state=plan_path.with_suffix('.status.json')
    if state.exists():raise ValueError('Preserve existing lifecycle; never restart this insertion blindly')
    if plan['host']!=os.uname().nodename:raise ValueError('Reservation is host-specific')
    live=dict(status='checking',completed=[],coordinator_reserved=False,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def save():
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps(live,indent=2)+'\n');temp.replace(state)
    def frozen():
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='main':raise ValueError('Main required')
        for name,sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen source changed: '+name)
        for job in plan['jobs']:
            if hashlib.sha256((ROOT/job['queue']).read_bytes()).hexdigest()!=job['queue_sha256']:raise ValueError('Queue changed: '+job['tag'])
    def publish(paths,message):
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m',message,'--',*paths],cwd=ROOT,check=True)
    coordinator=plan['coordinator'];guard=plan['predecessor_guard'];reserved=False
    save()
    def interrupted(signum,frame):raise InterruptedError('Supervisor signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    try:
        frozen();c=identity(coordinator['pid']);g=identity(guard['pid'])
        if not c or c['state'] in ('T','t','Z') or c['start_ticks']!=coordinator['start_ticks'] or coordinator['command_fragment'] not in c['command']:
            raise ValueError('Coordinator identity changed')
        if not g or g['start_ticks']!=guard['start_ticks'] or g['ppid']!=coordinator['pid']:
            raise ValueError('Current guard no longer belongs to coordinator; do not reserve a new stage')
        os.kill(coordinator['pid'],signal.SIGSTOP);reserved=True
        live.update(status='waiting_for_preserved_trainer',coordinator_reserved=True);save()
        deadline=time.monotonic()+plan['maximum_boundary_wait_s']
        while True:
            g=identity(guard['pid'])
            if not g or g['state']=='Z':break
            if g['start_ticks']!=guard['start_ticks']:raise ValueError('Guard PID reused')
            if time.monotonic()>deadline:raise TimeoutError('Bounded wait expired; resume original coordinator')
            time.sleep(5)
        predecessor=json.loads((ROOT/plan['predecessor_result']).read_text())
        if predecessor['status']!='completed':raise ValueError('Preserved predecessor did not complete')
        for job in plan['jobs']:
            frozen();live.update(status='running',current_job=job['tag']);save()
            env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},
                WAIT='1',WAIT_TIMEOUT_S='120',JOB_TIMEOUT_S=str(job['timeout_s']))
            subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env,check=True)
            row=validate_result(ROOT,job)
            if job['stage']=='smoke' and row['max_rss_kb']>.85*plan['resource_caps']['MEM_CAP_RSS_KB']:
                raise ValueError('Smoke near RSS cap; do not scale')
            publish([job['result']],'Record completed '+job['tag'])
            live['completed'].append(dict(tag=job['tag'],result=job['result'],wall_s=row.get('wall_s'),max_rss_kb=row.get('max_rss_kb')));save()
        pilots=[validate_result(ROOT,j) for j in plan['jobs'] if j['stage']=='pilot']
        baseline,teacher=pilots
        if baseline['data_sha256']!=teacher['data_sha256']:raise ValueError('Matched data required')
        accuracy_gain=teacher['final']['dev']['accuracy']-baseline['final']['dev']['accuracy']
        nll_gain=baseline['final']['dev']['nll']-teacher['final']['dev']['nll']
        ratio=teacher['work']['total_training_unit_special_flops']/baseline['work']['total_training_unit_special_flops']
        analysis=plan_path.with_name(plan_path.stem+'_analysis.json')
        summary=dict(status='completed',kind='exploratory_state_credit_comparison',
            source_results=[j['result'] for j in plan['jobs'] if j['stage']=='pilot'],
            accuracy_gain=accuracy_gain,nll_gain=nll_gain,whole_fitting_work_ratio=ratio,
            followup_gate_passed=accuracy_gain>=.05 and nll_gain>=.02 and ratio<=2,
            scope='Single seed, reused development populations; no benchmark or unbiased-gradient claim. No automatic larger fit.')
        analysis.write_text(json.dumps(summary,indent=2)+'\n')
        publish([str(analysis.relative_to(ROOT))],'Compare matched integrated addressed-state credit pilots')
        subprocess.run([str(ROOT/'.venv-docker/bin/python'),'report/make_pdf.py'],cwd=ROOT,check=True)
        report_paths=['REPORT.md','report/sleeping_machines_status.pdf']
        figure='report/figures/state_credit_quality_work.png'
        if (ROOT/figure).exists():report_paths.append(figure)
        publish(report_paths,'Publish completed state-write credit quality and work')
        live.update(status='completed',comparison=summary,current_job=None);save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise
    finally:
        c=identity(coordinator['pid'])
        if reserved and c and c['start_ticks']==coordinator['start_ticks']:os.kill(coordinator['pid'],signal.SIGCONT)
        live['coordinator_reserved']=False;save()


if __name__=='__main__':main()
