"""Bounded priority insertion; preserve active fit and resume frozen campaign."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-stem',required=True)
    stem=parser.parse_args().manifest_stem
    if Path(stem).name!=stem:raise ValueError('Local stem required')
    plan=json.loads((ROOT/f'experiments/queue/{stem}.json').read_text())
    path=ROOT/f'experiments/queue/{stem}.status.json'
    if path.exists():raise ValueError('Never duplicate a priority coordinator')
    live=dict(status='checking',completed=[],decisions=[],current_job=None,predecessor_suspended=False)
    child=None;held=False
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(live,indent=2)+'\n');temporary.replace(path)
    def repository_ready():
        # A pull/rebase must not turn a completed fit into a lost publication.
        deadline=time.monotonic()+600
        artifacts=['REPORT.md','report/sleeping_machines_status.pdf','report/figures',
                   'report/readable_report.py','report/make_pdf.py']
        while True:
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
            gitdir=Path(subprocess.check_output(['git','rev-parse','--absolute-git-dir'],cwd=ROOT,text=True).strip())
            busy=any((gitdir/name).exists() for name in ('rebase-merge','rebase-apply','MERGE_HEAD','index.lock'))
            staged=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode
            edited=subprocess.run(['git','diff','--quiet','--',*artifacts],cwd=ROOT).returncode
            if branch=='main' and not busy and not staged and not edited:return
            if time.monotonic()>deadline:raise TimeoutError('Git/report busy; completed evidence retained for review')
            time.sleep(5)
    def frozen():
        repository_ready()
        for name,sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen source changed: '+name)
    def validate(name):
        r=json.loads((ROOT/name).read_text())
        if r['status']!='completed':raise ValueError('Completed stage required: '+name)
        for name,sha in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Completed source changed: '+name)
        return r
    def publish(job):
        repository_ready()
        result=validate(job['result'])
        changes=subprocess.check_output(['git','status','--porcelain','--',job['result']],cwd=ROOT,text=True)
        if changes.strip():
            publisher='scripts/publish_native_research_stage.py' if '/native_language/' in job['result'] or '/native_event/' in job['result'] else 'scripts/publish_delay_feature_stage.py'
            subprocess.run([sys.executable,publisher,job['result']],cwd=ROOT,check=True)
        live['completed'].append(job['result']);save();return result
    def run(job,timeout=None):
        nonlocal child
        frozen();live.update(status='running',current_job=job['tag']);save()
        env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},
            JOB_TIMEOUT_S=str(timeout or job['timeout_s']),WAIT='1',WAIT_TIMEOUT_S='86400')
        child=subprocess.Popen(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env)
        rc=child.wait();child=None
        if rc:raise ValueError(f"Guarded stage {job['tag']} failed with {rc}")
        return publish(job)
    def interrupted(signum,frame):
        raise RuntimeError(f'Priority coordinator received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        save()
        frozen()
        # Recovery starts only after the old coordinators and trainers exited.
        processes=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
        forbidden=('scripts/run_native_research_campaign.py','scripts/run_delay_feature_campaign.py',
                   'experiments/clock_feature_language_benchmark.py','experiments/native_event_benchmark.py',
                   'experiments/native_language_benchmark.py')
        if any(any(token in line for token in forbidden) for line in processes.splitlines()):
            raise ValueError('Another coordinator/trainer is alive; inspect before recovery')
        predecessor=json.loads((ROOT/plan['archived_predecessor_status']).read_text())
        if predecessor['status']!='needs_review':raise ValueError('Recorded exited predecessor failure required')
        baseline=run(plan['native_language_reference'])
        for job in plan['contracts']:run(job)
        for job in plan['smokes']:run(job)
        pilots=[run(job) for job in plan['pilots']]
        bpc=lambda r:r['final']['dev']['bpc']
        work=lambda r:r['work']['projected_event_architecture']['total_training_unit_special_flops']
        best=min(pilots,key=bpc);clock=best['args']['clock_features']
        variant=str(clock)+'_'+best['args']['clock_allocation']
        control=run(plan['waiting_controls'][variant])
        matched=all(r['fitting_data_sha256']==baseline['fitting_data_sha256'] and
            r['development_data_sha256']==baseline['development_data_sha256'] for r in pilots+[control])
        if not matched:raise ValueError('Pilot data hashes differ')
        live.update(selected_clocks=clock,selected_allocation=best['args']['clock_allocation'],gain_vs_native_bpc=bpc(baseline)-bpc(best),
            gain_vs_waiting_bpc=bpc(control)-bpc(best),projected_fit_work_ratio=work(best)/work(baseline));save()
        if (live['gain_vs_native_bpc']>=plan['minimum_gain_bpc'] and live['gain_vs_waiting_bpc']>=plan['minimum_readout_gain_bpc']
                and live['projected_fit_work_ratio']<=plan['maximum_work_ratio']):
            timeout=math.ceil(best['wall_s']*4*1.5+1800)
            if timeout<=plan['maximum_timeout_s'] and best['max_rss_kb']<=plan['maximum_pilot_rss_kb']:
                eight=run(plan['eight_k'][variant],timeout)
                live.update(data_gain_bpc=bpc(best)-bpc(eight));save()
                # Larger fits await this completed stage and the preserved
                # native timing/capacity/replication campaign, not speculative extrapolation.
            else:live['decisions'].append('8K deferred by measured timeout/memory gate');save()
        else:live['decisions'].append('Pilot misses quality/readout/work gate; no larger delay-feature fit');save()
        # A separate predeclared near-quality/resource envelope tests whether
        # the small native parent benefits from more data despite a slight
        # quality regression against the much larger saved KV construction.
        reference=validate(plan['saved_language_reference'])
        if bpc(baseline)<=bpc(reference)+plan['native_quality_tolerance_bpc'] and work(baseline)<=work(reference)*plan['native_maximum_work_ratio']:
            timeout=math.ceil(baseline['wall_s']*4*1.5+1800)
            if timeout<=plan['maximum_timeout_s']:run(plan['native_eight_k'],timeout)
        for job in plan.get('tabular_contracts',[]):run(job)
        for job in plan.get('tabular_smokes',[]):run(job)
        for job in plan.get('tabular_pilots',[]):run(job)
        live.update(status='completed_bounded_delay_feature_campaign',current_job=None);save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        # Reuse immutable parent queues. Its recovery has a distinct status file;
        # successful unchanged jobs are skipped, not retrained.
        if live.get('status')=='completed_bounded_delay_feature_campaign':
            frozen()
            subprocess.run(['tmux','new-session','-d','-s',plan['native_recovery_stem'],
                sys.executable+' scripts/run_native_research_campaign.py --manifest-stem '+plan['native_recovery_stem']+
                ' >> experiments/queue/'+plan['native_recovery_stem']+'.out 2>&1'],cwd=ROOT,check=True)
            live['native_recovery_started']=plan['native_recovery_stem'];save()


if __name__=='__main__':main()
