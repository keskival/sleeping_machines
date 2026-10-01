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
    owner=plan['predecessor_pid'];child=None;held=False
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(live,indent=2)+'\n');temporary.replace(path)
    def frozen():
        for name,sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen source changed: '+name)
    def validate(name):
        r=json.loads((ROOT/name).read_text())
        if r['status']!='completed':raise ValueError('Completed stage required: '+name)
        for name,sha in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Completed source changed: '+name)
        return r
    def publish(job):
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
        command=(Path(f'/proc/{owner}/cmdline')).read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/run_native_research_campaign.py' not in command or plan['predecessor_stem'] not in command:
            raise ValueError('Exact live predecessor process required')
        suspended='T' in Path(f'/proc/{owner}/status').read_text().split('State:')[1].splitlines()[0]
        if suspended:
            reservation=json.loads((ROOT/plan['reservation']).read_text())
            start=Path(f'/proc/{owner}/stat').read_text().split()[21]
            if reservation['predecessor_pid']!=owner or reservation['command']!=command or reservation['start_ticks']!=start:
                raise ValueError('Suspended predecessor reservation identity differs')
            held=True
        frozen()
        prior_path=ROOT/f"experiments/queue/{plan['predecessor_stem']}.status.json"
        prior=json.loads(prior_path.read_text())
        if prior['status']!='running' or not prior.get('current_job'):raise ValueError('Active guarded fit required')
        # Suspend only its coordinator, never the runner/watchdog/trainer.
        if not suspended:os.kill(owner,signal.SIGSTOP)
        held=True
        live.update(predecessor_suspended=True,predecessor_current_job=prior['current_job'],status='awaiting_preserved_fit');save()
        def jobs(obj):
            if isinstance(obj,dict):
                if all(k in obj for k in ('tag','result','queue','timeout_s')):yield obj
                else:
                    for value in obj.values():yield from jobs(value)
            elif isinstance(obj,list):
                for value in obj:yield from jobs(value)
        predecessor=json.loads((ROOT/f"experiments/queue/{plan['predecessor_stem']}.json").read_text())
        current=next(j for j in jobs(predecessor) if j['tag']==prior['current_job'])
        deadline=time.monotonic()+current['timeout_s']+1800
        while not (ROOT/current['result']).exists():
            log=ROOT/f"experiments/queue/runner_{Path(current['queue']).stem}.out"
            if log.exists():
                text=log.read_text()
                failures=[line for line in text.splitlines() if ('done '+current['tag']+' (exit ') in line and '(exit 0)' not in line]
                if failures or 'STOP '+current['tag'] in text:raise ValueError('Preserved guarded fit failed; inspect its log')
            if time.monotonic()>deadline:raise TimeoutError('Preserved fit completion timed out')
            time.sleep(10)
        # run_safe also waits for release and skips the exact successful job,
        # preserving its watchdog/exit status before publication.
        run(current)
        baseline=validate(plan['native_language_reference']['result']) if current['tag']==plan['native_language_reference']['tag'] else run(plan['native_language_reference'])
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
        if held:
            os.kill(owner,signal.SIGCONT)
            live['predecessor_suspended']=False;live['predecessor_resumed']=True;save()


if __name__=='__main__':main()
