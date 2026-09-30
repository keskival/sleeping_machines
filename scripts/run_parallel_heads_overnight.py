"""Prioritized, serial full-architecture language campaign with stage commits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM='local_parallel_heads_overnight_20260930T231500Z'


def main():
    plan=json.loads((ROOT/f'experiments/queue/{STEM}.json').read_text())
    status_path=ROOT/f'experiments/queue/{STEM}.status.json'
    live=dict(status='running',completed=[],decisions=[])
    def save():status_path.write_text(json.dumps(live,indent=2)+'\n')
    def validate(path):
        r=json.loads((ROOT/path).read_text())
        if r['status']!='completed':raise ValueError('Completed record required: '+path)
        for name,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Source changed: '+name)
        return r
    def publish():
        # Initial reviewed commit already contains reused checks/pilot records.
        changes=subprocess.check_output(['git','status','--porcelain','--',*live['completed']],cwd=ROOT,text=True)
        if not changes.strip():return
        subprocess.run([sys.executable,'scripts/update_episodic_language_report.py',
                        *live['completed']],cwd=ROOT,check=True)
    def run(job):
        for name,digest in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Preserve queued model/driver sources: '+name)
        live['current_job']=job['tag'];save()
        env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},
                 JOB_TIMEOUT_S=str(job['timeout_s']),WAIT='1',WAIT_TIMEOUT_S='86400')
        subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env,check=True)
        r=validate(job['result']);live['completed'].append(job['result']);save()
        publish();return r
    try:
        # These existing guarded jobs are skipped on success; an active pilot
        # finishes under its original lock before any later work can start.
        run(plan['head2_contracts']);run(plan['head2_smoke'])
        baseline=run(plan['head2_reference_pilot'])
        run(plan['accumulation_contracts']);run(plan['accumulation_smoke'])
        trials={name:run(job) for name,job in plan['optimizer_pilots'].items()}
        quality=lambda r:r['final']['dev']['bpc']
        best=min(map(quality,trials.values()))
        acceptable={name:r for name,r in trials.items()
            if quality(r)<=best+plan['optimizer_quality_tolerance_bpc'] and
               quality(r)<=quality(baseline)+plan['maximum_optimizer_regression_bpc']}
        if not acceptable:
            raise ValueError('No stable accumulated-optimizer pilot; keep reference and diagnose')
        chosen=min(acceptable,key=lambda name:trials[name]['work']['cpu_emulator']['total_training_unit_special_flops'])
        live['optimizer_config']=chosen;live['decisions'].append('Choose lowest work within declared quality tolerance');save()
        head2=run(plan['head2_eight_k'][chosen])
        run(plan['head4_contracts']);run(plan['head4_smoke'][chosen])
        head4pilot=run(plan['head4_pilots'][chosen])
        eligible={2:head2}
        if quality(head4pilot)<=plan['maximum_pilot_bpc']:
            eligible[4]=run(plan['head4_eight_k'][chosen])
        else:
            live['decisions'].append('Four-head pilot exceeds absolute stability gate; larger four-head fit deferred');save()
        best=min(map(quality,eligible.values()))
        contenders={h:r for h,r in eligible.items() if quality(r)<=best+plan['head_quality_tolerance_bpc']}
        heads=min(contenders,key=lambda h:contenders[h]['work']['cpu_emulator']['total_training_unit_special_flops'])
        live['selected_heads']=heads;save()
        control=validate(plan['single_head_eight_k'])
        if quality(eligible[heads])>quality(control)+plan['maximum_full_model_regression_bpc']:
            raise ValueError('Parallel-head 8K quality requires diagnosis before longer fitting')
        larger=run(plan['thirtytwo_k'][str(heads)][chosen])
        # Repeatability is more useful than another small hyperparameter grid.
        run(plan['repeat_eight_k'][str(heads)][chosen])
        gain=quality(eligible[heads])-quality(larger)
        live['thirtytwo_k_data_gain_bpc']=gain
        # Estimate added K/V + position/index array payload before longer work;
        # measured 32K RSS supplies the rest of the actual workload footprint.
        extra_bytes=(131072-32768)*8*heads*(2*32*4+16)
        estimated_rss=larger['max_rss_kb']+extra_bytes/1024
        live['projected_131k_peak_rss_kb']=estimated_rss;save()
        if gain>=plan['minimum_larger_data_gain_bpc'] and estimated_rss<=plan['maximum_projected_rss_kb']:
            run(plan['onehundredthirtyone_k'][str(heads)][chosen])
        else:
            live['decisions'].append('131K deferred: quality gain or measured-workload memory gate not satisfied');save()
        live.update(status='completed_bounded_parallel_head_campaign',current_job=None);save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise


if __name__=='__main__':main()
