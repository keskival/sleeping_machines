"""Priority-ordered native capability/language/capacity ladder; one job per host."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest-stem',required=True)
    stem=parser.parse_args().manifest_stem
    if Path(stem).name!=stem:raise ValueError('Local manifest stem required')
    plan=json.loads((ROOT/f'experiments/queue/{stem}.json').read_text())
    status=ROOT/f'experiments/queue/{stem}.status.json'
    if status.exists():raise ValueError('Do not duplicate supervisor or overwrite its status')
    live=dict(status='checking',completed=[],decisions=[],current_job=None)
    def save():
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(live,indent=2)+'\n');temp.replace(status)
    def frozen():
        for name,sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Queued source changed: '+name)
    def validate(name):
        r=json.loads((ROOT/name).read_text())
        if r['status']!='completed':raise ValueError('Completed record required: '+name)
        for source,sha in r['source_sha256'].items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=sha:raise ValueError('Result source changed: '+source)
        return r
    def run(job,timeout=None):
        frozen();live.update(status='running',current_job=job['tag']);save()
        env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},
            JOB_TIMEOUT_S=str(timeout or job['timeout_s']),WAIT='1',WAIT_TIMEOUT_S='86400')
        subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env,check=True)
        r=validate(job['result']);live['completed'].append(job['result']);save()
        changes=subprocess.check_output(['git','status','--porcelain','--',job['result']],cwd=ROOT,text=True)
        if changes.strip():subprocess.run([sys.executable,'scripts/publish_native_research_stage.py',job['result']],cwd=ROOT,check=True)
        return r
    def event_gate(r,task):
        final=r['final'];quality=final['dev']['accuracy'];memory_gap=quality-final['cleared_state']['accuracy']
        return quality>=plan['minimum_order_accuracy'] and memory_gap>=plan['minimum_memory_gap'] if task=='order' else quality>=plan['minimum_timing_accuracy']
    try:
        save();frozen()
        prior=json.loads((ROOT/plan['prioritized_status']).read_text())
        if prior['status']!='completed_bounded_write_credit_campaign' or prior.get('current_job'):
            raise ValueError('Predecessor is not finished')
        for path in plan['completed_contracts']:validate(path)
        reference=validate(plan['language_reference'])
        for job in plan['smokes']:run(job)
        order=run(plan['order_pilot'])
        language=run(plan['language_pilot'])
        timing=run(plan['timing_pilot'])
        eligible={};large_language=None
        for task,r in (('order',order),('timing',timing)):
            # Refit diagnostics even if the full pilot is weak: a failed
            # capability gate does not reveal which mechanism caused it.
            control=run(plan['controls'][task])
            live[task+'_control_gain']=r['final']['dev']['accuracy']-control['final']['dev']['accuracy'];save()
            if event_gate(r,task):
                eligible[task]=r
            else:
                live['decisions'].append(task+' misses native capability gate; retain result and diagnose before its scaling');save()
        # Independent language branch: compare the same data/selection/work boundary.
        bpc=lambda r:r['final']['dev']['bpc']
        work=lambda r:r['work']['cpu_emulator']['total_training_unit_special_flops']
        if (bpc(language)<=bpc(reference) and work(language)<work(reference)
                and language['fitting_data_sha256']==reference['fitting_data_sha256']
                and language['development_data_sha256']==reference['development_data_sha256']):
            eight=run(plan['language_eight_k'])
            control=validate(plan['indexed_control_eight_k'])
            gain=bpc(language)-bpc(eight);live['native_language_data_gain_bpc']=gain;save()
            if gain>=plan['minimum_language_data_gain_bpc'] and bpc(eight)<=bpc(control)+plan['maximum_language_control_regression_bpc']:
                larger=run(plan['language_thirtytwo_k'])
                repeats=[run(job) for job in plan['language_repeats']]
                gain=bpc(eight)-bpc(larger);timeout=math.ceil(larger['wall_s']*4*1.5+1800)
                live.update(native_language_larger_gain_bpc=gain,projected_131k_timeout_s=timeout);save()
                if (gain>=plan['minimum_larger_data_gain_bpc'] and larger['max_rss_kb']<=plan['maximum_rss_for_large_language_kb']
                        and max(bpc(r) for r in repeats)<=bpc(control)+plan['maximum_language_control_regression_bpc']
                        and timeout<=plan['maximum_timeout_s']):
                    # Occupied-capacity and independent event replications
                    # precede a potentially day-long language extension.
                    large_language=(plan['language_onehundredthirtyone_k'],timeout)
                else:live['decisions'].append('Native language131K deferred by measured quality/replication/memory/time gate');save()
            else:live['decisions'].append('Native language8K misses data/control gate; no32K');save()
        else:live['decisions'].append('Native language2K misses same-data quality/work gate; no larger fit');save()
        if 'order' in eligible:
            previous=order
            for stage in plan['capacity_stages']:
                run(stage['smoke']);r=run(stage['fit'])
                if r['final']['dev']['accuracy']<previous['final']['dev']['accuracy']-plan['maximum_capacity_regression']:
                    live['decisions'].append('Occupied capacity quality regresses beyond tolerance; stop population ladder');save();break
                previous=r
        for task in eligible:
            if task=='timing' and live.get('timing_control_gain',0)<plan['minimum_time_control_gain']:
                live['decisions'].append('Elapsed-time refit gap misses replication gate');save();continue
            for job in plan['event_repeats'][task]:run(job)
        if large_language is not None:run(*large_language)
        live.update(status='completed_bounded_native_research_campaign',current_job=None);save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise


if __name__=='__main__':main()
