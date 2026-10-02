"""Freeze same-budget replication of the two successful full-core mechanisms."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stamp',default=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    a=p.parse_args();sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
    from experiments.split_event_benchmark import sources
    from scripts.run_aws_matrix_recovery import validate_result
    parent_path=ROOT/'experiments/gym/plans/aws_split_event_20261001T230029Z/manifest.json'
    old=json.loads(parent_path.read_text());old_jobs={j['tag']:j for j in old['jobs']}
    replication=json.loads((ROOT/'experiments/gym/plans/aws_event_replication_20261002T005408Z/manifest.json').read_text())
    replication_pilots={j['tag']:j for j in replication['jobs'] if j['stage']=='pilot'}
    cells=(('paired_timing_observed','paired_timing_S4_private_P0_observed'),
           ('paired_timing_rank','paired_timing_S4_private_P0_rank'),
           ('order_S16_shared','order_S16_shared_P0'),('order_S16_private','order_S16_private_P0'))
    prefix='aws_native_confirmation_'+a.stamp
    directory=ROOT/'experiments/gym/plans'/prefix
    if directory.exists():raise ValueError('Never overwrite a prepared plan')
    directory.mkdir();jobs=[];fingerprints=sources()
    caps=dict(MEM_CAP_KB=4000768,MEM_CAP_RSS_KB=2499584,MIN_AVAIL_MB=8192)

    def add(tag,variant,stage,seed,driver,args,result_dir,requires,timeout):
        queue='experiments/queue/'+tag+'.txt';text=' '.join([tag,driver,*args])+'\n'
        if (ROOT/queue).exists() or (ROOT/result_dir/(tag+'.json')).exists():raise ValueError('Unique artifacts required')
        (ROOT/queue).write_text(text)
        jobs.append(dict(tag=tag,domain='temporal',variant=variant,stage=stage,seed=seed,
            driver=driver,queue=queue,queue_sha256=hashlib.sha256(text.encode()).hexdigest(),
            result=result_dir+'/'+tag+'.json',requires=requires,timeout_s=timeout,
            resource_caps=caps,hypothesis='Replicate useful elapsed time or shared processing/private state at fixed complete fitting budget'))

    for variant,parent_variant in cells:
        old_prefix=old['run_prefix']+'_'+parent_variant+'_s6_'
        prereqs=[]
        for stage in ('contracts','smoke'):
            job=old_jobs[old_prefix+stage];validate_result(ROOT,job);jobs.append(job);prereqs.append(job['tag'])
        pilot=old_jobs[old_prefix+'pilot'];row=validate_result(ROOT,pilot)
        fingerprints[pilot['result']]=hashlib.sha256((ROOT/pilot['result']).read_bytes()).hexdigest()
        restore_tag=prefix+'_'+variant+'_s6_restore'
        add(restore_tag,variant,'contracts',6,'experiments/split_event_confirmation.py',
            ['--tag',restore_tag,'--parent',pilot['result'],'--contracts-only'],
            'experiments/results/event_confirmation',prereqs,600)
        tag=prefix+'_'+variant+'_s6_pilot'
        add(tag,variant,'pilot',6,'experiments/split_event_confirmation.py',
            ['--tag',tag,'--parent',pilot['result']],
            'experiments/results/event_confirmation',[restore_tag],600)
        settings=row['args']
        for seed in (7,8):
            tag=prefix+'_'+variant+f'_s{seed}_pilot'
            if variant!='order_S16_private':
                name=parent_variant+('_observed' if settings['task']=='order' else '')
                parent_job=replication_pilots[replication['run_prefix']+'_'+name+f'_s{seed}_pilot']
                restore_tag=prefix+'_'+variant+f'_s{seed}_restore'
                add(restore_tag,variant,'contracts',seed,'experiments/split_event_confirmation.py',
                    ['--tag',restore_tag,'--parent',parent_job['result'],'--contracts-only'],
                    'experiments/results/event_confirmation',prereqs,600)
                add(tag,variant,'pilot',seed,'experiments/split_event_confirmation.py',
                    ['--tag',tag,'--parent',parent_job['result']],
                    'experiments/results/event_confirmation',[restore_tag],600)
                continue
            args=['--tag',tag,'--task',settings['task'],'--sources',str(settings['sources']),
                  '--payload','8','--depth','8','--heads','2','--pool','2','--protected-pairs','0',
                  '--seed',str(seed),'--time-input',settings['time_input'],'--fit-targets','128',
                  '--dev-targets','256','--epochs','4','--update-targets','64','--lr','.003','--confirm']
            if settings['shared_maps']:args.append('--shared-maps')
            add(tag,variant,'pilot',seed,'experiments/split_event_benchmark.py',args,
                'experiments/results/event_variants',prereqs,3600)
    for name in ('experiments/split_event_confirmation.py','experiments/split_confirmation_analysis.py',
                 'scripts/plan_split_confirmation.py','scripts/run_aws_matrix_recovery.py','experiments/queue/run_safe.sh'):
        fingerprints[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    plan=dict(schema=1,status='prepared_not_started',run_prefix=prefix,jobs=jobs,source_sha256=fingerprints,
        parallel_host='ip-172-31-47-132',max_parallel_jobs=3,
        protocol='experiments/AWS_NATIVE_CONFIRMATION.md',
        predecessor_completed_summary='experiments/gym/plans/aws_event_replication_20261002T005408Z/completed_summary.json',
        admission='Confirm prior worker has exited and ordinary host lock is free; inspect memory/GPU/processes. '
                  'No second worker. Original completed contracts/smokes are validated/reused; four checkpoint '
                  'restoration contracts precede all holdout scores. Existing AWS replication must finish '
                  'before this worker starts; six seed7/8 fitted checkpoints are reused, not duplicated.',
        confirmation='1024 queries, seed3201, untouched until frozen protocol; seeds6/7/8, same fit/dev data and '
                     '128-query/four-pass budget. Seed6 reuses selected weights, zero new updates, original full-fit charge.',
        new_training_jobs=2,checkpoint_reuse_jobs=10,
        promotion='Two predeclared accuracy contrasts with crossed seed/population97.5% intervals; timing '
                  'lower gain>=25pp and every observed seed>=85%; sharing positive lower gain, all seed gains '
                  'positive and each full-work ratio<=1. No automatic scaling or supremacy claim.',
        timeout_basis='Observed three-slot S4/S16 pilots795–879s, RSS<976MiB. Added1024-query frozen '
                      'confirmation uses existing evaluator; retain3600s pilot timeout and600s checkpoint evaluation.',
        report_ownership='Local host updates canonical report from completed results')
    path=directory/'manifest.json';path.write_text(json.dumps(plan,indent=2)+'\n');print(path.relative_to(ROOT))


if __name__=='__main__':main()
