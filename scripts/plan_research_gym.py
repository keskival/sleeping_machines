"""Offline first-wave planner: unique guarded queues, prerequisites and hypotheses."""
import argparse
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]


def specifications():
    cells=[]
    for task,credit,time_input,sources,depth,name in (
        ('timing','counterfactual','observed',4,8,'timing_full'),
        ('timing','counterfactual','rank',4,8,'timing_rank'),
        ('order','counterfactual','observed',4,8,'order_full'),
        ('order','pathwise','observed',4,8,'order_pathwise'),
        ('order','counterfactual','observed',16,8,'order_sources16'),
        ('order','counterfactual','observed',64,8,'order_sources64'),
        ('order','counterfactual','observed',4,4,'order_shallow_control')):
        cells.append(dict(name=name,domain='temporal',driver='experiments/native_event_benchmark.py',
            result_dir='native_event',timeout_s=3600,hypothesis='Elapsed-time credit, hard-route learning or occupied capacity at fixed events/activity',
            args=['--task',task,'--credit',credit,'--time-input',time_input,'--sources',str(sources),
                  '--depth',str(depth),'--payload','8','--heads','2','--pool','2'],
            pilot=['--fit-targets','128','--dev-targets','64','--epochs','4','--update-targets','64'],
            smoke=['--fit-targets',str(sources),'--dev-targets',str(sources),'--epochs','1','--update-targets',str(sources)]))
    for features,allocation,readout,name in ((0,'uniform','on','language_native'),(2,'uniform','on','language_reception'),
                                           (4,'late','on','language_late'),(2,'uniform','off','language_waiting')):
        cells.append(dict(name=name,domain='language',driver='experiments/clock_feature_language_benchmark.py',
            result_dir='clock_feature_language',timeout_s=5400,hypothesis='Useful delay readout and depth allocation at matched added budget',
            args=['--clock-features',str(features),'--clock-allocation',allocation,'--clock-readout',readout,
                  '--payload','8','--depth','8','--heads','2','--pool','2'],
            pilot=['--fit','512','--dev','1024','--epochs','4'],
            smoke=['--fit','65','--dev','128','--epochs','1']))
    for dataset in ('banknote','wine_red'):
        for model,features in (('ours',0),('ours',2),('trees',0)):
            cells.append(dict(name=dataset+'_'+model+'_R'+str(features),domain='tabular',
                driver='experiments/native_tabular_benchmark.py',result_dir='native_tabular',timeout_s=3600,
                hypothesis='Static conditional feature interactions; temporal reception versus native parent and strong trees',
                args=['--dataset',dataset,'--model',model,'--clock-features',str(features),'--payload','8','--depth','8','--heads','2','--pool','2'],
                pilot=['--fit','128','--dev','128','--epochs','4'],smoke=['--fit','16','--dev','16','--epochs','1']))
    return cells


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-prefix',required=True)
    p.add_argument('--domains',nargs='+',choices=('temporal','language','tabular'),default=['temporal','language','tabular'])
    p.add_argument('--seeds',nargs='+',type=int,default=[6]);a=p.parse_args()
    if not re.fullmatch(r'aws_[A-Za-z0-9_]+',a.run_prefix):raise ValueError('Unique safe AWS prefix required')
    if len(set(a.seeds))!=len(a.seeds):raise ValueError('Repeated seed')
    directory=ROOT/'experiments/gym/plans'/a.run_prefix
    if directory.exists():raise ValueError('Never overwrite a plan; use a new prefix')
    # Include the frozen integrated lineage and every new adapter/planning source.
    parent=json.loads((ROOT/'experiments/queue/local_delay_feature_campaign_20261001T202000Z.json').read_text())
    names={n for n in parent['source_sha256'] if n.endswith('.py') and not n.startswith('scripts/publish')}
    names.update(('scripts/plan_research_gym.py','scripts/run_research_gym.py','experiments/native_tabular_benchmark.py',
        'experiments/native_tabular_model.py','experiments/native_tabular_data.py','experiments/tabular_data_manifest.json'))
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
    jobs=[];queues={}
    for cell in specifications():
        if cell['domain'] not in a.domains:continue
        for seed in a.seeds:
            previous=[]
            for stage in ('contracts','smoke','pilot'):
                tag=f"{a.run_prefix}_{cell['name']}_s{seed}_{stage}"
                args=cell['args']+['--seed',str(seed)]+cell['pilot' if stage=='contracts' else stage]
                if stage=='contracts':args+=['--contracts-only']
                command=' '.join([tag,cell['driver'],'--tag',tag,*args])+'\n'
                queue='experiments/queue/'+tag+'.txt'
                if (ROOT/queue).exists():raise ValueError('Queue exists: '+queue)
                queues[queue]=command
                jobs.append(dict(tag=tag,domain=cell['domain'],variant=cell['name'],stage=stage,seed=seed,
                    driver=cell['driver'],queue=queue,queue_sha256=hashlib.sha256(command.encode()).hexdigest(),
                    result=f"experiments/results/{cell['result_dir']}/{tag}.json",requires=list(previous),
                    timeout_s=1800 if stage!='pilot' else cell['timeout_s'],hypothesis=cell['hypothesis']))
                previous.append(tag)
    directory.mkdir(parents=True)
    for name,command in queues.items():(ROOT/name).write_text(command)
    plan=dict(schema=1,status='prepared_not_started',run_prefix=a.run_prefix,jobs=jobs,source_sha256=hashes,
        concurrency='One guarded training job per host; concurrent workers on different hosts only',
        protocol='experiments/AWS_EARLY_INDICATION_MATRIX.md',promotion='Separate paired-seed confirmation, no automatic large-model promotion',
        blocked=[dict(domain='robotics',dataset='MIT planar pushing',reason='Raw unequal-timestamp causal regression adapter and immutable trial split not yet implemented'),
                 dict(domain='robotics',dataset='UCI Robot Execution Failures',reason='Trial-sequence adapter and per-problem vocabulary/splits not yet implemented')])
    (directory/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(f"Prepared {len(jobs)} guarded stages ({sum(j['stage']=='pilot' for j in jobs)} pilots); 2 explicitly blocked robotics adapters.")
    print(str((directory/'manifest.json').relative_to(ROOT)))


if __name__=='__main__':main()
