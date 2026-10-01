"""Prepare guarded banknote confirmation; do not displace the event battery."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.tabular_confirmation_benchmark import sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-prefix',required=True)
    a=parser.parse_args()
    if not re.fullmatch(r'aws_banknote_confirmation_[A-Za-z0-9_]+',a.run_prefix):
        raise ValueError('Unique AWS-specific confirmation prefix required')
    directory=ROOT/'experiments/gym/plans'/a.run_prefix
    if directory.exists():raise ValueError('Never overwrite a prepared plan')
    old='experiments/results/native_tabular/aws_fast_matrix_recovery_20261001T213409Z_banknote_ours_R0_s6_pilot.json'
    jobs=[];queues={}
    for stage in ('contracts','smoke','pilot'):
        for family in ('ours','trees','catboost','logistic'):
            for seed in ((6,7,8) if stage=='pilot' else (6,)):
                tag=f'{a.run_prefix}_{family}_s{seed}_{stage}'
                args=['experiments/tabular_confirmation_benchmark.py','--tag',tag,'--model',family,'--seed',str(seed)]
                if stage=='contracts':args+=['--contracts-only','--fit','16','--dev','16','--epochs','1']
                elif stage=='smoke':args+=['--fit','16','--dev','16','--epochs','1']
                else:
                    args+=['--fit','128','--dev','128','--epochs','4','--confirm']
                    if family=='ours' and seed==6:args+=['--reuse-result',old]
                queue=f'experiments/queue/{tag}.txt'
                command=shlex.join([tag,*args])+'\n'
                if (ROOT/queue).exists():raise ValueError('Existing queue: '+queue)
                queues[queue]=command
                base=f'{a.run_prefix}_{family}_s6'
                requires=[] if stage=='contracts' else [base+'_contracts']
                if stage=='pilot':requires+=[base+'_smoke']
                jobs.append(dict(tag=tag,domain='tabular',variant=family,stage=stage,seed=seed,
                    driver=args[0],queue=queue,queue_sha256=hashlib.sha256(command.encode()).hexdigest(),
                    result=f'experiments/results/tabular_confirmation/{tag}.json',requires=requires,
                    timeout_s=1800 if stage!='pilot' else 5400,
                    resource_caps=dict(MEM_CAP_KB=4000768,MEM_CAP_RSS_KB=2499584,MIN_AVAIL_MB=8192),
                    hypothesis='Frozen native quality confirmation versus three conventional controls; full learning cost retained'))
    hashes=sources()
    for name in ('scripts/plan_tabular_confirmation.py','scripts/run_aws_matrix_recovery.py',
                 'scripts/run_research_gym.py','experiments/queue/run_safe.sh',
                 'experiments/tabular_confirmation_analysis.py'):
        hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    directory.mkdir(parents=True)
    for name,command in queues.items():(ROOT/name).write_text(command)
    plan=dict(schema=1,status='prepared_not_started',run_prefix=a.run_prefix,jobs=jobs,
        source_sha256=hashes,parallel_host='ip-172-31-47-132',protocol='experiments/AWS_BANKNOTE_CONFIRMATION.md',
        prerequisites='Finish/retire split-event worker first; pinned control dependencies; seed6 checkpoint available; all contracts/smokes before pilots',
        preserved_priority='experiments/gym/plans/aws_split_event_20261001T230029Z/manifest.json',
        reuse=dict(result=old,checkpoint=old.replace('.json','.progress.pt'),new_native_fits=2),
        confirmation=dict(seeds=[6,7,8],fit_rows=128,dev_rows=128,epochs=4,
            test='All reserved feature-group-disjoint banknote rows; no architecture/tuning choices from test',
            primary='NLL against each control, Bonferroni three-comparison intervals; accuracy descriptive',
            scope='One fixed data split, three initialization seeds, no ensemble and no physical-energy comparison'))
    (directory/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(manifest=str((directory/'manifest.json').relative_to(ROOT)),stages=len(jobs),pilots=12,new_native_fits=2)))


if __name__=='__main__':main()
