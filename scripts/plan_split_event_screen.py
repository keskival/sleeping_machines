"""Prepare immutable one-job queues for theory-driven native event variations."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.split_event_benchmark import sources


def cells():
    rows = []
    for population in (4, 16):
        for shared in (False, True):
            for protected in (0, 2):
                rows.append(dict(name=f"order_S{population}_{'shared' if shared else 'private'}_P{protected}",
                    task='order', sources=population, shared=shared, protected=protected, time='observed'))
    for name, shared, protected, clock in (
        ('paired_shared_temporal', True, 0, 'observed'),
        ('paired_shared_rank', True, 0, 'rank'),
        ('paired_shared_protected', True, 2, 'observed')):
        rows.append(dict(name=name, task='paired_timing', sources=4, shared=shared,
                         protected=protected, time=clock))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-prefix', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'(local|aws)_split_screen_[A-Za-z0-9_]+', args.run_prefix):
        raise ValueError('Unique host-prefixed split-screen tag required')
    directory = ROOT/'experiments/gym/plans'/args.run_prefix
    if directory.exists():
        raise ValueError('Preserve prepared plans; use a new run prefix')
    jobs = []
    queues = {}
    for stage in ('contracts', 'smoke', 'pilot'):
        for cell in cells():
            stem = args.run_prefix+'_'+cell['name']+'_s6'
            tag = stem+'_'+stage
            minimum = 2*cell['sources'] if cell['task']=='paired_timing' else cell['sources']
            budget = (['--fit-targets','128','--dev-targets','256','--epochs','4','--update-targets','64']
                      if stage=='pilot' else ['--fit-targets',str(minimum),'--dev-targets',str(minimum),
                          '--epochs','1','--update-targets',str(minimum)])
            extra = ['--shared-maps'] if cell['shared'] else []
            if stage=='contracts':
                extra += ['--contracts-only']
            command = ' '.join([tag, 'experiments/split_event_benchmark.py','--tag',tag,
                '--task',cell['task'],'--sources',str(cell['sources']),'--payload','8','--depth','8',
                '--heads','2','--pool','2','--seed','6','--time-input',cell['time'],
                '--protected-pairs',str(cell['protected']), *budget, *extra])+'\n'
            queue = 'experiments/queue/'+tag+'.txt'
            if (ROOT/queue).exists():
                raise ValueError('Existing queue: '+queue)
            queues[queue] = command
            requirements = [] if stage=='contracts' else [stem+'_contracts']
            if stage=='pilot':
                requirements += [stem+'_smoke']
            jobs.append(dict(tag=tag, domain='temporal', variant=cell['name'], stage=stage,
                seed=6, driver='experiments/split_event_benchmark.py', queue=queue,
                queue_sha256=hashlib.sha256(command.encode()).hexdigest(),
                result='experiments/results/event_variants/'+tag+'.json',requires=requirements,
                timeout_s=1800 if stage!='pilot' else 5400,
                resource_caps=dict(MEM_CAP_KB=4000768,MEM_CAP_RSS_KB=2499584,MIN_AVAIL_MB=8192),
                hypothesis='Protected silent retention x shared processing/private state; paired timing identification'))
    hashes = sources()
    for path in ('scripts/plan_split_event_screen.py','scripts/run_aws_matrix_recovery.py',
                 'scripts/run_research_gym.py','experiments/queue/run_safe.sh'):
        hashes[path] = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    directory.mkdir(parents=True)
    for path, command in queues.items():
        (ROOT/path).write_text(command)
    plan = dict(schema=1,status='prepared_not_started',run_prefix=args.run_prefix,
        jobs=jobs,source_sha256=hashes,parallel_host='ip-172-31-47-132' if args.run_prefix.startswith('aws_') else None,
        protocol='experiments/ADVANTAGE_BATTERY.md',
        concurrency='Serial locally; up to three bounded CPU slots only on the explicitly authorized AWS host',
        publication='Existing AWS worker commits raw results; report publisher ingests completed event_variants only',
        promotion='No automatic long run; paired seeds and independent held-out confirmation before benchmark claims',
        prerequisites='Finish and retire current matrix/coordinator before launch; all contracts/smokes precede pilots',
        blocked=[dict(domain='robotics',reason='Unequal-timestamp whole-trial adapter pending'),
                 dict(domain='recognized_event',reason='Integrated SHD/DVS adapter and matched modern controls pending'),
                 dict(domain='language',reason='Protected variant language adapter pending event evidence'),
                 dict(domain='hardware',reason='No synthesized/trained clockless ASIC or measured energy implementation')])
    (directory/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(manifest=str((directory/'manifest.json').relative_to(ROOT)),stages=len(jobs),pilots=11)))


if __name__=='__main__':
    main()
