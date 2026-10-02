"""Freeze matched development replications; no new holdout or architecture."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(ROOT / 'experiments'))
    from split_event_benchmark import sources
    prefix = 'aws_event_replication_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = ROOT / 'experiments/gym/plans' / prefix
    directory.mkdir()
    jobs = []
    cells = [('paired_timing', 4, False, 0, 'observed'),
             ('paired_timing', 4, False, 0, 'rank'),
             ('order', 16, True, 0, 'observed'),
             ('order', 16, True, 2, 'observed')]
    for seed in (7, 8):
        for task, population, shared, protected, mode in cells:
            name = f'{task}_S{population}_{"shared" if shared else "private"}_P{protected}_{mode}'
            requirements = []
            for stage in ('contracts', 'smoke', 'pilot'):
                tag = f'{prefix}_{name}_s{seed}_{stage}'
                small = population * (2 if task == 'paired_timing' else 1)
                fit, dev, epochs, update = (small, small, 1, small) if stage == 'smoke' else (128, 256, 4, 64)
                args = ['--tag', tag, '--task', task, '--sources', str(population),
                        '--protected-pairs', str(protected), '--time-input', mode,
                        '--payload', '8', '--depth', '8', '--heads', '2', '--pool', '2',
                        '--seed', str(seed), '--fit-targets', str(fit), '--dev-targets', str(dev),
                        '--epochs', str(epochs), '--update-targets', str(update)]
                if shared:
                    args.append('--shared-maps')
                if stage == 'contracts':
                    args.append('--contracts-only')
                queue = f'experiments/queue/{tag}.txt'
                command = ' '.join([tag, 'experiments/split_event_benchmark.py', *args]) + '\n'
                with (ROOT / queue).open('x') as stream:
                    stream.write(command)
                jobs.append(dict(tag=tag, domain='temporal', variant=name, stage=stage, seed=seed,
                    queue=queue, queue_sha256=hashlib.sha256(command.encode()).hexdigest(),
                    result=f'experiments/results/event_variants/{tag}.json', requires=list(requirements),
                    timeout_s=3600 if stage == 'pilot' else 1800,
                    resource_caps=dict(MEM_CAP_KB=4000768, MEM_CAP_RSS_KB=2499584, MIN_AVAIL_MB=8192)))
                requirements.append(tag)
    fingerprints = sources()
    for name in ('scripts/plan_aws_event_replication.py', 'scripts/run_aws_matrix_recovery.py', 'experiments/queue/run_safe.sh'):
        fingerprints[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    plan = dict(schema=1, status='prepared_not_started', run_prefix=prefix, jobs=jobs,
        source_sha256=fingerprints, parallel_host=os.uname().nodename, max_parallel_jobs=3,
        protocol='experiments/AWS_EVENT_REPLICATION.md',
        scope='Seed7/8 training replication on unchanged fit/dev populations; selected after seed6; no fresh confirmation evidence',
        measured_timeout_basis='Completed same-settings seed6 pilots797–879s under three slots;3600s allows margin',
        predecessor='experiments/gym/plans/aws_banknote_confirmation_20261001T234000Z/completed_summary.json')
    path = directory / 'manifest.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
