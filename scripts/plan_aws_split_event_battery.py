"""Freeze the first protected/shared-state battery specified by theory section357."""
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
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = 'aws_split_event_' + stamp
    directory = ROOT / 'experiments/gym/plans' / prefix
    directory.mkdir()
    cells = []
    for population in (4, 16):
        for shared in (False, True):
            for protected in (0, 2):
                cells.append(dict(name=f'order_S{population}_{"shared" if shared else "private"}_P{protected}',
                                  task='order', sources=population, shared=shared, protected=protected, time='observed'))
    for protected, mode in ((0, 'observed'), (0, 'rank'), (2, 'observed')):
        cells.append(dict(name=f'paired_timing_S4_private_P{protected}_{mode}', task='paired_timing',
                          sources=4, shared=False, protected=protected, time=mode))
    jobs = []
    for cell in cells:
        requirements = []
        for stage in ('contracts', 'smoke', 'pilot'):
            tag = prefix + '_' + cell['name'] + '_s6_' + stage
            smoke_targets = cell['sources'] * (2 if cell['task'] == 'paired_timing' else 1)
            fit, dev, epochs, update = (smoke_targets, smoke_targets, 1, smoke_targets) if stage == 'smoke' else (128, 256, 4, 64)
            arguments = ['--tag', tag, '--task', cell['task'], '--sources', str(cell['sources']),
                         '--protected-pairs', str(cell['protected']), '--time-input', cell['time'],
                         '--payload', '8', '--depth', '8', '--heads', '2', '--pool', '2', '--seed', '6',
                         '--fit-targets', str(fit), '--dev-targets', str(dev), '--epochs', str(epochs),
                         '--update-targets', str(update)]
            if cell['shared']:
                arguments.append('--shared-maps')
            if stage == 'contracts':
                arguments.append('--contracts-only')
            queue = 'experiments/queue/' + tag + '.txt'
            command = ' '.join([tag, 'experiments/split_event_benchmark.py', *arguments]) + '\n'
            if (ROOT / queue).exists():
                raise ValueError('Never overwrite a queue')
            (ROOT / queue).write_text(command)
            jobs.append(dict(tag=tag, domain='temporal', variant=cell['name'], stage=stage, seed=6,
                driver='experiments/split_event_benchmark.py', queue=queue,
                queue_sha256=hashlib.sha256(command.encode()).hexdigest(),
                result='experiments/results/event_variants/' + tag + '.json', requires=list(requirements),
                timeout_s=3600 if stage == 'pilot' else 1800,
                resource_caps=dict(MEM_CAP_KB=4000768, MEM_CAP_RSS_KB=2499584, MIN_AVAIL_MB=8192),
                hypothesis='Protected silent retention versus temporal sensitivity; shared rules versus private addressed state'))
            requirements.append(tag)
    fingerprints = sources()
    for name in ('scripts/plan_aws_split_event_battery.py', 'scripts/run_aws_matrix_recovery.py', 'experiments/queue/run_safe.sh'):
        fingerprints[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    plan = dict(schema=1, status='prepared_not_started', run_prefix=prefix, jobs=jobs,
                source_sha256=fingerprints, parallel_host=os.uname().nodename, max_parallel_jobs=3,
                protocol='experiments/theory/53_protected_state_and_shared_processing.md',
                concurrency='User-authorized3 CPU slots; reserved global lock, slot guards,8GiB floor, serial result publication',
                promotion='No automatic confirmation or language scaling; assess completed quality/work first',
                comparison_scope='Sharing also replaces private source embeddings; no map-only attribution without embedding-only control',
                predecessor='experiments/gym/plans/aws_fast_matrix_recovery_20261001T213409Z/worker_recovery.status.json',
                measured_timeout_basis='Saved3-slot S4/S16 order pilots789/829s at64 dev queries; new256-dev work proxy2.59x,1.5x margin fits3600s',
                independent_seeds='One architecture seed6; paired seeds/confirmation deferred',
                mechanisms='Eight temporal/sparse addressed blocks,H2,d8, separate key/value roles, counterfactual learning retained; no KV bank')
    path = directory / 'manifest.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
