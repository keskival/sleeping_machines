#!/usr/bin/env python3
"""Run the reviewed AWS manifest serially via run_safe.sh; publish each outcome."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/workspace')
QUEUE = ROOT / 'experiments/queue'
PLAN = QUEUE / 'aws_plan_20260929.json'
PROGRESS = QUEUE / 'aws_progress_20260929.json'
BRANCH = 'aws/non-shd-benchmarks-20260929'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def publish_result():
    """Keep retrying transient remote failures and rebase after concurrent main updates."""
    delay = 15
    while True:
        fetched = subprocess.run(['git', 'fetch', 'origin'], cwd=ROOT, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if fetched.returncode:
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'fetch failed before publish; retrying:', fetched.stdout.strip(), flush=True)
        else:
            rebased = subprocess.run(['git', 'rebase', 'origin/main'], cwd=ROOT, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if rebased.returncode:
                subprocess.run(['git', 'rebase', '--abort'], cwd=ROOT, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'rebase onto main failed; retrying:', rebased.stdout.strip(), flush=True)
            else:
                pushed = subprocess.run(['git', 'push', 'origin',
                                         'HEAD:refs/heads/' + BRANCH], cwd=ROOT, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if pushed.returncode == 0:
                    print(pushed.stdout.strip(), flush=True)
                    return
                print(datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'push failed; retrying:', pushed.stdout.strip(), flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 300)


def main():
    os.chdir(ROOT)
    if git('branch', '--show-current') != BRANCH:
        raise RuntimeError('Use the dedicated AWS benchmark branch')
    jobs = json.loads(PLAN.read_text())
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    supplemental_progress = QUEUE / 'aws_progress_90m_baselines.json'
    if supplemental_progress.exists():
        progress.update(json.loads(supplemental_progress.read_text()))
    for job in jobs:
        tag = job['run_tag']
        if tag in progress:
            continue
        arguments = list(job['arguments'])
        if job.get('prepare'):
            dependency = ROOT / 'experiments/results/aws_20260929' / job['dependency_tag']
            provenance = dependency / 'provenance.json'
            if not provenance.exists() or json.loads(provenance.read_text())['status'] != 'completed':
                progress[tag] = dict(status='blocked', dependency=str(dependency),
                                     reason='Required benchmark did not complete')
                PROGRESS.write_text(json.dumps(progress, indent=2)+'\n')
                print(tag, 'blocked on', dependency, flush=True)
                continue
            if job['prepare'] == 'e78':
                arrays = list(dependency.glob('*_ptrue_valid.npy'))
                if len(arrays) != 1:
                    raise RuntimeError(f'Expected one E77 probability array in {dependency}')
                arguments += ['--e77_dir', str(dependency), '--e77', arrays[0].name.removesuffix('_ptrue_valid.npy')]
            elif job['prepare'] == 'e80':
                results = list(dependency.glob('tv_market_val_*.json'))
                if len(results) != 1:
                    raise RuntimeError(f'Expected one E80 validation result in {dependency}')
                result = json.loads(results[0].read_text())
                arguments += ['--mode', 'test', '--epochs', str(result['best_epoch']), '--policy_file', str(results[0])]
                for key in ('L', 'fine', 'cross', 'mag', 'd', 'M', 'lr', 'batch', 'fees'):
                    arguments += ['--' + key, str(result['args'][key])]
        out = ROOT / 'experiments/results/aws_20260929' / tag
        if out.exists():
            raise RuntimeError(f'Unreviewed existing output: {out}')
        q = QUEUE / (tag + '.txt')
        command = ['experiments/aws_benchmark.py', '--run-tag', tag, '--script', job['script']]
        if job.get('stdout_only'):
            command.append('--stdout-only')
        command += ['--', *arguments]
        q.write_text(tag + ' ' + shlex.join(command) + '\n')
        env = dict(os.environ, WAIT='1', PYTHONUNBUFFERED='1', MIN_AVAIL_MB='8192',
                   MEM_CAP_KB=str(job.get('mem_cap_kb', 6000000)),
                   MEM_CAP_RSS_KB=str(job.get('rss_cap_kb', 3500000)),
                   JOB_TIMEOUT_S=str(job.get('timeout_s', 21600)))
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), tag, flush=True)
        subprocess.run(['free', '-h'], check=True)
        processes = subprocess.check_output(['ps', '-eo', 'pid,etime,args'], text=True)
        print('\n'.join(line for line in processes.splitlines()
                        if 'run_safe.sh' in line or 'python experiments/' in line), flush=True)
        if Path('/usr/bin/nvidia-smi').exists():
            subprocess.run(['nvidia-smi'], check=True)
        started = time.monotonic()
        result = subprocess.run(['./experiments/queue/run_safe.sh', str(q.relative_to(ROOT))], env=env)
        lifecycle = QUEUE / ('runner_' + tag + '.out')
        log = QUEUE / 'logs' / (tag + '.log')
        record = dict(exit_code=result.returncode, wall_s=round(time.monotonic()-started, 3),
                      status='completed' if result.returncode == 0 else 'failed',
                      command=shlex.join(command), limits={k: env[k] for k in
                      ('MEM_CAP_KB', 'MEM_CAP_RSS_KB', 'MIN_AVAIL_MB', 'JOB_TIMEOUT_S')})
        provenance = out / 'provenance.json'
        if provenance.exists():
            meta = json.loads(provenance.read_text())
            record['peak_rss_kb'] = meta.get('peak_rss_kb')
            if result.returncode != 0 and meta['status'] == 'running':
                meta.update(status='terminated', exit_code=result.returncode)
                provenance.write_text(json.dumps(meta, indent=2)+'\n')
        progress[tag] = record
        PROGRESS.write_text(json.dumps(progress, indent=2)+'\n')
        files = [q, PROGRESS, lifecycle, log, *out.glob('*.json')]
        files = [str(p.relative_to(ROOT)) for p in files if p.exists()]
        if git('branch', '--show-current') != BRANCH:
            raise RuntimeError('Branch changed; refusing to commit benchmark results')
        subprocess.run(['git', 'add', '--', *files], check=True)
        subprocess.run(['git', 'commit', '--only', '-m', f'Record {tag}: {record["status"]}', '--', *files], check=True)
        publish_result()
        # A host-wide memory shortage stops the batch. A per-model failure is
        # recorded and independent configurations may still run safely.
        if result.returncode in (130, 143) or (lifecycle.exists() and 'below 8192MB' in lifecycle.read_text()):
            raise RuntimeError('Host pressure or interruption: batch stopped')
    print('All scheduled non-SHD jobs attempted; review progress and unresolved audit entries.', flush=True)


if __name__ == '__main__':
    main()
