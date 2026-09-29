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


def main():
    os.chdir(ROOT)
    if git('branch', '--show-current') != BRANCH:
        raise RuntimeError('Use the dedicated AWS benchmark branch')
    jobs = json.loads(PLAN.read_text())
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    for job in jobs:
        tag = job['run_tag']
        if tag in progress:
            continue
        out = ROOT / 'experiments/results/aws_20260929' / tag
        if out.exists():
            raise RuntimeError(f'Unreviewed existing output: {out}')
        q = QUEUE / (tag + '.txt')
        command = ['experiments/aws_benchmark.py', '--run-tag', tag, '--script', job['script'], '--', *job['arguments']]
        q.write_text(tag + ' ' + shlex.join(command) + '\n')
        env = dict(os.environ, WAIT='1', PYTHONUNBUFFERED='1', MIN_AVAIL_MB='8192',
                   MEM_CAP_KB=str(job.get('mem_cap_kb', 6000000)),
                   MEM_CAP_RSS_KB=str(job.get('rss_cap_kb', 3500000)),
                   JOB_TIMEOUT_S=str(job.get('timeout_s', 21600)))
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), tag, flush=True)
        subprocess.run(['free', '-h'], check=True)
        subprocess.run(['ps', '-eo', 'pid,etime,comm'], stdout=subprocess.DEVNULL, check=True)
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
        subprocess.run(['git', 'push', 'origin', f'HEAD:refs/heads/{BRANCH}'], check=True)
        # A host-wide memory shortage stops the batch. A per-model failure is
        # recorded and independent configurations may still run safely.
        if result.returncode in (130, 143) or (lifecycle.exists() and 'below 8192MB' in lifecycle.read_text()):
            raise RuntimeError('Host pressure or interruption: batch stopped')
    print('All scheduled non-SHD jobs attempted; review progress and unresolved audit entries.', flush=True)


if __name__ == '__main__':
    main()
