#!/usr/bin/env python3
"""Commit a supplemental 90M result before run_safe releases its global queue lock."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path('/workspace')
QUEUE = ROOT / 'experiments/queue'
PLAN = QUEUE / 'aws_plan_20260929.json'
PROGRESS = QUEUE / 'aws_progress_90m_baselines.json'
BRANCH = 'main'


def git(*args, check=True):
    return subprocess.run(['git', *args], cwd=ROOT, check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def publish():
    delay = 5
    while True:
        if git('branch', '--show-current').stdout.strip() != BRANCH:
            raise RuntimeError('AWS development must stay on main; refusing to publish')
        fetched = git('fetch', 'origin', check=False)
        if fetched.returncode:
            print(stamp(), 'fetch failed; retrying:', fetched.stdout.strip(), flush=True)
        else:
            rebased = git('rebase', 'origin/main', check=False)
            if rebased.returncode:
                git('rebase', '--abort', check=False)
                raise RuntimeError('Rebase onto origin/main failed; resolve it on main '
                                   'before restarting publication:\n' + rebased.stdout.strip())
            else:
                if git('branch', '--show-current').stdout.strip() != BRANCH:
                    raise RuntimeError('Branch changed; refusing to publish benchmark results')
                pushed = git('push', 'origin', 'HEAD:refs/heads/main', check=False)
                if pushed.returncode == 0:
                    return
                print(stamp(), 'push failed; retrying:', pushed.stdout.strip(), flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 300)


def main():
    if git('branch', '--show-current').stdout.strip() != BRANCH:
        raise RuntimeError('AWS development and benchmark commits must happen on main')
    tag = os.environ['AFTER_JOB_NAME']
    exit_code = int(os.environ.get('AFTER_JOB_EXIT_CODE', '1'))
    jobs = {job['run_tag']: job for job in json.loads(PLAN.read_text())}
    job = jobs[tag]
    result_dir = ROOT / 'experiments/results/aws_20260929' / tag
    provenance_path = result_dir / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    status = 'completed' if exit_code == 0 and provenance.get('status') == 'completed' else 'failed'
    if status != 'completed' and provenance.get('status') == 'running':
        provenance.update(status='terminated', exit_code=exit_code)
        provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')

    args = list(job['arguments'])
    command = ['experiments/aws_benchmark.py', '--run-tag', tag, '--script', job['script'], '--', *args]
    qfile = QUEUE / f'{tag}.txt'
    qfile.write_text(tag + ' ' + shlex.join(command) + '\n')
    record = dict(exit_code=exit_code, wall_s=provenance.get('wall_s'), status=status,
                  command=shlex.join(command), limits=provenance.get('limits'),
                  peak_rss_kb=provenance.get('peak_rss_kb'))
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    progress[tag] = record
    PROGRESS.write_text(json.dumps(progress, indent=2) + '\n')

    # Runner logs remain local; provenance and progress retain their metrics.
    paths = [qfile, PROGRESS, *result_dir.glob('*.json')]
    if status == 'completed':
        updated = subprocess.run([str(ROOT / '.venv-docker/bin/python'),
                                  'scripts/update_aws_benchmark_report.py'], cwd=ROOT,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(updated.stdout.strip(), flush=True)
        if updated.returncode == 0:
            paths.append(ROOT / 'report/figures/accomplishments.png')
            env = dict(os.environ, MPLCONFIGDIR='/tmp/mpl_sm_report')
            built = subprocess.run([str(ROOT / '.venv-docker/bin/python'), 'report/make_pdf.py'],
                                   cwd=ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if built.returncode == 0:
                paths.extend([ROOT / 'REPORT.md', ROOT / 'report/sleeping_machines_status.pdf'])
                print(stamp(), 'rebuilt report PDF', flush=True)
            else:
                print(stamp(), 'PDF rebuild deferred; upstream report inputs are missing:',
                      built.stdout.strip().splitlines()[-1], flush=True)

    paths = [str(path.relative_to(ROOT)) for path in paths if path.exists()]
    if git('branch', '--show-current').stdout.strip() != BRANCH:
        raise RuntimeError('Branch changed; refusing to commit benchmark results')
    git('add', '--', *paths)
    git('commit', '--only', '-m', f'Record {tag}: {status}', '--', *paths)
    publish()
    print(stamp(), 'published', tag, 'to main', flush=True)


if __name__ == '__main__':
    main()
