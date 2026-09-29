#!/usr/bin/env python3
"""Publish each AWS result on main, refresh the report PDF, and rebase between runs."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'refs/remotes/origin/aws/non-shd-benchmarks-20260929'
TARGET = 'refs/remotes/origin/main'
PUSH_TARGET = 'refs/heads/main'
BRANCH = 'aws/non-shd-benchmarks-20260929'
REPORT_WORKTREE = Path('/tmp/sm-aws-report-worktree')


def git(*args, cwd=ROOT, check=True, env=None):
    return subprocess.run(['git', *args], cwd=cwd, check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def remove_report_worktree():
    if REPORT_WORKTREE.exists():
        git('worktree', 'remove', '--force', str(REPORT_WORKTREE), check=False)
    git('worktree', 'prune', check=False)


def refresh_report():
    """Build report/PDF from the latest main snapshot, retrying concurrent main pushes."""
    for attempt in range(4):
        fetched = git('fetch', 'origin', check=False)
        if fetched.returncode:
            print(stamp(), 'report refresh fetch failed:', fetched.stdout.strip(), flush=True)
            time.sleep(3)
            continue
        base = git('rev-parse', TARGET).stdout.strip()
        remove_report_worktree()
        added = git('worktree', 'add', '--detach', str(REPORT_WORKTREE), base, check=False)
        if added.returncode:
            print(stamp(), 'report worktree failed:', added.stdout.strip(), flush=True)
            return False
        try:
            python = str(ROOT / '.venv-docker/bin/python')
            updated = subprocess.run([python, 'scripts/update_aws_benchmark_report.py'], cwd=REPORT_WORKTREE,
                                   check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if updated.returncode:
                print(stamp(), 'report markdown refresh failed:', updated.stdout.strip(), flush=True)
                return False
            env = dict(os.environ, MPLCONFIGDIR='/tmp/mpl_sm_report')
            built = subprocess.run([python, 'report/make_pdf.py'], cwd=REPORT_WORKTREE, check=False,
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            if built.returncode:
                print(stamp(), 'PDF build failed:', built.stdout.strip(), flush=True)
                return False
            paths = ['REPORT.md', 'experiments/FINDINGS.md', 'report/sleeping_machines_status.pdf']
            git('add', '--', *paths, cwd=REPORT_WORKTREE)
            changed = git('diff', '--cached', '--quiet', cwd=REPORT_WORKTREE, check=False)
            if changed.returncode == 0:
                print(stamp(), 'report and PDF already include all completed AWS results', flush=True)
                return True
            git('commit', '--only', '-m', 'Refresh research report and PDF from completed AWS runs',
                '--', *paths, cwd=REPORT_WORKTREE)
            pushed = git('push', 'origin', 'HEAD:' + PUSH_TARGET, cwd=REPORT_WORKTREE, check=False)
            if pushed.returncode:
                print(stamp(), 'main advanced during report build; rebuilding:', pushed.stdout.strip(), flush=True)
                continue
            print(stamp(), 'updated REPORT.md, findings, and PDF on main', flush=True)
            return True
        finally:
            remove_report_worktree()
    print(stamp(), 'could not refresh report after concurrent main updates', flush=True)
    return False


def main():
    mirrored = None
    reported = None
    while True:
        fetched = git('fetch', 'origin', check=False)
        if fetched.returncode:
            print(stamp(), 'fetch failed:', fetched.stdout.strip(), flush=True)
            time.sleep(15)
            continue
        source = git('rev-parse', '--verify', SOURCE, check=False)
        target = git('rev-parse', '--verify', TARGET, check=False)
        if source.returncode or target.returncode:
            print(stamp(), 'remote branch unavailable', flush=True)
            time.sleep(15)
            continue
        source_sha, target_sha = source.stdout.strip(), target.stdout.strip()
        main_sha = target_sha
        if source_sha != mirrored:
            source_in_main = git('merge-base', '--is-ancestor', source_sha, target_sha, check=False)
            if source_in_main.returncode == 0:
                mirrored = source_sha
            else:
                main_in_source = git('merge-base', '--is-ancestor', target_sha, source_sha, check=False)
                if main_in_source.returncode == 0:
                    candidate = source_sha
                else:
                    tree = git('merge-tree', '--write-tree', target_sha, source_sha, check=False)
                    if tree.returncode:
                        print(stamp(), 'merge conflict; stopped without changing main:', tree.stdout.strip(), flush=True)
                        return 1
                    commit = git('commit-tree', tree.stdout.strip(), '-p', target_sha, '-p', source_sha,
                                 '-m', 'Merge completed AWS benchmark work onto main')
                    candidate = commit.stdout.strip()
                pushed = git('push', 'origin', candidate + ':' + PUSH_TARGET, check=False)
                if pushed.returncode:
                    print(stamp(), 'main moved during push; retrying:', pushed.stdout.strip(), flush=True)
                    time.sleep(2)
                    continue
                mirrored = source_sha
                print(stamp(), 'published', source_sha, 'on main as', candidate, flush=True)
                git('fetch', 'origin', check=False)
                target_sha = git('rev-parse', TARGET).stdout.strip()
                main_sha = target_sha
                # Rebase the shared AWS branch to current main between benchmark jobs.
                if git('branch', '--show-current').stdout.strip() == BRANCH:
                    rebased = git('rebase', TARGET, check=False)
                    if rebased.returncode:
                        print(stamp(), 'rebase failed; aborting safely:', rebased.stdout.strip(), flush=True)
                        git('rebase', '--abort', check=False)
                    else:
                        synced = git('push', 'origin', 'HEAD:refs/heads/' + BRANCH, check=False)
                        if synced.returncode:
                            print(stamp(), 'branch sync deferred:', synced.stdout.strip(), flush=True)
                # Reports are based on the committed result tree, never on live files.
                refresh_report()
                git('fetch', 'origin', check=False)
                main_sha = git('rev-parse', TARGET).stdout.strip()
                reported = main_sha
        if main_sha != reported:
            refresh_report()
            reported = git('rev-parse', TARGET, check=False).stdout.strip()
        time.sleep(5)


if __name__ == '__main__':
    sys.exit(main())
