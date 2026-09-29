#!/usr/bin/env python3
"""Merge each published AWS batch commit onto main without losing concurrent work."""
import datetime
import subprocess
import sys
import time

SOURCE = 'refs/remotes/origin/aws/non-shd-benchmarks-20260929'
TARGET = 'refs/remotes/origin/main'

def git(*args, check=True):
    return subprocess.run(['git', *args], check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    mirrored = None
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
        if source_sha == mirrored:
            time.sleep(5)
            continue
        contained = git('merge-base', '--is-ancestor', source_sha, target_sha, check=False)
        if contained.returncode == 0:
            mirrored = source_sha
            time.sleep(5)
            continue
        tree = git('merge-tree', '--write-tree', target_sha, source_sha, check=False)
        if tree.returncode:
            print(stamp(), 'merge conflict; stopped without changing main:', tree.stdout.strip(), flush=True)
            return 1
        commit = git('commit-tree', tree.stdout.strip(), '-p', target_sha, '-p', source_sha,
                     '-m', 'Merge completed AWS benchmark work onto main')
        pushed = git('push', 'origin', commit.stdout.strip() + ':' + TARGET, check=False)
        if pushed.returncode:
            print(stamp(), 'main moved during push; retrying:', pushed.stdout.strip(), flush=True)
            time.sleep(2)
            continue
        mirrored = source_sha
        print(stamp(), 'published', source_sha, 'on main as', commit.stdout.strip(), flush=True)
        time.sleep(5)


if __name__ == '__main__':
    sys.exit(main())
