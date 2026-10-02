"""Publish the complete replication inventory after the reserved chain exits."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT='experiments/results/diagnostics/aws_event_replication_20261002T005408Z_analysis.json'


def main():
    proc=Path('/proc/788298')
    if proc.exists():
        if b'scripts/run_aws_independent_event_recovery.py' not in (proc/'cmdline').read_bytes():
            raise ValueError('Unexpected chain process')
        identity=(proc/'stat').read_text().split()[21]
        while proc.exists():
            try:fields=(proc/'stat').read_text().split()
            except FileNotFoundError:break
            if fields[21]!=identity:raise ValueError('Chain PID reused')
            if fields[2]=='Z':break
            time.sleep(10)
    # No experiment source changes. Reserve ordinary admission during publication.
    with open('/tmp/experiments-runner.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def run(*command):subprocess.run(command,cwd=ROOT,check=True)
        run(sys.executable,'-m','experiments.event_replication_analysis','--output',OUTPUT)
        run('git','add',OUTPUT);run('git','commit','-m','Publish complete three-seed event replication quality and resource inventory')
        run('git','pull','--rebase');run('git','push')


if __name__=='__main__':main()
