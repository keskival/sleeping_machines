"""Publish the completed capacity inventory after its worker releases the host."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PID=901862
DIRECTORY='experiments/gym/plans/aws_event_history_20261002T123343Z'
OUTPUT='experiments/results/diagnostics/aws_event_history_20261002T123343Z_analysis.json'


def main():
    proc=Path(f'/proc/{PID}')
    if proc.exists():
        if (DIRECTORY+'/manifest.json').encode() not in (proc/'cmdline').read_bytes():raise ValueError('Unexpected worker')
        identity=(proc/'stat').read_text().split()[21]
        while proc.exists():
            try:fields=(proc/'stat').read_text().split()
            except FileNotFoundError:break
            if fields[21]!=identity:raise ValueError('Worker PID reused')
            if fields[2]=='Z':break
            time.sleep(10)
    if json.loads((ROOT/DIRECTORY/'completed_summary.json').read_text())['status']!='completed':
        raise ValueError('Complete capacity battery required')
    with open('/tmp/experiments-runner.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def run(*command):subprocess.run(command,cwd=ROOT,check=True)
        run(sys.executable,'-m','experiments.event_history_analysis','--manifest',DIRECTORY+'/manifest.json','--output',OUTPUT)
        run('git','add',OUTPUT);run('git','commit','-m','Publish complete fresh native/history quality and resource comparison')
        run('git','pull','--rebase');run('git','push')


if __name__=='__main__':main()
