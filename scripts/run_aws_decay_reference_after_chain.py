"""One guarded diagnostic after the native chain and its summary publisher."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
TAG='aws_decay_timing_reference_20261002T024100Z'


def wait_process(pid,script):
    proc=Path(f'/proc/{pid}')
    if not proc.exists():return
    if script.encode() not in (proc/'cmdline').read_bytes():raise ValueError('Unexpected predecessor')
    identity=(proc/'stat').read_text().split()[21]
    while proc.exists():
        try:fields=(proc/'stat').read_text().split()
        except FileNotFoundError:return
        if fields[21]!=identity:raise ValueError('PID reused')
        if fields[2]=='Z':return
        time.sleep(10)


def main():
    wait_process(788298,'scripts/run_aws_independent_event_recovery.py')
    wait_process(801289,'scripts/run_aws_replication_summary.py')
    plan=json.loads((ROOT/f'experiments/queue/{TAG}.plan.json').read_text())
    for name,sha in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen diagnostic source changed')
    if hashlib.sha256((ROOT/plan['queue']).read_bytes()).hexdigest()!=plan['queue_sha256']:
        raise ValueError('Frozen diagnostic queue changed')
    subprocess.run(['free','-m'],check=True)
    env={**os.environ,'MEM_CAP_KB':'3000000','MEM_CAP_RSS_KB':'1250000',
         'MIN_AVAIL_MB':'8192','JOB_TIMEOUT_S':'300','WAIT':'1','WAIT_TIMEOUT_S':'600'}
    subprocess.run(['bash','experiments/queue/run_safe.sh',f'experiments/queue/{TAG}.txt'],cwd=ROOT,env=env,check=True)
    result=f'experiments/results/diagnostics/{TAG}.json'
    with open('/tmp/experiments-runner.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for command in (['git','add',result],['git','commit','-m','Publish accounted causal two-trace timing diagnostic'],['git','pull','--rebase'],['git','push']):
            subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
