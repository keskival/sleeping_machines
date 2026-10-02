"""Reserve only an identified coordinator; insert one guarded short audit."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_state_credit_boundary import identity
from scripts.run_aws_matrix_recovery import validate_result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', required=True)
    args = parser.parse_args(); path = ROOT/args.plan; plan = json.loads(path.read_text())
    status = path.with_suffix('.status.json')
    if status.exists(): raise ValueError('Preserve existing insertion lifecycle')
    coordinator, guard = plan['coordinator'], plan['predecessor_guard']
    live = dict(status='checking', coordinator_reserved=False, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    reserved = False; child = None

    def save():
        temporary = status.with_suffix('.tmp'); temporary.write_text(json.dumps(live, indent=2)+'\n'); temporary.replace(status)

    def verify():
        assert os.uname().nodename == plan['host']
        assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'main'
        for name, sha in plan['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha, name
        job = plan['job']; queue = ROOT/job['queue']
        assert hashlib.sha256(queue.read_bytes()).hexdigest() == job['queue_sha256']
        assert len([s for s in queue.read_text().splitlines() if s.strip() and not s.startswith('#')]) == 1

    def interrupted(signum, frame): raise InterruptedError('Signal '+str(signum))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT): signal.signal(sig, interrupted)
    save()
    try:
        verify(); c, g = identity(coordinator['pid']), identity(guard['pid'])
        assert c and c['state'] not in ('T','t','Z') and c['start_ticks'] == coordinator['start_ticks']
        assert coordinator['command_fragment'] in c['command']
        assert g and g['start_ticks'] == guard['start_ticks'] and g['ppid'] == coordinator['pid']
        os.kill(coordinator['pid'], signal.SIGSTOP); reserved = True
        # Verify the captured boundary again after reserving the coordinator.
        g = identity(guard['pid'])
        assert not g or g['start_ticks'] == guard['start_ticks']
        live.update(status='waiting_for_preserved_trainer', coordinator_reserved=True); save()
        deadline = time.monotonic()+plan['maximum_boundary_wait_s']
        while True:
            g = identity(guard['pid'])
            if not g or g['state'] == 'Z': break
            assert g['start_ticks'] == guard['start_ticks'], 'Guard PID reused'
            if time.monotonic() > deadline: raise TimeoutError('Bounded boundary wait expired')
            time.sleep(5)
        predecessor = json.loads((ROOT/plan['predecessor_result']).read_text())
        assert predecessor['status'] == 'completed', 'Preserve failed predecessor'
        verify(); job = plan['job']; live.update(status='running_audit'); save()
        env = dict(os.environ, **{k:str(v) for k,v in plan['resource_caps'].items()}, JOB_TIMEOUT_S=str(job['timeout_s']))
        for key in ('AWS_GYM_SLOT','AWS_GYM_HOST_LOCK_FD','WAIT','AFTER_JOB_HOOK','CONTINUE_ON_FAILURE'): env.pop(key, None)
        child = subprocess.Popen(['bash','experiments/queue/run_safe.sh',job['queue']], cwd=ROOT, env=env)
        rc = child.wait(); child = None
        if rc: raise RuntimeError('Guarded audit exit '+str(rc))
        row = validate_result(ROOT, job); verify()
        assert row['max_rss_kb'] <= .85*plan['resource_caps']['MEM_CAP_RSS_KB'], 'Review unexpectedly large diagnostic'
        subprocess.run(['git','add','--',job['result']], cwd=ROOT, check=True)
        subprocess.run(['git','commit','--only','-m','Record frozen language credit and context-dependence audit','--',job['result']], cwd=ROOT, check=True)
        live.update(status='completed', result=job['result'], wall_s=row['wall_s'], max_rss_kb=row['max_rss_kb']); save()
    except BaseException as error:
        live.update(status='needs_review', error=str(error)); save(); raise
    finally:
        try:
            if child is not None and child.poll() is None:
                child.terminate(); child.wait(timeout=30)
        finally:
            c = identity(coordinator['pid'])
            if reserved and c and c['start_ticks'] == coordinator['start_ticks']:
                os.kill(coordinator['pid'], signal.SIGCONT)
            live['coordinator_reserved'] = False; save()


if __name__ == '__main__': main()
