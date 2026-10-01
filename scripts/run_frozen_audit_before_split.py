"""Bounded diagnostic insertion; preserve live trainer and resume split waiter."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RESERVATION = ROOT / 'experiments/queue/aws_frozen_route_audit_reservation_20261001T233000Z.json'
TAG = 'aws_frozen_native_route_audit_20261001T233000Z'


def main():
    reservation = json.loads(RESERVATION.read_text()); pid = reservation['pid']
    proc = Path(f'/proc/{pid}')
    if (proc/'stat').read_text().split()[21] != reservation['start_ticks']:
        raise ValueError('Split waiter identity changed')
    state = ROOT / 'experiments/queue/aws_frozen_route_audit_20261001T233000Z.status.json'
    if state.exists():
        raise ValueError('Never duplicate this diagnostic lifecycle')
    live = dict(status='waiting_for_preserved_first_matrix', split_waiter_reserved=True)
    def save():
        state.write_text(json.dumps(live, indent=2)+'\n')
    save()
    try:
        predecessor = Path('/proc/696038')
        while predecessor.exists():
            if 'Z' in (predecessor/'status').read_text().split('State:')[1].splitlines()[0]:
                break
            time.sleep(5)
        live['status']='running_frozen_audit'; save()
        env = dict(os.environ, WAIT='1', WAIT_TIMEOUT_S='1800', MEM_CAP_KB='4000768',
                   MEM_CAP_RSS_KB='2499584', MIN_AVAIL_MB='8192', JOB_TIMEOUT_S='300')
        subprocess.run(['bash','experiments/queue/run_safe.sh',f'experiments/queue/{TAG}.txt'],cwd=ROOT,env=env,check=True)
        result=f'experiments/results/diagnostics/{TAG}.json'
        record=json.loads((ROOT/result).read_text())
        if record['status']!='completed':raise ValueError('Completed audit required')
        subprocess.run(['git','add','--',result],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','Publish frozen native route/state credit diagnostic','--',result],cwd=ROOT,check=True)
        subprocess.run(['git','pull','--rebase'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
        live.update(status='completed',summary=record['summary']);save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise
    finally:
        if proc.exists() and (proc/'stat').read_text().split()[21]==reservation['start_ticks']:
            os.kill(pid,signal.SIGCONT)
        live['split_waiter_reserved']=False;save()


if __name__=='__main__':main()
