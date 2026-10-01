"""Finish reserved depth4, publish it, then run the prescribed smoke/pilot screen."""
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
TAG = 'aws_e19_race_res_d4_P64000_20260929'
PLAN = 'experiments/gym/plans/aws_fast_matrix_v1_20261001T213000Z/manifest.json'
RESERVATION = ROOT / 'experiments/queue/aws_after_depth4_20261001T211941Z.reservation.json'
STATE = ROOT / 'experiments/queue/aws_fast_matrix_after_depth4.status.json'


def main():
    os.chdir(ROOT)
    if STATE.exists():
        raise RuntimeError('Preserve existing lifecycle; do not duplicate this supervisor')
    live = {'status': 'waiting_for_depth4', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}

    def save():
        temporary = STATE.with_suffix('.tmp')
        temporary.write_text(json.dumps(live, indent=2) + '\n')
        temporary.replace(STATE)

    def run(*args):
        subprocess.run(args, cwd=ROOT, check=True)

    def frozen():
        plan = json.loads((ROOT / PLAN).read_text())
        for name, sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('Frozen source changed: ' + name)
        for job in plan['jobs']:
            if hashlib.sha256((ROOT / job['queue']).read_bytes()).hexdigest() != job['queue_sha256']:
                raise RuntimeError('Queue changed: ' + job['queue'])

    def publish_matrix(state, message):
        files = state['completed']
        changes = subprocess.check_output(['git', 'status', '--porcelain', '--', *files], text=True)
        if changes.strip():
            run('git', 'add', '--', *files)
            run('git', 'commit', '--only', '-m', message, '--', *files)
            run('git', 'pull', '--rebase')
            run('git', 'push', 'origin', 'main')

    save()
    try:
        frozen()
        reservation = json.loads(RESERVATION.read_text())
        pid = reservation['predecessor_pid']
        proc = Path(f'/proc/{pid}')
        if (proc / 'stat').read_text().split()[21] != reservation['start_ticks']:
            raise RuntimeError('Reserved coordinator identity changed')
        if (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode() != reservation['command']:
            raise RuntimeError('Reserved coordinator command changed')
        if 'T' not in (proc / 'status').read_text().split('State:')[1].splitlines()[0]:
            raise RuntimeError('Coordinator must remain suspended until its child finishes')
        log = ROOT / f'experiments/queue/runner_{TAG}.out'
        deadline = time.monotonic() + 21600
        while f'done {TAG} (exit 0)' not in log.read_text():
            contents = log.read_text()
            if f'done {TAG} (exit ' in contents or f'STOP {TAG}' in contents:
                raise RuntimeError('Depth4 failed; preserve its logs and review')
            if time.monotonic() > deadline:
                raise TimeoutError('Depth4 did not finish within its original timeout')
            time.sleep(10)
        # Wait for the original runner to exit before retiring its parent.
        while True:
            children = (proc / 'task' / str(pid) / 'children').read_text().split()
            if not children or all('Z' in Path(f'/proc/{c}/status').read_text().split('State:')[1].splitlines()[0] for c in children):
                break
            time.sleep(1)
        out = ROOT / 'experiments/results/aws_20260929' / TAG
        meta = json.loads((out / 'provenance.json').read_text())
        if meta['status'] != 'completed':
            raise RuntimeError('Depth4 completion provenance required')
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        while proc.exists():
            time.sleep(1)
        progress_path = ROOT / 'experiments/queue/aws_progress_20260929.json'
        progress = json.loads(progress_path.read_text())
        queue = ROOT / f'experiments/queue/{TAG}.txt'
        progress[TAG] = dict(status='completed', exit_code=0, wall_s=meta['wall_s'],
                             wall_scope='Benchmark wrapper wall; original runner completed successfully',
                             command=queue.read_text().strip().split(' ', 1)[1],
                             limits=meta['limits'], peak_rss_kb=meta['peak_rss_kb'])
        progress_path.write_text(json.dumps(progress, indent=2) + '\n')
        files = [str(queue.relative_to(ROOT)), str(progress_path.relative_to(ROOT))]
        files += [str(p.relative_to(ROOT)) for p in out.glob('*.json')]
        run('git', 'add', '--', *files)
        run('git', 'commit', '--only', '-m', 'Record completed depth4 before AWS fast matrix', '--', *files)
        run('git', 'pull', '--rebase')
        run('git', 'push', 'origin', 'main')
        frozen()
        common = [sys.executable, 'scripts/run_research_gym.py', '--manifest', PLAN,
                  '--rss-mib', '2441', '--vms-mib', '3907']
        live['status'] = 'running_smokes'; save()
        run(*common, '--smokes-only')
        smoke_state = json.loads((ROOT / Path(PLAN).parent / 'worker_all_attempt1.status.json').read_text())
        if smoke_state['status'] != 'completed':
            raise RuntimeError('Every smoke must complete before pilots')
        results = [json.loads((ROOT / name).read_text()) for name in smoke_state['completed']]
        peaks = [r['max_rss_kb'] for r in results if 'max_rss_kb' in r]
        live['smoke_peak_rss_kb'] = max(peaks, default=0)
        if live['smoke_peak_rss_kb'] > 2000000:
            raise RuntimeError('Smoke memory near cap; review before pilots')
        publish_matrix(smoke_state, 'Preserve completed AWS fast matrix contracts and smokes')
        frozen()
        live['status'] = 'running_pilots'; save()
        run(*common, '--attempt', '2')
        pilot_state = json.loads((ROOT / Path(PLAN).parent / 'worker_all_attempt2.status.json').read_text())
        publish_matrix(pilot_state, 'Publish completed AWS fast matrix pilot evidence')
        live['status'] = 'completed'; save()
    except Exception as error:
        live.update(status='needs_review', error=str(error)); save()
        raise


if __name__ == '__main__':
    main()
