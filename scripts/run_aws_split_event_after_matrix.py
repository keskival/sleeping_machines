"""Wait for preserved first-wave jobs, then admit the independent next battery."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predecessor-pid', type=int, required=True)
    args = parser.parse_args()
    manifest = ROOT / args.manifest
    plan = json.loads(manifest.read_text())
    state = manifest.parent / 'worker_admission.status.json'
    if state.exists():
        raise ValueError('Never duplicate a waiting supervisor')
    live = dict(status='waiting_for_previous_matrix', predecessor_pid=args.predecessor_pid)
    def save():
        temporary = state.with_suffix('.tmp')
        temporary.write_text(json.dumps(live, indent=2) + '\n'); temporary.replace(state)
    save()
    try:
        proc = Path(f'/proc/{args.predecessor_pid}')
        if proc.exists():
            command = (proc / 'cmdline').read_bytes()
            if b'scripts/run_aws_matrix_recovery.py' not in command or b'aws_fast_matrix_recovery_20261001T213409Z' not in command:
                raise ValueError('Unexpected predecessor process')
            identity = (proc / 'stat').read_text().split()[21]
            while proc.exists():
                if (proc / 'stat').read_text().split()[21] != identity:
                    raise ValueError('Predecessor PID reused')
                if 'Z' in (proc / 'status').read_text().split('State:')[1].splitlines()[0]:
                    break
                time.sleep(10)
        previous = json.loads((ROOT / plan['predecessor']).read_text())
        if previous['status'] not in ('completed', 'needs_review'):
            raise ValueError('Predecessor has no terminal lifecycle')
        # These new cells have their own contracts and smokes; a preserved
        # failure in an unrelated wine-regression cell is no prerequisite.
        live.update(status='starting_independent_next_battery', predecessor_status=previous['status'],
                    predecessor_error=previous.get('error')); save()
        rc = subprocess.run([sys.executable, 'scripts/run_aws_matrix_recovery.py',
                             '--manifest', args.manifest, '--jobs', '3'], cwd=ROOT).returncode
        if rc:
            raise RuntimeError(f'Next battery stopped for review with exit {rc}')
        live['status'] = 'completed'; save()
    except Exception as error:
        live.update(status='needs_review', error=str(error)); save()
        raise


if __name__ == '__main__':
    main()
