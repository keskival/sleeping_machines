"""Publish completed gym evidence while preserving the running guarded worker."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'experiments/gym/plans/aws_fast_matrix_v1_20261001T213000Z/manifest.json'
OWNER = 693768


def main():
    os.chdir(ROOT)
    owner = Path(f'/proc/{OWNER}')
    identity = (owner / 'stat').read_text().split()[21]
    jobs = json.loads(PLAN.read_text())['jobs']
    while owner.exists():
        if (owner / 'stat').read_text().split()[21] != identity:
            raise RuntimeError('Supervisor identity changed')
        # Only intervene while the supervisor waits for a gym worker, never
        # during its own git operations or between worker phases.
        children = (owner / 'task' / str(OWNER) / 'children').read_text().split()
        worker_active = any(Path(f'/proc/{pid}/cmdline').exists() and
                            b'scripts/run_research_gym.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
                            for pid in children)
        if not worker_active:
            time.sleep(2)
            continue
        pending = []
        for job in jobs:
            path = ROOT / job['result']
            if not path.exists():
                continue
            if not subprocess.check_output(['git', 'status', '--porcelain', '--', job['result']], text=True).strip():
                continue
            result = json.loads(path.read_text())
            if result.get('status') != 'completed' or result['args']['tag'] != job['tag']:
                raise RuntimeError('Invalid completed result: ' + job['result'])
            for name, sha in result.get('source_sha256', {}).items():
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                    raise RuntimeError('Result source differs: ' + name)
            pending.append(job)
        if pending:
            # Reserve only the publication supervisor. Trainer/guard/watchdog
            # continue, and phase publication cannot race these git operations.
            os.kill(OWNER, signal.SIGSTOP)
            try:
                children = (owner / 'task' / str(OWNER) / 'children').read_text().split()
                if any(Path(f'/proc/{pid}/cmdline').exists() and
                       b'git' in Path(f'/proc/{pid}/cmdline').read_bytes() for pid in children):
                    continue
                for job in pending:
                    path = job['result']
                    if not subprocess.check_output(['git', 'status', '--porcelain', '--', path], text=True).strip():
                        continue
                    subprocess.run(['git', 'add', '--', path], check=True)
                    subprocess.run(['git', 'commit', '--only', '-m', 'Record completed ' + job['tag'], '--', path], check=True)
                    subprocess.run(['git', 'pull', '--rebase'], check=True)
                    subprocess.run(['git', 'push', 'origin', 'main'], check=True)
            finally:
                if owner.exists():
                    os.kill(OWNER, signal.SIGCONT)
        time.sleep(2)


if __name__ == '__main__':
    main()
