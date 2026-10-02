"""Publish complete prespecified confirmation analysis, then run event replications."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = 'experiments/gym/plans/aws_banknote_confirmation_20261001T234000Z/manifest.json'
ANALYSIS = 'experiments/results/diagnostics/aws_banknote_confirmation_20261001T234000Z_analysis.json'


def run(*args):
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predecessor-pid', required=True, type=int)
    args = parser.parse_args()
    manifest = ROOT / args.manifest
    state = manifest.parent / 'confirmation_admission.status.json'
    if state.exists():
        raise ValueError('Never duplicate a supervisor')
    live = dict(status='waiting', predecessor_pid=args.predecessor_pid)
    def save():
        temporary = state.with_suffix('.tmp')
        temporary.write_text(json.dumps(live, indent=2) + '\n')
        temporary.replace(state)
    save()
    try:
        proc = Path(f'/proc/{args.predecessor_pid}')
        if proc.exists():
            command = (proc / 'cmdline').read_bytes()
            if b'scripts/run_aws_matrix_recovery.py' not in command or PREVIOUS.encode() not in command:
                raise ValueError('Unexpected predecessor process')
            identity = (proc / 'stat').read_text().split()[21]
            while proc.exists():
                try:
                    fields = (proc / 'stat').read_text().split()
                except FileNotFoundError:
                    break
                if fields[21] != identity:
                    raise ValueError('Predecessor PID reused')
                if fields[2] == 'Z':
                    break
                time.sleep(10)
        previous = ROOT / PREVIOUS
        summary = json.loads((previous.parent / 'completed_summary.json').read_text())
        if summary['status'] != 'completed':
            raise ValueError('Complete confirmation battery required')
        live['status'] = 'analyzing_confirmation'; save()
        if (ROOT / ANALYSIS).exists():
            raise ValueError('Existing analysis requires reconciliation; never overwrite')
        run(sys.executable, 'experiments/tabular_confirmation_analysis.py',
            '--manifest', PREVIOUS, '--output', ANALYSIS)
        run('git', 'add', ANALYSIS)
        run('git', 'commit', '-m', 'Publish prespecified paired banknote confirmation analysis')
        run('git', 'pull', '--rebase')
        run('git', 'push')
        live['status'] = 'running_event_replication'; save()
        run(sys.executable, 'scripts/run_aws_matrix_recovery.py', '--manifest', args.manifest, '--jobs', '3')
        live['status'] = 'completed'; save()
    except Exception as error:
        live.update(status='needs_review', error=str(error)); save()
        raise


if __name__ == '__main__':
    main()
