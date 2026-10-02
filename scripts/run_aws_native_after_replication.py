"""Follow the reserved replication chain with fresh-data native confirmation."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'experiments/gym/plans/aws_native_confirmation_20261002T010500Z/manifest.json'
PREVIOUS = 'experiments/gym/plans/aws_event_replication_20261002T005408Z/completed_summary.json'
PID = 774142


def main():
    directory = (ROOT / PLAN).parent
    state = directory / 'worker_native_admission.status.json'
    if state.exists():
        raise ValueError('Never duplicate an admission supervisor')
    live = dict(status='waiting_for_reserved_replication', predecessor_pid=PID)
    def save():
        temp = state.with_suffix('.tmp')
        temp.write_text(json.dumps(live, indent=2)+'\n'); temp.replace(state)
    def run(*command):
        subprocess.run(command, cwd=ROOT, check=True)
    save()
    try:
        proc = Path(f'/proc/{PID}')
        if proc.exists():
            if b'scripts/run_aws_confirmation_then_replication.py' not in (proc/'cmdline').read_bytes():
                raise ValueError('Unexpected predecessor process')
            identity = (proc/'stat').read_text().split()[21]
            while proc.exists():
                try:
                    fields = (proc/'stat').read_text().split()
                except FileNotFoundError:
                    break
                if fields[21] != identity:
                    raise ValueError('Predecessor PID reused')
                if fields[2] == 'Z':
                    break
                time.sleep(10)
        if json.loads((ROOT/PREVIOUS).read_text())['status'] != 'completed':
            raise ValueError('Completed reserved replications required')
        live['status']='running_native_confirmation';save()
        run(sys.executable,'scripts/run_aws_matrix_recovery.py','--manifest',PLAN,'--jobs','3')
        if json.loads((directory/'completed_summary.json').read_text())['status'] != 'completed':
            raise ValueError('Complete confirmation results required')
        output = str((directory/'confirmation_analysis.json').relative_to(ROOT))
        run(sys.executable,'-m','experiments.split_confirmation_analysis','--manifest',PLAN,'--output',output)
        run('git','add',output)
        run('git','commit','-m','Publish prespecified fresh-data native confirmation analysis')
        run('git','pull','--rebase');run('git','push')
        live['status']='completed';save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save()
        raise


if __name__=='__main__':
    main()
