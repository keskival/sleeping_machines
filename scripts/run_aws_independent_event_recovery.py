"""Drain preserved anomalous control, then continue independent event protocols."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REPLICATION='experiments/gym/plans/aws_event_replication_20261002T005408Z/manifest.json'
NATIVE='experiments/gym/plans/aws_native_confirmation_20261002T010500Z/manifest.json'


def main():
    state=(ROOT/REPLICATION).parent/'worker_independent_recovery.status.json'
    if state.exists():raise ValueError('Never duplicate recovery supervisor')
    live=dict(status='waiting_for_control_drain',predecessor_pid=770997)
    def save():
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps(live,indent=2)+'\n');temp.replace(state)
    def run(*args):subprocess.run(args,cwd=ROOT,check=True)
    save()
    try:
        proc=Path('/proc/770997')
        if proc.exists():
            if b'aws_banknote_confirmation_20261001T234000Z/manifest.json' not in (proc/'cmdline').read_bytes():
                raise ValueError('Unexpected predecessor')
            identity=(proc/'stat').read_text().split()[21]
            while proc.exists():
                try:fields=(proc/'stat').read_text().split()
                except FileNotFoundError:break
                if fields[21]!=identity:raise ValueError('PID reused')
                if fields[2]=='Z':break
                time.sleep(10)
        prior=json.loads((ROOT/'experiments/gym/plans/aws_banknote_confirmation_20261001T234000Z/worker_recovery.status.json').read_text())
        if prior['status']!='needs_review':raise ValueError('Preserved control failure required')
        live.update(status='running_independent_replication',banknote_error=prior.get('error'));save()
        run(sys.executable,'scripts/run_aws_matrix_recovery.py','--manifest',REPLICATION,'--jobs','3')
        if json.loads(((ROOT/REPLICATION).parent/'completed_summary.json').read_text())['status']!='completed':
            raise ValueError('Complete replication required')
        live['status']='running_native_confirmation';save()
        run(sys.executable,'scripts/run_aws_matrix_recovery.py','--manifest',NATIVE,'--jobs','3')
        output=str(((ROOT/NATIVE).parent/'confirmation_analysis.json').relative_to(ROOT))
        run(sys.executable,'-m','experiments.split_confirmation_analysis','--manifest',NATIVE,'--output',output)
        run('git','add',output);run('git','commit','-m','Publish complete independent native confirmation analysis')
        run('git','pull','--rebase');run('git','push')
        live['status']='completed_banknote_still_incomplete';save()
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise


if __name__=='__main__':main()
