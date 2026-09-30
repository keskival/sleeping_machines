"""Separate online pilot, then the larger-message ladder; one guarded job at a time."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM='local_integrated_followups_20260930T200000Z'
ONLINE='local_integrated_online_backbone_D8192_20260930T200000Z'


def main():
    env=dict(os.environ,MEM_CAP_KB='4000000',MEM_CAP_RSS_KB='2500000',MIN_AVAIL_MB='8192',
        JOB_TIMEOUT_S='7200',WAIT='1',WAIT_TIMEOUT_S='86400')
    status=ROOT/f'experiments/queue/{STEM}_online_status.json'
    completed=ROOT/f'experiments/results/online_language/{ONLINE}.json'
    rc=subprocess.run(['bash','experiments/queue/run_safe.sh',f'experiments/queue/{ONLINE}.txt'],cwd=ROOT,env=env).returncode
    published=False
    if rc==0 and completed.exists():
        published=subprocess.run([sys.executable,'scripts/update_integrated_online_report.py',
            str(completed.relative_to(ROOT))],cwd=ROOT).returncode==0
    status.write_text(json.dumps(dict(status='completed' if rc==0 else 'needs_review',
        guarded_exit_code=rc,report_published=published,result=str(completed.relative_to(ROOT))),indent=2)+'\n')
    # Online adaptation is independent of the from-scratch capacity question.
    # A failed pilot/report hook preserves its evidence; the larger ladder continues.
    raise SystemExit(subprocess.run([sys.executable,'scripts/run_full_sparse_ladder.py',
        f'experiments/queue/{STEM}.json'],cwd=ROOT,env=env).returncode)


if __name__=='__main__':main()
