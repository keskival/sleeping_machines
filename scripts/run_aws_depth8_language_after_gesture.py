"""Wait for the reserved gesture matrix to drain, then admit guarded language."""
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'experiments/queue/aws_depth8_replay_20261002T235500Z/worker_recovery.status.json'
while True:
    state=json.loads(STATE.read_text())
    if state['status']=='completed':break
    if state['status']=='needs_review':raise RuntimeError('Review gesture lifecycle before starting language')
    time.sleep(10)
subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/run_aws_depth8_replay_matrix.py','--manifest','experiments/queue/aws_depth8_language_20261002T234100Z/manifest.json','--jobs','2'],cwd=ROOT,check=True)
