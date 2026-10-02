"""Drain language smokes; replace coordinator to confirm depth8 before10M jobs."""
import json,os,signal,time
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
for family in ('private','depth'):
    p=ROOT/f'experiments/results/aws_depth8_language/aws_depth8_language_20261002T234100Z_{family}_smoke_s7.json'
    while not p.exists():time.sleep(10)
    r=json.loads(p.read_text())
    if r['status']!='completed':raise RuntimeError('Smoke incomplete')
# The old admission coordinator is suspended; guards/trainers finish normally.
time.sleep(3)
os.kill(927989,signal.SIGKILL)
subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/run_aws_depth8_replay_matrix.py','--manifest','experiments/queue/aws_depth8_confirm_language_20261002T234300Z/manifest.json','--jobs','3'],cwd=ROOT,check=True)
