"""Drain existing smokes before replacing an admission coordinator, preserving trainers."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
plan=json.loads((ROOT/'experiments/queue/aws_language_credit_matrix_20261003T013500Z/manifest.json').read_text())
for job in plan['jobs']:
    if job['stage']!='smoke':continue
    p=ROOT/job['result'];log=ROOT/'experiments/queue'/('runner_'+Path(job['queue']).stem+'.out')
    while not p.exists():
        if log.exists() and 'exit 1)' in log.read_text():raise RuntimeError('Original smoke failed; review first')
        time.sleep(5)
    if json.loads(p.read_text())['status']!='completed':raise RuntimeError('Completed smoke required')
# Guards release slot descriptors after writing completion markers.
time.sleep(3)
pid=int(Path('/tmp/aws_language_matrix_coordinator.pid').read_text());os.kill(pid,signal.SIGKILL)
subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/run_aws_depth8_replay_matrix.py','--manifest','experiments/queue/aws_language_winner_matrix_20261003T014100Z/manifest.json','--jobs','3'],cwd=ROOT,check=True)
