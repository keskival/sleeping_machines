"""Wait for driver numerical admission then run source-scoped guarded language matrix."""
from pathlib import Path
import json,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'experiments/results/diagnostics/aws_language_credit_driver_contracts_20261003T013300Z.json'
while not p.exists():
    log=ROOT/'experiments/queue/runner_aws_language_credit_driver_contracts_20261003T013300Z.out'
    if log.exists() and 'exit 1)' in log.read_text():raise RuntimeError('Driver contracts failed; review required')
    time.sleep(5)
assert json.loads(p.read_text())['contracts_passed']
# Wait for safe queue to release its ordinary host lock after result publication.
time.sleep(3)
subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/run_aws_depth8_replay_matrix.py','--manifest','experiments/queue/aws_language_credit_matrix_20261003T013500Z/manifest.json','--jobs','3'],cwd=ROOT,check=True)
