"""Wait for immutable AWS controls, validate lineage, publish fixed calibration."""
import json
import math
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_aws_matrix_recovery import validate_result
plan_path = Path(sys.argv[1])
plan = json.loads(plan_path.read_text())
state = plan_path.parent / 'worker_recovery.status.json'
while True:
    status = json.loads(state.read_text())['status'] if state.exists() else 'waiting'
    if status == 'needs_review':
        raise RuntimeError('Preserve failed battery; no completed calibration')
    running = False
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if b'scripts/run_aws_matrix_recovery.py' in args and str(plan_path).encode() in args:
            running = True
    if status == 'completed' and not running:
        break
    time.sleep(10)
table_job = next(j for j in plan['jobs'] if j['variant'] == 'table')
table = validate_result(ROOT, table_job)
bar = table['final']['confirmation']['accuracy']
rows = []
checkpoints = []
for job in plan['jobs']:
    if job['stage'] != 'pilot':
        continue
    result = validate_result(ROOT, job)
    assert result['data_sha256']['fit'] == table['data_sha256']['fit']
    assert result['data_sha256']['dev'] == table['data_sha256']['development']
    assert result['data_sha256']['confirmation'] == table['data_sha256']['confirmation']
    work = result['work']
    rows.append(dict(model=job['variant'], seed=job['seed'], result=job['result'],
        accuracy=result['final']['confirmation']['accuracy'], nll=result['final']['confirmation']['nll'],
        parameters=result['parameters'], selected_epoch=result['selected_epoch'],
        fitting_gflops=work['total_training_unit_special_flops']/1e9,
        fitting_mflops_per_query=work['fit_mflops_per_query'],
        inference_mflops_per_query=work['inference_unit_special_flops_per_query']/1e6,
        calibration_pass=result['final']['confirmation']['accuracy'] >= bar + .2,
        wall_s=result['wall_s'], max_rss_kb=result['max_rss_kb']))
    checkpoints.append(result['checkpoint']['path'])
assert len(rows) == 6
summary = dict(status='completed', table_accuracy=bar, threshold=bar+.2, rows=rows,
    all_seeds_pass=all(r['calibration_pass'] for r in rows),
    scope='Fixed base calibration. Unit-special convention; sampled-window fitting estimates, not energy. No history ladder admitted automatically.')
out = ROOT / 'experiments/results/diagnostics' / (plan['run_prefix'] + '_analysis.json')
out.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
note = ROOT / 'experiments' / (plan['run_prefix'] + '_FINDINGS.md')
lines = ['# Completed AWS joint-event controls', '', f'Current v2 table accuracy {bar:.6%}; fixed calibration threshold {bar+.2:.6%}.', '',
    '|Control|Seed|Accuracy|NLL|Whole-fit GFLOPs|Fit MFLOPs/query|Inference MFLOPs/query|Calibration|',
    '|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
    lines.append(f"|{r['model']}|{r['seed']}|{r['accuracy']:.6%}|{r['nll']:.6f}|{r['fitting_gflops']:.6f}|{r['fitting_mflops_per_query']:.6f}|{r['inference_mflops_per_query']:.6f}|{'PASS' if r['calibration_pass'] else 'FAIL'}|")
lines += ['', summary['scope'], 'Controls remain diagnostic; no native or overall supremacy claim. All three seeds reported, no best-seed selection. Failed prerequisites remain archived in preceding plans. Selected predictors saved; no confirmation tuning.']
note.write_text('\n'.join(lines)+'\n')
files = [str(out.relative_to(ROOT)), str(note.relative_to(ROOT)), *checkpoints]
subprocess.run(['git','add','-f','--',*files],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Record completed AWS joint-event calibration and selected predictors','--',*files],cwd=ROOT,check=True)
subprocess.run(['git','pull','--rebase'],cwd=ROOT,check=True)
subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
