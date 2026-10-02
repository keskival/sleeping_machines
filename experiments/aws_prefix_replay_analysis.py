"""Fail closed unless paired completed replay implementations agree."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_aws_matrix_recovery import validate_result
import dvs_native_contracts as E
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args()
plan=json.loads((ROOT/a.manifest).read_text());jobs=[j for j in plan['jobs'] if j['stage']=='smoke']
assert len(jobs)==2
results=[validate_result(ROOT,j) for j in jobs];full,reuse=results
for key in ('data','curve','final','selected_epoch'):E.equal(full[key],reuse[key])
for key in ('epochs','seed','fit','dev','payload','depth','heads','pool','lr','update_targets'):
    assert full['args'][key]==reuse['args'][key]
checkpoints=[]
for j in jobs:checkpoints.append(torch.load(ROOT/Path(j['result']).with_suffix('.progress.pt'),weights_only=False))
for key in ('online_model','optimizer','best_state','best','cursor','torch_rng'):E.equal(checkpoints[0][key],checkpoints[1][key])
rows=[]
for j,r in zip(jobs,results):
    work=r['work'];rows.append(dict(variant=j['variant'],result=j['result'],result_sha256=hashlib.sha256((ROOT/j['result']).read_bytes()).hexdigest(),accuracy=r['final']['accuracy'],nll=r['final']['nll'],whole_fit_gflops=work['whole_fit_unit_special_flops_estimate']/1e9,fit_mflops_per_target=work['fit_unit_special_flops_per_target_estimate']/1e6,inference_mflops_per_target=work['inference_unit_special_flops_per_target_estimate']/1e6,wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb'],activity=r['activity'],available_receivers=work['native_available_receivers']))
ratio=rows[1]['whole_fit_gflops']/rows[0]['whole_fit_gflops']
out=ROOT/'experiments/results/diagnostics'/(plan['run_prefix']+'_analysis.json')
assert not out.exists()
out.write_text(json.dumps(dict(status='completed',rows=rows,fit_work_ratio=ratio,all_curves_predictions_model_adam_cursor_rng_identical=True,scope='24fit/8dev/two-pass matched implementation smoke; no broad quality or model supremacy claim. Window-traced unit-special work convention; no measured energy.'),indent=2)+'\n')
lines=['# Completed exact prefix-replay saving','',f'Fitting work ratio {ratio:.6f}; {100*(1-ratio):.3f}% lower fitting work with identical entire curves, predictions, model weights, Adam and recovery state.','', '|Implementation|Accuracy|NLL|Whole fit GFLOPs|Fit MFLOPs/target|Inference MFLOPs/target|Wall seconds|Peak RSS KiB|','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"|{r['variant']}|{r['accuracy']:.3%}|{r['nll']:.6f}|{r['whole_fit_gflops']:.6f}|{r['fit_mflops_per_target']:.6f}|{r['inference_mflops_per_target']:.6f}|{r['wall_s']:.3f}|{r['max_rss_kb']}|")
lines+=['','Same24 real fitting gestures/two passes (48 targets),8 development, p16/L2/H2/pool2. Selected updates and scored keys below include alternate simulation; hard inference remains unchanged. Source/data hashes and executed stage ledgers accompany raw results. Wall time is one concurrent-host observation; the saving claim is charged arithmetic equivalence, not a general latency/energy claim. Snapshot/copy memory traffic is not measured. Initial small contract fits and failed/research work are distinct from these per-fit totals.','', 'Scope: improving the integrated actual-write learning implementation over its full-replay reference, with exact empirical quality/update retention. No claim over strongest gesture controls. Other hosts own substantive credit replication and capacity fitting; this optimization can support them without altering the estimator.']
(ROOT/'experiments/AWS_PREFIX_REPLAY_FINDINGS_20261002.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(ratio=ratio,rows=rows),indent=2))
