"""Fixed common-unit native resolution comparison, fail closed on mismatches."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_aws_matrix_recovery import validate_result
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args()
plan=json.loads((ROOT/a.manifest).read_text());jobs=[j for j in plan['jobs'] if j['stage']=='pilot'];results=[validate_result(ROOT,j) for j in jobs]
assert len(jobs)>=2
fine=next(r for r in results if r['args']['bins']==20)
rows=[]
for job,r in zip(jobs,results):
    for k in ('fit','dev','epochs','update_targets','payload','heads','depth','pool','seed','lr','terminal_risk'):assert r['args'][k]==fine['args'][k]
    for k in ('data_artifact_sha256','data_result_sha256','controls_sha256'):assert r['data'][k]==fine['data'][k]
    assert r['final']['targets']==fine['final']['targets']
    w=r['work'];fw=fine['work'];ratio=w['whole_fit_unit_special_flops_estimate']/fw['whole_fit_unit_special_flops_estimate']
    gain=fine['final']['nll']-r['final']['nll'];accgain=r['final']['accuracy']-fine['final']['accuracy']
    rows.append(dict(variant=job['variant'],seed=r['args']['seed'],result=job['result'],bins=r['args']['bins'],clock_step=r['args']['clock_step'],selected_epoch=r['selected_epoch'],accuracy=r['final']['accuracy'],nll=r['final']['nll'],whole_fit_gflops=w['whole_fit_unit_special_flops_estimate']/1e9,fit_mflops_per_target=w['fit_unit_special_flops_per_target_estimate']/1e6,inference_mflops_per_target=w['inference_unit_special_flops_per_target_estimate']/1e6,fit_work_ratio=ratio,nll_improvement=gain,accuracy_difference=accgain,gate_pass=r['args']['bins']!=20 and gain>=.02 and accgain>=-.01 and ratio<=.5,available_receivers=w['native_available_receivers'],activity=r['activity'],wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']))
out=ROOT/'experiments/results/diagnostics'/(plan['run_prefix']+'_analysis.json');assert not out.exists()
out.write_text(json.dumps(dict(status='completed',args=vars(a),rows=rows,passing_variants=[r['variant'] for r in rows if r['gate_pass']],scope='Fixed exploratory integrated screen, same data/passes/model, different packet precision. Whole fit arithmetic+unit specials estimates; coalescing/normalization/loading included in wall, NumPy preprocessing FLOPs/traffic/energy unmeasured. No strong-control superiority or fresh confirmation claim.'),indent=2)+'\n')
lines=['# Completed native temporal-resolution screen','','|Variant|Seed|Accuracy|NLL|Whole-fit GFLOPs|Fit MFLOPs/target|Inference MFLOPs/target|Gate|','|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:lines.append(f"|{r['variant']}|{r['seed']}|{r['accuracy']:.3%}|{r['nll']:.6f}|{r['whole_fit_gflops']:.6f}|{r['fit_mflops_per_target']:.6f}|{r['inference_mflops_per_target']:.6f}|{'reference' if r['bins']==20 else 'PASS' if r['gate_pass'] else 'FAIL'}|")
lines+=['','Same256 distinct fitting gestures,4passes/1024 fitting targets,192 development,p16/L2/H2/pool2,8 available receivers,4 selected updates/event. Coarse arms receive four250ms releases plus1s query; fine receives twenty50ms releases plus1s query. Actual scored keys and updates remain fully charged. Fine within-quarter timing is discarded. Old984-fit native/control references remain independent and stronger controls retain77.604%/.686661 at984fit. No fair quality claim from unequal data.','', 'Gate: >=.02 NLL improvement, <=1pp accuracy decline, <=.50 fitting-work ratio. All arms reported with whole-fit/per-target common units; NumPy preprocessing FLOPs/traffic and energy unknown. Full wall includes input coalescing/normalization and evaluations. Development selection over fixed4passes is exploratory. Passing arms nominate unchanged seed7 matched replication; failed arms do not escalate automatically.']
(ROOT/'experiments'/(plan['run_prefix']+'_FINDINGS.md')).write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
