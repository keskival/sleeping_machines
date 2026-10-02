"""Fixed quadratic native quality/work decision against saved nested control."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_aws_matrix_recovery import validate_result
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();plan=json.loads((ROOT/a.manifest).read_text())
job=next(j for j in plan['jobs'] if j['stage']=='pilot');new=validate_result(ROOT,job);seed=new['args']['seed']
prefix='aws_coarse_native_20261002T212600Z' if seed==6 else 'aws_coarse_native_replication_20261002T212900Z'
tag=f'{prefix}_coarse_matchedclock_s{seed}';old=validate_result(ROOT,dict(tag=tag,result=f'experiments/results/dvs_native/{tag}.json'))
assert new['data']==old['data']
for key in ('fit','dev','epochs','update_targets','seed','bins','clock_step','payload','depth','heads','pool','lr','terminal_risk'):assert new['args'][key]==old['args'][key]
fit_ratio=new['work']['whole_fit_unit_special_flops_estimate']/old['work']['whole_fit_unit_special_flops_estimate'];infer_ratio=new['work']['inference_unit_special_flops_per_target_estimate']/old['work']['inference_unit_special_flops_per_target_estimate'];nll_gain=old['final']['nll']-new['final']['nll'];acc_gain=new['final']['accuracy']-old['final']['accuracy']
rows=[]
for name,r in [('affine',old),('quadratic',new)]:
 w=r['work'];rows.append(dict(variant=name,seed=seed,accuracy=r['final']['accuracy'],nll=r['final']['nll'],parameters=r['parameters'],whole_fit_gflops=w['whole_fit_unit_special_flops_estimate']/1e9,fit_mflops_per_target=w['fit_unit_special_flops_per_target_estimate']/1e6,inference_mflops_per_target=w['inference_unit_special_flops_per_target_estimate']/1e6,available_receivers=w['native_available_receivers'],activity=r['activity'],wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']))
summary=dict(status='completed',args=vars(a),rows=rows,nll_gain=nll_gain,accuracy_gain=acc_gain,fit_work_ratio=fit_ratio,inference_work_ratio=infer_ratio,gate_pass=nll_gain>=.05 and acc_gain>=-.01 and fit_ratio<=1.25 and infer_ratio<=1.5,scope='Small integrated degree2 head screen; same native temporal/race/state paths, different local readout. Same256fit/192dev/4passes, initial logits/original gradients nested. Fixed local teacher remains approximate. No fresh confirmation or supremacy claim.')
out=ROOT/'experiments/results/diagnostics'/(plan['run_prefix']+'_analysis.json');assert not out.exists();out.write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Native quadratic local readout screen','',f"Frozen gate {'PASS' if summary['gate_pass'] else 'FAIL'}: NLL gain {nll_gain:.6f}, accuracy gain {acc_gain*100:.3f}pp, fitting ratio {fit_ratio:.6f}, inference ratio {infer_ratio:.6f}.",'', '|Readout|Seed|Accuracy|NLL|Parameters|Whole-fit GFLOPs|Fit MFLOPs/target|Inference MFLOPs/target|','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"|{r['variant']}|{r['seed']}|{r['accuracy']:.3%}|{r['nll']:.6f}|{r['parameters']}|{r['whole_fit_gflops']:.6f}|{r['fit_mflops_per_target']:.6f}|{r['inference_mflops_per_target']:.6f}|")
lines+=['',summary['scope'],'All4 passes paid;1024 fitting targets,8 available receivers,4 selected writes/event,5events/query. New5808 weights pay features/contractions/backward/Adam. Inference pays5 head evaluations per clip. Shared preprocessing wall included, NumPy preprocessing FLOPs/traffic/energy unknown. More powerful probe results are separate diagnostic controls, not installed prediction scores. Failed gates block unchanged escalation; passing firstseed nominates unchangedseed7 only.']
(ROOT/'experiments'/(plan['run_prefix']+'_FINDINGS.md')).write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
