"""Completed nine-arm full-data comparison under frozen per-seed gates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_aws_matrix_recovery import validate_result
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args()
plan=json.loads((ROOT/a.manifest).read_text());jobs=[j for j in plan['jobs'] if j['stage']=='pilot'];assert len(jobs)==9
results={j['tag']:validate_result(ROOT,j) for j in jobs};rows=[];predictions={}
for j in jobs:
 r=results[j['tag']];seed=r['args']['seed'];f=next(v for v in results.values() if v['args']['seed']==seed and v['args']['bins']==20)
 assert r['args']['fit']==984 and r['args']['dev']==192 and r['args']['epochs']==8
 for key in ('seed','epochs','fit','dev','payload','depth','pool','heads','update_targets','lr','terminal_risk'):assert r['args'][key]==f['args'][key]
 for key in ('data_result_sha256','data_artifact_sha256','controls_sha256'):assert r['data'][key]==f['data'][key]
 w=r['work'];fw=f['work'];ratio=w['whole_fit_unit_special_flops_estimate']/fw['whole_fit_unit_special_flops_estimate']
 gain=f['final']['nll']-r['final']['nll'];acc=r['final']['accuracy']-f['final']['accuracy']
 rows.append(dict(variant=j['variant'],seed=seed,result=j['result'],result_sha256=hashlib.sha256((ROOT/j['result']).read_bytes()).hexdigest(),selected_epoch=r['selected_epoch'],accuracy=r['final']['accuracy'],nll=r['final']['nll'],nll_improvement=gain,accuracy_difference=acc,fit_work_ratio=ratio,gate_pass=r['args']['bins']!=20 and gain>=.02 and acc>=-.01 and ratio<=.5,whole_fit_gflops=w['whole_fit_unit_special_flops_estimate']/1e9,fit_mflops_per_target=w['fit_unit_special_flops_per_target_estimate']/1e6,inference_mflops_per_target=w['inference_unit_special_flops_per_target_estimate']/1e6,available_receivers=w['native_available_receivers'],activity=r['activity'],wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']))
 predictions[(j['variant'],seed)]=np.array(r['final']['probabilities'])
variants=('fine','coarse_fastclock','coarse_matchedclock');groups=[]
for v in variants:
 rs=[r for r in rows if r['variant']==v];assert len(rs)==3
 groups.append(dict(variant=v,mean_accuracy=float(np.mean([r['accuracy'] for r in rs])),mean_nll=float(np.mean([r['nll'] for r in rs])),mean_whole_fit_gflops=float(np.mean([r['whole_fit_gflops'] for r in rs])),all_seed_gate_pass=all(r['gate_pass'] for r in rs) if v!='fine' else None))
# Diagnostic dependence-aware uncertainty; not a new gate/confirmation protocol.
meta=json.loads((ROOT/'experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json').read_text());z=np.load(ROOT/meta['data_artifact'],allow_pickle=False);labels=z['dev_labels'];users=np.array([str(x).split('_')[0] for x in z['dev_ids']]);names=sorted(set(users));blocks=[np.flatnonzero(users==u) for u in names];rng=np.random.default_rng(82103);intervals=[]
for v in variants[1:]:
 nd=[];ad=[]
 for seed in (6,7,8):
  p0=predictions[('fine',seed)];p1=predictions[(v,seed)]
  nd.append(np.log(p1[np.arange(192),labels])-np.log(p0[np.arange(192),labels]))
  ad.append((p1.argmax(1)==labels).astype(float)-(p0.argmax(1)==labels).astype(float))
 nd=np.array(nd);ad=np.array(ad);samples=[]
 for _ in range(10000):
  seeds=rng.integers(0,3,3);chosen=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]);samples.append((nd[seeds][:,chosen].mean(),ad[seeds][:,chosen].mean()))
 sample=np.array(samples);intervals.append(dict(variant=v,mean_nll_gain=float(nd.mean()),mean_accuracy_gain=float(ad.mean()),nll_gain_98_75_interval=np.quantile(sample[:,0],[.00625,.99375]).tolist(),accuracy_gain_98_75_interval=np.quantile(sample[:,1],[.00625,.99375]).tolist(),scope='Diagnostic crossed3 fitted seeds/development-user cluster bootstrap,10000 draws; Bonferroni4 contrasts. Few reused users, not independent confirmation. Not used to alter frozen gates.'))
controls=json.loads((ROOT/'experiments/results/diagnostics/aws_gesture_resolution_20261002T212000Z.json').read_text())
references=[dict(bins=r['bins'],accuracy=r['final']['accuracy'],nll=r['final']['nll'],model_storage_bytes=r['model_storage_bytes'],sequential_median_ms=r['sequential_median_ms_per_query'],fitting_flops=None,scope='RBF SVC selected on fitting-user GroupKFold NLL;984fit/192dev, kernel search+27fold/3final fits retained, solver FLOPs unknown.') for r in controls['rows']]
summary=dict(status='completed',args=vars(a),rows=rows,groups=groups,uncertainty=intervals,strong_controls=references,passing_variants=[g['variant'] for g in groups if g['all_seed_gate_pass']],scope='Full-data developmental architecture comparison. All eight passes/7872 fit targets and per-seed curves retained. Native arithmetic+unit-special estimates, preprocessing wall included but NumPy FLOPs/traffic/energy unmeasured. No official-test/fresh-confirmation or broad supremacy claim.')
out=ROOT/'experiments/results/diagnostics'/(plan['run_prefix']+'_analysis.json');assert not out.exists();out.write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Completed full-data native coarse comparison','','Same984 fitting gestures,8passes/7872 fitting targets,192 development,U16,Adam.003,clip1,p16/L2/H2/pool2. All seed rows/fixed gates below; no best-seed or ensemble selection.','','|Variant|Seed|Accuracy|NLL|Whole-fit GFLOPs|Fit MFLOPs/target|Inference MFLOPs/target|Gate|','|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:lines.append(f"|{r['variant']}|{r['seed']}|{r['accuracy']:.3%}|{r['nll']:.6f}|{r['whole_fit_gflops']:.6f}|{r['fit_mflops_per_target']:.6f}|{r['inference_mflops_per_target']:.6f}|{'reference' if r['variant']=='fine' else 'PASS' if r['gate_pass'] else 'FAIL'}|")
lines+=['','Strong fitting-user-selected RBF controls on the SAME984fit/192dev data:']
for r in references:lines.append(f"- {r['bins']}bins: {r['accuracy']:.3%},NLL {r['nll']:.6f}; serialized {r['model_storage_bytes']}bytes, sequential {r['sequential_median_ms']:.6f}ms/query. Solver fitting/inference FLOPs unknown.")
lines+=['',summary['scope'],'','Packets:4x250ms+1squery or20x50ms+1squery, all original raw counts retained, fine within-quarter timing discarded.8 available receivers and4 selected writes per observed event. No extra dense carrier/decoder or teacher substitution. Fixed per-seed advancement gate: >=.02NLL improvement, <=1pp accuracy decline, <=.50 native-fit work. Diagnostic intervals and all actual activity/wall/RSS remain in raw analysis. Successful screens do not erase a failed full-data seed. No automatic extension after a failure.']
(ROOT/'experiments/AWS_FULL_COARSE_FINDINGS_20261002.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(groups=groups,passing=summary['passing_variants']),indent=2))
