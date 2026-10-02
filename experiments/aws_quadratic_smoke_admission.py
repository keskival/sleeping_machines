"""Enforce fixed quadratic resource/learning smoke before the pilot."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_aws_matrix_recovery import validate_result
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();plan=json.loads((ROOT/a.manifest).read_text())
j=next(j for j in plan['jobs'] if j['stage']=='smoke');q=validate_result(ROOT,j)
name='experiments/results/dvs_native/aws_coarse_native_20261002T212600Z_coarse_matchedclock_smoke.json';ref=validate_result(ROOT,dict(tag='aws_coarse_native_20261002T212600Z_coarse_matchedclock_smoke',result=name))
assert q['small_fit_learning_passed'] and q['max_rss_kb']<900000 and q['data']==ref['data']
fit_ratio=q['work']['whole_fit_unit_special_flops_estimate']/ref['work']['whole_fit_unit_special_flops_estimate'];inference_ratio=q['work']['inference_unit_special_flops_per_target_estimate']/ref['work']['inference_unit_special_flops_per_target_estimate']
assert fit_ratio<=1.25 and inference_ratio<=1.5
out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists();out.write_text(json.dumps(dict(status='completed',args=vars(a),fit_work_ratio=fit_ratio,inference_work_ratio=inference_ratio,source_sha256=plan['source_sha256'],scope='Learning/resource smoke admission only; no quality claim.'),indent=2)+'\n')
