"""Require completed, learning, fully charged coarse native smokes before pilots."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_aws_matrix_recovery import validate_result
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()
plan=json.loads((ROOT/a.manifest).read_text());rows=[]
for job in plan['jobs']:
    if job['stage']!='smoke':continue
    result=validate_result(ROOT,job)
    assert result['small_fit_learning_passed'] and result['max_rss_kb']<900000
    assert result['args']['fit']==24 and result['args']['dev']==8 and result['args']['epochs']==2
    rows.append(dict(tag=job['tag'],max_rss_kb=result['max_rss_kb'],wall_s=result['wall_s'],work=result['work']))
assert len(rows)==3
out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
out.write_text(json.dumps(dict(status='completed',args=vars(a),rows=rows,source_sha256=plan['source_sha256'],scope='All three shape-specific learning and accounting smokes pass; admission only, not benchmark quality.'),indent=2)+'\n')
