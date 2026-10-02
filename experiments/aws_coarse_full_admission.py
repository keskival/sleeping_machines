"""Require both prescribed exploratory seed gates before full native fits."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_aws_matrix_recovery import validate_result
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()
rows=[]
for prefix in ('aws_coarse_native_20261002T212600Z','aws_coarse_native_replication_20261002T212900Z'):
    plan=json.loads((ROOT/f'experiments/gym/plans/{prefix}/manifest.json').read_text())
    result=json.loads((ROOT/f'experiments/results/diagnostics/{prefix}_analysis.json').read_text())
    assert result['status']=='completed' and sorted(result['passing_variants'])==['coarse_fastclock','coarse_matchedclock']
    for job in plan['jobs']:
        if job['stage']=='pilot':validate_result(ROOT,job)
    rows.append(result)
plan=json.loads((ROOT/a.manifest).read_text());out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
out.write_text(json.dumps(dict(status='completed',args=vars(a),screens=rows,source_sha256=plan['source_sha256'],scope='Full-data admission after two completed fixed quality/work screens; no new quality score.'),indent=2)+'\n')
